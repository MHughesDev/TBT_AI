# TBT Pricing Model — Design & Findings

Source: `archive_08-03-26_cleaned.csv` — 6,892 rows, 42 columns, quotes dated
2023-12 through 2026-12.

---

## 1. What the data actually is

The unit of analysis is not "a tank." It's **a tank line on a revision of a quote.**
That distinction drives every other decision in this document.

| | |
|---|---|
| Rows | 6,892 |
| Unique Quote # | 3,018 |
| Quotes with >1 revision | 855 |
| Quotes with >1 distinct tank | 870 |
| Rows sharing an identical spec signature | 4,789 (69%) |

So roughly half the file is repeat observations. 1,355 quotes have multiple rows,
and among those, the price varies by a **median of 26.8%** across rows of the same
quote. Some of that is genuine revision (scope changed), some is multiple different
tanks under one quote number.

### The noise floor

Take every group of rows that share an identical spec signature — same diameter,
height, material, deck style, floor style, use type, quantity. Their prices differ
by a **median of 14.9%**.

That is the ceiling on how well *any* model can do using only those columns. Two
identical tanks got priced 15% apart, and nothing in the spec explains it. The
explanation lives in location, labor market, quote date, and whatever the estimator
was thinking that week.

This number is the single most useful thing in the audit. It's the benchmark. A model
at 12% isn't "only 88% accurate" — it's operating below the spec-level noise floor by
using the contextual columns to resolve part of that spread.

### Price drift is real and large

Median $/sq-ft of tank surface, by quarter of due date:

```
2024Q1  $55.6      2025Q1  $65.9      2026Q1  $68.9
2024Q2  $59.6      2025Q2  $62.5      2026Q2  $76.9
2024Q3  $71.2      2025Q3  $67.4      2026Q3  $72.2
2024Q4  $68.2      2025Q4  $68.3
```

Roughly +35% from early 2024 to mid 2026, and it isn't a size-mix artifact — median
diameter stays in the 31–34 ft band the whole time. Any model that ignores date will
learn a blurred average and be systematically low on new work. Dropping the time
feature costs 0.8 points of accuracy (12.7% → 13.5%), which understates it, because
cross-validation lets the model see both past and future. In the honest
forward-looking test it matters considerably more.

---

## 2. Leakage — the columns that must never be inputs

Verified arithmetically against the file:

```
Proposal Total = Material + Fabrication + Construction
               + Insulation Material + Insulation Construction + Tax
               (median absolute difference: exactly $0.00)

Total Price    = Proposal Total + Freight       (exactly $0.00)
```

These are not relationships to be learned. They're identities. Feed any component
price in as a feature and you get a model that reports 99% accuracy and adds nothing,
because it has been handed the answer.

The percentage columns are a subtler trap. `Margin (%)`, `Contingency (%)`,
`Insulation Margin (%)`, `Commission (%)` are **pricing policy decisions applied
during quoting** — they're outputs of the estimating process, not attributes of the
tank. Using them means your model can only score a quote that has already been built,
which defeats the purpose.

Hard exclusion list is coded into `tbt_model.py` as `BANNED`.

---

## 3. Validation design

This is where the project succeeds or fails, and it's worth being blunt: the naive
setup produces a number that is wrong by 55%.

| Scheme | MedAPE | What it actually measures |
|---|---|---|
| Random 5-fold | **8.2%** | Fiction. Revision 2 trains, revision 3 tests. |
| GroupKFold by Quote # | **12.7%** | Honest for a new revision of a known job. |
| GroupKFold by Company | **13.7%** | Honest for a brand-new customer. |
| Rolling origin (most recent 2 quarters) | **10.1%** | Honest for tomorrow's quote. |

Group by `Quote #` at minimum. Group by `Company Name` if you want to know how the
model behaves on a customer you've never quoted before.

### Rolling origin — train on the past, score the next quarter

```
2024Q4   n=603    17.8%
2025Q1   n=724    20.6%
2025Q2   n=768    15.1%
2025Q3   n=792    13.4%
2025Q4   n=686    12.5%
2026Q1   n=726     9.8%
2026Q2   n=703    10.0%
2026Q3   n=193    10.2%
```

This is the number to quote to anyone who asks how good it is. It improves steadily
as history accumulates and has settled around **10%** for the last three quarters.
That's the realistic deployment accuracy today, and it's better than the
cross-validated 12.7% because the recent model is trained on more, and more relevant,
data.

---

## 4. Model selection

All scored under GroupKFold by Quote #, target = `log(Total Price / Quantity)`.

| Model | MedAPE | ≤10% | ≤20% |
|---|---|---|---|
| Ridge on log features | 21.6% | 25% | 47% |
| RandomForest | 14.7% | 38% | 61% |
| ExtraTrees | 13.9% | 39% | 62% |
| HistGBM `lr=.10 it=300` | 13.3% | 41% | 65% |
| HistGBM `lr=.06 it=500` | 13.1% | 41% | 66% |
| **HistGBM `lr=.03 it=1200 leaf=15`** | **12.7%** | **42%** | **66%** |

`HistGradientBoostingRegressor` from scikit-learn wins and is the right choice for a
second reason: it's in scikit-learn, so there is no XGBoost or LightGBM install to
manage on a locked-down corporate ThinkPad. It handles categoricals natively and
missing values without imputation. Training takes about 20 seconds on CPU.

Slow-and-many beats fast-and-few here, which is typical for noisy targets — the low
learning rate keeps it from chasing individual weird quotes.

### Target formulation

| Formulation | MedAPE |
|---|---|
| Raw dollars | 14.8% |
| **log(unit price)** | **12.7%** |
| log($/sq ft), multiplied back by area | 12.5% |

Log is essential — prices span $5K to $5.9M, and in raw dollars the loss function is
dominated entirely by the handful of million-dollar tanks. Normalizing by surface area
first is a hair better and is philosophically nicer (you're modeling a rate, which is
how estimators actually think), but 0.2 points is inside the noise. The shipped model
uses plain log(unit price) for simplicity.

Note the division by `Quantity` — a 4-tank line item should not train the model to
think a tank costs 4×.

---

## 5. Features

Raw columns get you part of the way. The engineered geometry is what closes the gap,
because tank cost is fundamentally a materials-and-hours problem with known physics:

```python
shell_area  = π · D · H                  # plate area → fabrication hours
floor_area  = π · (D/2)²
total_area  = shell_area + 2.02 · floor_area
hoop        = D · H                      # hoop stress ∝ D·H → sets plate gauge
steel_proxy = shell_area · √(hoop)       # area × thickness ≈ steel weight
aspect      = H / D                      # tall-thin vs short-wide erection difficulty
```

`steel_proxy` is the important one. A tree model can approximate it from D and H given
enough splits, but handing it over directly means the model spends its capacity on the
things it can't derive.

### Ablation — drop one feature group at a time

| Removed | MedAPE | Cost |
|---|---|---|
| *(nothing — baseline)* | 12.7% | — |
| Seismic (Ss, S1) | 12.6% | −0.1 |
| Sales Manager, Bid Type | 12.8% | +0.1 |
| Distance to site | 12.8% | +0.1 |
| Time (months) | 13.5% | +0.8 |
| Wage type, Country, State | 14.6% | +1.9 |
| Deck/Floor/Material/Use Type | 15.2% | +2.5 |
| **Geometry** | **34.8%** | **+22.1** |

And the mirror image — each group *alone*:

| Only | MedAPE |
|---|---|
| Geometry | 32.3% |
| Labor/geography | 43.2% |
| Sales Manager / Bid Type | 42.6% |
| Seismic | 45.2% |
| Distance | 46.7% |
| Config categoricals | 48.0% |
| Time | 54.1% |

Geometry alone gets to 32%. Everything else alone is near-useless. But geometry plus
everything else gets to 12.7% — the context columns are worth 20 points *conditional
on* knowing the size. That's the interaction structure gradient boosting is good at
and Ridge is not, and it's exactly why Ridge sits at 21.6%.

Seismic contributing nothing is worth a sanity check with your engineers. Either it's
genuinely already captured by State (likely — Ss/S1 are geographic lookups), or the
values aren't being applied the way you'd expect.

---

## 6. Where the model is weak

MedAPE by segment, with signed bias (positive = model overprices):

**By material** — CS 12.5%, 304SS 15.4%, 316SS 18.0% (bias −6.9%). Stainless is
underrepresented (140 rows of 316SS) and the model underprices it. Don't use it for
316SS without a manual check.

**By country** — MX 8.0%, PE 10.6%, CL 12.0%, US 14.0%, CO 16.4%, CA 22.5%,
**AR 38.7% (bias −16.7%)**. Argentina is a disaster and should be excluded or handled
separately; 116 rows with severe inflation distortion is not something a tree model
will untangle. Canada at 22.5% is also shaky.

**By use type** — Fire Protection 10.1% (the best-covered segment, 2,993 rows),
Waste Water 16.5%, Industrial Storage Tank 18.8%.

**By bid type** — Firm 12.0%, Budget 15.4%. Expected; budget numbers are rougher
by construction.

**By wage type** — Prevailing Wage is the worst at 18.7%. Prevailing-wage jobs vary
by locality in ways a single categorical can't capture.

### Regression to the mean

| Price band | MedAPE | Bias |
|---|---|---|
| Bottom 17% | 14.8% | **+7.8%** |
| p17–33 | 10.8% | +3.5% |
| p33–50 | 11.9% | +0.8% |
| p50–67 | 12.1% | −0.5% |
| p67–83 | 12.5% | −4.1% |
| Top 17% | 15.1% | **−9.1%** |

The model overprices cheap tanks and underprices expensive ones. This is inherent
to squared-error boosting on a log target and it is not a bug you can fully fix — but
it means the model is least trustworthy exactly at the extremes, which is often where
the commercially interesting quotes live. Flag it in the UI.

---

## 7. Prediction intervals

A point estimate on a 12%-error model invites false confidence. Ship a range.

Naive quantile regression (HistGBM with `loss="quantile"`) is **badly miscalibrated**
here — the nominal P10–P90 band achieved only 56% empirical coverage out-of-fold.
Don't use it.

Split conformal prediction works. Hold out whole quotes, measure absolute residuals on
the log scale, take the appropriate percentile as a multiplicative band:

| Nominal | Empirical coverage | Band |
|---|---|---|
| 80% | 76% | ÷1.34 to ×1.34 |
| 90% | 86% | ÷1.47 to ×1.47 |

Coverage comes in a few points under nominal because the calibration set is grouped,
not exchangeable — acceptable, and conservative in the right direction if you widen
slightly. These bands are computed at training time and stored in the model bundle.

A ±34% band is wide. That is the honest width, and communicating it is more valuable
than a tight-looking number that's wrong a third of the time.

---

## 8. What this model cannot do

**It cannot model win rate.** Status breaks down as Deliver To Customer 3,345,
Revised 2,437, Sales 783, **Won 173, Lost 142**, Bid Review 9. Fewer than 5% of rows
have a resolved outcome. You can't train a win/loss classifier on 315 labeled examples
spread across 33 countries and 14 use types.

This is the most important limitation to communicate upward, because "AI pricing
model" will be heard as "tells us what price wins." It does not. It tells you **what
TBT would have quoted**, learned from what TBT has quoted. If your historical pricing
has a systematic error, the model reproduces it faithfully.

If win-rate modeling is the actual goal, the data collection problem comes first —
you need outcome labels on a meaningful fraction of quotes, plus ideally the competing
bid. That's a process change, not a modeling change.

**It cannot extrapolate.** 196 ft diameter is the largest tank in the file. Ask for
250 ft and a tree model returns something near its largest leaf, confidently and
wrongly. Guard the input range.

---

## 9. Does more data help?

| Quotes | Rows | MedAPE |
|---|---|---|
| 444 | 909 | 18.0% |
| 888 | 2,046 | 15.1% |
| 1,481 | 3,391 | 15.2% |
| 2,221 | 5,052 | 13.2% |
| 3,018 | 6,679 | 12.7% |

Still declining, but flattening — doubling the archive might buy 1–1.5 points. You're
approaching the 14.9% spec-level noise floor, and further gains have to come from
*new columns*, not more rows. The highest-value additions, roughly in order:

1. **Steel cost index at quote date** — would replace the crude `months` proxy with
   the actual driver of the 35% drift.
2. **Actual plate thickness / shell course schedule** — turns `steel_proxy` from an
   approximation into a real weight.
3. **Nozzle, manway, ladder, platform counts** — appendage count is a known cost
   driver and is entirely absent from this file.
4. **Foundation scope and site access** — likely a large chunk of the unexplained
   construction-price variance.
5. **Outcome labels** — unlocks a different and more valuable class of model entirely.

---

## 10. Deployment

**Retrain monthly.** Rolling-origin shows accuracy improves as recent history
accumulates, and drift is fast enough that a stale model degrades. It's a 20-second
job — put it on a scheduled task.

**Monitor in production.** Log every prediction alongside the price the estimator
actually used. If median signed error drifts off zero for two months running, the
market has moved and the retrain isn't keeping up.

**The right first use is a check, not an oracle.** `TBT_FLAG` in `excel_udf.py`
returns OK / LOW / HIGH by comparing a human-built quote against the 80% band. At 10%
median error the model is not good enough to set prices, but it's easily good enough
to catch the quote where someone fat-fingered a diameter or forgot the stainless
premium. That catches real money and carries almost no downside risk, and it builds
the track record you'd need before anyone would trust it further.

### Files

| File | Purpose |
|---|---|
| `tbt_model.py` | Feature engineering, training, validation, scoring. Run `train`, `check`, or `predict`. |
| `excel_udf.py` | xlwings bridge — `TBT_PRICE`, `TBT_RANGE`, `TBT_FLAG` as worksheet functions. |
| `tbt_pricing_model.joblib` | Trained bundle: model, encoder, conformal bands. 4.8 MB. |

```
pip install pandas scikit-learn joblib xlwings
python tbt_model.py train archive_08-03-26_cleaned.csv
xlwings addin install          # then Import Functions from the Excel ribbon
```

One Excel-specific warning, carried over from earlier: mark the UDFs non-volatile
(they are, as written) or Excel will re-run inference on every recalculation of every
cell and the workbook will crawl. If you're scoring more than a few dozen rows, run
the batch in Python and paste values.

---
---

# v2 — Component Decomposition

Supersedes sections 4–7 above. The v1 findings on data structure, leakage, and
validation design (sections 1–3, 8–9) still hold.

## 11. Correction to the accounting identity

v1 stated `Total Price = Proposal Total + Freight`. That was checked on the median
and the median was zero — but it only holds on 62% of rows. The true identities are:

```
Proposal Total = Material + Fabrication + Construction
               + InsulMaterial + InsulConstruction + Tax        (100.0% of rows)

Total Price    = Material + Fabrication + Construction
               + InsulMaterial + InsulConstruction + Freight     (100.0% of rows)
```

**Total Price is tax-exclusive and freight-inclusive. Proposal Total is
tax-inclusive and freight-exclusive.** They are not nested. The 2,522 rows where
v1's identity failed are exactly the 2,522 rows where tax was charged.

Worth confirming with accounting that this is intended — quoting a customer a
"Total Price" that excludes sales tax but includes freight, alongside a "Proposal
Total" that does the reverse, is at minimum a naming hazard.

### Margin is already inside the components

Median Material Price per sq-ft of shell, for CS tanks, by margin band:

| Margin % | n | $/sf |
|---|---|---|
| 16% | 75 | 22.11 |
| 18% | 2,497 | 23.03 |
| 20% | 2,679 | 24.05 |
| 22% | 108 | 32.53 |
| 26% | 38 | 42.21 |

The rate climbs with margin, so the component prices are already marked up — they
are sell prices, not costs. You were right to ask. The practical consequence: the
model learns *what TBT charges*, margin included, and you cannot recover underlying
cost from this file. If you want a cost model, you need the pre-margin figures.

### Tax should never have been modelled

Effective tax rate by state is nearly deterministic — LA 6.26%, AL 6.21%, TN 6.12%,
TX 5.30%, GA 4.82%, NC 4.58%, PA 3.82%, WI 3.32%, with standard deviations mostly
under 1 point. This is a lookup table, not a regression. v2 stores it as a dict and
applies `rate x taxable_base`. Modelling it would have added error for nothing.

Also note: only 53% of US rows carry tax at all (exemption status), and outside the
US it's ~0% except Missouri. Tax presence is a customer attribute the file doesn't
contain, so `est_tax` is best-effort.

## 12. The decomposition result

Six regressors, one per component, each on `log(unit price | component present)`.
Summed. Compared against the v1 single-model approach on identical folds:

| Approach | MedAPE | <=10% | <=20% |
|---|---|---|---|
| v1 direct model on Total Price | 12.7% | 42% | 66% |
| Component sum, scope inferred | 11.6% | 45% | 68% |
| v1 direct + scope flags | 9.9% | 50% | 74% |
| Component sum, scope known | 9.5% | 52% | 75% |
| **Blend: 70% components + 30% direct** | **9.3%** | **52%** | **76%** |

**Your instinct was right, and it beat the direct model in both scenarios.** The
decomposition wins by 1.1 points when scope has to be guessed and 0.4 points when
it's known — and on identical information (direct+flags vs components+flags, 9.9%
vs 9.5%) the split is still ahead, which means component errors are partially
cancelling rather than compounding. That was the real risk in the idea and it
didn't materialise.

Blending the two at 70/30 recovers another 0.2. Weights between 0.3 and 0.7 all
land at 9.3-9.5%, so the exact number doesn't matter; the blend is just insurance
against one architecture failing on an unusual tank.

### Per-component accuracy

| Component | n present | MedAPE |
|---|---|---|
| Fabrication | 6,679 | 9.9% |
| Material | 6,679 | 11.3% |
| Insulation Material | 1,539 | 12.1% |
| Construction | 4,570 | 14.9% |
| Freight | 5,491 | 14.9% |
| Insulation Construction | 1,456 | 16.0% |

Fabrication and Material are the most learnable — they're close to a steel-weight
calculation. Construction and Freight carry the site-specific variance.

Insulation at 12.1% is a striking improvement on v1's implicit handling: when the
model was forced to guess whether insulation existed at all, the effective error on
insulated tanks was 34%. Separating "is there insulation" from "how much does it
cost" cut that by two thirds.

## 13. Scope flags are the single biggest lever

Larger than the decomposition itself. Three booleans — erection included, insulation
included, freight included — move the model from 11.5% to 9.3%.

This is not a modelling trick, it's recognising that scope is **an input the
estimator already knows and the spec columns cannot contain.** Whether a customer
bought insulation is a commercial decision, not a property of the tank. Asking the
model to infer it was always the wrong framing.

How well can they be inferred when not supplied? Presence-classifier AUC:

| Component | AUC | base rate |
|---|---|---|
| Construction | 0.988 | 0.68 |
| Freight | 0.950 | 0.82 |
| Insulation Construction | 0.923 | 0.22 |
| Insulation Material | 0.914 | 0.23 |

Construction is nearly determined — `Wage Type = "No Erection Included"` implies zero
construction 82% of the time, and a blank Wage Type implies it 97% of the time.
Freight is geographic: 57% of Mexico rows and 85-92% of Peru/Chile/Colombia rows carry
no freight, because the customer arranges export shipping. Insulation is the genuinely
uncertain one — 34% of fire-protection tanks have it, 23% of waste-water, and nothing
in the spec columns predicts which.

### Hurdle handling: hard threshold, not expected value

The natural implementation is `E[component] = P(present) x E[value | present]`. That is
correct for forecasting a portfolio and **wrong for quoting a tank.** A 50% insulation
probability produces half an insulation package, which matches no tank anyone will
ever build.

Empirically the two are indistinguishable (11.6% vs 11.5%), so there's no accuracy
argument for the incoherent one. v2 thresholds at P > 0.5 and returns the inferred
probabilities in `scope_inferred` so the estimator can see when the model was
guessing and override it.

**Implementation trap:** the presence classifiers must be trained on a feature matrix
that excludes the scope flags. Otherwise a blank flag reads as "no insulation" and the
classifier confirms it — the prediction becomes self-fulfilling. v2 keeps two encoders
(`encoder`, `encoder_nf`) and runs prediction in two stages: resolve scope on the
flag-free matrix, then price with flags set. This bit during development and produced
quotes with all scope zeroed out.

## 14. Rolling-origin, v2

Train on everything before a quarter, score that quarter:

| Quarter | n | direct | components | blend |
|---|---|---|---|---|
| 2024Q4 | 603 | 14.1% | 13.4% | 12.6% |
| 2025Q1 | 724 | 16.8% | 16.2% | 16.0% |
| 2025Q2 | 768 | 10.7% | 10.2% | 10.2% |
| 2025Q3 | 792 | 9.2% | 9.2% | 8.9% |
| 2025Q4 | 686 | 10.7% | 8.8% | 9.4% |
| 2026Q1 | 726 | 9.1% | 8.7% | 8.4% |
| 2026Q2 | 703 | 7.5% | 6.5% | 6.7% |
| 2026Q3 | 193 | 5.6% | 5.1% | 5.1% |
| **mean** | | 10.5% | 9.8% | **9.6%** |
| **last 3** | | 7.4% | 6.8% | **6.7%** |

**Deployment accuracy is now ~7% on recent quarters**, down from ~10% in v1. The
blend wins in 6 of 8 quarters.

Conformal bands tighten correspondingly: 80% coverage at /x 1.24 (was 1.34), 90% at
/x 1.40 (was 1.47).

## 15. What the decomposition buys beyond accuracy

The accuracy gain is real but modest. The bigger win is **auditability**, and it
changes what the tool is for.

A single number that says $238,916 is unarguable and therefore unusable — an
estimator who disagrees has no way to engage with it. A breakdown that says Material
$77,865 / Fabrication $78,673 / Construction $79,741 / Freight $2,363 is something a
human can check line by line. When the estimator's own fabrication number is 40%
above the model's, that's a specific, answerable question.

`TBT_WORST_COMPONENT` in the Excel bridge does exactly this: given the estimator's
component figures, it names the line that deviates most. That turns "this quote looks
off" into "check the construction line," which is the difference between a tool people
use and a tool people ignore.

The out-of-fold audit across the whole archive (`oof_audit.py`) shows where the
disagreements concentrate. Of 918 rows more than +/-30% from the model:

| Driver | rows |
|---|---|
| Material | 437 |
| Construction | 355 |
| Fabrication | 78 |
| Freight | 23 |
| Insulation Construction | 14 |
| Insulation Material | 11 |

Material and Construction account for 86% of large disagreements. That's where to
look for historical mispricing, transposed dimensions, and missing scope lines — and
it's a list of about 800 specific quotes, not a vague concern.

## 16. Revised file list

| File | Purpose |
|---|---|
| `tbt_model_v2.py` | Component architecture: 6 regressors, 4 presence classifiers, tax lookup, direct model, blending, conformal calibration. `train` / `check`. |
| `excel_udf.py` | xlwings bridge — `TBT_PRICE`, `TBT_BREAKDOWN`, `TBT_RANGE`, `TBT_FLAG`, `TBT_WORST_COMPONENT`. |
| `batch_score.py` | Score a CSV of quotes to xlsx. Use for volume; don't drag a UDF down 6,000 rows. |
| `oof_audit.py` | Honest out-of-fold re-scoring of the archive, sorted worst-first, with the driving component named. |
| `tbt_pricing_model_v2.joblib` | Trained bundle. |
| `audit_archive.xlsx` | Output of `oof_audit.py` on the current archive. |

```
pip install pandas scikit-learn joblib openpyxl xlwings
python tbt_model_v2.py train archive_08-03-26_cleaned.csv
python oof_audit.py archive_08-03-26_cleaned.csv audit.xlsx
xlwings addin install
```

## 17. What to do next

1. **Add the three scope flags to the quote sheet as explicit inputs.** Biggest single
   accuracy lever available, and it costs three checkboxes.
2. **Work the `audit_archive.xlsx` top 50 with an estimator.** If the model is right
   about most of them, you have a business case. If it's wrong, you learn what column
   is missing — and that's worth more than a tenth of a point of MedAPE.
3. **Confirm the tax/freight accounting with finance** before anyone quotes off
   `proposal_total`.
4. **Then** consider the steel cost index and appendage counts from section 9. Those
   are the next real accuracy gains, and they require data collection rather than
   modelling.

None of this changes the section 8 limitation: with 173 Won and 142 Lost, this still
models what TBT quotes, not what wins.

---
---

# v3 — Fixes, Extrapolation, and Honest Numbers

Supersedes v2. Sections 1-3 and 8-9 of v1 still hold; section 11's accounting
identity still holds; sections 12-15 are superseded by what follows.

## 18. Three things we got wrong

### 18.1 The Quantity division was a bug

v1 and v2 both divided every price column by `Quantity` to get a per-tank figure.
That was wrong. The columns are **already per-tank**.

The evidence is unambiguous. Median $/sq-ft of shell, by order quantity:

| Quantity | raw $/sf | after dividing by qty | ratio to qty=1 |
|---|---|---|---|
| 1 | 76.1 | 76.1 | 1.00 |
| 2 | 76.8 | 38.4 | 0.50 |
| 3 | 64.3 | 21.4 | 0.33 |
| 4 | 82.2 | 20.6 | 0.25 |
| 5 | 57.1 | 11.4 | 0.20 |
| 6 | 83.3 | 8.7 | 0.17 |

Raw $/sf holds flat across order sizes. After division it collapses as exactly
1/Quantity. That is the signature of dividing something that was never extended.
The same pattern appears on Material Price alone, so it is not an artifact of the
total.

This corrupted the 15% of rows with multi-tank orders — and those rows are 38% of
the top-5%-by-value quotes, which is precisely the segment we most need right.

A side effect: with the bug fixed, only **173** rows fail the $20-250/sq-ft
plausibility gate, not 455. Most of the previously "implausible" rows were
multi-tank orders that the division had turned into nonsense. The data is cleaner
than v2 made it look.

**Lesson:** when a derived quantity varies as exactly 1/n with some column, the
arithmetic is wrong, not the world. We should have run that check in v1.

### 18.2 We trained on rows we knew were bad

v2 filtered implausible rows out of *scoring* but left them in *training*. Removing
them from the fit as well is worth 0.4 points of mean error and nearly 3 points of
large-tank bias on its own. Filtering evaluation but not training measures the model
more kindly without making it better — exactly backwards.

### 18.3 Trees cannot extrapolate, and we never checked

A gradient-boosted tree predicts a constant beyond its largest leaf. v2 priced a
111 ft x 51 ft waste-water tank at $1.5M against an actual $3.1M, and 96% of
top-5%-by-value rows were underpriced. We spent a whole round attributing that to
price drift; it was not. The bias sat at -25% at *every* train/test boundary tried,
and an oracle price index with perfect foresight moved it only to -24.0%.

## 19. What actually fixed it, and what did not

Every change below measured on the same forward holdout: train on everything before
2026Q1, score the 1,595 plausible quotes after it.

| Change | median | mean | p90 | aggregate | top 5% |
|---|---|---|---|---|---|
| v2 as shipped | 8.5% | 11.5% | 24.8% | -10.4% | -25.7% |
| + Quantity fix | 8.2% | 11.2% | 24.2% | -7.5% | -22.4% |
| + exclude bad rows from training | 7.9% | 10.8% | 23.5% | -7.9% | -19.7% |
| + ridge log-log backbone | 8.2% | 11.0% | 23.2% | **-4.0%** | **-15.5%** |
| + recency weighting (1-yr half-life) | **7.5%** | **10.2%** | **22.2%** | -5.4% | -16.8% |

### Things that did not work

Worth recording so nobody retries them:

- **Dollar-weighted training** (weights from value^0.25 to value^1.0) — no effect
  on the aggregate bias it was designed to fix.
- **Isotonic and log-log recalibration** — the model is unbiased *in sample*
  (log-log slope 1.007), so there is nothing for a calibrator to learn.
- **Monotonic constraints** on size features — slightly worse.
- **Absolute-error loss** — better p90, notably worse aggregate bias.
- **A dedicated large-tank specialist model** — worse on every measure; 744 training
  rows is not enough to support a separate model.
- **Hedonic index deflation**, even with an oracle index — 1.7 points of large-tank
  bias, nowhere near enough to matter.

The only thing that moved the large-tank bias was the parametric backbone, and it
moved it because it addresses the actual mechanism rather than the symptom.

## 20. The architecture

```
                  ridge on [log D, log H, log D x log H, t, material]
                                    |
      log(component) =        backbone  +  GBM(everything else)
                                    |
                          extrapolates    captures interactions
```

Six component regressors, each of that form, each trained only on rows where the
component is present. Four hurdle classifiers decide scope when the estimator has
not. Tax is a state-rate lookup and is never modelled. The final estimate blends
70% component-sum with 30% a direct whole-price model of the same form.

The backbone is deliberately minimal. `log D` and `log H` span all the geometry —
shell area, floor area and volume are products of powers of them — so the basis is
identified. An earlier attempt that also fed in `log(shell_area)` produced
coefficients like -12.657 from pure collinearity.

The backbone alone is a poor model (40% median error; it knows nothing about scope
or configuration). Its job is not to be accurate, it is to give the ensemble a
sane slope outside the training envelope.

## 21. Where v3 actually lands

Rolling origin, retraining each quarter on all prior data — which is how it will be
run in production:

| Quarter | n | median | mean | p90 | aggregate | top 5% |
|---|---|---|---|---|---|---|
| 2026Q1 | 719 | 7.6% | 10.0% | 21.3% | -3.3% | -13.1% |
| 2026Q2 | 686 | 6.4% | 9.2% | 19.8% | -6.7% | -23.6% |
| 2026Q3 | 190 | 4.5% | 6.9% | 15.4% | -1.4% | -1.6% |
| **mean** | | **6.1%** | **8.7%** | **18.8%** | **-3.8%** | **-12.8%** |

Out-of-fold across the whole archive (grouped by Quote #): median 7.9%, mean 13.4%,
with 644 rows beyond +/-30% — down from 918 under v2.

Conformal bands tightened at each version: 80% coverage needed /x 1.34 in v1,
1.24 in v2, and **1.19** in v3.

**Quote the 8.7% mean, not the 6.1% median.** Half your quotes land inside 6%, but
the mean is what a portfolio experiences, and the distribution has a real tail.

## 22. What is still wrong

**Large tanks are still underpriced by roughly 13-17% in aggregate.** The backbone
cut this from -25.7%, but did not close it. The residual is genuine: the backbone
*alone* is -52% on this segment, which means large tanks scale **superlinearly** —
they cost more than any power law in D and H predicts. That is real engineering,
not a fitting artifact: thicker plate, more shell courses, larger crane classes,
specialised rigging. Nothing in these 42 columns captures it.

v3 handles this by refusing to be quiet about it. `predict()` returns a `warnings`
list, and any quote in the top 5% by value carries *"treat as a floor, not an
estimate."* `TBT_FLAG` appends the same caution, and `batch_score.py` writes it to
a `caution` column.

**Do not use v3 unsupervised on the big end, and do not use it to value a backlog.**

**Win rate remains unmodellable.** 173 Won and 142 Lost out of 6,892. v3 models what
TBT quotes, not what wins.

## 23. How much headroom is left

Conditioning on progressively more of what we know, and measuring the spread that
remains within each group:

| Conditioned on | median | mean |
|---|---|---|
| Identical spec | 5.9% | 18.3% |
| + state | 2.2% | 6.3% |
| + state + wage type + year | 1.5% | **4.6%** |

(Note this corrects the 14.9% figure quoted in section 1, which was computed before
the Quantity fix and without the plausibility gate.)

The floor is around **4.6% mean**; v3 sits at 8.7%. So there is roughly 2x of
headroom and we are *not* at the limit of the data. The caveat is that these groups
include revisions of the same quote, so the true floor is somewhat higher.

That headroom is in **features, not algorithms**. Everything algorithmic that could
be tried has been tried; the list in section 19 is not short. What would move the
number, in order of expected value:

1. **Plate thickness / shell course schedule.** Converts the superlinear scaling
   from something the model must infer into something it is told. This is the single
   highest-value addition and it should already exist in your engineering output.
2. **Crane class and site access.** The best remaining hypothesis for the large-tank
   jump, and it is a field an estimator already knows at quote time.
3. **Nozzle, manway, ladder and platform counts.** Known cost drivers, absent
   entirely from the export.
4. **Steel index at quote date.** Replaces the crude `months` proxy with the actual
   driver of the observed drift.
5. **Outcome labels on a meaningful share of quotes.** Unlocks a different and more
   valuable class of model. This is a process change, not a modelling one.

## 24. Running it

```
pip install pandas scikit-learn joblib openpyxl xlwings

python tbt_model.py train archive.csv          # fit + validate + save (~10 min)
python tbt_model.py train archive.csv --fast   # skip validation (~2 min)
python tbt_model.py check archive.csv          # validation only
python oof_audit.py archive.csv audit.xlsx     # honest archive re-scoring
python batch_score.py new_quotes.csv out.xlsx  # score a batch

xlwings addin install    # then Import Functions from the Excel ribbon
```

**Retrain monthly.** Accuracy improves as history accumulates and drift is fast
enough that a stale model degrades. `--fast` makes it a two-minute scheduled task.
`TBT_MODEL_INFO()` puts the training date on the sheet so nobody quotes off a stale
model without noticing.

**Monitor.** Log every prediction beside the price the estimator actually used. If
median signed error drifts off zero two months running, retrain more often or
investigate what changed.

## 25. Recommended rollout

1. **Deploy `TBT_FLAG` only.** Not `TBT_PRICE`. At 8.7% mean error the model is not
   good enough to set prices, but it is easily good enough to catch a transposed
   dimension or a forgotten stainless premium. That catches real money at almost no
   risk, and builds the track record needed before anyone would trust more.
2. **Work the top 50 rows of `audit_archive.xlsx` with an estimator.** Material and
   Construction drive 87% of large disagreements. This is the cheapest way to find
   out whether the model is seeing something real.
3. **Add the three scope checkboxes to the quote sheet.** Two points of accuracy for
   three booleans, and the only change on this list that costs nothing.
4. **Confirm the tax and freight accounting with finance** before anyone quotes off
   `proposal_total`. Total Price is tax-exclusive and freight-inclusive; Proposal
   Total is the reverse. That is a naming hazard regardless of the model.
5. **Add a $/sq-ft gate to data entry** so partial quotes stop entering the archive.
6. **Then** collect plate thickness and crane class. That is where the next real
   accuracy lives.

## 26. Files

| File | Purpose |
|---|---|
| `tbt_model.py` | The model. Feature engineering, backbone+GBM stages, hurdle classifiers, tax lookup, conformal calibration, guard rails. `train` / `check` / `predict`. |
| `excel_udf.py` | xlwings bridge: `TBT_PRICE`, `TBT_PRICE_PER_TANK`, `TBT_BREAKDOWN`, `TBT_RANGE`, `TBT_FLAG`, `TBT_WORST_COMPONENT`, `TBT_MODEL_INFO`. |
| `batch_score.py` | Score a CSV to xlsx. Use for volume. |
| `oof_audit.py` | Out-of-fold re-scoring of the archive, sorted worst-first. |
| `tbt_pricing_model.joblib` | Trained bundle. |
| `audit_archive.xlsx` | Audit output on the current archive. |
| `DESIGN.md` | This document. |
