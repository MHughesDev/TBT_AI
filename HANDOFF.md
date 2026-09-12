# HANDOFF

For whoever — or whatever — picks this up next. Written to be read cold.

---

## What this is

A pricing model for TBT steel tanks. Given specs (diameter, height, material,
deck/floor style, use type, location, wage type, seismic, mileage) it predicts
what TBT would quote. It is a **check on human estimators, not a replacement**.

Current accuracy: **8.7% mean absolute error, 6.1% median**, measured by
retraining each quarter on prior data and scoring the next quarter.

Read `README.md` for how to run it, `DESIGN.md` for the full technical record
including everything that was tried and failed.

---

## State of the system

| | |
|---|---|
| Status | Working, validated, not yet deployed |
| Training data | 6,679 usable rows from 6,892; Dec 2023 – Dec 2026 |
| Runtime | ~2 min to train on CPU, no GPU needed |
| Deps | pandas, scikit-learn, joblib, openpyxl, xlwings |
| Model artifact | 39 MB, gitignored — rebuild with `train --fast` |

### Architecture in one paragraph

Six gradient-boosted regressors, one per price component (Material, Fabrication,
Construction, Insulation Material, Insulation Construction, Freight), each fitted
on `log(component)` over rows where that component is present. Each sits on a ridge
log-log backbone in `[log D, log H, log D × log H, t, material]` so it can
extrapolate past the largest tank it has seen — the tree learns only the residual.
Four hurdle classifiers decide scope when the estimator has not supplied it. Tax is
a state-rate lookup, never modelled. The final estimate blends 70% component-sum
with 30% a direct whole-price model of the same form. Conformal bands from a
grouped holdout.

---

## The three bugs that were found, so they are not reintroduced

**1. Do not divide price columns by `Quantity`.** They are already per-tank. The
tell: `Total Price / Quantity / shell_area` falls as exactly 1/Quantity across
order sizes (1.00, 0.50, 0.33, 0.25, 0.20, 0.17) while the undivided version stays
flat. v1 and v2 both had this and it corrupted 15% of rows — which were 38% of the
top-5%-by-value quotes.

**2. Filter implausible rows out of TRAINING, not just scoring.** v2 filtered
evaluation only, which measures the model more kindly without making it better. The
gate is `$20 ≤ Total Price / shell_area ≤ 250`; 173 rows fail it, mostly partial
quotes missing scope lines.

**3. Presence classifiers must not see the scope flags.** They predict those flags.
If a blank flag reads as "no insulation," the classifier confirms it and the
prediction is self-fulfilling. Hence two encoders (`encoder`, `encoder_nf`) and the
two-stage `predict()`: resolve scope on the flag-free matrix, then price with flags
set.

### Leakage guard

Never use as inputs: any component price, `Total Tax`, `Proposal Total`,
`Total Price`, or any `Margin (%)` / `Contingency (%)` / `Commission (%)` column.
The first group are arithmetic identities; the second are pricing-policy decisions
made during quoting, not tank attributes. `BANNED` in `tbt_model.py` is the list.

### Validation guard

Never use a random train/test split. 6,892 rows are only 3,018 quotes — revisions
are near-duplicates and a random split leaks them, reporting ~3% instead of ~8%.
Group by `Quote #` at minimum; prefer rolling origin by quarter, which is what
`validate()` does.

---

## Open problem #1 — large tanks (the real one)

**The model underprices the largest 5% of quotes by 13–17% in aggregate.**

This is not shrinkage and not drift. Evidence:

- The model is **unbiased in-sample** (log-log slope 1.007), so there is nothing
  for a calibrator to learn.
- The bias sat at −25% at *every* train/test boundary tried, so it is not a
  one-off structural break.
- A hedonic price index with **perfect foresight** moved it only 1.7 points.
- The ridge backbone alone is **−52%** on this segment.

That last number is the diagnosis. Large tanks scale **superlinearly** — they cost
more than any power law in D and H predicts. Physically that is thicker plate, more
shell courses, larger crane classes, specialised rigging. None of it is in the 42
available columns.

### Do not retry these

All tested on the forward holdout, all failed:

| Attempt | Result |
|---|---|
| Dollar-weighted training (value^0.25 … value^1.0) | No effect on the bias it targeted |
| Isotonic recalibration | −0.8 pts, within noise |
| Linear log-log recalibration | −1.2 pts, within noise |
| Monotonic constraints on size features | Slightly worse |
| Absolute-error loss | Better p90, notably worse aggregate bias |
| Dedicated large-tank specialist model | Worse on every measure (744 train rows) |
| Hedonic index deflation, oracle index | −1.7 pts |
| Deeper trees / more capacity | No change |

### What would actually work

Get plate thickness or the shell course schedule into the export. That converts the
superlinear term from something the model must infer into something it is told. This
is an engineering-data problem, not a modelling problem, and TBT almost certainly
already generates it.

Until then the model warns on every top-5% prediction and says to treat the number
as a floor. **Keep that warning.** Removing it is the single easiest way to make
this system cause harm.

---

## Open problem #2 — no win-rate model is possible

`Status` breaks down as Deliver To Customer 3,345, Revised 2,437, Sales 783,
**Won 173, Lost 142**, Bid Review 9. Under 5% of rows have a resolved outcome.

Expect to be asked for "a model that tells us what price wins." It cannot be built
from this data. 315 labelled examples across 33 countries and 14 use types is not
enough, and no amount of modelling fixes it. It needs outcome capture on a
meaningful share of quotes first — a process change.

Say this early and plainly. "AI pricing model" gets heard as "tells us what wins,"
and the gap between that and what this does is where trust gets destroyed.

---

## Headroom

Spread remaining within groups, conditioning on more of what we know:

| Conditioned on | median | mean |
|---|---|---|
| Identical spec | 5.9% | 18.3% |
| + state | 2.2% | 6.3% |
| + state + wage type + year | 1.5% | **4.6%** |

Floor is around **4.6% mean**; we are at 8.7%. Roughly 2× headroom, and we are not
at the limit of the data. (Caveat: these groups include quote revisions, so the true
floor is somewhat higher.)

**The headroom is in features, not algorithms.** The algorithmic search is
exhausted — see the table above and section 19 of `DESIGN.md`.

---

## Next tasks, in priority order

### 1. Ship `TBT_FLAG` — nothing else
**Why:** At 8.7% mean error the model cannot set prices, but it reliably catches a
transposed dimension, a missing scope line or a forgotten stainless premium. Real
money, almost no risk, and it builds the track record needed before anyone would
trust more.
**Done when:** estimators see an OK/LOW/HIGH column on live quotes and at least one
genuine error has been caught.
**Do not** ship `TBT_PRICE` to estimators in the first phase.

### 2. Work the top 50 of `audit_archive.xlsx` with an estimator
**Why:** 644 archive rows sit beyond ±30% out-of-fold. Material and Construction
drive 87% of them. Either the model is finding real mispricing (business case) or
it is wrong in a patterned way (tells you which column is missing). Both outcomes
are worth more than another tenth of a point of accuracy.
**Done when:** 50 rows are adjudicated and categorised as model-wrong vs quote-wrong.

### 3. Add three scope checkboxes to the quote sheet
**Why:** erection / insulation / freight included. Worth ~2 points of accuracy
(11.5% → 9.3% in the v2 tests) for three booleans. Cheapest win available.
**Done when:** the flags are populated in the export and `predict()` is called with
them.

### 4. Add a $/sq-ft gate to data entry
**Why:** stops partial quotes entering the archive. 173 rows are currently
unusable. Gate at $20–250/sq-ft with a soft warning.
**Done when:** new rows cannot be saved outside the band without an override.

### 5. Confirm tax and freight accounting with finance
**Why:** both verified on 100% of rows, and they are not nested:
```
Total Price    = 5 components + Freight    (tax-EXCLUSIVE)
Proposal Total = 5 components + Tax        (freight-EXCLUSIVE)
```
Quoting a "Total Price" that excludes sales tax but includes freight, beside a
"Proposal Total" that does the reverse, is a naming hazard independent of any model.
**Done when:** finance confirms intent, or the columns get renamed.

### 6. Get plate thickness / shell course schedule into the export
**Why:** the only known path to fixing the large-tank bias. See open problem #1.
**Done when:** the column exists and a retrain shows the top-5% aggregate bias
moving materially off −13%.

### 7. Set up monthly retraining
**Why:** accuracy improves as history accumulates; drift is fast enough that a
stale model degrades. `train --fast` is a 2-minute scheduled task.
**Done when:** it runs unattended and `TBT_MODEL_INFO()` shows a recent date.

---

## Things that look like good ideas and are not

- **Swapping in XGBoost/LightGBM.** sklearn's `HistGradientBoostingRegressor` is
  the same algorithm family and won the bake-off. The extra dependency buys nothing
  and is a hassle on a locked-down corporate laptop.
- **Adding `Company Name` as a feature.** Tempting — repeat customers get
  consistent pricing — but it makes the model useless on new customers, which is
  where an estimate is most valuable. If you try it, validate with
  `GroupKFold` by company, not by quote.
- **Using Python-in-Excel (`=PY()`).** It runs in an Azure sandbox with no
  filesystem or network access, so it cannot reach a local model. xlwings is the
  right bridge. This was checked.
- **Dropping the 70/30 blend for pure component-sum.** Blend weights between 0.3
  and 0.8 all land within 0.2 points, so the exact number does not matter — but the
  blend is cheap insurance against one architecture failing on an unusual tank.
- **Reporting the median.** 6.1% is real but flattering. The mean (8.7%) is what a
  portfolio experiences.

---

## Where the bodies are buried

- `tbt_pricing_model.joblib` pickles the `Stage` class. The CLI routes through
  `import tbt_model as _M` specifically so it pickles as `tbt_model.Stage` rather
  than `__main__.Stage`, which would fail to unpickle from any importing process.
  **Do not "simplify" that back to a direct `train()` call.**
- `validate()` skips the hurdle classifiers (`with_clfs=False`) because true scope
  is known during validation and they would be ~40% of the compute for nothing.
- The component prices **already include margin**. They are sell prices, not costs.
  No cost model can be built from this file; `Material Price / sq-ft` rises with the
  `Margin (%)` band, which is how this was confirmed.
- `Freight Price` is zero for 57% of Mexico rows and 85–92% of Peru/Chile/Colombia
  rows because the customer arranges export shipping. That is correct, not missing
  data.
- Argentina (116 rows) was 38.7% error with −16.7% bias under v1. It has not been
  re-checked under v3. Treat AR output with suspicion until someone does.
