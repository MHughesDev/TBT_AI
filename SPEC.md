# TBT Tank Pricing Estimator — System Specification

Status: v1.0, clean-sheet design. This document is the deliverable. A separate team
builds from it. Nothing in the repository's existing `tbt_model.py`, `DESIGN.md` or
`HANDOFF.md` is binding; those are the prior investigation's field notes and were
used as evidence only.

How to read this document:

- **[SETTLED]** — implement as written. Where a settled choice is also marked
  **[LOAD-BEARING]**, simplifying it away breaks something specific that the text
  names. Do not simplify it without re-running the measurement in §3.
- **[OPEN]** — a genuine fork. The options, the experiment, and the decision rule
  are written down here, before any data is seen. Run the experiment once, apply
  the rule, record the result in the build log. Do not add options.
- **[PREDICTION]** — a number we expect but cannot measure from here. Every
  prediction states the reasoning and the result that would falsify it.

Document map (matches the brief's §6):

| Brief | Here |
|---|---|
| 6.1 Architecture | §2 |
| 6.3 Measurement | §3 |
| 6.2 Data contract | §4 |
| 6.4 Build sequence | §5 |
| 6.5 Acceptance criteria | §6 |
| 6.6 User surface | §7 |
| 6.7 Risk register | §8 |
| 6.8 Data wishlist | §9 |

§1 states the objective and the reasoning shared by all three roles. §10 records
where the roles disagreed and how each disagreement was resolved.

---

## 1. Objective, ceiling, and honest scope

### 1.1 The objective

Minimise **mean** absolute percentage error (APE) on per-tank Total Price,
measured forward in time on the protocol in §3. Secondary objectives, in order:
dollar-weighted aggregate calibration; honest prediction intervals; auditability.

### 1.2 The number to beat and the target

| | mean APE | median APE |
|---|---|---|
| Incumbent, as reported | 8.5% | 5.9% |
| Must-pass (no regression) | ≤ 8.5% | — |
| **Target** | **≤ 8.0%** | ≤ 5.7% |
| Stretch | ≤ 7.6% | — |

The incumbent's 8.5% was measured on three scoring quarters (2026Q1–Q3). The
targets above are absolute, on the §3 headline window. The builder also
re-measures a reference model on the exact harness (§5, increment B3) for two
reasons: it validates the harness (a reference far from the prior 9.99% means the
harness, not the model, is wrong) and it attributes the gain to each increment.
The relative figures in §6.1 are diagnostics, not the pass mark.

**[PREDICTION] The gap we expect to close on mean APE is 0.4–0.9 points**, from:
size-dependent drift handling (0.1–0.3), a physics term in the extrapolation
backbone (0.1–0.3 on mean, more on large-tank bias), and treating the calibration
problem separately from the point-estimate problem so neither is compromised for the
other (0–0.1 on mean, 2–3 points on aggregate bias). Falsified if increment B8
(§5) lands above 8.3% on the headline window.

We do **not** expect to reach the 4.6% floor. That floor was computed within groups
that include revisions of the same quote, so the true floor is higher, and the
remaining variance is concentrated in engineered-to-order work whose drivers
(appendages, mixers, covers, plate schedule) are not in the 42 columns. The
remaining gap is a data-collection problem; §9 ranks what to collect.

### 1.3 Two point estimates, not one — the resolution of the first conflict

The north star (mean APE) and secondary objective 1 (aggregate calibration) are
minimised by **different numbers**. In log space the model fits the conditional
median. Exponentiating a log-space fit returns the median of a log-normal, which
sits *below* the mean by roughly `exp(σ²/2) − 1`: about 1% for commodity work
(σ ≈ 0.12) and about 3% for bespoke work (σ ≈ 0.25). The mean is what a book of
work sums to. MAPE, on the other hand, is minimised by an estimate at or slightly
*below* the median. Forcing one number to serve both purposes silently sacrifices
one of them, and that is the current −4% aggregate bias, in part.

**[SETTLED] The system produces two figures from one model:**

- `point` — the conditional median, the number shown beside a single quote and
  the number the north star is scored on.
- `book` — `point × s`, where `s` is a retransformation ("smearing") factor
  estimated from *forward* residuals (§2.7). Used for summing over a portfolio and
  scored on the aggregate-bias metrics.

Both are reported on every protocol run (§3.4). This is the explicit statement the
brief asks for: we do not trade calibration for MAPE; we decline to make one number
do two jobs. §2.7 specifies an [OPEN] experiment on whether the top-value-band
factor should also be applied to `point`.

---

## 2. Architecture (Architect)

### 2.1 Decomposition — [SETTLED] [LOAD-BEARING]

Six component models plus one direct model plus a deterministic tax lookup.

```
component c ∈ {Material, Fabrication, Construction, InsulMaterial, InsulConstruction, Freight}

log(price_c)  =  backbone_c(x)  +  residual_c(x)         fitted on rows where c is present
point_components = Σ_c  present_c · exp(ŷ_c)             present_c is a scope input, never inferred
point_direct     = exp(ŷ_total)                          same form, fitted on Total Price
point            = w · point_components + (1 − w) · point_direct        w = 0.7 (see 2.8)
tax              = IS_TAXABLE · rate[State] · Σ_{c ≠ Freight} present_c · exp(ŷ_c)
```

Why component-wise: it earned −0.4 points and, more importantly, it makes the
output auditable line by line, which §7 depends on. The direct model is
insurance against the component sum failing on an unusual tank; it is cheap.

Why it is load-bearing: a builder who collapses to one total model loses the
breakdown (§7.3), the worst-component diagnostic, and the ability to fit each
component on the rows where it is present. The insulation components in particular
are present on 21–23% of rows; a single model must learn "is there insulation" and
"how much" at once, and the prior evidence puts the cost of that at roughly three
times the error on insulated tanks.

Material and Fabrication are always present. The four optional components are
gated by explicit scope inputs (§2.4). Tax is arithmetic, not a model.

### 2.2 Prediction target — [SETTLED]

Per component: `log(price_c)` per tank, in USD as recorded. Not divided by
Quantity (§4.5 of the brief; the columns are already per tank). Not divided by
area.

Rate targets (`log(price/shell_area)`) were considered and rejected as equivalent:
the backbone basis contains `log D` and `log H`, and `log(shell_area) = log π +
log D + log H` is linear in that basis, so dividing by area only reparameterises the
backbone and leaves the residual model's task unchanged. The prior measurement
found 0.2 points between them, inside noise. Choose the simpler form.

Log is mandatory. Prices span $15k to $6M; a raw-dollar loss is dominated by a
handful of rows.

### 2.3 Where physics lives and where learning lives — [SETTLED] with one [OPEN] term

**The backbone is parametric, per component, fitted by ridge regression.** Its job
is not accuracy — alone it has roughly 40% median error — it is to give the
ensemble a sane slope outside the training envelope, because a tree predicts a
constant beyond its largest leaf. That is why the prior found a −52% top-5% bias
without it and −15% with it. [LOAD-BEARING] A builder who removes the backbone
because "the GBM does the work" will reproduce the −25% large-tank failure.

Backbone basis, all components:

```
b0 = 1
b1 = log D                       D = Diameter (ft)
b2 = log H                       H = Height (ft)
b3 = log D · log H
b4 = t                           years since 2024-01-01, from Due Date; capped at inference (§2.9)
b5–b7 = one-hot Material         (CS reference; 304SS, 316SS, other)
b9 = t · log D                   [OPEN-1: size-dependent drift term, see below]
b10 = log steel_lb_est           [OPEN-2: physics term, see below]
```

Standardise `b1..b4` on the training window (store mean/sd in the bundle). Ridge
`alpha = 1.0` on standardised inputs. Do not add `log(shell_area)`, `log(volume)`
or any other product of powers of D and H to the basis: they are exactly collinear
with `b1, b2` and the prior investigation recorded coefficients of −12.7 from doing
so.

**The residual model is a gradient-boosted tree ensemble** on all features (§4.3)
including the backbone inputs, fitted to `log(price_c) − backbone_c(x)`. It
captures interactions (deck style × material × use type × wage type × geography)
that a linear form cannot, and it is where the contextual columns earn their 20
points.

**[OPEN-1] Size-dependent drift in the backbone.** The field notes show drift that
differs by size quartile. A single `t` cannot represent it; recency weighting and
monthly retraining absorb some of it implicitly. Options: (A) basis without `b9`;
(B) basis with `b9 = t · log D`. Experiment: full §3 protocol, both variants,
everything else identical. Decision rule: take B if it improves headline mean APE
by ≥ 0.15 points **and** does not worsen top-5% dollar-weighted bias by more than 1
point; otherwise A. [PREDICTION] B wins by 0.1–0.3. Falsified if B is worse or
within 0.1.

**[OPEN-2] A physics term in the backbone.** The brief's diagnosis is that large
tanks scale superlinearly and that the mechanism is plate thickness set by hoop
stress, with more courses and heavier rigging. A power law in D and H cannot bend at
the point where hoop-stress thickness overtakes the code minimum thickness; a
one-foot-method steel-weight estimate can, because it has exactly that knee.

```
H_liq       = max(H − Freeboard_in/12, 1.0)                        ft
n_courses   = ceil(H / 8)                                            8 ft plate courses
for course k = 1..n_courses (k = 1 at the bottom):
    h_k     = max(H_liq − 8·(k−1) − 1, 0)                            design head at 1 ft above course bottom
    t_hoop  = 2.6 · D · h_k · G / (S · E)        inches, G = 1.0, S = 20000 psi, E = 0.85
    t_min   = 0.1875 if D < 50 else 0.25 if D < 120 else 0.3125 if D < 200 else 0.375
    t_k     = max(t_hoop, t_min)
    lb_k    = π · D · min(8, H − 8·(k−1)) · t_k · 40.8               40.8 lb per ft² per inch of steel
shell_lb    = Σ_k lb_k
floor_lb    = π · (D/2)² · 0.25 · 40.8
steel_lb_est = shell_lb + floor_lb
```

The constants are nominal AWWA D100 / API 650 values and are **not** to be tuned
to the data; the term's purpose is the shape, and the ridge coefficient sets the
scale. The formula is wrong for dry-bulk silos (Janssen loading) and for bolted
tanks; the residual model conditions on Use Type and Material and will absorb
that. Roofs are left to the residual model via Deck Style.

The prior investigation tried "shell-course geometry" as *tree features* and it
was worse (10.10% against 9.99%). That result does not bear on this experiment:
a tree cannot extrapolate whatever features it is given, and the failure being
addressed is extrapolation. The term belongs in the backbone and nowhere else.

Options: (A) basis without `b10`; (B) basis with `b10`. Experiment: full §3
protocol. Decision rule: take B if the top-5% dollar-weighted bias improves by ≥ 3
points **and** headline mean APE does not worsen by more than 0.15 points;
otherwise A. [PREDICTION] B moves the top-5% bias from about −14% to −8% to
−11% and mean APE by −0.1 to −0.3. Falsified if the top-5% bias moves by less
than 2 points, in which case the superlinearity is not in the plate schedule and
§9 item 1 is the only remaining path.

Run OPEN-1 and OPEN-2 as a 2×2 (four protocol runs), apply each rule on its own
margin, and if the rules conflict prefer the variant with the better top-5% bias
because that is where the money is.

### 2.4 Scope is an input — [SETTLED] [LOAD-BEARING]

Five booleans arrive with every request and are never inferred, defaulted, or
imputed:

| Input | Gates |
|---|---|
| `construction` | Construction component |
| `insulation` | Insulation Material component |
| `insulation_erection` | Insulation Construction component; requires `insulation = true` |
| `freight` | Freight component |
| `taxable` | Tax on the five non-freight components |

Absent or blank → the system refuses (§6.5). A classifier here scored AUC 0.91–0.99
in a prior version and was still the wrong design: it can only agree with what the
estimator already knows, or be confidently wrong in a way that changes the price
with nothing visible on the sheet. Removing it cost nothing measurable. The
Designer's objection (five extra questions is friction) is answered in §7.2 by
asking them once per quote, not once per tank row. The objection does not reopen
the decision.

Why load-bearing: this is the single most likely good-faith regression (§8, R1).
"Default freight to yes when blank" is the same bug in different clothes.

### 2.5 Model family — [SETTLED]

`sklearn.ensemble.HistGradientBoostingRegressor` for the residual models,
`sklearn.linear_model.Ridge` for backbones. Two GBM variants per component,
averaged in log space (geometric mean of prices).

Starting hyperparameters (from prior evidence; the search is in §5, B9):

| Variant | learning_rate | max_iter | min_samples_leaf | l2_regularization | max_leaf_nodes | random_state |
|---|---|---|---|---|---|---|
| 1 | 0.03 | 1200 | 15 | 0.0 | 31 | 1 |
| 2 | 0.05 | 700 | 25 | 0.5 | 31 | 2 |

`loss = "squared_error"`, `categorical_features` set for every categorical column,
`early_stopping = False` (early stopping on a random validation split would leak
revisions; do not enable it).

Alternatives considered and why they lost:

- **LightGBM / XGBoost / CatBoost.** Same family; CatBoost's ordered target
  statistics are attractive for the high-cardinality geography columns, but each is
  a compiled dependency on a locked-down laptop, and the prior bake-off found no gap
  worth that. Not an open question; revisit only if a §9 column arrives that changes
  the feature mix.
- **Random forest / extra trees.** 1–2 points worse in the prior bake-off; no
  extrapolation advantage.
- **Explainable boosting (GAM-style).** Interpretability is delivered by the
  component decomposition instead; an extra dependency for a second route to the
  same property.
- **Bayesian hierarchical model.** Partial pooling on residuals was tried and
  always chose maximal shrinkage; the field notes are explicit that the hard
  segments are noisy, not biased, and pooling fixes bias. Uncertainty is delivered
  by conformal (§2.6). Would also miss the retrain-in-minutes constraint.
- **Neural networks.** 6,700 rows of heavily categorical tabular data. No.
- **Quantile boosting for intervals.** 56% empirical coverage at nominal 80% in
  the prior investigation. Rejected.

Recency weighting: sample weight `w_i = 0.5 ^ (age_years / 1.0)` where age is
measured from the training window's latest Due Date. Half-life 1.0 year
[SETTLED] — per-component half-lives were tried and lost. Weights apply to the GBM
fits and to the backbone ridge fits alike.

### 2.6 Uncertainty — [SETTLED]

Group-conditional (Mondrian) split conformal prediction on log residuals, using
**forward residuals only** (§2.7 defines the ledger that supplies them).

```
group(row) = use_family(row) × big(row)
use_family ∈ {Fire, Potable, Waste, Industrial, Other}     mapping in §4.4
big        = point ≥ T_big,   T_big = P95 of training-window per-tank Total Price
```

`big` is defined on the **predicted** value in calibration and at inference alike,
because the actual is unknown when the band is drawn; a group that cannot be
computed at inference is not a group.

For each group with ≥ 150 ledger residuals `r = log(actual) − log(point)`:

```
q80(group) = quantile(|r|, 0.80 · (1 + 1/n))
band80     = [ point / exp(q80),  point · exp(q80) ]
```

Groups with fewer than 150 residuals fall back to the use-family group; a family
with fewer than 150 falls back to the global pool. The bundle stores the table of
`q80` (and `q90`) per group with its `n`.

This is what makes "narrower for commodity work than for bespoke work" a property
of the design rather than a hope: Fire Protection's IQR is 12% against Waste Water's
27%, so their bands differ by construction. [PREDICTION] Empirical 80% coverage on
the protocol lands in 76–84% overall and in every group with n ≥ 150; Fire's band
half-width is at least 8 points narrower than Waste's. Falsified by coverage
outside 72–88% in any large group, which would mean the ledger residuals are not
exchangeable with new quotes (most likely cause: the ledger was scored in-sample,
§8 R9).

A heteroscedastic "difficulty model" (a second GBM on |residual|) was considered
and rejected: it adds a fitted component whose failure is invisible; the Mondrian
table is a lookup the estimator can read.

### 2.7 The residual ledger and the retransformation factor — [SETTLED] [LOAD-BEARING]

The **ledger** is an append-only table of forward predictions:

```
ledger(quote_key, due_date, bundle_id, point, book, band80_lo, band80_hi, group, actual_total, actual_components…, scored_at)
```

Rule: **every row is scored by a bundle trained strictly before that row's Due
Date.** In production this is automatic if the monthly job does two things in this
order: (1) score the newly arrived month's archive rows with the *currently
deployed* bundle and append to the ledger; (2) retrain. Reversing the order makes
the residuals in-sample, the conformal bands collapse to a fraction of their honest
width, and the retransformation factor drifts toward 1. That ordering is the
load-bearing detail. At first deployment the ledger is bootstrapped by running the
§3 backtest once and writing its forward predictions.

The ledger is also the production monitor (§7.6) and the source of the §3 metrics
going forward; the backtest and the monitor are the same computation.

**Retransformation factor.** From the most recent four quarters of ledger rows (or
all rows if fewer than 800):

```
s_band = mean( exp(r) )  over ledger rows in band,  r = log(actual) − log(point)
band ∈ {top5, rest}      top5 = point ≥ T_big   (predicted, same threshold as §2.6)
book  = point × s_band(row)
```

The band is assigned on the predicted value so that it can be assigned at
inference. The §3.4 metric `top5_bias` is defined on the actual value; the two
thresholds serve different purposes and are both recorded in the report.

Two bands, not one, because the aggregate bias is concentrated in the largest
quotes. `s` is clipped to [0.90, 1.25]; a value outside that range is a monitoring
alarm, not a correction to apply.

[PREDICTION] `s_rest` ≈ 1.01–1.03 and `s_top5` ≈ 1.08–1.16 before OPEN-2; if OPEN-2
adopts the physics term, `s_top5` should fall toward 1.03–1.08. Aggregate bias on
`book` lands within ±2%. Falsified if `book` aggregate bias remains beyond ±3% on
the headline window.

**[OPEN-3] Apply `s_top5` to `point` as well?** The prior investigation's
recalibration attempts were fitted in-sample, where the model is unbiased, so they
had nothing to learn; a factor fitted on forward residuals is a different object.
Options: (A) `point` unadjusted; (B) `point × s_top5` for rows in the top band
only. Decision rule: take B if top-5% dollar-weighted bias of `point` improves by
≥ 5 points and headline mean APE does not worsen by more than 0.10; otherwise A.
[PREDICTION] B improves top-5% bias by 5–10 points and worsens mean APE by
0.0–0.2, so the rule will probably reject it for `point` — which is fine, because
`book` already carries it. Record the trade either way.

### 2.8 Blend weight — [OPEN-4]

`point = w · point_components + (1 − w) · point_direct`. Prior evidence: anything
in 0.3–0.8 lands within 0.2 points. Options: `w ∈ {1.0, 0.7}`. Decision rule: take
0.7 unless 1.0 is within 0.1 points on headline mean APE **and** within 0.5 on
top-5% bias, in which case take 1.0 because it is simpler and the breakdown then
sums exactly to the point estimate. This is the only open question where the
tie-break favours the *less* insured option, and the reason is auditability: an
estimator asking "why doesn't the breakdown add up" is a real cost.

If `w < 1` is kept, the breakdown shown to the estimator is rescaled so its
components sum to `point` (each component multiplied by `point /
point_components`), and the sheet says so.

### 2.9 Edges of competence — [SETTLED]

The system's failure mode outside its competence must be **visible on the sheet**,
never a plausible-looking number. Three tiers:

1. **Refuse** (returns an error token, no number): missing or blank scope;
   `insulation_erection` without `insulation`; Diameter or Height outside the hard
   envelope `[0.5 × p0.5, 1.5 × p99.5]` of the training window; unknown Material;
   unknown Use Type; non-numeric geometry; Quantity < 1; a bundle that fails its
   integrity check.
2. **Warn** (returns the number plus a warning code the sheet must display):
   geometry outside `[p0.5, p99.5]` but inside the hard envelope (`EXTRAP`);
   predicted value in the top-5% band (`BIG`: "treat as a floor"); unknown State or
   Country (`GEO`: treated as missing, geography contribution unlearned); taxable
   with no rate on file for the State (`TAXRATE`: Total Price returned, Proposal
   Total withheld); Argentina (`AR_BIAS`); Wage Type missing when `construction =
   true` (`WAGE`); bundle older than 45 days (`STALE`); bundle older than 120 days
   (`STALE_HARD`, and the reliability tier is forced to C).
3. **Degrade gracefully**: unknown Deck Style, Floor Style, Bid Type or Sales
   Manager → treated as missing (the GBM handles missing natively), warning
   `UNSEEN:<field>`.

Time feature at inference: `t = min(t_due, t_train_max + 0.25)`. A linear drift
term extrapolated a year past the last training row is a guess about the market
that nobody made; capping it at one quarter is the conservative default and monthly
retraining makes the cap rarely bind. [LOAD-BEARING] for any deployment where
retraining lapses.

The reliability tier shown to the user (§7.4) is derived from the band width and
the warnings, so competence is communicated on every row, not only at the edges.

---

## 3. Measurement protocol (Engineer) — non-negotiable

Two people implementing this section independently must produce the same number.
Every number reported anywhere in the build log comes from this section. Numbers
from a random split, a grouped K-fold, an in-sample fit, or the training archive
scored by the bundle trained on it, are not results and must not appear in a
report.

### 3.1 Row eligibility — identical for fitting and scoring

A row is **usable** iff all of:

1. `Due Date` parses to a date.
2. `Diameter (ft) > 0` and `Height (ft) > 0`.
3. `Material Price > 0` and `Fabrication Price > 0` (these are always-present
   processes; a zero means an incomplete quote).
4. `Total Price > 0` and the identity `|Total Price − (Material + Fabrication +
   Construction + InsulMaterial + InsulConstruction + Freight)| ≤ max($1, 0.001 ×
   Total Price)` holds (blank components read as 0).
5. **Plausibility gate:** `20 ≤ Total Price / shell_area ≤ 250` where
   `shell_area = π · D · H` in ft², using the *undivided* per-tank Total Price.

Rows failing any test are excluded from **both** fitting and scoring. Filtering
one and not the other is the trap the brief names; the harness must have a single
`usable` mask and no second path. [PREDICTION] On the 2026-08-03 archive this
yields 6,679 ± 60 usable rows before the gate and about 173 gate failures. The
loader must print both counts and the count of rows failing each test; a
discrepancy of more than 5% against these figures must be explained in the build
log before any model is fitted (the most likely cause is a changed export format,
§4.6).

Nothing else is filtered. No de-duplication of revisions, no dropping of Budget
bids, Argentina, alternates, or any segment. Every such cleaning step was measured
and made the forward number worse; the only cleaning that helped was removing rows
whose *label* is known to be incomplete. The principle for any future filter:
remove rows whose label is wrong, never rows whose label is right but inconvenient.

### 3.2 Splits — rolling origin by calendar quarter

```
quarters   = calendar quarters of Due Date, from 2024Q4 to the last quarter with ≥ 100 usable rows
for Q in quarters:
    train  = usable rows with Due Date <  first day of Q
    score  = usable rows with Due Date in Q
    fit the entire pipeline on train only   — including the tax-rate table, the categorical
                                              vocabularies, the geometry envelope, the P95 "big"
                                              threshold, the standardisation constants, the
                                              recency weights (age measured from max Due Date in train),
                                              and the conformal table (from the ledger of prior quarters' forward predictions)
    predict score with the true scope flags (§4.2 backfill rule) and true Quantity
```

Refit from scratch each quarter. A pipeline object that carries any state from a
previous fit is a leak. In particular the tax table, the conformal table and the
`big` threshold are the three pieces of state a builder is most likely to compute
once on the whole archive; the harness must assert that each is rebuilt inside the
loop (a unit test fits on 2024 rows only and checks that no 2026 state appears in
the tax table).

Conformal calibration inside the backtest: the calibration pool for quarter Q is the
forward predictions from quarters before Q. For the first two quarters the pool is
small; coverage is reported from 2025Q2 onward.

### 3.3 Windows

- **Full series:** every quarter from 2024Q4. Reported as a table, one row per
  quarter, with `n_rows` and `n_quotes`.
- **Headline window:** quarters from **2026Q1** onward. This is the window on which
  the incumbent's 8.5% was measured and on which the training set exceeds ~4,500
  rows. All acceptance criteria in §6 are on the headline window. As the archive
  grows, the headline window grows; it never slides its start.
- A quarter is included only if it has ≥ 100 usable rows.

### 3.4 Metrics — all required on every reported result

Per quarter, over usable scored rows, per-tank (no Quantity extension anywhere):

```
APE_i          = |point_i − actual_i| / actual_i                       actual = per-tank Total Price (tax-exclusive, freight-inclusive)
mean APE       = mean_i APE_i
median APE     = median_i APE_i
p90 APE        = 90th percentile of APE_i  (numpy default linear interpolation)
agg_bias(x)    = (Σ_i x_i − Σ_i actual_i) / Σ_i actual_i               reported for x = point AND x = book
top5_bias(x)   = agg_bias over rows with actual_i ≥ T                  T = P95 of actual over ALL usable rows in the headline window (one fixed threshold)
coverage80     = share of rows with band80_lo ≤ actual ≤ band80_hi
```

Headline figures are the **unweighted mean across quarters** of each per-quarter
metric (the brief's "report the mean across quarters"), except `top5_bias`, which
is computed once pooled over the headline window because a per-quarter top-5% is
~35 rows. Report the pooled-over-rows versions alongside, labelled `pooled`.

Additional required diagnostics:

- **New-quote subset:** the same metrics restricted to rows whose `Quote #` does
  not appear in the training window. The rolling protocol legitimately lets
  revision 3 see revision 1; this subset shows how much of the headline number is
  that. [PREDICTION] New-quote mean APE is 1.5–3 points above the headline.
- **By segment:** mean APE, median signed error, IQR of signed error, and n, for
  use family, country (top 6), wage type, material, and `big`.
- **Coverage by conformal group.**
- **Retrain-cadence simulation** (secondary, not the protocol): month-by-month
  over the headline window with the bundle refit monthly. Comparable to the prior
  8.69% figure, and the number the deployment will actually experience.

A report missing any required metric fails review. A report quoting the median in
its summary line fails review.

### 3.5 Deciding between variants

The protocol cannot support many comparisons: with ~3–4 headline quarters the
paired standard error of a mean-APE difference between two variants is roughly
0.1–0.15 points. Therefore:

- A variant "beats" another only if it is better by the margin stated in its
  decision rule **and** wins in at least half of the headline quarters plus the
  three most recent full-series quarters (i.e. wins ≥ 4 of the ~6 quarters
  compared).
- The total experiment budget for the build is **12 full protocol runs** beyond the
  increments B3–B7 in §5. OPEN-3 and OPEN-4 need no refit: both are post-hoc
  arithmetic on the stored per-component predictions of an existing run, and the
  harness must store those predictions so that this is possible. Every run is logged (variant, commit, seed, all §3.4 metrics)
  whether or not it is adopted. A thirteenth run requires a written reason in the
  build log.
- Fixed seeds throughout. Determinism test: two runs of the same commit on the
  same file produce byte-identical metric tables.

### 3.6 What a random split does here, for the record

6,892 rows are 3,018 quotes. A random split puts revision 2 in training and
revision 3 in test, and reports roughly 3% where the truth is roughly 8%. Grouping by
`Quote #` fixes duplication but lets the model see future prices, which drift; it
reports ~12.7% median in the prior investigation for a different reason (less
relevant training data). Neither is a deployment number. The harness must not
expose a random or grouped K-fold option at all; if a developer wants a quick
smoke test, the harness offers `--quarters 2026Q2` (one quarter of the real
protocol), not a different protocol.

---

## 4. Data contract (Engineer)

### 4.1 Training input: the archive export

One CSV, UTF-8, header row, one row per tank line per revision. The 42 columns of
the brief. The loader reads the columns below and **ignores every other column**
(an unknown extra column is logged, not used; a model that silently picks up a new
column is how leakage arrives in year two).

Type key: `num` = parsed as float, blank → NaN; `cat` = string, stripped, blank →
missing; `date` = parsed with `dayfirst=False` then `MM/DD/YYYY` fallback; `bool01`
= derived 0/1.

| Column | Type | Role | Required in archive | On absence |
|---|---|---|---|---|
| `Quote #` | cat | grouping key, new-quote diagnostic | yes | reject file |
| `Revision #` | num | **excluded** (calendar features hurt) | no | — |
| `Due Date` | date | time axis `t`, quarter assignment | yes | row unusable |
| `Tank Name` | cat | free-text features (§4.4) | no | features = "generic" |
| `Bid Type` | cat | feature | no | missing |
| `Status` | cat | **excluded** (post-hoc outcome) | no | — |
| `Company Name`, `Customer Name`, `Project Name`, `City`, `Sales Rep` | cat | **excluded** (§4.5) | no | — |
| `Country` | cat | feature | no | missing + `GEO` warn |
| `State` | cat | feature; tax-rate key | no | missing + `GEO` warn |
| `Sales Manager` | cat | feature, [OPEN-5] | no | missing |
| `Wage Type` | cat | feature after recoding (§4.4) | no | missing |
| `Miles to Site (From TBT)`, `Miles to Site (From GT)` | num | features | no | NaN |
| `Ss`, `S1` | num | features | no | NaN |
| `Quantity` | num | feature (`log Quantity`); extension at output only | no | 1 |
| `Material` | cat | feature; backbone one-hot | yes | row unusable |
| `Use Type` | cat | feature; conformal family | yes | row unusable |
| `Deck Style`, `Floor Style` | cat | features | no | missing |
| `Diameter (ft)`, `Height (ft)` | num | geometry | yes | row unusable |
| `Freeboard (in)` | num | feature; physics term | no | 0 |
| `Usable Capacity` | num | **not a feature**; consistency check only (§4.6) | no | — |
| `Commission (%)`, `Margin (%)`, `Contingency (%)`, `Insulation Margin (%)`, `Insulation Contingency (%)` | num | **banned** (policy) | no | — |
| `Material Price`, `Fabrication Price`, `Construction Price`, `Insulation Material Price`, `Insulation Construction Price`, `Freight Price` | num | **targets**; scope backfill; banned as features | yes | row unusable |
| `Total Tax` | num | tax-rate table only; banned as feature | no | 0 |
| `Proposal Total`, `Total Price` | num | `Total Price` is the scoring target; both banned as features | yes | row unusable |

### 4.2 Scope backfill from the archive — [SETTLED]

The archive has no scope columns. For training and backtesting only:

```
IS_CONSTRUCTION         = Construction Price            > 0
IS_INSULATION           = Insulation Material Price     > 0
IS_INSULATION_ERECTION  = Insulation Construction Price > 0
IS_FREIGHT              = Freight Price                 > 0
IS_TAXABLE              = Total Tax                     > 0
```

Assert `IS_INSULATION_ERECTION ⇒ IS_INSULATION` on every row; a violation is a
loader error (the archive has zero such rows; one appearing means the export
changed). If the export ever carries real scope columns (§9 item 3), they take
precedence and the backfill is used only to cross-check them, with disagreements
logged.

A blank price and a genuinely excluded scope are indistinguishable after the fact.
This is a known limitation of the backfill and is why §9 ranks capturing scope at
quote time so highly.

### 4.3 Feature matrix — [SETTLED]

Every model (six components, direct) sees the same matrix. Row order and column
order are fixed in the bundle.

Numeric:

```
log_D, log_H, logD_x_logH, t, t_x_logD (if OPEN-1 adopted), log_steel_lb_est (if OPEN-2 adopted)
log_shell_area   = log(π · D · H)
log_floor_area   = log(π · (D/2)²)
aspect           = H / D
freeboard_in
log_quantity     = log(Quantity)
miles_tbt, miles_gt          (NaN allowed)
Ss, S1                       (NaN allowed)
name_len, name_words, name_has_digits, name_generic     (§4.4)
```

Categorical (native categorical handling; vocabulary fixed at fit time from the
training window; unseen at inference → missing):

```
Material, Use Type, Deck Style, Floor Style, Country, State, Wage Type (recoded),
Bid Type, Sales Manager (OPEN-5), use_family, name_family (§4.4)
```

Scope (0/1 numeric):

```
IS_CONSTRUCTION, IS_INSULATION, IS_INSULATION_ERECTION, IS_FREIGHT
```

`IS_TAXABLE` is **not** a feature of any price model; it only gates the tax
arithmetic. Tax status has no bearing on the pre-tax price of a tank, and letting
the model see it invites it to learn customer type by proxy.

Scope flags are features of every component model, including Material and
Fabrication: a tank that TBT also erects may be fabricated differently (field-weld
versus shop-weld splits), and the prior evidence is that the flags are the largest
single lever. The direct model also sees them.

### 4.4 Derived fields — formulas

```
t                = (Due Date − 2024-01-01).days / 365.25
shell_area       = π · D · H
use_family       = Fire        if Use Type contains "Fire"
                 = Potable     if contains "Potable" or "Water Storage" and not "Waste"
                 = Waste       if contains "Waste" or "Sewage" or "Sludge" or "Digest"
                 = Industrial  if contains "Industrial" or "Silo" or "Process" or "Chemical"
                 = Other       otherwise
                 (matching is case-insensitive on Use Type only; the vocabulary of Use Type values is
                  13 strings — the builder enumerates them in the bundle and this mapping is a fixed table,
                  not a regex applied at runtime)

Wage Type recoding:
    "No Erection Included"  → missing, and IS_CONSTRUCTION must be 0 on that row (else row is flagged
                              CONFLICT and the flag wins: the row trains with Wage Type missing)
    everything else         → unchanged
    Reason: this value is a scope statement disguised as a labour-rate class. With scope explicit it is
    redundant at best and, at inference, a way for the estimator's Wage Type dropdown to contradict the
    scope block.

Tank Name features (all computed on lower-cased, whitespace-normalised name; missing → generic):
    name_len         = number of characters
    name_words       = number of whitespace-separated tokens
    name_has_digits  = 1 if any digit
    name_generic     = 1 if name matches ^(tank|t|tk)?\s*[-#]?\s*\d*[a-z]?$ or is empty
    name_family      = first match in this ordered list, else "none":
        backwash | (ext.? rafter|eccs|hdg) | (dome|geodesic|membrane) | (digest|anaerob|aerob) |
        (leachate|landfill) | (sludge|slurry|thicken) | (equaliz|detention) | (process|chemical|oil) |
        (dual|combo|hybrid|zone)
    The list is fixed. Do not extend it from the data during the build; that is a target-encoding
    search wearing a regex.

Tax-rate table (fitted on the training window only):
    base_i   = Material + Fabrication + Construction + InsulMaterial + InsulConstruction   (row i)
    rate_i   = Total Tax / base_i                        over rows with Total Tax > 0 and base > 0
    rate[State] = median(rate_i) over rows in that State with n ≥ 5
    rate[Country] fallback for non-US rows with n ≥ 5
    no entry → warning TAXRATE at inference when taxable = true
```

Backbone basis, steel-weight estimate, standardisation: §2.3.

### 4.5 Leakage exclusion list — complete

Banned as features under any name, transformation, or encoding:

1. The six component prices, `Total Tax`, `Proposal Total`, `Total Price`.
   Arithmetic identities of the target.
2. `Margin (%)`, `Contingency (%)`, `Insulation Margin (%)`, `Insulation
   Contingency (%)`, `Commission (%)`. Pricing-policy decisions applied during
   quoting; margin is verifiably baked into the component prices. A model using
   them can only score a quote already built.
3. `Status`. Post-hoc outcome; "Revised" also encodes revision count.
4. `Revision #`. Not leakage in the strict sense, but a calendar feature that
   measured worse and that encodes how many times a quote has been reworked, which
   is not known for a fresh quote.
5. `Company Name`, `Customer Name`. Target encoding learned the customer and
   stopped learning the tank (11.02% vs 9.99%); a new-customer quote is where an
   estimate is most valuable.
6. `Project Name` (85% empty), `Sales Rep` (73% empty), `City` (cardinality with
   no support).
7. `Quote #` as a numeric. Quote numbers are sequential and encode time and
   customer.
8. `Usable Capacity`. Derived from geometry; redundant, and if entered by hand it
   is an estimator's judgement that may already reflect the price.
9. `IS_TAXABLE` as a price-model feature (§4.3).
10. `Wage Type = "No Erection Included"` as a category (§4.4).
11. **Any state computed on rows outside the training window**: the tax table,
    categorical vocabularies, standardisation constants, the `big` threshold, the
    geometry envelope, the conformal table, the retransformation factor. These are
    leakage of the future, and they are the subtler kind because each is "just a
    lookup".
12. **Any prediction from a bundle trained on or after a row's Due Date**, in the
    ledger or in a report.

The harness enforces 1–10 by name: the feature builder raises if any banned column
is present in the feature matrix. Enforce 11–12 by construction (§3.2) and by the
unit tests in §6.4.

### 4.6 Load-time validation — reject, do not repair

The loader **rejects the file** (no model is trained, the previous bundle stays
deployed, the scheduler reports failure) when any of these fail:

| Check | Threshold | What it catches |
|---|---|---|
| Required columns present | all of §4.1 "yes" | export format change |
| `Due Date` parse rate | ≥ 98% | date format change |
| Identity `Total Price = Σ six components` | holds on ≥ 99.5% of rows with all present | a re-defined Total Price column |
| Identity `Proposal Total = five ex-freight + Total Tax` | ≥ 99.5% | same |
| **Per-tank tripwire:** median of `Total Price / shell_area` by Quantity ∈ {1,2,3,4} | ratio of qty-k median to qty-1 median within [0.6, 1.4] for each k | someone upstream started extending prices by Quantity; the 1/n signature returns |
| Usable-row count | ≥ 90% of the previous accepted file's count, and ≥ 1,000 | truncated export |
| Insulation nesting | zero violations | changed component semantics |
| Duplicate `(Quote #, Revision #, Tank Name)` rows with different prices | ≤ 1% | export joined wrongly |
| Newest Due Date | not more than 120 days in the future | future-dated garbage |

Warnings (logged, file accepted): `Usable Capacity` disagrees with `π (D/2)² (H −
Freeboard/12) × 7.48` by more than 25% on more than 5% of rows; any categorical
vocabulary gains a new value; any single State's tax rate moves by more than 1
point against the previous bundle.

The loader never fills, clips, or corrects a price column. A row that fails §3.1 is
dropped and counted; a file that fails this table is refused.

### 4.7 Inference input contract

One record per tank line. Same names as the Excel surface (§7.5).

| Field | Type | Required | Validation |
|---|---|---|---|
| `diameter_ft` | float | yes | > 0; envelope per §2.9 |
| `height_ft` | float | yes | > 0; envelope per §2.9 |
| `freeboard_in` | float | no (0) | ≥ 0, ≤ 0.5 × height in inches |
| `quantity` | int | no (1) | ≥ 1; used only to extend the output |
| `material` | enum | yes | in bundle vocabulary, else refuse |
| `use_type` | enum | yes | in bundle vocabulary, else refuse |
| `deck_style`, `floor_style`, `bid_type`, `sales_manager` | enum | no | unseen → missing + `UNSEEN` warning |
| `country`, `state` | enum | no | unseen → missing + `GEO` warning |
| `wage_type` | enum | no | unseen → missing; "No Erection Included" → recoded per §4.4 and must agree with `construction` |
| `miles_tbt`, `miles_gt`, `ss`, `s1` | float | no | NaN allowed |
| `due_date` | date | no (today) | `t` capped per §2.9 |
| `tank_name` | str | no | free text |
| `construction`, `insulation`, `insulation_erection`, `freight`, `taxable` | bool | **yes, all five** | blank → refuse `SCOPE`; nesting enforced |

---

## 5. Build sequence (Engineer)

Each increment is independently testable and has an acceptance criterion in §6.
Do not start an increment until the previous one's criterion is met and logged.
Order matters: measurement is built before any model, because the prior effort's
worst mistakes were measurement mistakes that looked like progress.

| # | Increment | Demonstrates | Throw away if |
|---|---|---|---|
| **B0** | Loader, data contract, §4.6 validation, scope backfill, `usable` mask | Row counts match §3.1 predictions; every §4.6 check has a failing fixture | — |
| **B1** | Protocol harness: rolling origin, all §3.4 metrics, report format, determinism, leakage tests of §6.4. Trivial baseline: median `$/shell_area` by `use_family × Material` from the training window, times shell area | The harness works end to end on a model that cannot cheat. [PREDICTION] 25–35% mean APE | — |
| **B2** | Ridge backbone only (basis of §2.3 without OPEN terms), six components + direct, scope-gated sum, `w = 0.7` | Extrapolation floor; the top-5% bias of a pure power law. [PREDICTION] 35–45% median, top-5% bias ≈ −50% | — |
| **B3** | Reference model: backbone + single GBM (variant 1), no recency weights, no name features. **This is the re-measured reference for every relative target in §6** | [PREDICTION] 9.5–10.5% headline mean, comparable to the prior 9.99% | — |
| **B4** | Recency weighting, half-life 1 yr | [PREDICTION] −0.2 to −0.4 | < 0.1 improvement → keep anyway (cost is one line); log it |
| **B5** | Second GBM variant, log-space average | [PREDICTION] −0.1 to −0.2 | < 0.05 improvement → drop the second variant (halves retrain time) |
| **B6** | Tank Name features (§4.4) | [PREDICTION] −0.05 to −0.15 | < 0.05 improvement → drop all name features; they are a maintenance surface |
| **B7** | Ledger, Mondrian conformal, `book` factor, reliability tiers | Coverage 76–84%; tier ordering; `book` bias within ±2% | — |
| **B8** | OPEN-1 × OPEN-2 (four runs), then OPEN-3, OPEN-4, OPEN-5 (see below) | Decision rules applied; results logged | per rule |
| **B9** | Hyperparameter search: 4 runs total, on variant 1 only: `(lr, iter) ∈ {(0.03,1200), (0.02,1800)} × min_samples_leaf ∈ {15, 30}`. Keep the best by headline mean if it beats the default by ≥ 0.15, else keep the default | The search is small on purpose | — |
| **B10** | Guard rails: §2.9 refusals and warnings, §6.5 negative tests, input contract | Every negative test returns the specified token | — |
| **B11** | Retrain job: unattended, scheduled, ledger-then-retrain ordering, bundle integrity, rollback to previous bundle on any failure | Runs on the target laptop in ≤ 5 min; a deliberately corrupted file leaves the old bundle deployed | — |
| **B12** | Batch scorer (CSV/xlsx in, xlsx out) | 5,000 rows in ≤ 10 s after bundle load | — |
| **B13** | Excel surface, Phase 1 functions only (§7) | Non-volatile; no per-recalculation inference; error tokens render | — |
| **B14** | Archive audit workbook (every usable row re-scored *forward* from the ledger, sorted worst-first, worst component named) | The adoption evidence for §7.7 Phase 2 | — |
| **B15** | Phase 2 and Phase 3 surfaces, gated by §7.7 evidence | — | — |

**[OPEN-5] Sales Manager as a feature.** It is known at quote time and is a proxy
for region and customer segment; it is also a person, and a model that learns a
person's habits has a shelf life. Options: (A) exclude; (B) include as
categorical. Decision rule: include only if headline mean APE improves by ≥ 0.15
and new-quote-subset mean APE also improves by ≥ 0.1; otherwise exclude.
[PREDICTION] Improvement of 0.0–0.1; the rule excludes it.

Runtime budget, all on a 4-core laptop CPU: one full fit of the pipeline (7 models
× 2 variants + 7 ridges + tables) ≤ 3 min; one protocol run (~8 refits) ≤ 25 min;
inference ≤ 2 ms per row after bundle load, bundle load ≤ 2 s. [PREDICTION] based on
the prior implementation's ~2 min fits with the same estimator family. Falsified if
a fit exceeds 6 min, in which case reduce `max_iter` of variant 2 before touching
anything else.

Dependencies: `pandas`, `numpy`, `scikit-learn ≥ 1.3`, `joblib`, `openpyxl`,
`xlwings`. Nothing else. Pin exact versions in a lockfile; the bundle records them
and refuses to load under a different scikit-learn major.minor.

---

## 6. Acceptance criteria (Engineer)

All on the §3 protocol, headline window, both `point` and `book` reported.
"Reference" = B3 as re-measured on the same harness and same file.

### 6.1 Accuracy

| Stage | Pass | Fail |
|---|---|---|
| B3 reference | mean APE within 10.5%; if outside 9.0–11.0% the harness is suspect, stop and audit before continuing | — |
| B7 (full system before open experiments) | mean APE ≤ 8.5% absolute; expected ≈ reference − 0.6 | > 8.7% |
| B8 final | mean APE ≤ 8.0% (target); must-pass ≤ 8.5%; expected ≈ reference − 0.9 | > 8.5% |
| Stretch | ≤ 7.6% | — |
| Median APE, final | ≤ 5.9% | — |
| p90 APE, final | ≤ 18.0% (prior: 18.3%) | > 19.5% |

### 6.2 Calibration

| Metric | Pass | Fail |
|---|---|---|
| `agg_bias(book)`, headline | within ±2.0% | outside ±3.0% |
| `top5_bias(book)`, headline | within ±5% | outside ±8% |
| `agg_bias(point)` | reported; expected −2% to −4%; no criterion | — |
| `top5_bias(point)` | ≥ −10% if OPEN-2 adopted; reported otherwise | — |
| `s_top5`, `s_rest` | inside [0.90, 1.25] | outside → alarm, factor not applied |

### 6.3 Uncertainty

| Metric | Pass | Fail |
|---|---|---|
| `coverage80`, headline overall | 76–84% | outside 72–88% |
| `coverage80` per group with n ≥ 150 | 74–86% | any group outside 70–90% |
| Band half-width, Fire vs Waste | Fire narrower by ≥ 8 points | Fire not narrower |
| Tier A rows | mean APE ≤ 6% | > 8% (tiers are mislabelled) |
| Tier C rows | mean APE ≥ tier A mean APE + 6 points | — |

### 6.4 Harness integrity tests (unit tests, must pass on every commit)

1. **Leakage tripwire:** a test passes a frame containing `Total Price` (and, in
   turn, each banned column of §4.5) to the feature builder and asserts it raises
   before any fit. (Left unguarded, the model would report under 1% mean APE and
   look finished; that is why the guard is a test and not a code comment.)
2. **Future-state tripwire:** fitting on rows before 2025-01-01 produces a tax
   table, vocabulary, envelope and `big` threshold identical to those from a file
   truncated at 2025-01-01.
3. **Quantity tripwire:** a fixture whose prices are extended by Quantity fails the
   §4.6 per-tank check.
4. **Filter symmetry:** the row set used for fitting quarter Q and the row set used
   for scoring quarter Q+1's training are produced by the same function; a test
   mutates the gate bounds and asserts both counts change.
5. **Ledger ordering:** the retrain job on a fixture with a new month appends
   ledger rows whose `bundle_id` is the *previous* bundle.
6. **Determinism:** two protocol runs, same commit, byte-identical reports.
7. **No random split:** the harness module exposes no function that accepts a
   random seed for splitting.

### 6.5 Negative acceptance — the system must refuse

Each returns the named error token and **no number**, in both the Python API and
the Excel surface:

| Input | Token |
|---|---|
| Any of the five scope fields blank or non-boolean | `#SCOPE` |
| `insulation_erection = true, insulation = false` | `#SCOPE_NEST` |
| `wage_type = "No Erection Included"` with `construction = true` | `#SCOPE_WAGE` |
| Diameter or Height outside the hard envelope, or ≤ 0, or non-numeric | `#RANGE` |
| Material not in vocabulary | `#UNKNOWN:Material` |
| Use Type not in vocabulary | `#UNKNOWN:UseType` |
| Quantity < 1 or non-integer | `#QTY` |
| Bundle missing, corrupt, hash mismatch, or wrong library version | `#MODEL` |
| Request for `proposal_total` when `taxable = true` and no rate for the State | `#TAXRATE` (Total Price still returned in its own field) |
| Request for a win probability, or any field named like one | `#UNSUPPORTED` — the API has no such field and the docs say why (§4.13 of the brief) |

And it must **not** refuse (returns a number plus a warning code): geometry between
p99.5 and the hard envelope (`EXTRAP`); unseen State (`GEO`); Argentina
(`AR_BIAS`); stale bundle (`STALE` / `STALE_HARD`); top-band prediction (`BIG`).

A warning code that does not render on the sheet is a failed acceptance test;
§7.4 specifies how it renders.

### 6.6 Operational

| Requirement | Pass |
|---|---|
| Unattended retrain on the target laptop | ≤ 5 min wall clock, exit 0, new bundle passes integrity, old bundle retained |
| Retrain on a file that fails §4.6 | exit non-zero, old bundle still deployed, failure reason in the log and in `TBT_MODEL_INFO()` |
| Batch scoring | 5,000 rows ≤ 10 s |
| Excel | no function is volatile; filling a 200-row range does not exceed 5 s; a 2,000-row range is refused by the UDF with `#BATCH` and directed to the batch scorer |
| Data locality | no network call anywhere in the code path; a test runs with networking disabled |

---

## 7. User surface and adoption (Designer)

### 7.1 Who the user is and what the tool is for

A tank estimator working in Excel, building a quote line by line. They are not a
data scientist and will not read a probability. At 8% mean error the model cannot
set a price. It can reliably catch a transposed dimension, a forgotten stainless
premium, a missing scope line, and a quote that drifted from what TBT has been
charging. **The product is a second opinion with a reason attached**, and it earns
the right to be more than that by being right in public for a while.

### 7.2 What the estimator supplies and what the system derives

**Supplied by the human (a "scope block", once per quote, copied to its tank
rows):**

| Field | Why the human, not the system |
|---|---|
| Erection included? | Commercial decision made in the room |
| Insulation supplied? / Insulation installed by us? | Same; and the archive says nothing predicts it |
| Freight included? | Same; 57% of Mexico rows carry none because the customer ships |
| Sales tax applies to this customer? | Exemption status is a customer attribute; 0% Oregon, 38% Texas, 91% New Jersey |

**Supplied by the human (per tank row):** diameter, height, freeboard, quantity,
material, use type, deck style, floor style, country, state, wage type, bid type,
tank name (optional, encouraged), due date (defaults to today).

**Derived by the system, never asked:** all geometry, the time feature, the tax
rate, the steel-weight estimate, the reliability tier, warnings.

A system that guesses any row of the first table is confidently wrong in a way
that is invisible on the sheet. That is why the scope block is five dropdowns with
no default and the function returns `#SCOPE` until they are filled. Five clicks
per quote, not per tank.

### 7.3 What appears on the sheet

Phase 1 (ships first), one block per tank row, all values not formulas after
scoring (§7.6):

```
| Model check | Reason                     | Reliability | Notes                     |
| LOW −22%    | Construction −38%          | B           | BIG: treat as floor       |
| OK  +3%     |                            | A           |                           |
| #SCOPE      |                            |             | fill scope block          |
| HIGH +31%   | Material +44%, Fab +19%    | C           | EXTRAP, GEO: state unseen |
```

- **Model check** compares the estimator's own Total Price to the model's 80%
  band: `OK` inside the band, `LOW` / `HIGH` outside, with the signed gap as a
  percentage of the estimator's own number. The model's price is not displayed in
  Phase 1. It can be backed out from the gap, but it is not on the sheet, and not
  being on the sheet is what matters for anchoring (§8, R14).
- **Reason** names the one or two components whose gap against the estimator's
  own component lines is largest (requires the estimator's component cells as
  inputs; if not provided, blank). This is what makes a disagreement actionable:
  "check the construction line", not "this looks off".
- **Reliability** is the tier (§7.4).
- **Notes** renders every warning code with a short phrase. Codes are the same
  strings as in §2.9 so the log and the sheet agree.

Phase 2 adds a component breakdown block (six values plus tax and the band).
Phase 3 adds `point`, the band, and a `book` column for portfolio sheets.

### 7.4 Reliability tiers — [SETTLED]

Derived, not fitted:

```
half_width = exp(q80(group)) − 1            from the conformal table
tier = A  if half_width ≤ 0.12 and no warnings
     = B  if half_width ≤ 0.20 and no warning in {EXTRAP, BIG, GEO, AR_BIAS, STALE_HARD}
     = C  otherwise
```

Rendered as a letter with a fixed legend on the sheet: "A — commodity work, model
usually within ±12%. B — standard work, within ±20%. C — bespoke or unfamiliar,
treat as a rough check." The thresholds are stated on the sheet, and the tier's
legend is regenerated from the bundle so it never lies about the current bands.
[PREDICTION] Fire Protection tanks in CS at common sizes land in A; most Waste
Water in B or C; everything with a warning in C.

### 7.5 Functions — signatures, returns, errors

Python core (`tbt` package). All prices in USD, per tank unless the field says
`extended`.

```python
@dataclass
class QuoteInput:            # field names and validation per §4.7
    diameter_ft: float; height_ft: float; material: str; use_type: str
    construction: bool; insulation: bool; insulation_erection: bool; freight: bool; taxable: bool
    freeboard_in: float = 0.0; quantity: int = 1
    deck_style: str | None = None; floor_style: str | None = None; bid_type: str | None = None
    country: str | None = None; state: str | None = None; wage_type: str | None = None
    sales_manager: str | None = None; miles_tbt: float | None = None; miles_gt: float | None = None
    ss: float | None = None; s1: float | None = None
    due_date: date | None = None; tank_name: str | None = None

@dataclass
class Estimate:
    point: float                    # conditional median, per tank, tax-exclusive, freight-inclusive
    book: float                     # point × s_band
    band80: tuple[float, float]
    band90: tuple[float, float]
    components: dict[str, float]    # six keys, absent scope → 0.0; rescaled to sum to point (§2.8)
    tax: float | None               # None when taxable and no rate (warning TAXRATE)
    proposal_total: float | None    # five ex-freight components + tax, or None
    extended_point: float           # point × quantity
    tier: Literal["A", "B", "C"]
    group: str                      # conformal group used
    warnings: list[str]             # codes from §2.9
    bundle_id: str; trained_through: date

class RefusalError(ValueError): code: str      # one of the §6.5 tokens

def estimate(q: QuoteInput, bundle=None) -> Estimate                      # raises RefusalError
def estimate_many(df: pd.DataFrame, bundle=None) -> pd.DataFrame          # one row per input; refusals become an `error` column, never an exception
def check(q: QuoteInput, quoted_total: float, quoted_components: dict | None = None) -> Check
    # Check(flag: "OK"|"LOW"|"HIGH", gap_pct: float, reasons: list[tuple[str, float]], tier, warnings)
def train(archive_csv: Path, out_dir: Path) -> TrainReport                # §5 B11; never overwrites a bundle in place
def backtest(archive_csv: Path, out_dir: Path, quarters: list[str] | None = None) -> MetricsReport   # §3, writes the report and ledger bootstrap
def ledger_append(archive_csv: Path, bundle_dir: Path) -> int             # scores rows newer than the ledger with the deployed bundle
def model_info(bundle_dir: Path) -> ModelInfo                             # bundle_id, trained_through, age_days, row counts, s factors, band table, last retrain status
```

Excel (xlwings UDFs, all `@xw.func`, none volatile, all with a per-session memo
cache keyed on the argument tuple):

```
=TBT_CHECK(quoted_total, diameter, height, material, use_type, scope_range, [options_range], [components_range])
    → 1×4 array: check, reason, tier, notes        (Phase 1)
=TBT_BREAKDOWN(diameter, height, material, use_type, scope_range, [options_range])
    → 1×9 array: six components, tax, band_lo, band_hi     (Phase 2)
=TBT_ESTIMATE(diameter, height, material, use_type, scope_range, [options_range])
    → 1×5 array: point, band_lo, band_hi, tier, notes       (Phase 3)
=TBT_BOOK(range_of_estimate_rows) → extended book total for a portfolio sheet, with the s factors used   (Phase 3)
=TBT_MODEL_INFO() → 1×6: bundle_id, trained_through, age_days, last_retrain_status, s_rest, s_top5
```

`scope_range` is a 1×5 range of Yes/No cells; `options_range` is a 1×N range
whose header row names the optional fields (§4.7 names) so columns can be added
without changing signatures. Errors return the §6.5 token as a string in the first
cell of the array; Excel shows it, and conditional formatting colours it. Any
function called on a range of more than 500 rows returns `#BATCH` and the ribbon
button is the path.

Ribbon (xlwings `RunPython`): **Score sheet** — scores every row of the active
table via `estimate_many` in one call and writes values (not formulas) into the
model block; **Retrain now**; **Open audit**.

### 7.6 Recalculation and staleness

Excel recalculates aggressively. The design assumes it:

- UDFs are non-volatile and memoised; the same arguments never trigger a second
  inference in a session.
- The recommended surface is the **Score sheet** button, which writes values. The
  UDFs exist for single-row what-ifs.
- The bundle is loaded once per Excel session and re-loaded only when
  `bundle_id` changes on disk.
- `TBT_MODEL_INFO()` sits in a visible cell of the template; conditional
  formatting turns it amber at `STALE` and red at `STALE_HARD`.

### 7.7 Adoption path — [SETTLED]

| Phase | Ships | Withheld | Evidence that unlocks the next phase |
|---|---|---|---|
| **0 — Shadow** (weeks 1–4) | Nothing on estimators' sheets. The retrain job runs monthly; the ledger accumulates; the audit workbook (B14) is produced | Everything | Two monthly retrains completed unattended; ledger coverage80 within 72–88% on live rows; an estimator has adjudicated the 50 worst audit rows and classified each as model-wrong / quote-wrong / can't-tell |
| **1 — Check** | `TBT_CHECK` with tier and notes. No model price visible | Point estimate, band, breakdown, `book` | ≥ 200 live quotes checked; of those flagged LOW/HIGH and adjudicated, ≥ 50% judged "worth a second look" by the estimator; at least one genuine error caught and documented; no estimator reports a `#SCOPE`-free wrong scope (i.e. the block was filled but wrong) more than twice — if that happens, the scope block UI is redesigned before Phase 2 |
| **2 — Breakdown** | `TBT_BREAKDOWN`, worst-component reasons on every check | Point total, `book` | One further quarter with ledger headline mean APE ≤ 9% on live rows and `book` aggregate bias within ±3% |
| **3 — Estimate** | `TBT_ESTIMATE`, `TBT_BOOK` on portfolio sheets | Anything resembling win probability, forever, until §9 item 7 exists | — |

Why the point estimate is withheld until Phase 3: a number on the sheet becomes an
anchor. Once estimators anchor on the model, the archive starts to contain the
model's own output, and the next retrain learns itself (§8, R14). Withholding the
number until the check has proven itself keeps the archive independent for the
period in which the model most needs honest data. When Phase 3 ships, every
quote that displayed a model price is tagged in the ledger (`model_shown = true`)
so the feedback loop is measurable.

### 7.8 Monitoring in production

From the ledger, monthly, in `model_info` and in a one-page report:

- Rolling three-month mean APE, median signed error, coverage80, by tier.
- Alarm if median signed error is outside ±3% for two consecutive months (market
  moved faster than the retrain).
- Alarm if coverage80 is below 70% or above 90% for two months.
- Alarm if `s_top5` leaves [0.90, 1.25].
- Alarm if retrain failed, with the §4.6 reason.

Alarms render in `TBT_MODEL_INFO()` and in the retrain log. Nothing emails
anything; the data does not leave the machine.

---

## 8. Risk register

Likelihood: H/M/L. Each entry names its detection and its mitigation.

### 8.1 Good-faith regressions a maintainer is likely to introduce

| # | Regression | Why it looks right | L | Detected by | Mitigation |
|---|---|---|---|---|---|
| R1 | Default a blank scope flag ("freight is almost always yes") or reintroduce a scope classifier "as a fallback" | AUC 0.91–0.99 tests beautifully | H | §6.5 negative tests fail | Tests are in CI; §2.4 and this row explain why; `#SCOPE` is a feature |
| R2 | Divide prices by Quantity "to get per-tank" | Every other pricing dataset needs it | H | §4.6 tripwire; §6.4 test 3 | The 1/n diagnostic is codified |
| R3 | Filter implausible rows from scoring but not training (or vice versa) | Looks like a reporting choice | H | §6.4 test 4 | Single `usable` mask |
| R4 | De-duplicate revisions "to remove near-duplicates" | Statistically virtuous | M | Headline number jumps ~2.5 points worse | §3.1 forbids; §4.11 of the brief recorded 12.49% |
| R5 | Drop Argentina / Budget bids / alternates "to clean the data" | Each looks like noise | M | Headline worse | §3.1 principle: remove wrong labels, never inconvenient ones |
| R6 | A "quick" random or grouped K-fold check | Faster; every tutorial does it | H | Reported number ~3–5 points too good | Harness exposes no such split (§3.6); reviewers reject any number not from §3 |
| R7 | Compute the tax table / vocab / `big` threshold / conformal table once on the whole file | It's "just a lookup" | H | §6.4 test 2 | Rebuild inside the loop, asserted |
| R8 | Make a UDF volatile, or fire inference per cell on a 2,000-row sheet | Live-updating feels better | M | Workbook locks; `#BATCH` | Non-volatile + memo + 500-row cap + button |
| R9 | Retrain first, then score the new month into the ledger | Natural order of operations | H | Bands collapse; coverage > 90%; `s` → 1 | §6.4 test 5; ordering is load-bearing (§2.7) |
| R10 | Remove the ridge backbone because it alone has 40% error | It looks useless in isolation | M | Top-5% bias returns to −25% | §2.3 marks it load-bearing; acceptance §6.2 catches it |
| R11 | Let `t` extrapolate unbounded when retraining lapses | Nobody notices for months | M | `STALE_HARD`; prices drift up linearly | Cap at one quarter (§2.9) |
| R12 | Quote the median in a summary | It's a nicer number | H | Review | §3.4 fails any report whose summary line is the median |
| R13 | Extend the `name_family` regex list from what looks predictive in the data | Feels like feature engineering | M | New-quote subset worsens while headline improves | §4.4 fixes the list; changes require a protocol run and log entry |
| R14 | Feedback loop: estimators anchor on the model, the archive fills with model output, the model learns itself | Invisible; accuracy appears to improve | M (rises with Phase 3) | Ledger `model_shown` share; live APE falling faster than backtest APE | Phase gating (§7.7); tag shown rows; report live metrics split by `model_shown` |
| R15 | Tune the physics constants (S, E, t_min) to the data | Improves the fit | L | Coefficients drift each retrain | §2.3 fixes them; the ridge coefficient is the only free scale |
| R16 | Add `Company Name` because repeat customers are consistent | Improves grouped K-fold | M | New-quote subset collapses | §4.5 bans; the new-quote diagnostic exists for this |
| R17 | Score the training archive with the trained bundle and call it accuracy | It's the obvious smoke test | H | ~2% APE reported | `batch_score` prints a banner when any input row's Due Date precedes `trained_through`; the audit workbook uses the ledger only |

### 8.2 External and structural risks

| # | Risk | L | Detection | Response |
|---|---|---|---|---|
| S1 | Export format changes (column renamed, prices extended by quantity, Total Price redefined) | M per year | §4.6 rejects | Old bundle stays; log names the failed check; fix the loader deliberately, not by relaxing the check |
| S2 | Drift regime change (steel price shock) faster than monthly retraining | M | §7.8 signed-error alarm | Retrain fortnightly for the duration; do not add a steel index by hand mid-quarter — put it on §9 |
| S3 | The physics term (OPEN-2) does not close the large-tank bias | M | OPEN-2 rule rejects | `book` still carries `s_top5`; the sheet still warns `BIG`; §9 item 1 becomes the only path and should be escalated as a data request |
| S4 | The ledger is lost or edited | L | Ledger hash in `model_info`; row counts monotone | Rebuild by re-running `backtest`; bands widen slightly for one quarter |
| S5 | Excel or xlwings version change breaks the add-in | M | `TBT_MODEL_INFO()` errors | Pin versions; the batch scorer is the fallback surface |
| S6 | Someone requests a win-rate model | H | — | §6.5 `#UNSUPPORTED`; 309 resolved outcomes across 33 countries and 13 use types cannot support one; §9 item 7 |
| S7 | The 4.6% floor is quoted as the target | M | — | §1.2: it includes revisions; the real floor is higher; the gap is a data problem |
| S8 | Waste-water variance never falls because its drivers are absent | H | Tier C share of Waste rows stays > 50% | Expected; §9 item 2 |
| S9 | Estimator fills the scope block wrong (not blank) | M | Phase 1 adjudication | Scope block shows the resulting components as text ("erected, insulated, shipped, taxed") so a wrong answer reads wrong |

---

## 9. Data wishlist

Ranked by expected value. Values are [PREDICTION]s with reasoning; each names the
metric it should move and the result that would show it did not.

| Rank | Column(s) to capture | Expected value | Reasoning | Would be falsified if |
|---|---|---|---|---|
| 1 | **Shell course schedule: plate thickness per course, or total shell steel weight** | Top-5% bias from ~−14% to within ±5%; mean APE −0.3 to −0.5 | The one diagnosed mechanism of the largest error in the book; converts an inferred superlinearity into a stated quantity; TBT's engineering output already computes it | Top-5% bias moves < 3 points with the column in the backbone |
| 2 | **Appendage and internals schedule**: counts of nozzles, manways, ladders, platforms, mixers, baffles, launders; cover type | Mean APE −0.8 to −1.2, almost all from Waste and Industrial; Waste IQR from 27% toward 18% | Fire tanks (IQR 12%) are commodities; waste tanks are engineered-to-order and their spread is exactly the absent internals; Tank Name is a weak proxy worth 0.1 | Waste-family mean APE improves < 0.5 |
| 3 | **Scope flags captured at quote time** (the five booleans) | No change on the backtest; prevents a silent training corruption | The backfill cannot distinguish "not yet filled in" from "excluded"; going forward every partially-built revision in the export teaches the model that erection is cheap | — (this is hygiene, not accuracy) |
| 4 | **Crane class / site access / foundation scope** | Construction-component mean APE −1 to −2 points; mean total −0.2 to −0.4 | Construction and Material drive 87% of large disagreements; construction is the site-specific half | Construction APE unchanged |
| 5 | **Pre-margin cost lines** (or margin as a separate field that is not applied to the sell columns) | Mean APE −0.3 to −0.6 | Margin bands move material $/sf from 22 to 42; that spread is policy noise inside the target; a cost target plus a margin input is cleaner and unlocks a cost model | Within-spec spread does not fall when conditioning on margin |
| 6 | **Quote creation date** (not just due date) and **steel price index at that date** | Mean APE −0.1 to −0.3; smaller reliance on monthly retraining | Drift is the largest single lever; today it is inferred from a due date and a 1-year half-life | Retrain-cadence gap (never vs monthly) does not shrink |
| 7 | **Outcome on every quote** (won / lost / no-bid, and the competing price when known) | Unlocks a different model; no effect on this one | 309 resolved rows cannot support a win model; a process that resolves ≥ 60% of quotes could within two years | — |
| 8 | **Customer tax-exempt status as a CRM field** | Removes one of the five questions | It is a customer attribute, stable over time | — |
| 9 | **Design standard** (AWWA D100 / API 650 / NFPA 22 / bolted), **coating / lining spec**, **roof live load** | Mean APE −0.1 to −0.3 | Small but cheap; partially present in Deck Style and Use Type | — |
| 10 | **Estimated shop and field hours** | Would replace much of the model | If the estimator's own hours are captured, the model becomes a rate check rather than a price guess; the highest ceiling and the hardest to obtain | — |

Items 1 and 2 together are [PREDICTION] worth more than everything in §2 combined.
That is the honest scope statement the brief asks for: with the 42 columns the
system specified here should land near 8%; the road to 5–6% runs through the
quoting system, not the model.

---

## 10. Where the roles disagreed, and how it was resolved

| Disagreement | Architect | Engineer | Designer | Resolution |
|---|---|---|---|---|
| One number or two | Conditional mean for calibration | Conditional median for the north star | One number, please | Two figures from one model, `point` and `book`, each scored on its own objective (§1.3). The Designer's concern is met by showing only one on any given sheet: `point` beside a quote, `book` on a portfolio |
| Scope questions | Inputs, never inferred | Agrees; classifiers were cost without benefit | Five questions is friction | Inputs. Friction reduced by asking once per quote and rendering the answers as a sentence (§7.2, S9). The accuracy consequence of guessing is invisible on the sheet, which is the worst kind |
| Physics in the model | Steel-weight term in the backbone | Shell-course geometry already failed as GBM features | — | Open experiment with a decision rule; the prior failure was in the wrong place (tree, not backbone) and does not settle it (§2.3 OPEN-2) |
| Large-tank calibration | Fix the mechanism (physics) | Fix the symptom from forward residuals (`s_top5`), because it is measurable now | The estimator needs a warning either way | Both: `s_top5` on `book` immediately, physics term if it earns its place, `BIG` warning regardless. If the physics term works, `s_top5` shrinks — which is also the test of whether it worked |
| How much to ship | Full breakdown is the point of the decomposition | Withhold anything not yet measured live | Point estimate builds trust fastest | Phased (§7.7): check first, breakdown second, estimate third. The Designer's own concern about anchoring (R14) decided it |
| Experiment budget | Try several backbone forms | 12 runs, rules fixed in advance | — | 12 runs (§3.5). Deciding in advance is what stops the build from becoming a search for a flattering number |
| Interval method | Heteroscedastic model | Something a test can verify | Something an estimator can read | Mondrian conformal from the forward ledger: verifiable coverage, a table a human can read, narrower for commodity work by construction (§2.6) |
| Retrain cadence vs run time | — | Must be unattended and minutes | — | Monthly, ≤ 5 min; the ledger makes the honest residuals free (§2.7). Quarterly full backtest is a separate, longer job |

---

## Appendix A — Prediction register

Every [PREDICTION] in this document, in one place, for the build log to fill in.

| ID | Where | Prediction | Falsified if |
|---|---|---|---|
| P1 | §1.2 | Final headline mean APE 7.6–8.1% | > 8.3% |
| P2 | §2.3 OPEN-1 | `t·log D` term worth 0.1–0.3 | ≤ 0.1 or worse |
| P3 | §2.3 OPEN-2 | Physics term moves top-5% bias to −8% to −11%, mean −0.1 to −0.3 | top-5% bias moves < 2 points |
| P4 | §2.6 | coverage80 76–84% overall and per large group; Fire band ≥ 8 points narrower than Waste | Any large group outside 72–88% |
| P5 | §2.7 | `s_rest` 1.01–1.03; `s_top5` 1.08–1.16 before OPEN-2; `book` aggregate bias within ±2% | `book` bias outside ±3% |
| P6 | §2.7 OPEN-3 | Applying `s_top5` to `point` improves top-5% bias 5–10 points, worsens mean 0.0–0.2 | — (recorded either way) |
| P7 | §3.1 | 6,679 ± 60 usable pre-gate, ~173 gate failures | > 5% discrepancy |
| P8 | §3.4 | New-quote subset 1.5–3 points above headline | outside 0.5–5 |
| P9 | §5 B1 | Trivial baseline 25–35% mean | — |
| P10 | §5 B2 | Backbone-only 35–45% median, top-5% bias ≈ −50% | — |
| P11 | §5 B3 | Reference 9.5–10.5% mean | outside 9.0–11.0 → audit the harness |
| P12 | §5 B4 | Recency weighting −0.2 to −0.4 | — |
| P13 | §5 B5 | Second variant −0.1 to −0.2 | < 0.05 → drop |
| P14 | §5 B6 | Name features −0.05 to −0.15 | < 0.05 → drop |
| P15 | §5 OPEN-5 | Sales Manager 0.0–0.1 | — |
| P16 | §5 | Full fit ≤ 3 min; protocol run ≤ 25 min | fit > 6 min |
| P17 | §7.4 | Fire/CS common sizes → tier A; most Waste → B/C | — |
| P18 | §9 | Items 1+2 worth more than all of §2 combined | — |

## Appendix B — Glossary

- **APE** — absolute percentage error, `|estimate − actual| / actual`.
- **Headline window** — scoring quarters from 2026Q1 onward (§3.3).
- **Ledger** — the append-only table of forward predictions (§2.7).
- **`point` / `book`** — conditional-median estimate / retransformed aggregate
  estimate (§1.3).
- **Mondrian conformal** — split conformal with a separate residual quantile per
  group (§2.6).
- **Backbone** — the ridge log-log model each component's GBM is fitted on the
  residual of (§2.3).
- **Usable row** — a row passing every test in §3.1.
- **Scope** — the five commercial booleans (§2.4).
