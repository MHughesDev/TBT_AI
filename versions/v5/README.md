# TBT Tank Pricing — v5

A rebuild aimed at one number: **mean absolute percentage error**.

v4 sits at 8.5% mean / 5.9% median on rolling-origin quarterly validation. v5
attacks that from three directions its own handoff points at without following:

1. **It computes the shell course schedule** instead of waiting for TBT to export
   plate thickness. The superlinear cost growth that makes v4 underprice big
   tanks by 13–17% is a design-code step function, and it is derivable from
   diameter and height. See `DESIGN.md` §2.
2. **It optimises the metric it is judged on.** Squared loss on `log(price)`
   predicts the geometric mean, which is not the minimiser of mean APE. The
   minimiser is the `1/y`-weighted median, and it shrinks in proportion to the
   *conditional* variance — hard where the model is unsure, barely at all where
   it is confident. See `DESIGN.md` §3.
3. **It builds the comparables model the headroom table already describes.**
   v4's "4.6% floor, conditioned on spec + state + wage + year" is a
   nearest-neighbour result reported as a noise floor. See `DESIGN.md` §4.

---

## Read this before anything else

**v5 has not been measured on the real archive, and on synthetic data it does not
beat v4 on mean APE.**

The archive is customer data and is not in this repository — `*.csv` is
gitignored, which is correct. Nothing here has been run against it. Every number
in this folder is carried from v4's own measurements, derived analytically, or
measured on synthetic data from `tbt5/synth.py`.

Here is what the synthetic bench said, four rolling-origin quarters, identical
rows and folds:

| entrant | mean | median | **$top5%** | secs |
|---|---|---|---|---|
| v4 component GBM | **6.6%** | **5.3%** | −5.6% | **380** |
| v5 `objective=mape` | 6.7% | 5.4% | **−3.7%** | 896 |
| v5 ablation: no physics | 6.7% | 5.4% | −6.7% | 721 |

Read honestly, that says three things:

- **The physics layer works.** Removing it nearly doubles the large-tank bias
  (−3.7% → −6.7%). That is the thing v4's handoff calls open problem #1, tested
  directly, and it holds.
- **It buys calibration, not mean error.** 6.7% either way.
- **v5 is behind v4 on the headline metric** by 0.1 points, and costs 2.4× the
  training time.

Synthetic data is weak evidence — it is a guess at how pricing works, and it
happens to be a smooth guess with homogeneous noise, which is the regime least
favourable to two of v5's three ideas. But it is the evidence there is, and it
does not currently favour v5.

One command settles it properly, and it takes about twenty minutes:

```bash
python ../../bench/compare.py archive_prepared.csv
```

**Run that before deploying anything here.** If v5 does not win, v4 is still
there and still works.

---

## Start here

```bash
pip install -r requirements.txt

python cli.py selftest                             # end-to-end, no data needed
python prepare_data.py archive.csv archive_prepared.csv
python cli.py validate archive_prepared.csv        # the real number
python cli.py train    archive_prepared.csv --fast
python cli.py predict --diameter 32 --height 30 --material CS --state MO \
       --construction yes --insulation no --freight yes --taxable no
```

Everything runs on CPU. Training is about two and a half minutes — no GPU, no
cloud, no data leaves the machine.

---

## The one rule that has not changed

**The model prices scope. It does not guess scope.**

Whether a quote includes erection, insulation, freight or sales tax is a
commercial decision the estimator already knows. It is not a property of the tank
and it is not inferable from the specs. Every call states it:

```python
predict(diameter=40, height=36, material="CS", state="TX",
        construction=True,      # do we erect it
        insulation=True,        # is insulation supplied
        freight=True,           # do we ship it
        taxable=False)          # does sales tax apply to this customer
```

Omit any and you get a `ScopeError`, not a number.

This will be proposed again as "just default freight to yes when it's blank", or
"bring the classifier back as a fallback". It will be proposed in good faith and
it will test beautifully — v3's scope classifiers ran AUC 0.91–0.99. That is the
problem. A confidently wrong scope silently changes the price with nothing
visible on the sheet to show it. A `#SCOPE` error costs one question and stops a
wrong number going out the door.

Material and Fabrication carry no flag. They are present on 6,679 of 6,679 rows —
processes, not options.

### Pass the tank name too

`tank_name` is optional and worth supplying. The process descriptor is the only
proxy available for appendages that appear in no column: a digester has mixers,
covers and gas handling; an equalization basin is a bare shell. Median $/sq-ft
runs 1.35× base for digesters against 0.68× for multi-zone configurations. A
generic "Tank 1" is itself informative — it usually means a commodity shell.

---

## Pick the right objective

New in v5, and the one piece of the API that needs a decision rather than a
default. An estimate tuned to minimise per-quote percentage error leans low on
purpose, so summing those estimates undershoots.

| `objective` | Returns | Use it for |
|---|---|---|
| `mape` *(default)* | the `1/y`-weighted median | per-quote accuracy |
| `median` | the conditional median | when median error is the target |
| `unbiased` | the conditional mean | anything that gets **summed** |

```python
predict(..., objective="unbiased")     # valuing a backlog
```

v4's handoff says "do not use it to value a backlog". With `objective="unbiased"`
that is addressed rather than inherited — but you have to ask for it.

---

## What the model tells you beyond a price

```python
r = predict(...)
r["unit_price"]        # the estimate
r["p80_low"], r["p80_high"]
r["sigma"]             # conditional log spread — how unsure the model is
r["components"]        # the six price components
r["learners"]          # what each base learner said before blending
r["warnings"]          # every reason to distrust this number
```

`sigma` is the useful new one. It is the model's own uncertainty for this
specific tank, and it is what drives the shrink. A `sigma` above 0.30 means the
specs are an unusual combination and the number deserves a human.

---

## For Excel

```bash
xlwings addin install    # then: Excel ribbon -> xlwings -> Import Functions
```

Keep `tbt5/`, `tbt5_bundle.joblib` and `excel_udf.py` beside the workbook. Add
four Yes/No dropdown columns for scope, then:

```
=TBT5_FLAG(H2, B2, C2, D2, E2, F2, G2)   -> "OK +3%" / "LOW -22%" / "HIGH +31%"
=TBT5_BREAKDOWN(B2, C2, D2, E2, F2, G2)  -> components, band, spread, warnings
=TBT5_PRICE(B2, C2, D2, E2, F2, G2)      -> a single number
=TBT5_INFO()                              -> how stale the model is
```

Blank scope returns `#SCOPE`, not a guess.

Past a few dozen rows use `python cli.py score quotes.csv scored.csv` — Excel
fires inference on every recalculation and will crawl.

---

## Before anyone quotes off it

**Deploy `TBT5_FLAG` first, not `TBT5_PRICE`.** At single-digit mean error the
model cannot set prices, but it reliably catches a transposed dimension, a
missing scope line or a forgotten stainless premium. Real money at almost no
risk, and it builds the track record you would need before trusting it further.

**Keep the large-tank warning until the bench says otherwise.** v4 underprices
the top 5% by value by 13–17% in aggregate. v5's physics features target exactly
that, but *target* is not *fixed* — it has not been measured on the archive.
Until `bench/compare.py` says the bias has moved, treat top-5% predictions as a
floor.

**It models what TBT quotes, not what wins.** The archive has 173 Won and 142
Lost out of 6,892 rows. There is no win-rate model here and the data cannot
support one. Say this early — "AI pricing model" gets heard as "tells us what
wins", and the gap between those is where trust gets destroyed.

**Retrain monthly.** Never-retrain 9.99% / quarterly 9.28% / monthly 8.69% mean.
That 1.3-point gap is bigger than every modelling change in this project
combined, and it costs two minutes of CPU. If only one thing here gets done, make
it this. `=TBT5_INFO()` shows how stale the model is.

---

## Two accounting facts worth confirming with finance

Both verified on 100% of rows, and they are **not nested**:

```
Total Price    = 5 components + Freight     (tax-EXCLUSIVE)
Proposal Total = 5 components + Tax         (freight-EXCLUSIVE)
```

Quoting a "Total Price" that excludes sales tax but includes freight, beside a
"Proposal Total" that does the reverse, is a naming hazard independent of any
model.

The component prices already include margin — they are sell prices, not costs, so
no cost model can be built from this file.

---

## Layout

```
cli.py              train / validate / predict / score / selftest
prepare_data.py     one-time scope backfill from the archive
excel_udf.py        xlwings bridge
tbt5/
  config.py         column names, scope rules, tuning constants
  data.py           loading, scope validation, sample weighting
  physics.py        API-650-style shell course schedule and steel weight
  features.py       feature assembly and the parametric backbone basis
  comparables.py    causal nearest-neighbour pricing by analogy
  learners.py       ridge backbone + boosted residual + quantile heads
  decision.py       conditional distribution -> the mean-APE minimiser
  model.py          the pipeline: blend, calibrate, fit, score
  evaluate.py       rolling-origin harness and segment reports
  synth.py          synthetic archive, for running without the real one
tests/              invariants that correspond to bugs actually shipped
```

`DESIGN.md` has the full technical record, including what would falsify each
claim. Read §0 first.
