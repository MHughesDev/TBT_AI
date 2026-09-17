# TBT Tank Pricing Estimator

Predicts what TBT would quote for a welded or bolted steel storage tank, from
the tank's specification plus the commercial scope of the job.

Two documents sit beside this one and they have different jobs:

- **`SPEC.md`** is the design and the reasoning. It is the authority. Every
  module names the section it implements.
- **`IMPLEMENTATION.md`** is the build record: what exists, how to run it, the
  bugs found on the way, and an honest statement of what has and has not been
  measured.

`DESIGN.md` and `HANDOFF.md` are the prior investigation's field notes. They
are evidence, not the plan.

---

## The one thing to understand

**The model prices scope. It does not guess scope.**

Whether a quote includes erection, insulation, freight or sales tax is a
commercial decision the estimator already knows. It is not a property of the
tank and it is not inferable from the specification. Every call states it:

```python
from tbt import QuoteInput, estimate, load_bundle

est = estimate(QuoteInput(
    diameter_ft=40, height_ft=36, material="CS",
    use_type="Fire Protection Storage Tank", state="TX",
    construction=True,          # do we erect it
    insulation=True,            # is insulation supplied
    insulation_erection=True,   # do we install it
    freight=True,               # do we ship it
    taxable=False,              # does sales tax apply to this customer
), bundle=load_bundle("models"))
```

Leave any of the five blank and you get `#SCOPE`, not a number. That is
deliberate. A classifier in that position can only agree with what the
estimator already knows, or be confidently wrong in a way that silently
changes the price with nothing visible on the sheet.

Material and Fabrication carry no flag. They are present on every usable row:
processes, not options.

## Quick start

```bash
pip install -r requirements.txt

# the real archive is not in this repo; generate a structurally faithful stand-in
python tools/make_synthetic_archive.py archive.csv

python -m tbt backtest archive.csv --report report.txt     # the only source of accuracy numbers
python -m tbt train    archive.csv --out models --bootstrap-ledger
python -m tbt predict  --diameter 32 --height 30 --material CS \
        --use-type "Fire Protection Storage Tank" --state MO \
        --construction yes --insulation no --insulation-erection no \
        --freight yes --taxable no --out models
python -m pytest tests -q
```

Everything runs on CPU. A full fit is about a minute on four cores; scoring a
few thousand rows takes a couple of seconds. No GPU, no cloud, no network call
anywhere in the code path.

## What comes back

A point estimate is not the product. Each call returns:

| Field | What it is for |
|---|---|
| `point` | The conditional median. The number shown beside one quote. |
| `book` | `point` corrected by a factor measured from forward residuals. Use this to sum a portfolio; the sum of medians is not the expected sum. |
| `band80`, `band90` | Conformal bands, computed per segment, so commodity work gets a narrower range than engineered-to-order work. |
| `components` | The six price lines. This is what makes a disagreement answerable: "check the construction line" rather than "this looks off". |
| `tier` | A, B or C. Derived from the band width and the warnings, not fitted. |
| `warnings` | Codes that must render on the sheet. A number with a hidden caveat is the failure mode this design exists to prevent. |

## Retraining

```bash
python -m tbt train archive.csv --out models      # put this on a schedule
```

The job scores the newly arrived rows with the **currently deployed** bundle,
appends them to the ledger, and only then refits. That order is load-bearing.
Reverse it and the ledger's residuals become in-sample: the prediction bands
collapse to a fraction of their honest width and the portfolio correction
drifts to one, while every number on the sheet still looks entirely plausible.

If the archive fails validation the job exits non-zero, the previous bundle
stays deployed, and the reason is recorded. The loader rejects a bad file
rather than repairing it.

## For Excel

```bash
xlwings addin install     # then Excel ribbon -> xlwings -> Import Functions
```

Phase 1 ships `TBT_CHECK` and `TBT_MODEL_INFO` only. The breakdown and the
point estimate are withheld until the check has earned them; `SPEC.md` section
7.7 sets out the evidence that unlocks each phase. Functions are non-volatile
and memoised, and anything over 500 rows returns `#BATCH` and points at the
batch scorer, because Excel recalculates aggressively and per-cell inference
will lock the workbook.

## Read this before anyone quotes off it

**No real-world accuracy number is claimed here.** The archive this was built
for is not in the repository, so every number produced during the build comes
from a synthetic stand-in. `IMPLEMENTATION.md` says exactly what that does and
does not establish. Run the backtest against the real file and quote that.

**Quote the mean, never the median.** The median flatters this problem by
roughly two and a half points. The mean is what a book of work experiences.

**It models what TBT quotes, not what wins.** Fewer than five percent of rows
have a resolved outcome. There is no win-rate model here, the data cannot
support one, and the API has no field that looks like one.

**Treat the largest quotes as a floor.** Large tanks scale superlinearly and
the available columns do not fully explain it. Every prediction in the top
band carries a warning.

**The first use is a check, not an oracle.** The model is not good enough to
set prices. It is good enough to catch a transposed dimension, a missing scope
line or a forgotten stainless premium, and that is real money at almost no
risk.
