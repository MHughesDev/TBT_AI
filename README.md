# TBT Tank Pricing

Predicts what TBT would quote for a steel tank, from its specifications.
Trained on the TBT quote archive (Dec 2023 – Dec 2026, 6,892 rows / 3,018
quotes).

This repository holds **several complete, independent versions** of the system.
They do not share code. Each folder under `versions/` is a standalone
application: its own loader, features, model, CLI, Excel bridge, tests and
documentation. You can delete any one of them without touching the others.

That is deliberate. Pricing models are judged by a single number, and the only
honest way to choose between two designs is to run both on the same rows and
compare. Shared code makes that comparison quietly dishonest — a "shared"
feature module changes under one version when the other one needs it to.

---

## The versions

| Folder | Architecture | Mean APE | Status |
|---|---|---|---|
| [`versions/v4`](versions/v4) | Six component GBMs on a ridge log-log backbone, 70/30 blend with a direct model | **8.5%** measured | Shipped, validated |
| [`versions/v5`](versions/v5) | Computed shell-course physics, causal comparables kNN, quantile heads, mean-APE decision layer | **not yet measured** | Complete, unmeasured |

**v4 is the incumbent and the only version with a real number against its name.**
v5 is a complete system built on a specific argument about where v4's remaining
error lives; that argument has not been tested against the archive, because the
archive is customer data and is not in this repository.

---

## Choosing between them

```bash
pip install -r versions/v5/requirements.txt
python bench/compare.py archive_prepared.csv
```

`bench/compare.py` owns the rows and the folds. It loads the archive once,
applies one filter, cuts one set of quarter boundaries, and hands every version
exactly the same training mask and the same scoring mask. Each version supplies
only a `fit(train) → estimates(test)` function.

This matters more than it sounds. Every version ships its own validator and every
one reports a number, but those numbers are not comparable: the versions disagree
slightly about which rows are usable and about how the folds are cut. A
comparison where the yardstick moves with the model is not a comparison.

The leaderboard sorts on **mean** absolute percentage error, which is what a
portfolio of quotes actually experiences. Median is flattering — v4 reports 5.9%
median against 8.5% mean — and should not be used to pick between versions.

No archive to hand? `python bench/compare.py --synthetic` runs the whole thing on
generated data. That proves the pipeline works. It is not an accuracy result.

---

## Facts that outrank every version

These came out of the data, not out of any model, and they apply whichever
version is running.

**Retrain monthly.** Never-retrain 9.99% / quarterly 9.28% / monthly 8.69% mean.
That 1.3-point gap is larger than every modelling change in this project
combined, and it costs two minutes of CPU. If exactly one thing from this
repository gets done, make it a scheduled task.

**The model prices scope, it does not guess scope.** Whether a quote includes
erection, insulation, freight or sales tax is a commercial decision the estimator
already knows. Both versions raise `ScopeError` rather than defaulting. This will
be proposed again as a "fallback for blanks"; it will test at AUC 0.91–0.99, and
it will silently change prices with nothing visible on the sheet.

**Deploy the flag before the price.** At single-digit mean error neither version
can set prices, but both reliably catch a transposed dimension, a missing scope
line or a forgotten stainless premium. Real money at almost no risk, and it
builds the track record you would need before trusting anything further.

**No win-rate model is possible.** `Status` is Won 173, Lost 142 out of 6,892 —
under 5% resolved. That cannot support a model of what price wins, and no amount
of modelling fixes it. It needs outcome capture first, which is a process change.

**Never use a random train/test split.** 6,892 rows are only 3,018 quotes;
revisions are near-duplicates. A random split reports about 3% instead of about
8%, and the 3% is not real.

---

## Data

The archive is not in this repository and should not be — it contains customer
names, project names and real pricing. `*.csv` is gitignored.

Both versions expect a *prepared* archive, with five Yes/No scope columns added:

```bash
python versions/v5/prepare_data.py archive.csv archive_prepared.csv
```

That backfills scope by reading which components carry a non-zero price. It is
the right migration for history and the wrong thing to rely on going forward: a
blank price and a genuinely excluded scope look identical after the fact, and
only the estimator knows which it was. Five Yes/No dropdowns at quote time is the
fix, and they are the same five answers the model needs anyway.

---

## Adding a version

1. `mkdir versions/v6` and build it. Copy from an existing version or start
   clean — do not import across version folders.
2. Give it a `fit(train_df) → estimates(test_df)` entry point.
3. Add an adapter to `bench/compare.py` and register it in `entrants`.
4. Run the bench. If it does not win on mean APE, say so in its own `DESIGN.md`
   rather than deleting it — a version that lost is a result, and the next person
   needs to know it was tried.

`versions/v5/DESIGN.md` §8 is a worked example of the last point: it states in
advance what result would falsify each of v5's claims, so the bench run is a test
rather than a search for confirmation.
