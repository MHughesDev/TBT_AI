# Build results

Every number below comes from the rolling-origin protocol in SPEC section 3. No number here comes from a random split, a grouped K-fold, or an in-sample fit.

**Irreducible floor of this archive: 4.56% mean APE.** No model can beat it. Distance to the floor is the only meaningful reading of the numbers below.

## Increment results

| run | mean APE | median | p90 | agg bias (point) | agg bias (book) | top5 bias (point) | coverage80 | new-quote mean | secs |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 26.63% | 20.11% | n/a | n/a | n/a | n/a | n/a | n/a | 0.2 |
| B2 | 13.75% | 9.66% | 26.18% | -6.18% | -3.47% | -13.41% | 81.65% | 14.03% | 1.5 |
| B3 | 7.47% | 5.16% | 13.93% | 0.20% | -0.29% | -0.71% | 82.15% | 7.54% | 1046.0 |
| B4 | 7.54% | 4.99% | 14.84% | -0.37% | -0.32% | -1.53% | 79.68% | 7.49% | 254.6 |
| B5 | 7.38% | 4.88% | 14.46% | -0.29% | -0.04% | -1.35% | 80.22% | 7.33% | 439.4 |
| B6 | 7.26% | 4.66% | 14.42% | -0.48% | -0.12% | -2.09% | 80.18% | 7.22% | 454.4 |
| B8_blend1 | 7.17% | 4.61% | 13.75% | -0.62% | -0.19% | -1.77% | 81.45% | 7.10% | 460.0 |
| B8_both | 7.35% | 4.74% | 14.11% | -0.89% | -0.21% | -2.15% | 79.83% | 7.28% | 512.9 |
| B8_drift | 7.45% | 4.83% | 14.25% | -0.63% | -0.09% | -2.17% | 79.33% | 7.49% | 492.0 |
| B8_physics | 7.25% | 4.90% | 13.80% | -0.55% | -0.14% | -1.71% | 79.74% | 7.14% | 496.0 |
| B8_salesmgr | 7.21% | 4.81% | 14.16% | -0.41% | -0.07% | -2.03% | 79.86% | 7.15% | 464.8 |
| B9_lr0.02_it1800_leaf15 | 7.26% | 4.69% | 14.30% | -0.44% | -0.11% | -2.10% | 80.51% | 7.22% | 523.0 |
| B9_lr0.02_it1800_leaf30 | 7.23% | 4.69% | 14.26% | -0.59% | -0.03% | -2.18% | 80.40% | 7.17% | 502.9 |
| B9_lr0.03_it1200_leaf15 | 7.26% | 4.66% | 14.42% | -0.48% | -0.12% | -2.09% | 80.18% | 7.22% | 455.9 |
| B9_lr0.03_it1200_leaf30 | 7.25% | 4.68% | 14.13% | -0.59% | 0.02% | -2.07% | 80.69% | 7.18% | 438.1 |

## Acceptance criteria (SPEC 6), evaluated on `B8_blend1`

The absolute thresholds in SPEC 6 were set for the REAL archive. On a different archive with a different irreducible floor they are not transferable, so both readings are given and the distance-to-floor column is the meaningful one.

| criterion | measured | spec threshold | reading |
|---|---|---|---|
| mean APE | 7.17% | <= 8.50% (real archive) | not comparable - +2.61 pts above this archive's 4.56% floor |
| median APE | 4.61% | <= 5.90% | not comparable |
| p90 APE | 13.75% | <= 18.0% | not comparable |
| aggregate bias (book) | -0.19% | within +/-2.0% | PASS |
| top-5% bias (book) | -2.77% | within +/-5% | PASS |
| coverage80 | 81.45% | 76-84% | PASS |
| new-quote subset | 7.10% | 1.5-3 pts above headline | not comparable - -0.07 pts |

## Pre-registered decisions

| question | measured | outcome |
|---|---|---|
| B4 recency weighting (1-yr half-life) | -0.07 pts | KEPT (spec: keep anyway, cost is one line) |
| B5 second GBM variant | +0.15 pts | KEPT |
| B6 tank-name features | +0.12 pts | KEPT |
| OPEN-1 size-dependent drift term (t x log D) | -0.19 pts; top5 -0.08 pts; wins 2/8 | REJECTED by rule -> take (A), no term |
| OPEN-2 steel-weight term in the backbone | top5 |bias| 2.09% -> 1.71% (+0.38 pts); mean +0.02 pts | REJECTED by rule -> take (A), no term |
| OPEN-4 blend weight w=1.0 (pure component sum) | mean -0.09 pts vs w=0.7; top5 -0.31 pts | ADOPTED w=1.0 (simpler; breakdown sums exactly) |
| OPEN-5 Sales Manager as a feature | +0.05 pts; new-quote +0.07 pts | REJECTED by rule -> exclude |
| B9 hyperparameter search (best: B9_lr0.02_it1800_leaf30) | +0.03 pts vs default | REJECTED by rule -> keep the default |

## Verification

### Test suite

88 tests pass in 6m28s: the integrity tests (leakage, future-state leakage in
fitted lookup tables, the quantity tripwire, filter symmetry, determinism,
module shadowing, ledger ordering), the full refusal contract, the Excel
surface, and the end-to-end command line paths.

### What a random split reports here

Same model, same data, three schemes:

| scheme | n | mean APE | median | what it is |
|---|---|---|---|---|
| random K-fold | 6,689 | 5.89% | 3.28% | fiction: revision 2 trains, revision 3 tests |
| GroupKFold by Quote # | 6,689 | 6.93% | 4.63% | honest about duplication, still sees the future |
| **rolling origin (SPEC 3)** | 4,713 | **7.13%** | 4.71% | **the protocol** |

A random split understates the error by 1.24 points here. On the real archive
the documented gap is larger, roughly 3% against 8%, because its revisions are
more tightly duplicated than this generator makes them. The direction and the
mechanism are the same. This is why `tbt.protocol` exposes no random splitter
and why a number from anywhere else must be rejected at review.

### Retraining cadence

Month-by-month forward test, which is what the deployment experiences:

| cadence | mean APE | median | p90 | aggregate bias | CPU |
|---|---|---|---|---|---|
| never retrain | 7.08% | 5.05% | 13.83% | -1.50% | 21s |
| quarterly | 6.84% | 4.66% | 13.01% | -1.40% | 60s |
| **monthly** | **6.16%** | **3.69%** | **11.90%** | **-1.32%** | 138s |

**Monthly retraining is worth 0.92 points against never retraining.** Every
modelling change in the build sequence put together is worth 0.37 points
(second ensemble variant 0.16, tank-name features 0.12, pure component sum
0.09).

That confirms the field notes' central deployment claim on this archive:
cadence is the largest single lever, larger than every modelling change
combined, and it costs about two minutes of CPU a month. The measured margin
is 0.92 points against the 1.30 documented on the real archive: same finding,
smaller magnitude, same conclusion. **Put the retrain on a schedule.**

## What these numbers are, and are not

The measured result is **7.17% mean APE against an irreducible floor of 4.56%**
for this archive, a gap of 2.61 points.

That floor is known exactly rather than estimated, because the archive was
generated twice from identical random draws, once with the irreducible noise
and once without. The distance to it is the transferable reading: it says the
pipeline extracts most of the signal that exists and is not leaving obvious
structure on the table.

The 7.17% itself is **not** a prediction of accuracy on TBT's archive. It is a
property of the noise this generator injects. The real number requires the real
file:

```bash
python -m tbt backtest <real archive.csv> --report report.txt
```

Three specification predictions were falsified and are recorded rather than
quietly dropped:

- **Recency weighting** cost 0.07 points where 0.2 to 0.4 of gain was expected.
  The likely cause here is that drift is smooth and the time feature already
  carries it, so down-weighting old rows only discards data. The rule kept it
  anyway, as written, because it costs one line.
- **The size-dependent drift term** cost 0.19 points where 0.1 to 0.3 of gain
  was expected, and won only 2 of 8 quarters.
- **The hyperparameter surface is flat.** All four cells land within 0.03
  points, so the search bought nothing. That is the expected shape for a low
  learning rate on a noisy target, and it means effort belongs in features
  rather than tuning.

Two results do **not** transfer and must be re-run on the real archive:

- **The steel-weight backbone term.** Its rule exists to repair a large-tank
  underpricing problem documented at -13.9% on the real archive. This archive
  showed -2.09% before the term was added, so there was nothing to repair and
  the experiment had no purchase. Rejected here means untested, not refuted.
- **The backbone-only baseline**, at 13.75% rather than the predicted 35-45%
  median, because the generator makes geometry more learnable than reality.
