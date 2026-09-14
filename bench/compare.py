"""Head-to-head: every version, identical folds, identical rows.

    python bench/compare.py archive_prepared.csv
    python bench/compare.py --synthetic            # no archive needed

Why a separate harness
----------------------
Each version ships its own validator, and each one reports a number. Those
numbers are not comparable, because the versions disagree slightly about which
rows are usable and about how the folds are cut, and a comparison where the
yardstick moves with the model is not a comparison.

So this file owns the rows and the folds. It loads the archive once, applies one
filter, cuts one set of quarter boundaries, and hands every version exactly the
same training mask and the same scoring mask. The versions supply only a
`fit(train) -> estimates(test)` function.

The headline column is **mean**: mean absolute percentage error, which is what a
portfolio of quotes actually experiences. Median is flattering and should not be
used to pick between versions.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "versions" / "v4"
V5 = ROOT / "versions" / "v5"


# ------------------------------------------------------------------ adapters
def load_common(source):
    """One filtered, engineered frame. v5's loader owns the row selection; v4's
    features are then computed over the identical rows."""
    sys.path.insert(0, str(V5))
    from tbt5 import data as v5data
    df = (source if isinstance(source, pd.DataFrame)
          else pd.read_csv(source, low_memory=False))
    return v5data.load(df, verbose=True)


def v4_adapter(d5):
    """Prepare v4's view of the same rows and return a fold-scoring closure."""
    sys.path.insert(0, str(V4))
    import tbt_model as v4

    d4 = v4.engineer(d5.copy())
    for c in v4.COMPS:
        d4[c + "_u"] = d4[c]
    d4["total_u"] = d4["Total Price"]

    # v4 crashes on an export that is missing an optional numeric column
    # outright: HistGradientBoosting's binner raises "window shape cannot be
    # larger than input array shape" on an all-NaN feature, and
    # `Miles to Site (From GT)` is absent from some exports. Neutralised here
    # rather than patched into v4, so that the entrant in this bench is v4 as
    # shipped. v5's Encoder drops such columns at fit time instead.
    for c in v4.NUM:
        if c in d4 and not np.isfinite(pd.to_numeric(d4[c], errors="coerce")).any():
            d4[c] = 0.0

    X, _ = v4.encode(d4)
    ci = v4._ci(X)
    B = v4.backbone(d4)
    w = v4.recency_weights(d4)

    def score(tr_mask, te_mask):
        regs, direct = v4.fit_models(d4, X, B, ci, tr_mask, w)
        return v4.combine(regs, direct, d4, X, B, te_mask)

    return score


def v5_adapter(d5, objective="mape", **kw):
    sys.path.insert(0, str(V5))
    from tbt5.model import TBT5

    def score(tr_mask, te_mask):
        m = TBT5(objective=objective, **kw).fit(d5[tr_mask].copy(), verbose=False)
        return m.estimate(d5[te_mask].copy())["estimate"]

    return score


# ---------------------------------------------------------------------- run
def metrics(est, actual):
    est, actual = np.asarray(est, float), np.asarray(actual, float)
    ok = np.isfinite(est) & np.isfinite(actual) & (actual > 0)
    est, actual = est[ok], actual[ok]
    ape = np.abs(est - actual) / actual
    big = actual >= np.percentile(actual, 95)
    return dict(n=len(est), median=np.median(ape) * 100, mean=ape.mean() * 100,
                p90=np.percentile(ape, 90) * 100,
                agg=(est.sum() / actual.sum() - 1) * 100,
                top5=(est[big].sum() / actual[big].sum() - 1) * 100)


def run(d5, entrants, n_folds=6, min_test=50):
    q = pd.to_datetime(d5["Due Date"], errors="coerce").dt.to_period("Q")
    pl = d5["plausible"].values
    actual = d5["Total Price"].values
    qs = sorted(q.dropna().unique())

    folds = []
    for i in range(max(6, len(qs) - n_folds), len(qs)):
        tr = (q < qs[i]).values & pl
        te = (q == qs[i]).values & pl
        if te.sum() >= min_test and tr.sum() >= 300:
            folds.append((str(qs[i]), tr, te))
    if not folds:
        sys.exit("no usable folds — check the date range in the archive")

    print(f"\n{len(folds)} folds: " + ", ".join(f for f, _, _ in folds))
    print(f"train sizes {[int(t.sum()) for _, t, _ in folds]}")
    print(f"test sizes  {[int(t.sum()) for _, _, t in folds]}\n")

    results = {}
    for name, score in entrants.items():
        print(f"--- {name} " + "-" * (58 - len(name)))
        print(f"  {'quarter':<9}{'n':>6}{'median':>9}{'mean':>8}{'p90':>8}"
              f"{'$agg':>9}{'$top5%':>9}")
        per = []
        t0 = time.time()
        for label, tr, te in folds:
            m = metrics(score(tr, te), actual[te])
            per.append(m)
            print(f"  {label:<9}{m['n']:>6}{m['median']:>8.1f}%{m['mean']:>7.1f}%"
                  f"{m['p90']:>7.1f}%{m['agg']:>8.1f}%{m['top5']:>8.1f}%", flush=True)
        avg = {k: float(np.mean([p[k] for p in per]))
               for k in ("median", "mean", "p90", "agg", "top5")}
        avg["secs"] = time.time() - t0
        results[name] = avg
        print(f"  {'MEAN':<9}{'':>6}{avg['median']:>8.1f}%{avg['mean']:>7.1f}%"
              f"{avg['p90']:>7.1f}%{avg['agg']:>8.1f}%{avg['top5']:>8.1f}%"
              f"   ({avg['secs']:.0f}s)\n")

    print("=" * 66)
    print("LEADERBOARD — by mean absolute percentage error, lower is better")
    print("=" * 66)
    print(f"  {'version':<28}{'mean':>8}{'median':>9}{'p90':>8}{'$agg':>9}")
    for name, a in sorted(results.items(), key=lambda kv: kv[1]["mean"]):
        print(f"  {name:<28}{a['mean']:>7.1f}%{a['median']:>8.1f}%"
              f"{a['p90']:>7.1f}%{a['agg']:>8.1f}%")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="prepared archive CSV")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate a synthetic archive instead (plumbing only — "
                         "the resulting numbers are NOT accuracy results)")
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of entrant names")
    a = ap.parse_args()

    if a.synthetic:
        sys.path.insert(0, str(V5))
        from tbt5 import synth
        print("SYNTHETIC DATA — this measures that the pipeline runs, not how "
              "accurate it is.\n")
        src = synth.generate()
    elif a.csv:
        src = a.csv
    else:
        ap.error("give a CSV or pass --synthetic")

    d5 = load_common(src)
    # The two ablations isolate v5's thesis: drop the computed shell schedule
    # (backbone falls back to v4's basis, physics features withheld from the
    # trees), and drop the comparables engine. If either matches the full model,
    # that half of the argument is wrong — see versions/v5/DESIGN.md section 8.
    entrants = {
        "v4 component GBM": v4_adapter(d5),
        "v5 (objective=median)": v5_adapter(d5, objective="median"),
        "v5 (objective=mape)": v5_adapter(d5, objective="mape"),
        "v5 ablation: no physics": v5_adapter(d5, objective="mape",
                                              with_physics=False),
        "v5 ablation: no comparables": v5_adapter(d5, objective="mape",
                                                  with_comparables=False),
    }
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        entrants = {k: v for k, v in entrants.items()
                    if any(w.lower() in k.lower() for w in want)}
    run(d5, entrants, n_folds=a.folds)


if __name__ == "__main__":
    main()
