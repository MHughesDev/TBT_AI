"""Rolling-origin validation.

The protocol is v4's, unchanged, because changing the yardstick at the same time
as the model is how projects convince themselves of improvements that are not
there. Train on everything before quarter Q, score quarter Q, walk forward.

Never use a random train/test split on this archive. 6,892 rows are only 3,018
quotes; a revision is a near-duplicate of its parent, and a random split puts
parent and child on opposite sides of it. That reports roughly 3% error instead
of roughly 8%, and the 3% is not real.
"""
import numpy as np
import pandas as pd

from . import data as datamod
from .config import TARGET
from .model import TBT5


def metrics(est, actual):
    """The five numbers. `mean` is the headline: it is what a portfolio feels."""
    est = np.asarray(est, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ok = np.isfinite(est) & np.isfinite(actual) & (actual > 0)
    est, actual = est[ok], actual[ok]
    if len(est) == 0:
        return dict(n=0, median=np.nan, mean=np.nan, p90=np.nan,
                    agg=np.nan, top5=np.nan)
    ape = np.abs(est - actual) / actual
    big = actual >= np.percentile(actual, 95)
    return dict(
        n=int(len(est)),
        median=float(np.median(ape) * 100),
        mean=float(ape.mean() * 100),
        p90=float(np.percentile(ape, 90) * 100),
        agg=float((est.sum() / actual.sum() - 1) * 100),
        top5=float((est[big].sum() / actual[big].sum() - 1) * 100),
    )


HEADER = (f"  {'quarter':<9}{'n':>6}{'median':>9}{'mean':>8}{'p90':>8}"
          f"{'$agg':>9}{'$top5%':>9}")


def _row(label, m):
    return (f"  {label:<9}{m['n']:>6}{m['median']:>8.1f}%{m['mean']:>7.1f}%"
            f"{m['p90']:>7.1f}%{m['agg']:>8.1f}%{m['top5']:>8.1f}%")


def rolling_origin(df, n_folds=6, fit_fn=None, verbose=True, min_test=50):
    """Walk forward a quarter at a time. Returns the per-fold metric dicts.

    `fit_fn(train_df, test_df) -> estimates` lets the bench drop another version
    into the identical folds. The default fits v5.
    """
    if fit_fn is None:
        def fit_fn(tr, te):
            return TBT5().fit(tr, verbose=False).estimate(te)["estimate"]

    pl = df["plausible"].values
    q = datamod.quarters(df)
    qs = sorted(q.dropna().unique())

    if verbose:
        print(HEADER, flush=True)
    folds = []
    for i in range(max(6, len(qs) - n_folds), len(qs)):
        tr_m = (q < qs[i]).values & pl
        te_m = (q == qs[i]).values
        # Score only plausible rows: an implausible row is a partial quote with a
        # scope line missing, and holding the model to it measures data entry.
        te_scored = te_m & pl
        if te_scored.sum() < min_test or tr_m.sum() < 300:
            continue
        est = fit_fn(df[tr_m].copy(), df[te_scored].copy())
        m = metrics(est, df[TARGET].values[te_scored])
        m["quarter"] = str(qs[i])
        folds.append(m)
        if verbose:
            print(_row(str(qs[i]), m), flush=True)

    if verbose and folds:
        print()
        print(_row("mean", _avg(folds)))
        print(_row("last 3", _avg(folds[-3:])))
    return folds


def _avg(folds):
    keys = ["median", "mean", "p90", "agg", "top5"]
    out = {k: float(np.mean([f[k] for f in folds])) for k in keys}
    out["n"] = int(np.sum([f["n"] for f in folds]))
    return out


def segment_report(df, est, actual, by, top=12):
    """Mean APE broken out by a column, worst first. Where to look next."""
    d = pd.DataFrame({"seg": df[by].astype(str).values,
                      "ape": np.abs(est - actual) / actual,
                      "est": est, "act": actual})
    g = d.groupby("seg").agg(n=("ape", "size"), mean=("ape", "mean"),
                             med=("ape", "median"),
                             bias=("est", lambda s: np.nan))
    sums = d.groupby("seg")[["est", "act"]].sum()
    g["bias"] = (sums["est"] / sums["act"] - 1)
    g = g[g["n"] >= 20].sort_values("mean", ascending=False)
    return g.head(top).assign(mean=lambda x: x["mean"] * 100,
                              med=lambda x: x["med"] * 100,
                              bias=lambda x: x["bias"] * 100)
