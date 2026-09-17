"""A comparables engine — pricing by analogy, done causally.

Why this is here
----------------
v4's handoff includes a headroom table that is usually read as a noise-floor
estimate. It is also, read the other way, a performance claim for a model nobody
built:

    conditioned on         median   mean
    identical spec          5.9%    18.3%
    + state                 2.2%     6.3%
    + state + wage + year   1.5%     4.6%

Those rows were produced by grouping rows that match exactly and measuring the
spread within the group. But "find rows matching on spec, state, wage and year,
and take their price" is not a description of the noise floor. It is a
description of a nearest-neighbour model, and the table says such a model lands
near 4.6% mean where the shipped gradient boosting lands at 8.7%.

The catch is that exact matching only fires when an exact match exists. This
module generalises it: a distance-weighted average over the k most similar
*prior* quotes, which degrades gracefully to "nothing close enough" instead of
degrading to "no match".

Two rules make it honest rather than a leak
-------------------------------------------
1. Strict causality. A row may only be compared against rows dated before it. The
   expanding-window build below is slower than one global index and is the whole
   reason the number means anything.
2. No self-quotes. Revisions of the same Quote # are near-duplicates of each
   other; letting revision 3 find revision 2 as its comparable reports about 3%
   error and predicts nothing. Same-quote rows are excluded from every
   neighbour set.

The output is fed to the gradient booster as features rather than blended in at
a fixed weight, so the booster can learn *when* to trust the analogy: cmp_n and
cmp_spread tell it how many comparables were found and how much they disagreed,
which is exactly the information needed to discount a thin or contradictory
neighbourhood.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import (
    SCOPE, TARGET, GROUP_COL, DATE_COL, CMP_K, CMP_MIN_NEIGHBOURS,
    CMP_HALFLIFE_YEARS,
)

# Distance weights. Size dominates, then scope (which moves price ~2x), then the
# context columns in roughly the order v4's ablation ranked them.
W_LOG_D = 3.0
W_LOG_H = 2.0
W_SCOPE = 1.5
W_CAT = {
    "Material": 1.2,
    "Country": 1.0,
    "Use Type": 0.8,
    "Wage Type": 0.8,
    "State": 0.5,
    "Deck Style": 0.5,
    "Floor Style": 0.4,
}
W_TIME_PER_YEAR = 0.3

FEATURES = ["cmp_logpsf", "cmp_n", "cmp_dist", "cmp_spread", "cmp_logprice"]


class Comparables:
    """Distance-weighted kNN over log $/sq-ft, restricted to earlier quotes."""

    def __init__(self, k=CMP_K, halflife=CMP_HALFLIFE_YEARS):
        self.k = k
        self.halflife = halflife
        self.levels_ = None

    # ------------------------------------------------------------ embedding
    def _learn_levels(self, df):
        self.levels_ = {c: sorted(df[c].astype(str).fillna("?").unique())
                        for c in W_CAT}

    def _embed(self, df):
        """Map rows into a space where Euclidean distance means 'comparable'.

        Categoricals become weighted one-hot columns, so a mismatch contributes a
        fixed distance penalty of w*sqrt(2) and a match contributes nothing. An
        unseen level matches nothing, which is the correct behaviour: a material
        we have never quoted has no comparables.
        """
        cols = [
            W_LOG_D * np.log(pd.to_numeric(df["Diameter (ft)"], errors="coerce")
                             .clip(lower=0.1).values),
            W_LOG_H * np.log(pd.to_numeric(df["Height (ft)"], errors="coerce")
                             .clip(lower=0.1).values),
        ]
        for c in SCOPE:
            cols.append(W_SCOPE * pd.to_numeric(df[c], errors="coerce").fillna(0).values)
        for c, w in W_CAT.items():
            v = df[c].astype(str).fillna("?").values
            for lv in self.levels_[c]:
                cols.append(w * (v == lv).astype(float))
        cols.append(W_TIME_PER_YEAR * (df["months"].astype(float).fillna(0).values / 12.0))
        return np.nan_to_num(np.column_stack(cols), nan=0.0)

    # ------------------------------------------------------------------ fit
    def fit(self, df):
        """Store the reference set. Only plausible rows are eligible as neighbours."""
        ok = df["plausible"].values if "plausible" in df else np.ones(len(df), bool)
        ref = df[ok]
        self._learn_levels(ref)
        self.Z_ = self._embed(ref)
        self.logpsf_ = np.log(ref[TARGET].values / ref["shell_area"].values)
        self.date_ = pd.to_datetime(ref[DATE_COL], errors="coerce").values
        self.quote_ = (ref[GROUP_COL].astype(str).values if GROUP_COL in ref
                       else np.array([f"__{i}" for i in range(len(ref))]))
        self.n_ref_ = len(ref)
        return self

    # ------------------------------------------------------------- querying
    def transform(self, dfq):
        """Comparables for rows scored *after* the reference set — the live path.

        No per-row date filter is needed here because the reference set is by
        construction entirely in the past: either the training fold of a rolling
        -origin split, or the whole archive when scoring a live quote.
        """
        return self._neighbours(dfq, self._embed(dfq), self.Z_, self.logpsf_,
                                self.quote_, self.date_)

    def fit_transform_causal(self, df, block="M"):
        """Comparables for the training rows themselves, without seeing the future.

        Walks forward in time by block. For each block the reference set is every
        plausible row strictly before it, so a row is never compared to a quote
        that had not happened yet. This is the expensive path — roughly one index
        build per month of archive — and it is the reason the cmp_* features carry
        the same meaning at fit time as they do in production.
        """
        self._learn_levels(df[df["plausible"]] if "plausible" in df else df)
        out = {f: np.full(len(df), np.nan) for f in FEATURES}

        d = pd.to_datetime(df[DATE_COL], errors="coerce")
        per = d.dt.to_period(block)
        ok = (df["plausible"].values if "plausible" in df
              else np.ones(len(df), bool)) & d.notna().values

        Z_all = self._embed(df)
        logpsf_all = np.log(np.maximum(df[TARGET].values, 1e-9)
                            / np.maximum(df["shell_area"].values, 1e-9))
        quote_all = (df[GROUP_COL].astype(str).values if GROUP_COL in df
                     else np.array([f"__{i}" for i in range(len(df))]))
        date_all = d.values

        for p in sorted(per.dropna().unique()):
            qm = (per == p).values
            rm = (per < p).values & ok
            if not qm.any() or rm.sum() < CMP_MIN_NEIGHBOURS:
                continue
            sub = self._neighbours(df[qm], Z_all[qm], Z_all[rm], logpsf_all[rm],
                                   quote_all[rm], date_all[rm])
            for f in FEATURES:
                out[f][qm] = sub[f]

        self.fit(df)
        return out

    def _neighbours(self, dfq, Zq, Zref, logpsf, quote_ref, date_ref):
        """Distance-weighted neighbour average of log $/sq-ft.

        The only caller-supplied guarantee is that `Zref` contains no rows dated
        after the query rows; same-quote exclusion is handled here.
        """
        n = len(dfq)
        out = {f: np.full(n, np.nan) for f in FEATURES}
        if len(Zref) < CMP_MIN_NEIGHBOURS:
            return out
        # Over-fetch: some neighbours will be dropped as same-quote revisions.
        k = min(len(Zref), self.k * 4)
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(Zref)
        dist, idx = nn.kneighbors(Zq)

        qid = (dfq[GROUP_COL].astype(str).values if GROUP_COL in dfq
               else np.array([None] * n))
        qdate = pd.to_datetime(dfq[DATE_COL], errors="coerce").values
        area = dfq["shell_area"].values

        for i in range(n):
            j, d = idx[i], dist[i]
            keep = quote_ref[j] != qid[i]
            j, d = j[keep][: self.k], d[keep][: self.k]
            if len(j) < CMP_MIN_NEIGHBOURS:
                continue
            # Gaussian kernel with a bandwidth set by the neighbourhood itself,
            # so a dense region weights sharply and a sparse one weights flatly.
            h = max(np.median(d), 1e-6)
            w = np.exp(-0.5 * (d / h) ** 2)
            age = (qdate[i] - date_ref[j]) / np.timedelta64(365, "D")
            w = w * np.power(0.5, np.clip(age, 0, None) / self.halflife)
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            v = logpsf[j]
            m = float(np.sum(w * v))
            out["cmp_logpsf"][i] = m
            out["cmp_n"][i] = len(j)
            out["cmp_dist"][i] = float(np.sum(w * d))
            out["cmp_spread"][i] = float(np.sqrt(max(np.sum(w * (v - m) ** 2), 0.0)))
            out["cmp_logprice"][i] = m + np.log(max(area[i], 1e-9))
        return out


def attach(df, cmp_features):
    out = df.copy()
    for f in FEATURES:
        out[f] = cmp_features[f]
    return out
