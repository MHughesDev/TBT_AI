"""Base learners: a parametric backbone, boosted residuals, and quantile heads."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder

from .config import GBM_BASE, GBM_ALT, QUANTILES
from .features import NUMERIC, CATEGORICAL


class Encoder:
    """Ordinal codes for the categoricals, so HistGBM can split on them natively.

    Unknown levels map to -1 rather than raising. A material or state never seen
    in training is a real thing that happens in production, and the right
    response is to fall back on the rest of the row, not to refuse the quote.

    Numeric columns that are entirely missing or constant across the training
    fold are dropped and stay dropped at score time. They carry no fitted
    information either way, and an all-missing column is not merely useless —
    HistGradientBoosting's binner raises on one. `Miles to Site (From GT)` is
    absent from some exports and is exactly this case.
    """

    def __init__(self, drop=()):
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                  unknown_value=-1, encoded_missing_value=-1)
        self.drop = set(drop)

    def fit(self, df):
        cols = [c for c in NUMERIC if c not in self.drop]
        num = df[cols].astype(float).to_numpy()
        keep = []
        for j, c in enumerate(cols):
            col = num[:, j]
            fin = col[np.isfinite(col)]
            if fin.size and np.unique(fin).size > 1:
                keep.append(c)
        self.num_cols = keep
        self.cat_idx = [len(keep) + i for i in range(len(CATEGORICAL))]
        self.enc.fit(df[CATEGORICAL].astype(str))
        return self

    def transform(self, df):
        num = df[self.num_cols].astype(float).to_numpy()
        cat = self.enc.transform(df[CATEGORICAL].astype(str))
        return np.hstack([num, cat])

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class LogPriceModel:
    """Ridge log-log backbone plus a boosted residual, with optional quantiles.

    The backbone is what lets the model extrapolate. A gradient-boosted tree
    predicts a constant beyond its largest leaf, so without a parametric stage a
    tank bigger than anything in training is priced at the biggest thing we have
    seen — v2 quoted a 111 x 51 ft tank at $1.5M against an actual $3.1M for
    exactly this reason. The backbone keeps rising; the trees learn only what it
    missed.

    v5's backbone includes the computed shell steel weight and the
    minimum-thickness regime indicator (see features.backbone), which is the
    change that matters. v4's basis was a smooth power law in D and H and ran
    -52% on the large segment because the true cost curve has a kink in it.

    Losses are absolute rather than squared throughout. For small errors
    APE ~= |log(yhat) - log(y)|, so absolute loss in log space is a first-order
    match to the objective; squared loss is not, and chases the few quotes with
    the largest log residuals at the expense of the many.
    """

    def __init__(self, cat_idx, params=(GBM_BASE, GBM_ALT), quantiles=QUANTILES,
                 with_quantiles=True):
        self.cat_idx = cat_idx
        self.params = list(params)
        self.quantiles = tuple(quantiles) if with_quantiles else ()

    def fit(self, X, B, y, w=None):
        self.ridge_ = Ridge(alpha=1.0).fit(B, y, sample_weight=w)
        resid = y - self.ridge_.predict(B)

        self.gbms_ = [
            HistGradientBoostingRegressor(categorical_features=self.cat_idx, **p)
            .fit(X, resid, sample_weight=w) for p in self.params
        ]

        # Quantile heads run on the residual too, and deliberately smaller: they
        # only have to describe the *spread*, which the decision layer converts
        # into a shrink. Spending base-model capacity on them buys nothing.
        self.qheads_ = {}
        for q in self.quantiles:
            if abs(q - 0.5) < 1e-9:
                continue
            self.qheads_[q] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, categorical_features=self.cat_idx,
                learning_rate=0.08, max_iter=200, min_samples_leaf=30,
                l2_regularization=1.0, random_state=7,
            ).fit(X, resid, sample_weight=w)
        return self

    def predict(self, X, B):
        """Central estimate, in log dollars.

        The GBM variants are averaged in log space, which is a geometric mean of
        prices — the right average for a multiplicative target.
        """
        return self.ridge_.predict(B) + np.mean(
            [g.predict(X) for g in self.gbms_], axis=0)

    def predict_quantiles(self, X, B):
        """(levels, matrix) of conditional log-price quantiles, or (None, None)."""
        if not self.qheads_:
            return None, None
        base = self.ridge_.predict(B)
        med = self.predict(X, B)
        levels, cols = [], []
        for q in sorted(self.quantiles):
            if abs(q - 0.5) < 1e-9:
                levels.append(0.5)
                cols.append(med)
            else:
                levels.append(q)
                cols.append(base + self.qheads_[q].predict(X))
        return np.array(levels), np.column_stack(cols)


def fit_blend(preds, actual, prior=None, grid=21):
    """Non-negative blend weights over base-learner predictions, on mean APE.

    `preds` is {name: array of dollar predictions}. v4 blended component-sum and
    direct at a hardcoded 70/30 and noted that anything between 0.3 and 0.8 lands
    within 0.2 points — true for two learners that fail in similar places, and
    not something to assume for three that do not. Fitted out-of-fold on the
    metric we actually care about, with a coarse simplex search because the
    surface is flat and a fine one would be fitting noise.
    """
    names = [k for k in preds if np.isfinite(preds[k]).any()]
    if not names:
        raise ValueError("no usable base predictions to blend")
    actual = np.asarray(actual, dtype=float)

    # A learner that abstains on some rows (comparables, when nothing is close
    # enough) falls back to the blend of the others on those rows rather than
    # poisoning them with a NaN.
    P = {k: np.asarray(preds[k], dtype=float) for k in names}
    ok = np.isfinite(actual) & (actual > 0)
    for k in names:
        ok &= np.isfinite(P[k])
    if ok.sum() < 50:
        return dict(prior or {k: 1.0 / len(names) for k in names})

    best, best_w = np.inf, None
    for parts in _compositions(len(names), grid):
        w = [p / grid for p in parts]
        est = sum(wi * P[k][ok] for wi, k in zip(w, names))
        e = float(np.mean(np.abs(est - actual[ok]) / actual[ok]))
        if e < best:
            best, best_w = e, w
    return {k: float(v) for k, v in zip(names, best_w)}


def _compositions(n, total):
    """Every way to split `total` units across `n` non-negative bins."""
    if n == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in _compositions(n - 1, total - i):
            yield (i,) + rest
