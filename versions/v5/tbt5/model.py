"""The v5 pipeline: three base learners, a fitted blend, a metric-matched head.

    comparables ─┐
    components  ─┼─> blend fitted on mean APE ─> shrink by conditional variance
    direct      ─┘                                        │
                                                          └─> global shift
Scope is an input at every stage and is never inferred. That rule came from v4
and it stays: a scope classifier tests beautifully (AUC 0.91-0.99) and its
failure mode is a confidently wrong scope silently changing the price with
nothing visible on the sheet. `#SCOPE` is a feature.
"""
import numpy as np
import pandas as pd
import joblib

from . import comparables as cmpmod
from . import data as datamod
from . import decision, features
from .config import (
    ALWAYS, BLEND_PRIOR, BUNDLE, COMPONENTS, DATE_COL, GBM_ALT, GBM_BASE,
    GBM_COMPONENT, GROUP_COL, OPTIONAL, QUANTILES, TARGET, VERSION,
)
from .data import ScopeError
from .learners import Encoder, LogPriceModel, fit_blend

# Number of trailing quarters of the training window held out to fit the blend
# weights, the shrink and the bands. Two quarters is enough rows to fit three
# weights without starving the models that are being blended.
CALIB_QUARTERS = 2


class TBT5:
    def __init__(self, objective="mape", with_components=True, with_comparables=True):
        self.objective = objective
        self.with_components = with_components
        self.with_comparables = with_comparables

    # ---------------------------------------------------------------- fitting
    def _fit_core(self, df, mask, light=False):
        """Fit every base learner on `mask`. Returns a dict of fitted parts.

        `light` halves the boosting budget and drops the second GBM variant. It
        is used for the calibration pass only, whose outputs are three blend
        weights on a grid of 21, one scalar shift and two band multipliers —
        none of which move on the last few hundred boosting rounds. It roughly
        halves total training time, which is what keeps a monthly retrain a
        two-minute scheduled task rather than something nobody schedules.
        """
        tr = df[mask]
        w = datamod.weights(tr)

        scale = 0.5 if light else 1.0
        direct_params = ([_scaled(GBM_BASE, scale)] if light
                         else [GBM_BASE, GBM_ALT])
        comp_params = (_scaled(GBM_COMPONENT, scale),)
        quantiles = (0.10, 0.50, 0.90) if light else QUANTILES

        cmp_ = None
        if self.with_comparables:
            cmp_ = cmpmod.Comparables()
            feats = cmp_.fit_transform_causal(tr)
            tr = cmpmod.attach(tr, feats)
        else:
            for f in cmpmod.FEATURES:
                tr[f] = np.nan

        enc = Encoder().fit(tr)
        X = enc.transform(tr)
        B = features.backbone(tr)
        ci = enc.cat_idx

        direct = LogPriceModel(ci, params=direct_params, quantiles=quantiles).fit(
            X, B, np.log(tr[TARGET].values), w)

        comps = {}
        if self.with_components:
            for c in COMPONENTS:
                if c not in tr.columns:
                    continue
                m = tr[c].fillna(0).values > 0
                if m.sum() < 100:
                    continue
                comps[c] = LogPriceModel(ci, params=comp_params,
                                         with_quantiles=False).fit(
                    X[m], B[m], np.log(tr[c].values[m]), w[m])

        return {"enc": enc, "direct": direct, "comps": comps, "cmp": cmp_}

    def _raw_predict(self, parts, df):
        """Per-learner dollar predictions plus the conditional log-scale."""
        d = df
        if parts["cmp"] is not None:
            d = cmpmod.attach(d, parts["cmp"].transform(d))
        else:
            d = d.copy()
            for f in cmpmod.FEATURES:
                d[f] = np.nan

        X = parts["enc"].transform(d)
        B = features.backbone(d)

        log_direct = parts["direct"].predict(X, B)
        levels, logq = parts["direct"].predict_quantiles(X, B)
        sigma = (decision.sigma_from_quantiles(logq, levels) if logq is not None
                 else np.zeros(len(d)))

        preds = {"direct": np.exp(log_direct)}

        if parts["comps"]:
            present = _presence(d)
            total = np.zeros(len(d))
            per = {}
            for c, m in parts["comps"].items():
                v = np.exp(m.predict(X, B)) * present[c]
                per[c] = v
                total += v
            preds["components"] = total
            self._last_components = per
        if parts["cmp"] is not None:
            preds["comparables"] = np.exp(d["cmp_logprice"].values)

        return preds, log_direct, logq, levels, sigma

    def fit(self, df, verbose=True):
        """Fit on every plausible row, calibrating on a trailing inner holdout."""
        pl = df["plausible"].values
        q = datamod.quarters(df)
        qs = sorted(q.dropna().unique())

        # --- inner split: fit on the earlier part, calibrate on the tail -------
        blend, shift, bands = dict(BLEND_PRIOR), 0.0, {80: 1.6, 90: 2.0}
        if len(qs) > CALIB_QUARTERS + 4:
            cut = qs[-CALIB_QUARTERS]
            inner = pl & (q < cut).values
            cal = pl & (q >= cut).values
            if inner.sum() > 500 and cal.sum() > 100:
                if verbose:
                    print(f"  calibrating on {int(cal.sum())} rows from {cut} onward…")
                p0 = self._fit_core(df, inner, light=True)
                preds, logmed, logq, levels, sigma = self._raw_predict(p0, df[cal])
                actual = df[TARGET].values[cal]

                blend = fit_blend(preds, actual, prior=BLEND_PRIOR)
                est = _apply_blend(preds, blend)
                # The shrink and the shift are fitted in that order: the shrink is
                # per-row and the shift mops up whatever is left model-wide.
                log_est = decision.point_estimate(
                    np.log(np.maximum(est, 1e-9)), objective=self.objective,
                    sigma=sigma)
                shift = decision.fit_global_shift(np.exp(log_est), actual)
                resid = np.abs(np.log(actual) - (log_est + shift))
                bands = {int((1 - a) * 100): float(np.exp(np.quantile(resid, 1 - a)))
                         for a in (0.20, 0.10)}
                if verbose:
                    print("  blend: " + ", ".join(f"{k} {v:.2f}"
                                                  for k, v in blend.items()))
                    print(f"  shrink objective={self.objective}, "
                          f"global shift x{np.exp(shift):.4f}")
                    print(f"  80% band /x {bands[80]:.2f}   90% band /x {bands[90]:.2f}")

        # --- final fit on everything plausible --------------------------------
        if verbose:
            print(f"\n  fitting final models on {int(pl.sum())} plausible rows…")
        self.parts_ = self._fit_core(df, pl)
        self.blend_ = blend
        self.shift_ = shift
        self.bands_ = bands
        self.tax_ = datamod.tax_table(df)
        self.meta_ = {
            "version": VERSION,
            "objective": self.objective,
            "rows": int(pl.sum()),
            "quotes": int(df.loc[pl, GROUP_COL].nunique()) if GROUP_COL in df else None,
            "trained": str(pd.Timestamp.today().date()),
            "data_through": str(pd.to_datetime(df[DATE_COL], errors="coerce").max().date()),
        }
        self.limits_ = {
            "Diameter (ft)": (float(df.loc[pl, "Diameter (ft)"].quantile(.005)),
                              float(df.loc[pl, "Diameter (ft)"].quantile(.995))),
            "Height (ft)": (float(df.loc[pl, "Height (ft)"].quantile(.005)),
                            float(df.loc[pl, "Height (ft)"].quantile(.995))),
            "shell_area_max": float(df.loc[pl, "shell_area"].max()),
            "big_threshold": float(df.loc[pl, TARGET].quantile(.95)),
        }
        return self

    # -------------------------------------------------------------- inference
    def estimate(self, df, objective=None):
        """Dollar estimates for an engineered frame. The scoring workhorse."""
        objective = objective or self.objective
        preds, logmed, logq, levels, sigma = self._raw_predict(self.parts_, df)
        est = _apply_blend(preds, self.blend_)
        log_est = decision.point_estimate(np.log(np.maximum(est, 1e-9)),
                                          objective=objective, sigma=sigma)
        return {
            "estimate": np.exp(log_est + self.shift_),
            "sigma": sigma,
            "learners": preds,
            "components": getattr(self, "_last_components", {}),
        }

    # -------------------------------------------------------------- persistence
    def save(self, path=BUNDLE):
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=BUNDLE):
        return joblib.load(path)


def _scaled(params, factor):
    """Same GBM settings with the iteration budget scaled."""
    p = dict(params)
    p["max_iter"] = max(50, int(p["max_iter"] * factor))
    return p


def _presence(df):
    """Which components this row's scope says are priced."""
    p = {c: np.ones(len(df), dtype=float) for c in ALWAYS}
    for comp, flag in OPTIONAL.items():
        p[comp] = pd.to_numeric(df[flag], errors="coerce").fillna(0).values.astype(float)
    return p


def _apply_blend(preds, blend):
    """Weighted sum, renormalising over whichever learners produced a number.

    The comparables learner abstains when nothing similar enough exists in the
    archive. Renormalising rather than substituting a default means those rows
    fall back cleanly onto the other two instead of being dragged toward an
    arbitrary constant.
    """
    keys = [k for k in blend if k in preds]
    n = len(next(iter(preds.values())))
    num, den = np.zeros(n), np.zeros(n)
    for k in keys:
        v = np.asarray(preds[k], dtype=float)
        ok = np.isfinite(v) & (v > 0)
        num[ok] += blend[k] * v[ok]
        den[ok] += blend[k]
    out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    if not np.isfinite(out).all():          # nothing fired at all: fall back
        fb = np.asarray(preds.get("direct", np.full(n, np.nan)), dtype=float)
        out = np.where(np.isfinite(out), out, fb)
    return out
