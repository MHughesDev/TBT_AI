"""The fitted objects: Stage, tax table and Bundle.

SPEC 2.1 (decomposition), 2.3 (backbone + residual), 2.5 (model family),
2.8 (blend), 4.4 (tax table).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from . import config as C
from . import features as F


def recency_weights(due: pd.Series, ref: pd.Timestamp,
                    half_life_years: float = C.RECENCY_HALF_LIFE_YEARS) -> np.ndarray:
    """SPEC 2.5. w = 0.5 ** (age_years / half_life). Uniform 1-year half-life;
    per-component half-lives were tried and lost."""
    age = (ref - pd.to_datetime(due)).dt.days.values / 365.25
    age = np.clip(np.nan_to_num(age, nan=0.0), 0.0, None)
    return 0.5 ** (age / half_life_years)


class Stage:
    """Ridge log-log backbone + an ensemble of gradient-boosted residual models.

    SPEC 2.3. The backbone extrapolates; the trees learn only the residual and
    capture interactions. LOAD-BEARING: removing the backbone reproduces the
    -25% large-tank failure (SPEC 8.1 R10).
    """

    def __init__(self, params=None, categorical_mask=None):
        # an explicitly empty list means backbone-only; [] is falsy, so test None
        self.params = C.GBM_VARIANTS if params is None else params
        self.categorical_mask = categorical_mask
        self.lin: Ridge | None = None
        self.gbms: list[HistGradientBoostingRegressor] = []

    def fit(self, X: pd.DataFrame, B: np.ndarray, y: np.ndarray,
            w: np.ndarray | None = None) -> "Stage":
        self.lin = Ridge(alpha=C.RIDGE_ALPHA)
        self.lin.fit(B, y, sample_weight=w)
        resid = y - self.lin.predict(B)
        self.gbms = []
        for p in self.params:
            g = HistGradientBoostingRegressor(
                loss="squared_error",
                categorical_features=self.categorical_mask,
                early_stopping=False,   # a random validation split would leak revisions
                **p,
            )
            g.fit(X, resid, sample_weight=w)
            self.gbms.append(g)
        return self

    def predict(self, X: pd.DataFrame, B: np.ndarray) -> np.ndarray:
        base = self.lin.predict(B)
        if not self.gbms:
            return base
        # log-space mean == geometric mean of prices, the right average for a
        # multiplicative target (SPEC 2.5)
        res = np.mean([g.predict(X) for g in self.gbms], axis=0)
        return base + res

    def predict_backbone(self, B: np.ndarray) -> np.ndarray:
        return self.lin.predict(B)


def fit_tax_table(df: pd.DataFrame, mask: pd.Series) -> dict:
    """SPEC 4.4. Geography sets the rate; the customer sets whether it applies.

    Near-constant within a state (median sd 0.008), so this is a lookup, not a
    regression. Modelling it would add error for nothing.
    """
    d = df.loc[mask]
    base = sum(pd.to_numeric(d[c], errors="coerce").fillna(0.0)
               for c in C.COMPONENTS if c != "Freight Price")
    tax = pd.to_numeric(d.get("Total Tax"), errors="coerce").fillna(0.0)
    ok = (tax > 0) & (base > 0)
    if not ok.any():
        return {}
    rate = (tax[ok] / base[ok])
    table: dict[str, float] = {}

    # Keys are NAMESPACED. "CA" is both California and Canada; an unnamespaced
    # table silently prices a Canadian job at the California rate.
    if "State" in d.columns:
        keys = d.loc[ok, "State"].astype(object).map(
            lambda v: str(v).strip() if v is not None else "")
        for k, grp in rate.groupby(keys):
            if k and len(grp) >= 5:
                table[f"state:{k}"] = float(np.median(grp))
    if "Country" in d.columns:
        ctry = d.loc[ok, "Country"].astype(object).map(
            lambda v: str(v).strip() if v is not None else "")
        for k, grp in rate.groupby(ctry):
            # SPEC 4.4: the country entry is a fallback for NON-US rows only.
            # A US row with an unfamiliar state must fall through to TAXRATE,
            # not silently inherit a national average.
            if k and k.upper() != "US" and len(grp) >= 5:
                table[f"country:{k}"] = float(np.median(grp))
    return table


@dataclass
class Bundle:
    """Everything needed to score, plus everything needed to audit the fit.

    Every table in here is fitted on the TRAINING WINDOW ONLY. Computing any of
    them once on the whole file is leakage of the future (SPEC 4.5 item 11).
    """
    encoder: F.Encoder
    backbone: F.Backbone
    regs: dict                      # component -> Stage
    direct: Stage
    tax: dict
    use_family_map: dict
    blend: float = C.BLEND_WEIGHT
    big_threshold: float = 0.0
    envelope: dict = field(default_factory=dict)
    conformal: dict = field(default_factory=dict)
    smear: dict = field(default_factory=lambda: {"rest": 1.0, "top5": 1.0})
    trained_through: date | None = None
    n_train_rows: int = 0
    use_drift: bool = False
    use_physics: bool = False
    use_sales_manager: bool = False
    use_names: bool = True
    sklearn_version: str = ""
    bundle_id: str = ""
    built_at: str = ""

    def fingerprint(self) -> str:
        payload = json.dumps({
            "through": str(self.trained_through),
            "n": self.n_train_rows,
            "drift": self.use_drift,
            "physics": self.use_physics,
            "sm": self.use_sales_manager,
            "blend": self.blend,
            "cols": list(self.encoder.columns),
            "big": round(float(self.big_threshold), 2),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _component_present(df: pd.DataFrame, comp: str) -> np.ndarray:
    if comp in C.ALWAYS_PRESENT:
        return np.ones(len(df), dtype=bool)
    flag = C.OPTIONAL_GATE[comp]
    return pd.to_numeric(df[flag], errors="coerce").fillna(0).values.astype(int) == 1


def fit_bundle(df: pd.DataFrame, mask: pd.Series, *, use_drift=False,
               use_physics=False, use_sales_manager=False,
               variants=None, blend=C.BLEND_WEIGHT,
               recency=True, use_names=True) -> Bundle:
    """Fit the whole pipeline on the rows selected by `mask`.

    `mask` is the training window intersected with the usable mask. Nothing in
    here may look at a row outside it.
    """
    import sklearn

    train = df.loc[mask].copy()
    if len(train) == 0:
        raise ValueError("no training rows")

    ufm = F.build_use_family_map(train.get("Use Type", pd.Series(dtype=object)))
    eng = F.engineer(train, use_family_map=ufm)

    enc = F.Encoder(use_drift=use_drift, use_physics=use_physics,
                    use_sales_manager=use_sales_manager,
                    use_names=use_names).fit(eng)
    bb = F.Backbone(use_drift=use_drift, use_physics=use_physics).fit(eng)

    X = enc.transform(eng)
    B = bb.transform(eng)

    ref = pd.to_datetime(eng["Due Date"]).max()
    w_all = recency_weights(eng["Due Date"], ref) if recency else np.ones(len(eng))

    params = C.GBM_VARIANTS if variants is None else variants
    regs: dict[str, Stage] = {}
    for comp in C.COMPONENTS:
        present = _component_present(eng, comp)
        val = pd.to_numeric(eng[comp], errors="coerce").fillna(0.0).values
        fit_rows = present & (val > 0)
        if fit_rows.sum() < 30:
            regs[comp] = None
            continue
        y = np.log(val[fit_rows])
        regs[comp] = Stage(params, enc.categorical_mask).fit(
            X.loc[fit_rows], B[fit_rows], y, w_all[fit_rows])

    total = pd.to_numeric(eng[C.TARGET], errors="coerce").values
    ok = total > 0
    direct = Stage(params, enc.categorical_mask).fit(
        X.loc[ok], B[ok], np.log(total[ok]), w_all[ok])

    D = pd.to_numeric(eng["Diameter (ft)"], errors="coerce")
    H = pd.to_numeric(eng["Height (ft)"], errors="coerce")
    envelope = {
        "D": (float(D.quantile(0.005)), float(D.quantile(0.995))),
        "H": (float(H.quantile(0.005)), float(H.quantile(0.995))),
    }

    b = Bundle(
        encoder=enc, backbone=bb, regs=regs, direct=direct,
        tax=fit_tax_table(df, mask), use_family_map=ufm, blend=blend,
        big_threshold=float(pd.to_numeric(eng[C.TARGET], errors="coerce")
                            .quantile(C.BIG_QUANTILE)),
        envelope=envelope,
        trained_through=ref.date() if pd.notna(ref) else None,
        n_train_rows=int(len(eng)),
        use_drift=use_drift, use_physics=use_physics,
        use_sales_manager=use_sales_manager, use_names=use_names,
        sklearn_version=sklearn.__version__,
        built_at=datetime.utcnow().isoformat(timespec="seconds"),
    )
    b.bundle_id = b.fingerprint()
    return b


def predict_frame(bundle: Bundle, df: pd.DataFrame, cap_time=True) -> pd.DataFrame:
    """Score an engineered-or-raw frame. Returns per-tank component and total
    columns. SPEC 2.1.
    """
    eng = F.engineer(df, use_family_map=bundle.use_family_map)

    if cap_time and bundle.trained_through is not None:
        # SPEC 2.9: cap the drift term a quarter past the last training row.
        t_max = ((pd.Timestamp(bundle.trained_through) - pd.Timestamp(C.EPOCH)).days
                 / 365.25) + C.T_CAP_YEARS
        eng["t"] = eng["t"].clip(upper=t_max)
        eng["t_x_logD"] = eng["t"] * eng["log_D"]

    X = bundle.encoder.transform(eng)
    B = bundle.backbone.transform(eng)

    out = pd.DataFrame(index=eng.index)
    comp_sum = np.zeros(len(eng))
    for comp in C.COMPONENTS:
        stage = bundle.regs.get(comp)
        present = _component_present(eng, comp).astype(float)
        if stage is None:
            vals = np.zeros(len(eng))
        else:
            vals = np.exp(stage.predict(X, B)) * present
        out["est_" + comp.replace(" Price", "")] = vals
        comp_sum += vals

    direct = np.exp(bundle.direct.predict(X, B))
    point = bundle.blend * comp_sum + (1.0 - bundle.blend) * direct

    out["components_sum"] = comp_sum
    out["direct"] = direct
    out["point"] = point

    # tax: state rate gated by IS_TAXABLE, applied to the ex-freight base
    state = eng.get("State", pd.Series([None] * len(eng), index=eng.index))
    state = state.astype(object).map(lambda v: str(v).strip() if v is not None else "")
    country = eng.get("Country", pd.Series([None] * len(eng), index=eng.index))
    country = country.astype(object).map(lambda v: str(v).strip() if v is not None else "")
    rate = state.map(lambda v: bundle.tax.get(f"state:{v}") if v else None)
    # country fallback for non-US rows only (SPEC 4.4)
    rate = rate.fillna(country.map(
        lambda v: bundle.tax.get(f"country:{v}")
        if v and str(v).upper() != "US" else None))
    taxable = pd.to_numeric(eng.get("IS_TAXABLE", 0), errors="coerce").fillna(0).astype(int)
    has_rate = rate.notna()
    ex_freight = comp_sum - out["est_Freight"].values
    out["tax_rate"] = rate.fillna(0.0).values
    out["has_tax_rate"] = has_rate.values
    out["est_tax"] = np.where((taxable.values == 1) & has_rate.values,
                              ex_freight * rate.fillna(0.0).values, 0.0)
    out["est_proposal_total"] = np.where((taxable.values == 1) & ~has_rate.values,
                                         np.nan,
                                         ex_freight + out["est_tax"].values)
    return out
