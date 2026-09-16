"""Derived fields, the physics term, the backbone basis and the feature matrix.

SPEC 2.3 (backbone + steel weight), 4.3 (feature matrix), 4.4 (formulas),
4.5 (leakage guard).
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------- use family
# SPEC 4.4. A fixed table, not a regex applied at runtime. The mapping is built
# once from the observed Use Type vocabulary by `build_use_family_map` and
# stored in the bundle, so it cannot drift between fit and inference.
_FAMILY_RULES = [
    ("Waste", ("waste", "sewage", "sludge", "digest")),
    ("Fire", ("fire",)),
    ("Industrial", ("industrial", "silo", "process", "chemical")),
    ("Potable", ("potable", "water storage")),
]


def use_family_of(use_type: str | None) -> str:
    """Map one Use Type string to its family. SPEC 4.4.

    Waste is tested before Fire and Potable because 'Waste Water Storage Tank'
    contains 'water storage'; the ordering is what keeps the families disjoint.
    """
    if use_type is None or (isinstance(use_type, float) and np.isnan(use_type)):
        return "Other"
    s = str(use_type).strip().lower()
    if not s:
        return "Other"
    for fam, needles in _FAMILY_RULES:
        if any(n in s for n in needles):
            return fam
    return "Other"


def build_use_family_map(use_types) -> dict[str, str]:
    """Enumerate the observed Use Type vocabulary into a fixed table."""
    vocab = sorted({str(u).strip() for u in pd.Series(use_types).dropna().unique()})
    return {u: use_family_of(u) for u in vocab}


# ------------------------------------------------------------- tank name
_NAME_FAMILIES = [
    ("backwash", r"backwash"),
    ("rafters_eccs_hdg", r"ext\.?\s*rafter|eccs|hdg"),
    ("dome", r"dome|geodesic|membrane"),
    ("digester", r"digest|anaerob|aerob"),
    ("leachate", r"leachate|landfill"),
    ("sludge", r"sludge|slurry|thicken"),
    ("equalization", r"equaliz|detention"),
    ("process", r"process|chemical|oil"),
    ("multizone", r"dual|combo|hybrid|zone"),
]
_NAME_FAMILY_RE = [(name, re.compile(pat)) for name, pat in _NAME_FAMILIES]
_GENERIC_RE = re.compile(r"^(tank|t|tk)?\s*[-#]?\s*\d*[a-z]?$")


def name_features(name) -> dict:
    """SPEC 4.4. The family list is fixed; do not extend it from the data
    during the build (SPEC 8.1 R13) -- that is a target-encoding search
    wearing a regex.
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        s = ""
    else:
        s = re.sub(r"\s+", " ", str(name).strip().lower())
    generic = 1 if (s == "" or _GENERIC_RE.match(s) is not None) else 0
    fam = "none"
    for fname, rx in _NAME_FAMILY_RE:
        if rx.search(s):
            fam = fname
            break
    return {
        "name_len": float(len(s)),
        "name_words": float(len(s.split())) if s else 0.0,
        "name_has_digits": 1.0 if any(ch.isdigit() for ch in s) else 0.0,
        "name_generic": float(generic),
        "name_family": fam,
    }


# --------------------------------------------------------------- wage recode
NO_ERECTION = "No Erection Included"


def recode_wage(df: pd.DataFrame) -> pd.Series:
    """SPEC 4.4. 'No Erection Included' is a scope statement disguised as a
    labour-rate class; with scope explicit it is redundant at best and at
    inference a way for the dropdown to contradict the scope block.
    """
    if "Wage Type" not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    w = df["Wage Type"].astype(object).where(df["Wage Type"].notna(), None)
    mask = w.map(lambda v: isinstance(v, str) and v.strip() == NO_ERECTION)
    return w.mask(mask, None)


# ------------------------------------------------------------ physics term
def steel_lb_est(D, H, freeboard_in=0.0) -> np.ndarray:
    """One-foot-method shell + floor steel weight estimate. SPEC 2.3 (OPEN-2).

    Its purpose is the SHAPE -- a power law in D and H cannot bend where
    hoop-stress thickness overtakes the code minimum, and this can. The ridge
    coefficient sets the scale. The constants are nominal AWWA D100 / API 650
    values and must NOT be tuned to the data (SPEC 8.1 R15).
    """
    D = np.asarray(D, dtype=float)
    H = np.asarray(H, dtype=float)
    fb = np.asarray(freeboard_in, dtype=float)
    fb = np.nan_to_num(fb, nan=0.0)

    H_liq = np.maximum(H - fb / 12.0, 1.0)
    n_courses = np.ceil(np.maximum(H, 1.0) / C.COURSE_HEIGHT_FT).astype(int)
    max_courses = int(n_courses.max()) if n_courses.size else 1

    # minimum plate thickness by diameter band
    t_min = np.where(D < 50, 0.1875,
             np.where(D < 120, 0.25,
              np.where(D < 200, 0.3125, 0.375)))

    shell_lb = np.zeros_like(D, dtype=float)
    for k in range(1, max_courses + 1):
        bottom = C.COURSE_HEIGHT_FT * (k - 1)
        active = (n_courses >= k)
        # design head one foot above the course bottom
        h_k = np.maximum(H_liq - bottom - 1.0, 0.0)
        t_hoop = 2.6 * D * h_k * C.STEEL_G / (C.STEEL_S * C.STEEL_E)
        t_k = np.maximum(t_hoop, t_min)
        course_h = np.clip(H - bottom, 0.0, C.COURSE_HEIGHT_FT)
        shell_lb += np.where(active,
                             np.pi * D * course_h * t_k * C.STEEL_LB_PER_SQFT_IN,
                             0.0)

    floor_lb = np.pi * (D / 2.0) ** 2 * C.FLOOR_THICKNESS_IN * C.STEEL_LB_PER_SQFT_IN
    return shell_lb + floor_lb


# ---------------------------------------------------------------- engineer
def engineer(df: pd.DataFrame, use_family_map: dict | None = None) -> pd.DataFrame:
    """Add every derived column. SPEC 4.4. Pure: returns a new frame."""
    out = df.copy()

    D = pd.to_numeric(out.get("Diameter (ft)"), errors="coerce").astype(float)
    H = pd.to_numeric(out.get("Height (ft)"), errors="coerce").astype(float)
    out["Diameter (ft)"] = D
    out["Height (ft)"] = H

    fb = pd.to_numeric(out.get("Freeboard (in)", 0.0), errors="coerce").fillna(0.0)
    out["Freeboard (in)"] = fb

    qty = pd.to_numeric(out.get("Quantity", 1), errors="coerce").fillna(1).clip(lower=1)
    out["Quantity"] = qty

    # geometry
    out["shell_area"] = np.pi * D * H
    out["floor_area"] = np.pi * (D / 2.0) ** 2
    out["log_D"] = np.log(D.where(D > 0))
    out["log_H"] = np.log(H.where(H > 0))
    out["logD_x_logH"] = out["log_D"] * out["log_H"]
    out["log_shell_area"] = np.log(out["shell_area"].where(out["shell_area"] > 0))
    out["log_floor_area"] = np.log(out["floor_area"].where(out["floor_area"] > 0))
    out["aspect"] = H / D.where(D > 0)
    out["log_quantity"] = np.log(qty)
    out["freeboard_in"] = fb

    # time axis
    due = pd.to_datetime(out.get("Due Date"), errors="coerce")
    out["Due Date"] = due
    out["t"] = (due - pd.Timestamp(C.EPOCH)).dt.days / 365.25
    out["t_x_logD"] = out["t"] * out["log_D"]

    # physics
    out["steel_lb_est"] = steel_lb_est(D.values, H.values, fb.values)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_steel_lb_est"] = np.log(np.where(out["steel_lb_est"].values > 0,
                                                  out["steel_lb_est"].values, np.nan))

    # pass-through numerics
    for src, dst in [("Miles to Site (From TBT)", "miles_tbt"),
                     ("Miles to Site (From GT)", "miles_gt")]:
        out[dst] = pd.to_numeric(out.get(src), errors="coerce") if src in out.columns \
            else np.nan
    for c in ("Ss", "S1"):
        out[c] = pd.to_numeric(out.get(c), errors="coerce") if c in out.columns else np.nan

    # families
    if use_family_map:
        out["use_family"] = (out.get("Use Type").astype(object)
                             .map(lambda u: use_family_map.get(str(u).strip(),
                                                               use_family_of(u))))
    else:
        out["use_family"] = out.get("Use Type").map(use_family_of)

    # tank name
    nf = pd.DataFrame([name_features(v) for v in out.get("Tank Name", pd.Series([None] * len(out)))],
                      index=out.index)
    for c in nf.columns:
        out[c] = nf[c]

    # wage recode
    out["Wage Type"] = recode_wage(out)

    return out


# ------------------------------------------------------------- feature sets
NAME_NUMERIC = ["name_len", "name_words", "name_has_digits", "name_generic"]


def numeric_features(use_drift: bool, use_physics: bool,
                     use_names: bool = True) -> list[str]:
    feats = [f for f in C.NUMERIC_FEATURES_BASE
             if use_names or f not in NAME_NUMERIC]
    if use_drift:
        feats.append(C.NUMERIC_FEATURE_DRIFT)
    if use_physics:
        feats.append(C.NUMERIC_FEATURE_PHYSICS)
    return feats


def categorical_features(use_sales_manager: bool,
                         use_names: bool = True) -> list[str]:
    feats = [f for f in C.CATEGORICAL_FEATURES_BASE
             if use_names or f != "name_family"]
    if use_sales_manager:
        feats.append(C.CATEGORICAL_FEATURE_SALES_MANAGER)
    return feats


def assert_no_banned(columns) -> None:
    """SPEC 4.5. The feature builder raises before any fit.

    Left unguarded the model reports under 1% mean APE and looks finished,
    which is why this is a guard and not a comment (SPEC 6.4 test 1).
    """
    banned = sorted(set(map(str, columns)) & set(C.BANNED))
    if banned:
        raise ValueError(
            "Banned columns in feature matrix (SPEC 4.5 leakage list): "
            + ", ".join(banned)
        )


class Encoder:
    """Fixed column order and categorical vocabularies. SPEC 4.3.

    Vocabularies are fixed at fit time FROM THE TRAINING WINDOW ONLY. Fitting
    them on the whole file is leakage of the future (SPEC 4.5 item 11) and is
    the subtle kind, because it is 'just a lookup'.
    """

    def __init__(self, use_drift=False, use_physics=False,
                 use_sales_manager=False, use_names=True):
        self.use_drift = use_drift
        self.use_physics = use_physics
        self.use_sales_manager = use_sales_manager
        self.use_names = use_names
        self.num = numeric_features(use_drift, use_physics, use_names)
        self.cat = categorical_features(use_sales_manager, use_names)
        self.scope = list(C.SCOPE_FEATURES)
        self.vocab: dict[str, list[str]] = {}
        self.columns = self.num + self.scope + self.cat

    def fit(self, df: pd.DataFrame) -> "Encoder":
        assert_no_banned(self.columns)
        self.vocab = {}
        for c in self.cat:
            vals = df[c].astype(object) if c in df.columns else pd.Series([], dtype=object)
            uniq = sorted({str(v).strip() for v in vals.dropna().unique()
                           if str(v).strip() != ""})
            self.vocab[c] = uniq
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert_no_banned(self.columns)
        X = pd.DataFrame(index=df.index)
        for c in self.num:
            X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
        for c in self.scope:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float) \
                if c in df.columns else 0.0
        for c in self.cat:
            raw = df[c].astype(object) if c in df.columns else pd.Series([None] * len(df),
                                                                        index=df.index)
            cleaned = raw.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
                              else (str(v).strip() or None))
            # unseen values become missing -- SPEC 2.9 tier 3
            cleaned = cleaned.map(lambda v: v if (v is None or v in self.vocab.get(c, []))
                                  else None)
            X[c] = pd.Categorical(cleaned, categories=self.vocab.get(c, []))
        return X[self.columns]

    @property
    def categorical_mask(self) -> list[bool]:
        return [c in self.cat for c in self.columns]

    def unseen_fields(self, df: pd.DataFrame) -> list[str]:
        """Which categorical fields carry a value not in the training vocabulary."""
        out = []
        for c in self.cat:
            if c not in df.columns:
                continue
            for v in df[c].astype(object).dropna().unique():
                s = str(v).strip()
                if s and s not in self.vocab.get(c, []):
                    out.append(c)
                    break
        return out


# ------------------------------------------------------------- backbone basis
class Backbone:
    """Ridge log-log basis. SPEC 2.3.

    LOAD-BEARING. A tree predicts a constant beyond its largest leaf; this is
    what gives the ensemble a sane slope outside the training envelope. Remove
    it and the top-5% bias returns to about -25% (SPEC 8.1 R10).

    Do NOT add log(shell_area), log(volume) or any other product of powers of
    D and H: they are exactly collinear with log_D and log_H, and the prior
    investigation recorded coefficients of -12.7 from doing so.
    """

    BASE = ["log_D", "log_H", "logD_x_logH", "t"]

    def __init__(self, use_drift=False, use_physics=False):
        self.use_drift = use_drift
        self.use_physics = use_physics
        self.cols = list(self.BASE)
        if use_drift:
            self.cols.append("t_x_logD")
        if use_physics:
            self.cols.append("log_steel_lb_est")
        self.materials: list[str] = []
        self.mu = None
        self.sd = None

    def fit(self, df: pd.DataFrame) -> "Backbone":
        mats = sorted({str(m).strip() for m in df.get("Material", pd.Series(dtype=object))
                       .dropna().unique() if str(m).strip()})
        # CS is the reference level and is dropped from the one-hot
        self.materials = [m for m in mats if m.upper() != "CS"]
        B = self._raw(df)
        self.mu = np.nanmean(B[:, :len(self.cols)], axis=0)
        sd = np.nanstd(B[:, :len(self.cols)], axis=0)
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.cols:
            v = pd.to_numeric(df[c], errors="coerce").values.astype(float) \
                if c in df.columns else np.full(len(df), np.nan)
            cols.append(v)
        M = df.get("Material", pd.Series([None] * len(df), index=df.index)).astype(object)
        M = M.map(lambda v: str(v).strip() if v is not None else "")
        for m in self.materials:
            cols.append((M == m).values.astype(float))
        return np.column_stack(cols) if cols else np.zeros((len(df), 0))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        B = self._raw(df)
        k = len(self.cols)
        B[:, :k] = (B[:, :k] - self.mu) / self.sd
        # a missing basis value would poison the ridge; the median of the
        # standardised basis is 0 by construction
        return np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
