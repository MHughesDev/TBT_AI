"""Feature construction for v5.

Three groups, in descending order of how much they matter:

1. Geometry and physics — what the tank *is*. v4's ablation put geometry at 22
   points of median error, an order of magnitude ahead of anything else. v5 adds
   the computed shell schedule on top (see physics.py).
2. Context — material, use type, wage regime, geography, time. Worth ~20 points
   *conditional on* size and near-useless alone, which is the interaction
   structure gradient boosting exists for.
3. Tank Name text — the process descriptor is the only proxy available for
   appendages that never appear as columns. A digester has mixers, covers and gas
   handling; an equalization basin is a bare shell.

Scope arrives as five explicit yes/no columns and is never inferred. That rule is
inherited from v4 unchanged and is not up for revision — see the note in
model.py.
"""
import numpy as np
import pandas as pd

from . import physics
from .config import SCOPE, EPOCH, DATE_COL

# Free-text mining of Tank Name. Carried over from v4.2, where it was measured as
# a real gain on engineered-to-order work: median $/sq-ft runs 1.35x base for
# digesters against 0.68x for multi-zone configurations.
TN_GROUPS = {
    "tn_digester":  ["digester", "anaerobic", "aerobic", "biogas"],
    "tn_bio":       ["mbbr", "sbr", "cmas", "biomass", "bioreactor", "activated", "moving bed"],
    "tn_sludge":    ["sludge", "slurry", "thickener", "thickening", "biosolid"],
    "tn_reactor":   ["reactor", "reaeration", "oxidation", "contact"],
    "tn_clarifier": ["clarifier", "settling", "sedimentation", "launder"],
    "tn_eq":        ["equalization", "equalisation", " eq ", "flow equal", "detention", "retention"],
    "tn_aeration":  ["aeration", "aerated", "diffuser", "blower"],
    "tn_leachate":  ["leachate", "landfill"],
    "tn_fire":      ["fire", "fp tank", "nfpa", "sprinkler"],
    "tn_potable":   ["potable", "drinking", "finished water"],
    "tn_process":   ["process", "chemical", "acid", "caustic", "brine", "glycol", "oil", "fuel", "diesel"],
    "tn_silo":      ["silo", "lime", "carbon", "bulk"],
    "tn_backwash":  ["backwash", "wash water", "washwater", "rinse"],
    "tn_coating":   ["epoxy", "coated", "coating", "lined", "liner", "glass", "galvan"],
    "tn_dome":      ["dome", "geodesic", "cover", "membrane"],
    "tn_rafter":    ["eccs", "hdg", "rafter", "ext."],
    "tn_config":    ["dual", "combo", "hybrid", "zone", "multi", "two-stage", "stage"],
    "tn_optional":  ["optional", "option", "opt.", "alternate", "alt "],
}
TN_FEATURES = list(TN_GROUPS) + ["tn_len", "tn_words", "tn_generic", "tn_has_num", "tn_cap_k"]

GEOMETRY = [
    "Diameter (ft)", "Height (ft)", "Freeboard (in)", "Quantity",
    "shell_area", "floor_area", "total_area", "vol_cf", "log_vol",
    "aspect", "hoop", "steel_proxy",
]
CONTEXT = [
    "months", "t_sq", "t_x_area", "mi", "Ss", "S1", "seismic",
    "Miles to Site (From TBT)", "Miles to Site (From GT)",
]

# Populated by the comparables engine at fit time; absent (NaN) is meaningful and
# HistGradientBoosting handles it natively without imputation.
COMPARABLE_FEATURES = ["cmp_logpsf", "cmp_n", "cmp_dist", "cmp_spread", "cmp_logprice"]

NUMERIC = (GEOMETRY + CONTEXT + physics.PHYSICS_FEATURES + TN_FEATURES
           + SCOPE + COMPARABLE_FEATURES)
CATEGORICAL = ["Material", "Use Type", "Deck Style", "Floor Style", "Wage Type",
               "Country", "State", "Bid Type", "Sales Manager"]


def as01(s):
    """Accept Yes/No, TRUE/FALSE, Y/N, 1/0 — return 1/0, preserving NaN.

    NaN survives on purpose: a blank scope answer must reach check_scope() and
    raise, not be quietly read as 'No'.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return np.where(s.isna(), np.nan, (s.fillna(0) > 0).astype(float))
    m = s.astype(str).str.strip().str.upper()
    return m.map({"YES": 1.0, "Y": 1.0, "TRUE": 1.0, "T": 1.0, "1": 1.0,
                  "NO": 0.0, "N": 0.0, "FALSE": 0.0, "F": 0.0, "0": 0.0})


def build(df):
    """Engineer every feature that depends only on the row itself.

    Comparables are NOT built here: they depend on which rows are in the training
    fold, so they are attached later by the comparables engine. Doing it here
    would leak the future into the past.
    """
    df = df.copy()

    for c in SCOPE:
        if c in df.columns:
            df[c] = as01(df[c])

    D = pd.to_numeric(df["Diameter (ft)"], errors="coerce").astype(float)
    H = pd.to_numeric(df["Height (ft)"], errors="coerce").astype(float)
    df["Diameter (ft)"], df["Height (ft)"] = D, H

    df["shell_area"] = np.pi * D * H
    df["floor_area"] = np.pi * (D / 2.0) ** 2
    df["total_area"] = df["shell_area"] + 2.02 * df["floor_area"]
    df["vol_cf"] = df["floor_area"] * H
    df["log_vol"] = np.log1p(df["vol_cf"])
    df["aspect"] = H / D.replace(0, np.nan)
    df["hoop"] = D * H
    df["steel_proxy"] = df["shell_area"] * np.sqrt(df["hoop"].clip(lower=0))

    # Time, in months from a fixed epoch so that train and score agree.
    if DATE_COL in df.columns:
        d = pd.to_datetime(df[DATE_COL], errors="coerce")
    else:
        d = pd.Series(pd.Timestamp.today(), index=df.index)
    df["months"] = (d - pd.Timestamp(EPOCH)).dt.days / 30.4
    df["t_sq"] = df["months"] ** 2
    # Drift is size-dependent — over the archive the largest quartile of tanks
    # rose 22% per sq-ft while the second-largest fell 4%. A lone time feature
    # cannot express that, so time is allowed to interact with size.
    df["t_x_area"] = df["months"] * np.log1p(df["shell_area"])

    for c in ["Ss", "S1", "Freeboard (in)", "Quantity",
              "Miles to Site (From TBT)", "Miles to Site (From GT)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    df["seismic"] = df["Ss"] * df["S1"]
    mc = [c for c in ["Miles to Site (From TBT)", "Miles to Site (From GT)"]
          if df[c].notna().any()]
    df["mi"] = df[mc].min(axis=1) if mc else np.nan

    ph = physics.compute(
        D.fillna(0).values, H.fillna(0).values,
        freeboard_in=df["Freeboard (in)"].fillna(0).values,
        Ss=df["Ss"].values,
        deck_style=df["Deck Style"].values if "Deck Style" in df else None,
    )
    for k, v in ph.items():
        df[k] = v

    t = (df["Tank Name"].astype(str).str.lower() if "Tank Name" in df
         else pd.Series("", index=df.index))
    for feat, keys in TN_GROUPS.items():
        df[feat] = np.array([int(any(k in s for k in keys)) for s in t])
    # A bespoke tank gets a long descriptive name; a commodity one is "Tank 1".
    df["tn_len"] = t.str.len().values
    df["tn_words"] = t.str.split().str.len().fillna(0).values
    df["tn_generic"] = t.str.match(r"^\s*tank\s*\d*\s*$").astype(int).values
    df["tn_has_num"] = t.str.contains(r"\d").astype(int).values
    df["tn_cap_k"] = pd.to_numeric(
        t.str.extract(r"(\d+(?:\.\d+)?)\s*k\b", expand=False),
        errors="coerce").fillna(0).values

    for c in NUMERIC + CATEGORICAL:
        if c not in df.columns:
            df[c] = np.nan
    return df


def backbone(df, with_physics=True):
    """Design matrix for the parametric log-log stage.

    v4's backbone is [log D, log H, log D x log H, t]. v5 adds log(shell steel
    weight), the min-governed fraction, and their interaction — which is the
    whole point: those carry the regime change that a pure power law in D and H
    cannot bend to, because the cost curve has a kink at a fixed diameter and a
    power law has none. The remaining terms are kept so that v5 nests v4's basis
    rather than replacing it: if the physics terms are worthless the ridge zeroes
    them and this degenerates to v4's.

    `with_physics=False` gives exactly v4's basis, for the ablation.
    """
    lD = np.log(df["Diameter (ft)"].astype(float).clip(lower=0.1).values)
    lH = np.log(df["Height (ft)"].astype(float).clip(lower=0.1).values)
    t = df["months"].astype(float).fillna(0).values / 12.0
    cols = [lD, lH, lD * lH, t, np.ones(len(df))]
    if with_physics:
        lsteel = np.log1p(df["ph_shell_lb"].astype(float).clip(lower=0).values)
        lgov = df["ph_min_governed_frac"].astype(float).fillna(1.0).values
        cols[3:3] = [lsteel, lsteel * lgov, lgov]
    return np.column_stack(cols)
