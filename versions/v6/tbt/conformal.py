"""Group-conditional conformal bands and the retransformation factor.

SPEC 2.6 (Mondrian conformal), 2.7 (ledger + smearing).

Both are computed from FORWARD residuals only. Residuals from rows the bundle
was fitted on collapse the bands to a fraction of their honest width and drift
the smearing factor toward 1 (SPEC 8.1 R9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

GLOBAL_GROUP = "ALL"


def group_of(use_family, is_big) -> str:
    """SPEC 2.6. `big` is defined on the PREDICTED value, because the actual is
    unknown when the band is drawn; a group that cannot be computed at
    inference is not a group."""
    fam = str(use_family) if use_family is not None else "Other"
    return f"{fam}|{'big' if bool(is_big) else 'std'}"


def family_of_group(group: str) -> str:
    return group.split("|")[0]


def _q(abs_resid: np.ndarray, level: float) -> float:
    """Conformal quantile with the finite-sample (1 + 1/n) correction."""
    n = len(abs_resid)
    if n == 0:
        return float("nan")
    k = min(1.0, level * (1.0 + 1.0 / n))
    return float(np.quantile(abs_resid, k))


def fit_conformal(ledger: pd.DataFrame,
                  min_group: int = C.CONFORMAL_MIN_GROUP) -> dict:
    """Build the band table from forward residuals.

    Groups with fewer than `min_group` residuals fall back to their use family;
    a family with fewer than `min_group` falls back to the global pool.
    """
    if ledger is None or len(ledger) == 0:
        return {}
    d = ledger.copy()
    # the ledger round-trips through CSV/object dtype; coerce before any maths
    for c in ("actual_total", "point"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["actual_total", "point"])
    d = d[(d["actual_total"] > 0) & (d["point"] > 0)]
    if len(d) == 0:
        return {}
    r = np.abs(np.log(d["actual_total"].values.astype(float))
               - np.log(d["point"].values.astype(float)))
    groups = d["group"].astype(str).values
    fams = np.array([family_of_group(g) for g in groups])

    table: dict[str, dict] = {}
    table[GLOBAL_GROUP] = {"q80": _q(r, 0.80), "q90": _q(r, 0.90),
                           "n": int(len(r)), "source": GLOBAL_GROUP}
    for fam in np.unique(fams):
        sel = fams == fam
        if sel.sum() >= min_group:
            table[f"fam:{fam}"] = {"q80": _q(r[sel], 0.80), "q90": _q(r[sel], 0.90),
                                   "n": int(sel.sum()), "source": f"fam:{fam}"}
    for g in np.unique(groups):
        sel = groups == g
        if sel.sum() >= min_group:
            table[g] = {"q80": _q(r[sel], 0.80), "q90": _q(r[sel], 0.90),
                        "n": int(sel.sum()), "source": g}
    return table


def lookup(table: dict, group: str) -> dict:
    """Resolve a group to its band entry, applying the fallback chain."""
    if not table:
        return {"q80": float("nan"), "q90": float("nan"), "n": 0, "source": "none"}
    if group in table:
        return table[group]
    fam_key = f"fam:{family_of_group(group)}"
    if fam_key in table:
        return table[fam_key]
    return table.get(GLOBAL_GROUP,
                     {"q80": float("nan"), "q90": float("nan"), "n": 0, "source": "none"})


def bands_for(table: dict, group: str, point: float) -> tuple[tuple[float, float],
                                                              tuple[float, float]]:
    e = lookup(table, group)
    q80, q90 = e.get("q80"), e.get("q90")
    if not np.isfinite(q80):
        return (float("nan"), float("nan")), (float("nan"), float("nan"))
    return ((point / np.exp(q80), point * np.exp(q80)),
            (point / np.exp(q90), point * np.exp(q90)))


def fit_smear(ledger: pd.DataFrame, big_threshold: float,
              quarters_back: int = 4, min_rows: int = 800) -> dict:
    """SPEC 2.7 retransformation factor, in two bands.

    Two bands, not one, because the aggregate bias is concentrated in the
    largest quotes. Clipped to SMEAR_CLIP; a value outside that range is a
    monitoring alarm, not a correction to apply.
    """
    out = {"rest": 1.0, "top5": 1.0, "n_rest": 0, "n_top5": 0, "alarm": []}
    if ledger is None or len(ledger) == 0:
        return out
    d = ledger.copy()
    for c in ("actual_total", "point"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["actual_total", "point"])
    d = d[(d["actual_total"] > 0) & (d["point"] > 0)]
    if len(d) == 0:
        return out

    if "due_date" in d.columns and len(d) > min_rows:
        due = pd.to_datetime(d["due_date"], errors="coerce")
        cutoff = due.max() - pd.DateOffset(months=3 * quarters_back)
        recent = d[due >= cutoff]
        if len(recent) >= min_rows:
            d = recent

    ratio = (d["actual_total"].values.astype(float)
             / d["point"].values.astype(float))
    # band assigned on the PREDICTED value so it can be assigned at inference
    is_big = d["point"].values.astype(float) >= big_threshold
    for name, sel in (("rest", ~is_big), ("top5", is_big)):
        if sel.sum() >= 30:
            raw = float(np.mean(ratio[sel]))
            out[f"{name}_raw"] = raw
            if not (C.SMEAR_CLIP[0] <= raw <= C.SMEAR_CLIP[1]):
                out["alarm"].append(
                    f"smear[{name}]={raw:.3f} outside {C.SMEAR_CLIP}; not applied")
            out[name] = float(np.clip(raw, *C.SMEAR_CLIP))
        out[f"n_{name}"] = int(sel.sum())
    return out
