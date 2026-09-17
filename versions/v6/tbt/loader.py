"""Archive loading, validation and the single `usable` mask.

SPEC 3.1 (row eligibility), 4.1 (columns), 4.2 (scope backfill),
4.6 (load-time validation: reject, do not repair).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C


class LoadRejected(Exception):
    """SPEC 4.6. The file is refused; no model is trained and the previous
    bundle stays deployed."""

    def __init__(self, failures: list[str]):
        self.failures = failures
        super().__init__("Archive rejected:\n  - " + "\n  - ".join(failures))


REQUIRED_COLUMNS = [
    "Quote #", "Due Date", "Material", "Use Type",
    "Diameter (ft)", "Height (ft)",
    *C.COMPONENTS,
    "Proposal Total", "Total Price",
]


@dataclass
class LoadReport:
    n_rows: int = 0
    n_quotes: int = 0
    n_usable: int = 0
    failures_by_test: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"rows={self.n_rows} quotes={self.n_quotes} usable={self.n_usable}"]
        for k, v in self.failures_by_test.items():
            if v:
                lines.append(f"  dropped by {k}: {v}")
        for w in self.warnings:
            lines.append(f"  WARN {w}")
        return "\n".join(lines)


def _num(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def read_archive(path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def backfill_scope(df: pd.DataFrame) -> pd.DataFrame:
    """SPEC 4.2. Training/backtesting only.

    A blank price and a genuinely excluded scope are indistinguishable after
    the fact. This is a known limitation and is why SPEC 9 item 3 ranks
    capturing scope at quote time so highly.
    """
    out = df.copy()
    pairs = {
        "IS_CONSTRUCTION": "Construction Price",
        "IS_INSULATION": "Insulation Material Price",
        "IS_INSULATION_ERECTION": "Insulation Construction Price",
        "IS_FREIGHT": "Freight Price",
        "IS_TAXABLE": "Total Tax",
    }
    for flag, col in pairs.items():
        if flag in out.columns and out[flag].notna().any():
            # real scope columns take precedence; cross-check and log
            derived = (_num(out, col).fillna(0.0) > 0).astype(int)
            given = pd.to_numeric(out[flag], errors="coerce").fillna(0).astype(int)
            out[flag] = given
            out.attrs.setdefault("scope_disagreements", {})[flag] = int((derived != given).sum())
        else:
            out[flag] = (_num(out, col).fillna(0.0) > 0).astype(int)
    return out


def usable_mask(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """SPEC 3.1. THE single eligibility mask, used for fitting AND scoring.

    LOAD-BEARING. Filtering one and not the other measures the model more
    kindly without making it better (SPEC 8.1 R3). There is exactly one
    implementation and both paths call it.
    """
    reasons: dict[str, int] = {}
    ok = pd.Series(True, index=df.index)

    due_ok = pd.to_datetime(df.get("Due Date"), errors="coerce").notna()
    reasons["due_date"] = int((~due_ok & ok).sum())
    ok &= due_ok

    D, H = _num(df, "Diameter (ft)"), _num(df, "Height (ft)")
    geom_ok = (D > 0) & (H > 0)
    reasons["geometry"] = int((~geom_ok & ok).sum())
    ok &= geom_ok

    mat, fab = _num(df, "Material Price"), _num(df, "Fabrication Price")
    proc_ok = (mat > 0) & (fab > 0)
    reasons["material_fabrication_zero"] = int((~proc_ok & ok).sum())
    ok &= proc_ok

    total = _num(df, "Total Price")
    comp_sum = sum(_num(df, c).fillna(0.0) for c in C.COMPONENTS)
    tol = np.maximum(1.0, 0.001 * total.abs())
    ident_ok = (total > 0) & ((total - comp_sum).abs() <= tol)
    reasons["identity"] = int((~ident_ok & ok).sum())
    ok &= ident_ok

    shell = np.pi * D * H
    psf = total / shell.where(shell > 0)
    gate_ok = psf.between(C.PSF_MIN, C.PSF_MAX)
    reasons["plausibility_gate"] = int((~gate_ok & ok).sum())
    ok &= gate_ok

    return ok.fillna(False), reasons


def _check_per_tank(df: pd.DataFrame) -> tuple[bool, str]:
    """SPEC 4.6 per-tank tripwire.

    Catches someone upstream starting to extend prices by Quantity. The
    signature is that $/sq-ft falls as exactly 1/Quantity (SPEC 8.1 R2).
    """
    D, H = _num(df, "Diameter (ft)"), _num(df, "Height (ft)")
    shell = np.pi * D * H
    total = _num(df, "Total Price")
    qty = _num(df, "Quantity", 1).fillna(1)
    psf = (total / shell.where(shell > 0))
    base = psf[qty == 1].median()
    if not np.isfinite(base) or base <= 0:
        return True, "per_tank: no qty=1 rows to compare"
    bad = []
    for k in (2, 3, 4):
        sub = psf[qty == k].dropna()
        if len(sub) < 10:
            continue
        ratio = float(sub.median() / base)
        if not (0.6 <= ratio <= 1.4):
            bad.append(f"qty={k} ratio={ratio:.2f}")
    if bad:
        return False, ("per-tank tripwire: $/sq-ft falls with Quantity ("
                       + "; ".join(bad) + "). Prices appear extended by Quantity; "
                       "they must be per tank (SPEC 4.5).")
    return True, "per_tank ok"


def validate(df: pd.DataFrame, previous_usable: int | None = None) -> LoadReport:
    """SPEC 4.6. Rejects the file rather than repairing it."""
    rep = LoadReport(n_rows=len(df))
    failures: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        failures.append(f"missing required columns: {missing}")
        raise LoadRejected(failures)

    rep.n_quotes = int(df["Quote #"].nunique())

    due = pd.to_datetime(df["Due Date"], errors="coerce")
    parse_rate = float(due.notna().mean())
    rep.checks["due_parse_rate"] = parse_rate
    if parse_rate < 0.98:
        failures.append(f"Due Date parse rate {parse_rate:.3f} < 0.98")

    total = _num(df, "Total Price")
    comp_sum = sum(_num(df, c).fillna(0.0) for c in C.COMPONENTS)
    all_present = total.notna()
    id1 = ((total - comp_sum).abs() <= np.maximum(1.0, 0.001 * total.abs()))
    rate1 = float(id1[all_present].mean()) if all_present.any() else 0.0
    rep.checks["identity_total"] = rate1
    if rate1 < 0.995:
        failures.append(f"Total Price identity holds on {rate1:.3f} of rows < 0.995")

    ex_freight = sum(_num(df, c).fillna(0.0) for c in C.COMPONENTS
                     if c != "Freight Price")
    prop = _num(df, "Proposal Total")
    id2 = ((prop - (ex_freight + _num(df, "Total Tax").fillna(0.0))).abs()
           <= np.maximum(1.0, 0.001 * prop.abs()))
    rate2 = float(id2[prop.notna()].mean()) if prop.notna().any() else 0.0
    rep.checks["identity_proposal"] = rate2
    if rate2 < 0.995:
        failures.append(f"Proposal Total identity holds on {rate2:.3f} of rows < 0.995")

    ok_pt, msg_pt = _check_per_tank(df)
    rep.checks["per_tank"] = msg_pt
    if not ok_pt:
        failures.append(msg_pt)

    mask, reasons = usable_mask(df)
    rep.n_usable = int(mask.sum())
    rep.failures_by_test = reasons
    if rep.n_usable < 1000:
        failures.append(f"usable rows {rep.n_usable} < 1000")
    if previous_usable is not None and rep.n_usable < 0.90 * previous_usable:
        failures.append(f"usable rows {rep.n_usable} < 90% of previous {previous_usable}")

    scoped = backfill_scope(df)
    nest_bad = int(((scoped["IS_INSULATION_ERECTION"] == 1)
                    & (scoped["IS_INSULATION"] == 0)).sum())
    rep.checks["insulation_nesting_violations"] = nest_bad
    if nest_bad > 0:
        failures.append(f"{nest_bad} rows have insulation erection without supply")

    key = [c for c in ("Quote #", "Revision #", "Tank Name") if c in df.columns]
    if key:
        g = df.groupby(key, dropna=False)["Total Price"].nunique(dropna=False)
        dup_rate = float((g > 1).sum() / max(len(g), 1))
        rep.checks["dup_key_rate"] = dup_rate
        if dup_rate > 0.01:
            failures.append(f"{dup_rate:.3f} of quote keys carry differing prices > 0.01")

    if due.notna().any():
        ahead = (due.max() - pd.Timestamp.today().normalize()).days
        rep.checks["max_due_days_ahead"] = int(ahead)
        if ahead > 120:
            failures.append(f"newest Due Date is {ahead} days in the future > 120")

    # warnings (file still accepted)
    if "Usable Capacity" in df.columns:
        cap = _num(df, "Usable Capacity")
        D, H = _num(df, "Diameter (ft)"), _num(df, "Height (ft)")
        fb = _num(df, "Freeboard (in)").fillna(0.0)
        implied = np.pi * (D / 2) ** 2 * (H - fb / 12.0) * 7.48
        rel = ((cap - implied).abs() / implied.where(implied > 0))
        share = float((rel > 0.25).mean(skipna=True))
        if np.isfinite(share) and share > 0.05:
            rep.warnings.append(f"Usable Capacity disagrees with geometry on {share:.1%} of rows")

    if failures:
        raise LoadRejected(failures)
    return rep


def load_training(path, previous_usable: int | None = None):
    """Read, validate, backfill scope, and return (frame, usable_mask, report)."""
    raw = read_archive(path)
    rep = validate(raw, previous_usable=previous_usable)
    df = backfill_scope(raw)
    df["Due Date"] = pd.to_datetime(df["Due Date"], errors="coerce")
    mask, _ = usable_mask(df)
    return df, mask, rep
