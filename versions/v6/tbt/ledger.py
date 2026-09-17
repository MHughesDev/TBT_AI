"""The append-only ledger of forward predictions.

SPEC 2.7. LOAD-BEARING ORDERING: the monthly job must (1) score the newly
arrived rows with the CURRENTLY DEPLOYED bundle and append here, then
(2) retrain. Reversing the order makes these residuals in-sample, the conformal
bands collapse and the smearing factor drifts to 1 (SPEC 8.1 R9).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LEDGER_COLUMNS = [
    "quote_key", "due_date", "bundle_id", "point", "book",
    "band80_lo", "band80_hi", "group", "tier", "actual_total",
    "scored_at", "model_shown",
]


def empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in LEDGER_COLUMNS})


def read(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return empty()
    df = pd.read_csv(p)
    for c in LEDGER_COLUMNS:
        if c not in df.columns:
            df[c] = None
    for c in ("point", "book", "band80_lo", "band80_hi", "actual_total"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["due_date"] = pd.to_datetime(df["due_date"], errors="coerce")
    return df[LEDGER_COLUMNS]


def append(path, rows: pd.DataFrame) -> int:
    p = Path(path)
    rows = rows.reindex(columns=LEDGER_COLUMNS)
    header = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(p, mode="a", header=header, index=False)
    return len(rows)


def coverage(ledger: pd.DataFrame) -> float:
    d = ledger.copy()
    for c in ("actual_total", "band80_lo", "band80_hi"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["actual_total", "band80_lo", "band80_hi"])
    if len(d) == 0:
        return float("nan")
    inside = (d["actual_total"] >= d["band80_lo"]) & (d["actual_total"] <= d["band80_hi"])
    return float(inside.mean())
