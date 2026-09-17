"""Batch scoring. SPEC 5 B12, 6.6.

Use this for volume. Excel will crawl if inference fires on every
recalculation of every cell (SPEC 8.1 R8).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import scoring as E
from .contract import WARNING_TEXT


def score_csv(in_path, out_path, bundle, quoted_total_col: str | None = None):
    df = pd.read_csv(in_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    # SPEC 8.1 R17: scoring the training archive with the bundle trained on it
    # is the obvious smoke test and reports ~2% APE. Say so loudly.
    banner = None
    if bundle.trained_through is not None and "Due Date" in df.columns:
        due = pd.to_datetime(df["Due Date"], errors="coerce")
        n_seen = int((due <= pd.Timestamp(bundle.trained_through)).sum())
        if n_seen > 0:
            banner = (f"WARNING: {n_seen} of {len(df)} rows have a Due Date at or "
                      f"before this bundle's training cutoff "
                      f"({bundle.trained_through}). Those rows were very likely "
                      f"IN the training set; any error measured on them is "
                      f"in-sample and is not accuracy. Use the ledger or a "
                      f"rolling-origin backtest for an honest read.")
            print(banner)

    out = E.estimate_many(df, bundle=bundle)
    res = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    res["notes"] = out["warnings"].fillna("").map(
        lambda w: "; ".join(WARNING_TEXT.get(c.split(":")[0], c)
                            for c in w.split(",") if c) if w else "")

    if quoted_total_col and quoted_total_col in df.columns:
        qt = pd.to_numeric(df[quoted_total_col], errors="coerce")
        lo, hi = out["band80_lo"], out["band80_hi"]
        res["check"] = np.where(out["error"].notna(), out["error"],
                        np.where(qt.isna(), "",
                         np.where(qt < lo, "LOW",
                          np.where(qt > hi, "HIGH", "OK"))))
        res["gap_pct"] = np.where(qt > 0, (qt - out["point"]) / qt * 100.0, np.nan)

    p = Path(out_path)
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        res.to_excel(p, index=False)
    else:
        res.to_csv(p, index=False)
    return res, banner
