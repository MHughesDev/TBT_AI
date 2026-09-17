"""The archive audit workbook. SPEC 5 B14.

Every usable row re-scored FORWARD from the ledger -- never by a bundle that
was fitted on it -- sorted worst-first, with the driving component named.

This is the adoption evidence for Phase 2 (SPEC 7.7): work the top 50 with an
estimator. Either the model is finding real mispricing, which is a business
case, or it is wrong in a patterned way, which tells you which column is
missing. Both are worth more than another tenth of a point of accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def build_audit(rows: pd.DataFrame, archive: pd.DataFrame,
                out_path) -> pd.DataFrame:
    """`rows` must be forward predictions (protocol.backtest().rows or the
    ledger). Passing in-sample predictions here defeats the purpose."""
    d = rows.copy()
    d["actual_total"] = pd.to_numeric(d["actual_total"], errors="coerce")
    d["point"] = pd.to_numeric(d["point"], errors="coerce")
    d = d[(d["actual_total"] > 0) & (d["point"] > 0)].copy()
    d["signed_pct"] = (d["point"] - d["actual_total"]) / d["actual_total"] * 100.0
    d["abs_pct"] = d["signed_pct"].abs()

    keep = [c for c in ("Quote #", "Tank Name", "Due Date", "Country", "State",
                        "Material", "Use Type", "Deck Style", "Diameter (ft)",
                        "Height (ft)", "Quantity", *C.COMPONENTS)
            if c in archive.columns]
    arch = archive[keep].copy()
    if "Quote #" in arch.columns and "quote_key" in d.columns:
        arch["quote_key"] = arch["Quote #"].astype(str)
        arch = arch.drop_duplicates("quote_key")
        d = d.merge(arch, on="quote_key", how="left", suffixes=("", "_arch"))

    # name the component driving the disagreement
    comp_cols = [c for c in C.COMPONENTS if c in d.columns]
    if comp_cols:
        share = {}
        for c in comp_cols:
            share[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
        tot = sum(share.values())
        biggest = pd.DataFrame(share).idxmax(axis=1)
        d["largest_component"] = biggest
        d["largest_component_share"] = (pd.DataFrame(share).max(axis=1)
                                        / tot.where(tot > 0))

    d = d.sort_values("abs_pct", ascending=False)
    cols = [c for c in ("quote_key", "Tank Name", "due_date", "quarter", "group",
                        "tier", "point", "actual_total", "signed_pct", "abs_pct",
                        "largest_component", "largest_component_share",
                        "Country", "State", "Material", "Use Type",
                        "Diameter (ft)", "Height (ft)", "Quantity")
            if c in d.columns]
    d = d[cols]

    p = Path(out_path)
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        with pd.ExcelWriter(p) as xl:
            d.to_excel(xl, sheet_name="worst_first", index=False)
            summ = pd.DataFrame({
                "metric": ["rows", "beyond +/-30%", "mean abs %", "median abs %"],
                "value": [len(d), int((d["abs_pct"] > 30).sum()),
                          round(float(d["abs_pct"].mean()), 2),
                          round(float(d["abs_pct"].median()), 2)],
            })
            summ.to_excel(xl, sheet_name="summary", index=False)
            if "largest_component" in d.columns:
                (d[d["abs_pct"] > 30].groupby("largest_component").size()
                 .rename("rows").reset_index()
                 .to_excel(xl, sheet_name="drivers", index=False))
    else:
        d.to_csv(p, index=False)
    return d
