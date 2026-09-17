"""Derive the five scope columns from a raw archive export.

    python prepare_data.py archive.csv archive_prepared.csv

Adds, by reading which components carry pricing:

    IS_CONSTRUCTION         Construction Price > 0
    IS_INSULATION           Insulation Material Price > 0
    IS_INSULATION_ERECTION  Insulation Construction Price > 0
    IS_FREIGHT              Freight Price > 0
    IS_TAXABLE              Total Tax > 0

This is a one-time backfill for history and the wrong thing to rely on going
forward. A blank price and a genuinely excluded scope look identical after the
fact: a quote where freight had not been filled in yet is indistinguishable from
one where the customer collects, and only the estimator knows which it was. Five
Yes/No dropdowns at quote time are the fix, and they are the same five answers
the model needs anyway.

Material and Fabrication get no column. They are present on every row of the
archive — processes, not options.

Existing scope columns are left alone; pass --force to rebuild from pricing.
"""
import argparse
import sys

import numpy as np
import pandas as pd

DERIVE = {
    "IS_CONSTRUCTION": "Construction Price",
    "IS_INSULATION": "Insulation Material Price",
    "IS_INSULATION_ERECTION": "Insulation Construction Price",
    "IS_FREIGHT": "Freight Price",
    "IS_TAXABLE": "Total Tax",
}


def prepare(src, dst, force=False, as_text=True):
    df = pd.read_csv(src, low_memory=False)
    print(f"  read {len(df)} rows from {src}")

    missing = [c for c in DERIVE.values() if c not in df.columns]
    if missing:
        sys.exit(f"  ERROR: source is missing {', '.join(missing)} — "
                 "cannot derive scope from it.")

    for flag, col in DERIVE.items():
        if flag in df.columns and not force:
            print(f"  {flag:<24} already present, left alone (--force to rebuild)")
            continue
        v = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
        df[flag] = np.where(v, "Yes", "No") if as_text else v.astype(int)
        print(f"  {flag:<24} Yes on {v.sum():>5} rows ({v.mean() * 100:5.1f}%)")

    yes = lambda s: df[s].astype(str).str.upper().isin(["YES", "1", "TRUE"])
    bad = yes("IS_INSULATION_ERECTION") & ~yes("IS_INSULATION")
    if bad.any():
        print(f"\n  WARNING: {bad.sum()} rows have insulation erection without "
              "insulation supply. Review these — they are almost certainly data "
              "errors, and the model will refuse them.")
        print("  " + ", ".join(str(q) for q in df.loc[bad, "Quote #"].head(10)))

    for c in ["Material Price", "Fabrication Price"]:
        if c in df.columns:
            z = int((pd.to_numeric(df[c], errors="coerce").fillna(0) <= 0).sum())
            if z:
                print(f"\n  WARNING: {z} rows have no {c}. It is a process, not an "
                      "option — these are incomplete quotes and will be dropped "
                      "at training time.")

    df.to_csv(dst, index=False)
    print(f"\n  wrote {dst}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default=None)
    ap.add_argument("--force", action="store_true",
                    help="rebuild scope columns even if they already exist")
    ap.add_argument("--numeric", action="store_true", help="write 1/0 not Yes/No")
    a = ap.parse_args()
    prepare(a.src, a.dst or a.src.replace(".csv", "_prepared.csv"),
            force=a.force, as_text=not a.numeric)
