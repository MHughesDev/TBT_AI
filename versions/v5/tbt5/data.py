"""Loading, cleaning, scope validation and sample weighting."""
import numpy as np
import pandas as pd

from . import features
from .config import (
    COMPONENTS, OPTIONAL, SCOPE, TARGET, PSF_LO, PSF_HI,
    GROUP_COL, DATE_COL, HALFLIFE_YEARS,
)


class ScopeError(ValueError):
    """Scope is missing or internally inconsistent.

    Deliberately loud. Guessing scope is the failure mode v4 was built to remove
    and v5 keeps removed: a confidently wrong scope silently changes the price
    with nothing visible on the sheet to show it.
    """


def check_scope(df):
    missing = [c for c in SCOPE if c not in df.columns]
    if missing:
        raise ScopeError(
            f"missing scope columns: {', '.join(missing)}. Run prepare_data.py on "
            "the archive, or supply them per quote. v5 does not infer scope.")
    blank = df[SCOPE].isna().any(axis=1)
    if blank.any():
        raise ScopeError(f"{int(blank.sum())} rows have a blank scope value. Every "
                         "quote needs an explicit yes/no on each of the five.")
    viol = (df["IS_INSULATION_ERECTION"] == 1) & (df["IS_INSULATION"] == 0)
    if viol.any():
        raise ScopeError(f"{int(viol.sum())} rows set IS_INSULATION_ERECTION without "
                         "IS_INSULATION. We cannot install insulation we are not "
                         "supplying.")


def load(source, verbose=True):
    """Read a prepared archive and return it engineered, filtered and flagged.

    Accepts a path or an already-read DataFrame. The DataFrame form exists so the
    head-to-head bench can hand every version byte-identical rows — a comparison
    where the versions quietly disagree about which rows are usable is not a
    comparison.
    """
    df = (source.copy() if isinstance(source, pd.DataFrame)
          else pd.read_csv(source, low_memory=False))
    n0 = len(df)

    for c in COMPONENTS + [TARGET, "Total Tax", "Proposal Total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = features.build(df)
    df = df[(df["Diameter (ft)"] > 0) & (df["Height (ft)"] > 0) & (df[TARGET] > 0)]
    if "Status" in df.columns:
        df = df[df["Status"] != "Unfinished"]
    check_scope(df)

    # Price columns are ALREADY per-tank. Do not divide by Quantity. v1 and v2
    # both did and it corrupted the 15% of rows with multi-tank orders, which are
    # 38% of the top-5%-by-value quotes. The tell is that Total Price / Quantity
    # / shell_area falls as exactly 1/Quantity across order sizes while the
    # undivided version stays flat.
    for comp, flag in OPTIONAL.items():
        if comp in df.columns:
            mism = (df[comp].fillna(0) > 0) != (df[flag] == 1)
            if mism.any():
                raise ScopeError(
                    f"{int(mism.sum())} rows where {flag} disagrees with {comp}. "
                    "Re-run prepare_data.py.")

    df["psf"] = df[TARGET] / df["shell_area"]
    df["plausible"] = df["psf"].between(PSF_LO, PSF_HI)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    df = df.sort_values(DATE_COL, kind="stable").reset_index(drop=True)
    if verbose:
        print(f"  {n0} rows -> {len(df)} usable "
              f"({int((~df['plausible']).sum())} implausible, excluded from fitting)")
        if GROUP_COL in df.columns:
            print(f"  {df[GROUP_COL].nunique()} distinct quotes across {len(df)} rows")
        print("  scope: " + ", ".join(
            f"{f.replace('IS_', '')} {df[f].mean() * 100:.0f}%" for f in SCOPE))
    return df


def weights(df, ref=None, halflife=HALFLIFE_YEARS, dedup=True):
    """Sample weights: recency decay, optionally divided across quote revisions.

    The dedup half is new in v5 and is not cosmetic. 6,892 archive rows are only
    3,018 quotes, because a revised quote is stored as another row. A quote
    revised eight times therefore contributes eight times the training weight of
    a quote accepted first time — and revision count correlates with how
    contested and unusual a job is. The fit tilts toward exactly the rows whose
    pricing is least representative.

    v4 treats the revision structure purely as a leakage hazard to be split
    around. It is also a weighting bug, and this is the fix: each *quote* gets
    one unit of weight, shared among its revisions.
    """
    d = pd.to_datetime(df[DATE_COL], errors="coerce")
    ref = ref if ref is not None else d.max()
    age = (ref - d).dt.days / 365.25
    w = np.power(0.5, age.clip(lower=0).fillna(1.0) / halflife).values

    if dedup and GROUP_COL in df.columns:
        n = df.groupby(GROUP_COL)[GROUP_COL].transform("size").values.astype(float)
        w = w / np.maximum(n, 1.0)

    # Normalise so that weight scale never interacts with regularisation strength.
    return w / w.mean()


def tax_table(df, min_rows=10):
    """Effective sales-tax rate by state, from rows where tax was actually charged.

    The rate is geography. Whether it applies at all is IS_TAXABLE, which is the
    customer's exemption status and is not derivable from the state — 0% of
    Oregon quotes are taxed against 91% of New Jersey ones.
    """
    if "Total Tax" not in df or "Proposal Total" not in df:
        return {}
    s = df[df["Total Tax"] > 0].copy()
    if s.empty:
        return {}
    s["rate"] = s["Total Tax"] / (s["Proposal Total"] - s["Total Tax"]).clip(lower=1)
    t = s.groupby("State")["rate"].agg(["median", "size"])
    return {str(k): float(v) for k, v in t[t["size"] >= min_rows]["median"].items()}


def quarters(df):
    return pd.to_datetime(df[DATE_COL], errors="coerce").dt.to_period("Q")
