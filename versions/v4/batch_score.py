"""
Score a CSV of quotes and write an xlsx with estimates, bands, component
breakdown and an OK/LOW/HIGH flag.

    python batch_score.py quotes_prepared.csv scored.xlsx

The input must carry the five scope columns. Run prepare_data.py first if it is
an archive export; if it is a live quote list, the scope columns must come from
the estimator. This script will refuse rather than assume.

Scoring the training archive with this looks far better than reality — the model
has seen those rows. For an honest read use oof_audit.py.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import joblib
import tbt_model as T


def score(csv_path):
    b = joblib.load(T.MODEL_PATH)
    df = pd.read_csv(csv_path, low_memory=False)
    for c in T.SCOPE:
        if c in df.columns:
            df[c] = T._as01(df[c])
    df = T.engineer(df)
    df = df[(df["Diameter (ft)"] > 0) & (df["Height (ft)"] > 0)].copy().reset_index(drop=True)
    T.check_scope(df)
    qty = df["Quantity"].clip(lower=1).values if "Quantity" in df else np.ones(len(df))

    X, _ = T.encode(df, b["encoder"])
    B = T.backbone(df)

    present = {"Material Price": np.ones(len(df), bool),
               "Fabrication Price": np.ones(len(df), bool)}
    for comp, flag in T.OPTIONAL.items():
        present[comp] = df[flag].values == 1

    out = pd.DataFrame(index=df.index)
    unit = np.zeros(len(df))
    for c in T.COMPS:
        v = np.exp(b["regs"][c].predict(X, B)) * present[c]
        out["est_" + c.replace(" Price", "")] = np.round(v)
        unit += v
    direct = np.exp(b["direct"].predict(X, B))
    est_unit = b["blend"] * unit + (1 - b["blend"]) * direct

    out["est_per_tank"] = np.round(est_unit)
    out["est_total"] = np.round(est_unit * qty)
    out["est_low80"] = np.round(est_unit * qty / b["bands"][80])
    out["est_high80"] = np.round(est_unit * qty * b["bands"][80])
    out["implied_psf"] = np.round(est_unit / df["shell_area"].values, 1)

    rate = df["State"].astype(str).map(b["tax"]).fillna(0.0).values * (df["IS_TAXABLE"].values == 1)
    ex_frt = unit - np.exp(b["regs"]["Freight Price"].predict(X, B)) * present["Freight Price"]
    out["tax_rate"] = np.round(rate, 4)
    out["est_tax"] = np.round(ex_frt * rate)
    out["est_proposal_total"] = np.round(ex_frt * (1 + rate))

    big = est_unit * qty >= b["big_threshold"]
    over = df["shell_area"].values > b["limits"]["shell_area_max"]
    out["caution"] = np.where(over, "OVERSIZE — extrapolating",
                     np.where(big, "LARGE — model reads ~13-17% low", ""))

    if "Total Price" in df:
        act = df["Total Price"].values
        out["actual_total"] = act
        with np.errstate(divide="ignore", invalid="ignore"):
            out["gap_pct"] = np.round((act - out["est_total"]) / out["est_total"] * 100, 1)
        out["flag"] = np.where(act < out["est_low80"], "LOW",
                      np.where(act > out["est_high80"], "HIGH", "OK"))
        ape = np.abs(out["est_total"] - act) / np.where(act > 0, act, np.nan)
        print(f"  scored {len(df)} rows   median {np.nanmedian(ape)*100:.1f}%   "
              f"flagged {(out['flag'] != 'OK').sum()} ({(out['flag'] != 'OK').mean()*100:.0f}%)")

    keep = [c for c in ["Quote #", "Revision #", "Tank Name", "Company Name", "Status",
                        "Due Date", "State", "Country", "Material", "Use Type",
                        "Wage Type", "Diameter (ft)", "Height (ft)", "Quantity"]
            + T.SCOPE if c in df]
    return pd.concat([df[keep], out], axis=1)


def write_xlsx(res, dst):
    with pd.ExcelWriter(dst, engine="openpyxl") as w:
        res.to_excel(w, index=False, sheet_name="scored")
        ws = w.sheets["scored"]
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col[:200])
            ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 34)


if __name__ == "__main__":
    try:
        res = score(sys.argv[1])
    except T.ScopeError as e:
        sys.exit(f"SCOPE ERROR: {e}")
    dst = sys.argv[2] if len(sys.argv) > 2 else "scored.xlsx"
    write_xlsx(res, dst) if dst.endswith(".xlsx") else res.to_csv(dst, index=False)
    print(f"  wrote {dst}")
