"""
Score a CSV of quotes and write an xlsx with estimates, bands, component
breakdown and an OK/LOW/HIGH flag.

    python batch_score.py quotes.csv scored.xlsx

Use this instead of dragging a UDF down thousands of rows — same model, seconds
instead of an hour, and it will not re-fire on every recalculation.

If the input has the component price columns (i.e. it is the archive), true scope
is used. Otherwise scope is inferred by the classifiers.

Scoring the training archive with this will look far better than reality, because
the model has seen those rows. For an honest read use oof_audit.py.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import joblib
import tbt_model as T


def score(csv_path):
    b = joblib.load(T.MODEL_PATH)
    df = pd.read_csv(csv_path, low_memory=False)
    df = T.engineer(df)
    df = df[(df["Diameter (ft)"] > 0) & (df["Height (ft)"] > 0)].copy().reset_index(drop=True)
    qty = df["Quantity"].clip(lower=1).values if "Quantity" in df else np.ones(len(df))

    # Stage 1 — scope
    have = all(c in df for c in T.COMPS)
    Xn, _ = T.encode(df, b["encoder_nf"], with_flags=False)
    present = {}
    for c in T.HURDLE:
        present[c] = ((df[c] > 0).values if have
                      else b["clfs"][c].predict_proba(Xn)[:, 1] > 0.5)
    df["f_constr"] = present["Construction Price"].astype(int)
    df["f_insul"] = present["Insulation Material Price"].astype(int)
    df["f_freight"] = present["Freight Price"].astype(int)

    # Stage 2 — price
    X, _ = T.encode(df, b["encoder"])
    B = T.backbone(df)
    out = pd.DataFrame(index=df.index)
    unit = np.zeros(len(df))
    for c in T.COMPS:
        v = np.exp(b["regs"][c].predict(X, B))
        if c in T.HURDLE:
            v = v * present[c]
        out["est_" + c.replace(" Price", "")] = np.round(v)
        unit += v
    direct = np.exp(b["direct"].predict(X, B))
    est_unit = b["blend"] * unit + (1 - b["blend"]) * direct

    out["est_per_tank"] = np.round(est_unit)
    out["est_total"] = np.round(est_unit * qty)
    out["est_low80"] = np.round(est_unit * qty / b["bands"][80])
    out["est_high80"] = np.round(est_unit * qty * b["bands"][80])
    out["implied_psf"] = np.round(est_unit / df["shell_area"].values, 1)
    rate = df["State"].astype(str).map(b["tax"]).fillna(0.0).values
    ex_frt = unit - np.exp(b["regs"]["Freight Price"].predict(X, B)) * present["Freight Price"]
    out["est_tax"] = np.round(ex_frt * rate)
    out["est_proposal_total"] = np.round(ex_frt * (1 + rate))

    big = est_unit * qty >= b["big_threshold"]
    oversize = df["shell_area"].values > b["limits"]["shell_area_max"]
    out["caution"] = np.where(oversize, "OVERSIZE — extrapolating",
                     np.where(big, "LARGE — model reads ~17% low", ""))

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
                        "Wage Type", "Diameter (ft)", "Height (ft)", "Quantity"] if c in df]
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
    res = score(sys.argv[1])
    dst = sys.argv[2] if len(sys.argv) > 2 else "scored.xlsx"
    write_xlsx(res, dst) if dst.endswith(".xlsx") else res.to_csv(dst, index=False)
    print(f"  wrote {dst}")
