"""
Honest out-of-fold audit of the historical archive.

    python oof_audit.py archive.csv audit.xlsx

batch_score.py on the training file is optimistic — the model memorised those rows.
This re-scores every row with a model that never saw its quote, so the outliers it
surfaces are real pricing anomalies rather than memorisation artifacts.

Output is sorted worst-first, with the component driving each gap named. That list
is the practical deliverable: work the top 50 with an estimator. If the model is
right about most of them you have a business case; if it is wrong you learn which
column is missing, which is worth more than another tenth of a point of accuracy.
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold
import tbt_model as T
from batch_score import write_xlsx

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else "audit.xlsx"

df = T.load_training(src)
X, _ = T.encode(df)
Xn, _ = T.encode(df, with_flags=False)
ci, cin = T._ci(X), T._ci(Xn)
B = T.backbone(df)
w = T.recency_weights(df)
g = df["Quote #"].values
act = df["Total Price"].values
pl = df["plausible"].values

print("  building out-of-fold predictions (grouped by Quote #)…")
unit = np.zeros(len(df))
parts = {}
direct = np.zeros(len(df))
for fold, (tr_i, te_i) in enumerate(GroupKFold(5).split(X, act, g), 1):
    tr = np.zeros(len(df), bool); tr[tr_i] = True; tr &= pl
    te = np.zeros(len(df), bool); te[te_i] = True
    regs, _, dm = T.fit_models(df, X, Xn, B, ci, cin, tr, w, with_clfs=False)
    for c in T.COMPS:
        p = np.exp(regs[c].predict(X[te], B[te])) * (df[c + "_u"].values[te] > 0)
        parts.setdefault(c, np.zeros(len(df)))[te] = p
    direct[te] = np.exp(dm.predict(X[te], B[te]))
    print(f"     fold {fold}/5 done", flush=True)

unit = sum(parts.values())
est = T.BLEND * unit + (1 - T.BLEND) * direct

out = pd.DataFrame({"est_total": np.round(est), "actual_total": act})
for c in T.COMPS:
    short = c.replace(" Price", "")
    out["est_" + short] = np.round(parts[c])
    out["act_" + short] = df[c].values
out["gap_pct"] = np.round((act - est) / est * 100, 1)
out["abs_gap"] = out["gap_pct"].abs()
dev = {c: df[c].values - parts[c] for c in T.COMPS}
worst = [max(T.COMPS, key=lambda c: abs(dev[c][i])) for i in range(len(df))]
out["driver"] = [c.replace(" Price", "") for c in worst]
out["driver_gap"] = [round(dev[worst[i]][i]) for i in range(len(df))]
out["plausible"] = pl

ape = np.abs(est - act) / act
s = pl
print(f"\n  out-of-fold median {np.median(ape[s])*100:.1f}%   mean {ape[s].mean()*100:.1f}%")
print(f"  rows beyond +/-30%: {(out['abs_gap'][s] > 30).sum()} "
      f"({(out['abs_gap'][s] > 30).mean()*100:.1f}%)")
print("\n  gap attributed to:")
print(out[s & (out['abs_gap'] > 30)]["driver"].value_counts().to_string())

keep = [c for c in ["Quote #", "Revision #", "Tank Name", "Company Name", "Status",
                    "Due Date", "State", "Country", "Material", "Use Type",
                    "Wage Type", "Diameter (ft)", "Height (ft)", "Quantity"] if c in df]
res = pd.concat([df[keep], out], axis=1).sort_values("abs_gap", ascending=False)
write_xlsx(res, dst)
print(f"\n  wrote {dst} — sorted worst-first")
