"""
TBT tank pricing model — v3.

  python tbt_model.py train  archive.csv      # fit + save
  python tbt_model.py check  archive.csv      # forward-looking validation only
  python tbt_model.py predict --diameter 32 --height 30 --material CS --state MO

Architecture
------------
Accounting identity, verified on 100% of rows:

    Total Price    = Material + Fabrication + Construction
                   + InsulMaterial + InsulConstruction + Freight      (tax-EXCLUSIVE)
    Proposal Total = the five above, ex-Freight, + Tax                (tax-INCLUSIVE)

Six component regressors, each log(component | present). Four hurdle classifiers
decide scope when the estimator hasn't. Tax is a state-rate lookup, never modelled.
Each regressor sits on a ridge log-log backbone so it can extrapolate past the
largest tank it has seen; the tree learns only the residual. Final estimate blends
70% component-sum with 30% a direct whole-price model.

Changes from v2, each validated on a forward holdout (train <2026Q1, score after)
------------------------------------------------------------------------------
1. Quantity bug fixed. The price columns are ALREADY per-tank. v2 divided them by
   Quantity, corrupting the 15% of rows with multi-tank orders — which are 38% of
   the top-5%-by-value quotes.
2. Implausible rows excluded from TRAINING, not just from scoring. 455 rows price
   below $20/sq-ft (partial quotes, missing scope lines) and were poisoning the fit.
3. Ridge log-log backbone. Trees clip at their largest leaf, so v2 priced a
   111ft x 51ft tank at $1.5M against an actual $3.1M. A power law extrapolates.
4. Size x time interaction features. Drift is size-dependent — XL tanks rose 22%
   over the period while L tanks fell 4%. One time feature cannot express that.
5. Recency weighting, 1-year half-life.

Forward-holdout result (1,595 unseen quotes):
    v2:  median 8.5%  mean 11.5%  p90 24.8%  aggregate -10.4%  top-5% -25.7%
    v3:  median 7.5%  mean 10.2%  p90 22.2%  aggregate  -5.4%  top-5% -16.8%

Known limitation: v3 still underprices the largest 5% of quotes by about 17% in
aggregate. Large tanks scale superlinearly and nothing in these columns explains
it. predict() flags this; do not use v3 unsupervised on the big end.
"""
import sys, json, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupShuffleSplit
import joblib
from pathlib import Path

HERE = Path(__file__).parent
MODEL_PATH = HERE / "tbt_pricing_model.joblib"
EPOCH = pd.Timestamp("2024-01-01")

COMPS = ["Material Price", "Fabrication Price", "Construction Price",
         "Insulation Material Price", "Insulation Construction Price", "Freight Price"]
HURDLE = ["Construction Price", "Insulation Material Price",
          "Insulation Construction Price", "Freight Price"]
FLAGS = ["f_constr", "f_insul", "f_freight"]

# Never inputs: derived from price, or pricing-policy knobs rather than tank attributes.
BANNED = COMPS + ["Total Tax", "Proposal Total", "Total Price", "Margin (%)",
                  "Contingency (%)", "Insulation Margin (%)",
                  "Insulation Contingency (%)", "Commission (%)"]

NUM = ["Diameter (ft)", "Height (ft)", "Freeboard (in)", "cap_gal", "vol_cf",
       "shell_area", "floor_area", "total_area", "aspect", "hoop", "steel_proxy",
       "log_vol", "Quantity", "mi", "Miles to Site (From TBT)",
       "Miles to Site (From GT)", "Ss", "S1", "seismic", "months",
       "t_x_area", "t_x_dia", "t_sq"]
CAT = ["Material", "Use Type", "Deck Style", "Floor Style", "Wage Type",
       "Country", "State", "Bid Type", "Sales Manager"]

RP = dict(learning_rate=0.03, max_iter=1200, min_samples_leaf=15,
          l2_regularization=1.0, random_state=0)
CP = dict(learning_rate=0.05, max_iter=400, min_samples_leaf=20, random_state=0)

BLEND = 0.70          # weight on component sum
HALFLIFE = 1.0        # years; recency weighting
PSF_LO, PSF_HI = 20, 250   # plausibility gate, $ per sq-ft of shell


# ----------------------------------------------------------------- features
def engineer(df):
    """Geometry + time features. Safe on 1 row or 10,000."""
    df = df.copy()
    if "Usable Capacity" in df:
        df["cap_gal"] = pd.to_numeric(
            df["Usable Capacity"].astype(str).str.replace("gal", "", regex=False)
            .str.replace(",", "", regex=False), errors="coerce")
    else:
        df["cap_gal"] = np.nan

    d = (pd.to_datetime(df["Due Date"], errors="coerce") if "Due Date" in df
         else pd.Series(pd.Timestamp.today(), index=df.index))
    df["months"] = (d - EPOCH).dt.days / 30.4

    D = df["Diameter (ft)"].astype(float)
    H = df["Height (ft)"].astype(float)
    df["vol_cf"] = np.pi * (D / 2) ** 2 * H
    df["shell_area"] = np.pi * D * H                    # plate area -> fab hours
    df["floor_area"] = np.pi * (D / 2) ** 2
    df["total_area"] = df["shell_area"] + 2.02 * df["floor_area"]
    df["aspect"] = H / D.replace(0, np.nan)
    df["hoop"] = D * H                                  # hoop stress -> plate gauge
    df["steel_proxy"] = df["shell_area"] * np.sqrt(df["hoop"])   # area x thickness
    df["log_vol"] = np.log1p(df["vol_cf"])
    df["seismic"] = df.get("Ss", np.nan) * df.get("S1", np.nan)

    # Drift is size-dependent, so time must be allowed to interact with size.
    df["t_x_area"] = df["months"] * np.log1p(df["shell_area"])
    df["t_x_dia"] = df["months"] * D
    df["t_sq"] = df["months"] ** 2

    mc = [c for c in ["Miles to Site (From TBT)", "Miles to Site (From GT)"] if c in df]
    df["mi"] = df[mc].min(axis=1) if mc else np.nan
    for c in NUM + CAT + FLAGS:
        if c not in df:
            df[c] = np.nan
    return df


def backbone(df):
    """log D and log H span all the geometry (shell_area etc. are products of
    powers of them), so this basis is identified — no collinearity. A power law
    in these extrapolates; a tree does not."""
    lD = np.log(df["Diameter (ft)"].astype(float).clip(lower=0.1).values)
    lH = np.log(df["Height (ft)"].astype(float).clip(lower=0.1).values)
    t = df["months"].values / 12.0
    return np.column_stack([lD, lH, lD * lH, t, np.ones(len(df))])


def load_training(path):
    df = pd.read_csv(path, low_memory=False)
    df = engineer(df)
    n0 = len(df)
    df = df[(df["Diameter (ft)"] > 0) & (df["Height (ft)"] > 0) & (df["Total Price"] > 0)]
    df = df[df["Status"] != "Unfinished"]

    # Prices are ALREADY per-tank. Do not divide by Quantity.
    for c in COMPS:
        df[c + "_u"] = df[c]
    df["total_u"] = df["Total Price"]
    df["f_constr"] = (df["Construction Price"] > 0).astype(int)
    df["f_insul"] = (df["Insulation Material Price"] > 0).astype(int)
    df["f_freight"] = (df["Freight Price"] > 0).astype(int)

    df["psf"] = df["Total Price"] / df["shell_area"]
    df["plausible"] = df["psf"].between(PSF_LO, PSF_HI)

    ident = (df[[c + "_u" for c in COMPS]].sum(axis=1) - df["total_u"]).abs()
    print(f"  {n0} rows -> {len(df)} usable "
          f"({(~df['plausible']).sum()} implausible, excluded from fitting)")
    print(f"  accounting identity holds on {(ident < .05).mean()*100:.1f}%")
    return df.reset_index(drop=True)


def encode(df, enc=None, with_flags=True):
    """with_flags=False is the matrix the presence classifiers see. They must not
    be shown the flags they are predicting, or a blank flag becomes self-fulfilling."""
    X = df[(NUM + FLAGS if with_flags else NUM) + CAT].copy()
    if enc is None:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                             encoded_missing_value=-1)
        X[CAT] = enc.fit_transform(X[CAT].astype(str))
    else:
        X[CAT] = enc.transform(X[CAT].astype(str))
    return X, enc


def _ci(X):
    return [X.columns.get_loc(c) for c in CAT]


def recency_weights(df, halflife=HALFLIFE, ref=None):
    ref = ref or pd.to_datetime(df["Due Date"], errors="coerce").max()
    age = (ref - pd.to_datetime(df["Due Date"], errors="coerce")).dt.days / 365.25
    return np.power(0.5, age.clip(lower=0).fillna(1.0) / halflife).values


def tax_table(df):
    s = df[df["Total Tax"] > 0].copy()
    s["rate"] = s["Total Tax"] / (s["Proposal Total"] - s["Total Tax"]).clip(lower=1)
    t = s.groupby("State")["rate"].agg(["median", "size"])
    return {k: float(v) for k, v in t[t["size"] >= 10]["median"].items()}


# ------------------------------------------------------------------ fitting
class Stage:
    """Ridge log-log backbone + gradient-boosted residual."""

    def __init__(self, ci):
        self.ci = ci

    def fit(self, X, B, y, w=None):
        self.lin = Ridge(alpha=1.0).fit(B, y)
        self.gbm = HistGradientBoostingRegressor(categorical_features=self.ci, **RP)
        self.gbm.fit(X, y - self.lin.predict(B), sample_weight=w)
        return self

    def predict(self, X, B):
        return self.lin.predict(B) + self.gbm.predict(X)


def fit_models(df, X, Xn, B, ci, cin, idx, w, with_clfs=True):
    """Fit every component, classifier and the direct model on rows `idx`.
    with_clfs=False during validation, where true scope is known and the
    classifiers would be 40% of the compute for nothing."""
    regs, clfs = {}, {}
    for c in COMPS:
        a = df[c + "_u"].values
        m = idx & (a > 0)
        regs[c] = Stage(ci).fit(X[m], B[m], np.log(a[m]), w[m])
    if with_clfs:
        for c in HURDLE:
            clfs[c] = HistGradientBoostingClassifier(categorical_features=cin, **CP)\
                .fit(Xn[idx], (df[c + "_u"].values[idx] > 0), sample_weight=w[idx])
    direct = Stage(ci).fit(X[idx], B[idx], np.log(df["total_u"].values[idx]), w[idx])
    return regs, clfs, direct


def combine(regs, direct, df, X, B, te):
    comp = np.zeros(int(te.sum()))
    for c in COMPS:
        p = np.exp(regs[c].predict(X[te], B[te]))
        comp += p * (df[c + "_u"].values[te] > 0)
    return BLEND * comp + (1 - BLEND) * np.exp(direct.predict(X[te], B[te]))


# ---------------------------------------------------------------- validate
def validate(df, n_folds=6):
    """Rolling origin. The only honest measure — a random split leaks revisions,
    and a grouped split still lets the model see the future."""
    X, _ = encode(df); Xn, _ = encode(df, with_flags=False)
    ci, cin = _ci(X), _ci(Xn)
    B = backbone(df)
    w = recency_weights(df)
    act = df["Total Price"].values
    pl = df["plausible"].values
    q = pd.to_datetime(df["Due Date"], errors="coerce").dt.to_period("Q")
    qs = sorted(q.dropna().unique())

    print(f"\n  {'quarter':<9}{'n':>6}{'median':>9}{'mean':>8}{'p90':>8}{'$agg':>9}{'$top5%':>9}",
          flush=True)
    rows = []
    for i in range(max(6, len(qs) - n_folds), len(qs)):
        tr = (q < qs[i]).values & pl
        te = (q == qs[i]).values
        if te.sum() < 50:
            continue
        regs, _, direct = fit_models(df, X, Xn, B, ci, cin, tr, w, with_clfs=False)
        est = combine(regs, direct, df, X, B, te)
        s = pl[te]; e, a = est[s], act[te][s]
        ape = np.abs(e - a) / a
        xl = a >= np.percentile(a, 95)
        rows.append((np.median(ape) * 100, ape.mean() * 100,
                     np.percentile(ape, 90) * 100,
                     (e.sum() / a.sum() - 1) * 100,
                     (e[xl].sum() / a[xl].sum() - 1) * 100))
        print(f"  {str(qs[i]):<9}{int(s.sum()):>6}{rows[-1][0]:>8.1f}%{rows[-1][1]:>7.1f}%"
              f"{rows[-1][2]:>7.1f}%{rows[-1][3]:>8.1f}%{rows[-1][4]:>8.1f}%", flush=True)
    r = np.array(rows)
    print(f"  {'mean':<9}{'':>6}{r[:,0].mean():>8.1f}%{r[:,1].mean():>7.1f}%"
          f"{r[:,2].mean():>7.1f}%{r[:,3].mean():>8.1f}%{r[:,4].mean():>8.1f}%")
    print(f"  {'last 3':<9}{'':>6}{r[-3:,0].mean():>8.1f}%{r[-3:,1].mean():>7.1f}%"
          f"{r[-3:,2].mean():>7.1f}%{r[-3:,3].mean():>8.1f}%{r[-3:,4].mean():>8.1f}%")
    return r


# ------------------------------------------------------------------- train
def train(path, skip_validate=False):
    print("Loading…")
    df = load_training(path)
    if not skip_validate:
        validate(df)

    X, enc = encode(df); Xn, encn = encode(df, with_flags=False)
    ci, cin = _ci(X), _ci(Xn)
    B = backbone(df)
    w = recency_weights(df)
    pl = df["plausible"].values
    g = df["Quote #"].values
    act = df["Total Price"].values

    # Conformal calibration on held-out whole quotes.
    print("\n  calibrating prediction bands…")
    tr_i, cal_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=0)
                       .split(X, act, g))
    tr = np.zeros(len(df), bool); tr[tr_i] = True; tr &= pl
    cal = np.zeros(len(df), bool); cal[cal_i] = True; cal &= pl
    regs, _, direct = fit_models(df, X, Xn, B, ci, cin, tr, w)
    pred = combine(regs, direct, df, X, B, cal)
    resid = np.abs(np.log(act[cal]) - np.log(pred))
    bands = {int((1 - a) * 100): float(np.exp(np.quantile(resid, 1 - a)))
             for a in (0.20, 0.10)}
    print(f"     80% band = /x {bands[80]:.2f}      90% band = /x {bands[90]:.2f}")

    print("\nFitting final models on all plausible rows…")
    regs, clfs, direct = fit_models(df, X, Xn, B, ci, cin, pl, w)

    # Guard rails: refuse to extrapolate silently.
    lim = {c: (float(df.loc[pl, c].quantile(.005)), float(df.loc[pl, c].quantile(.995)))
           for c in ["Diameter (ft)", "Height (ft)", "shell_area"]}
    lim["shell_area_max"] = float(df.loc[pl, "shell_area"].max())
    big = float(df.loc[pl, "Total Price"].quantile(.95))

    joblib.dump({"regs": regs, "clfs": clfs, "direct": direct,
                 "encoder": enc, "encoder_nf": encn, "bands": bands,
                 "tax": tax_table(df), "blend": BLEND, "limits": lim,
                 "big_threshold": big, "rows": int(pl.sum()),
                 "trained": str(pd.Timestamp.today().date()),
                 "data_through": str(pd.to_datetime(df["Due Date"],
                                                    errors="coerce").max().date())},
                MODEL_PATH)
    print(f"Saved -> {MODEL_PATH}")


# ----------------------------------------------------------------- predict
_B = None

ALIAS = {"diameter": "Diameter (ft)", "height": "Height (ft)",
         "freeboard": "Freeboard (in)", "material": "Material",
         "use_type": "Use Type", "deck": "Deck Style", "floor": "Floor Style",
         "wage": "Wage Type", "country": "Country", "state": "State",
         "bid_type": "Bid Type", "sales_manager": "Sales Manager",
         "miles": "Miles to Site (From TBT)", "quantity": "Quantity",
         "ss": "Ss", "s1": "S1", "due_date": "Due Date"}


def predict(has_construction=None, has_insulation=None, has_freight=None, **kw):
    """Estimate one tank.

    Supplying the three scope flags is worth ~2 points of accuracy — pass them
    whenever the estimator knows the scope. Returns a per-tank price plus the
    extended total, a component breakdown, conformal bands and any warnings.
    """
    global _B
    if _B is None:
        _B = joblib.load(MODEL_PATH)
    b = _B

    row = {ALIAS.get(k, k): v for k, v in kw.items()}
    row.setdefault("Quantity", 1)
    row.setdefault("Country", "US")
    row.setdefault("Bid Type", "Firm")
    row.setdefault("Due Date", pd.Timestamp.today().strftime("%m/%d/%Y"))
    d0 = engineer(pd.DataFrame([row]))

    # Stage 1 — settle scope on a matrix that contains no flags.
    scope = {"Construction Price": has_construction,
             "Insulation Material Price": has_insulation,
             "Insulation Construction Price": has_insulation,
             "Freight Price": has_freight}
    Xn, _ = encode(d0, b["encoder_nf"], with_flags=False)
    inferred, present = {}, {}
    for c in HURDLE:
        if scope.get(c) is not None:
            present[c] = bool(scope[c])
        else:
            # Hard threshold, not P x V. Expected value averages across scopes and
            # returns half an insulation package, which matches no real tank.
            p = float(b["clfs"][c].predict_proba(Xn)[0, 1])
            inferred[c] = round(p, 2)
            present[c] = p > 0.5

    # Stage 2 — price with the scope resolved.
    row["f_constr"] = int(present["Construction Price"])
    row["f_insul"] = int(present["Insulation Material Price"])
    row["f_freight"] = int(present["Freight Price"])
    d = engineer(pd.DataFrame([row]))
    X, _ = encode(d, b["encoder"])
    B = backbone(d)

    parts, total = {}, 0.0
    for c in COMPS:
        v = float(np.exp(b["regs"][c].predict(X, B)[0]))
        if c in HURDLE:
            v *= float(present[c])
        parts[c] = round(v, 0)
        total += v
    direct = float(np.exp(b["direct"].predict(X, B)[0]))
    est = b["blend"] * total + (1 - b["blend"]) * direct

    qty = float(row.get("Quantity", 1) or 1)
    rate = b["tax"].get(str(row.get("State", "")), 0.0)
    ex_frt = total - parts["Freight Price"]

    warn = []
    lo, hi = b["limits"]["Diameter (ft)"]
    if not lo <= float(row["Diameter (ft)"]) <= hi:
        warn.append(f"diameter outside the range the model was fitted on ({lo:.0f}-{hi:.0f} ft)")
    lo, hi = b["limits"]["Height (ft)"]
    if not lo <= float(row["Height (ft)"]) <= hi:
        warn.append(f"height outside the range the model was fitted on ({lo:.0f}-{hi:.0f} ft)")
    if float(d["shell_area"].iloc[0]) > b["limits"]["shell_area_max"]:
        warn.append("tank is larger than anything in the training data — extrapolating")
    if est >= b["big_threshold"]:
        warn.append("top-5% by value: the model underprices this band by roughly "
                    "17% in aggregate — treat as a floor, not an estimate")
    psf = est / float(d["shell_area"].iloc[0])
    if not PSF_LO <= psf <= PSF_HI:
        warn.append(f"implied ${psf:.0f}/sq-ft is outside the plausible range")

    return {"unit_price": round(est, 0), "total": round(est * qty, 0),
            "p80_low": round(est * qty / b["bands"][80], 0),
            "p80_high": round(est * qty * b["bands"][80], 0),
            "p90_low": round(est * qty / b["bands"][90], 0),
            "p90_high": round(est * qty * b["bands"][90], 0),
            "components": {k: round(v, 0) for k, v in parts.items()},
            "est_tax": round(ex_frt * rate, 0),
            "proposal_total": round(ex_frt * (1 + rate), 0),
            "psf": round(psf, 1),
            "direct_model_says": round(direct, 0),
            "component_sum_says": round(total, 0),
            "scope_inferred": inferred,
            "warnings": warn,
            "model_trained": b["trained"], "data_through": b["data_through"]}


if __name__ == "__main__":
    # Route through the imported module so that classes defined here pickle as
    # tbt_model.Stage rather than __main__.Stage, which would fail to unpickle
    # from any process that imports this file instead of running it.
    import tbt_model as _M

    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        _M.train(sys.argv[2], skip_validate="--fast" in sys.argv)
    elif cmd == "check":
        _M.validate(_M.load_training(sys.argv[2]))
    elif cmd == "predict":
        ap = argparse.ArgumentParser()
        for f in ["diameter", "height", "freeboard", "miles", "ss", "s1"]:
            ap.add_argument(f"--{f}", type=float)
        for f in ["material", "use-type", "deck", "floor", "wage", "country",
                  "state", "bid-type", "sales-manager", "due-date"]:
            ap.add_argument(f"--{f}")
        ap.add_argument("--quantity", type=int, default=1)
        for f in ["construction", "insulation", "freight"]:
            ap.add_argument(f"--{f}", choices=["yes", "no"])
        a = vars(ap.parse_args(sys.argv[2:]))
        sc = {f"has_{f}": (None if a.pop(f) is None else a_ == "yes")
              for f in ["construction", "insulation", "freight"]
              for a_ in [a.get(f)]} if False else {}
        for f in ["construction", "insulation", "freight"]:
            v = a.pop(f, None)
            sc[f"has_{f}"] = None if v is None else (v == "yes")
        kw = {k.replace("-", "_"): v for k, v in a.items() if v is not None}
        print(json.dumps(_M.predict(**sc, **kw), indent=2))
