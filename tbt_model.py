"""
TBT tank pricing model — v4.

    python prepare_data.py archive.csv archive_prepared.csv   # adds the IS_ columns
    python tbt_model.py train archive_prepared.csv --fast
    python tbt_model.py predict --diameter 32 --height 30 --material CS --state MO \
           --construction yes --insulation no --freight yes --taxable yes

Design principle
----------------
**The model prices scope. It does not guess scope.**

Whether a quote includes erection, insulation or freight is a commercial decision
the estimator knows at quote time. It is not a property of the tank and it is not
inferable from the specs. v3 tried to infer it with classifiers; that was wrong in
principle — a classifier that is confidently wrong silently changes the number
without changing anything visible on the sheet.

v4 deletes those classifiers. Scope arrives as explicit yes/no columns, derived
once from the archive by prepare_data.py and supplied by the estimator thereafter.
predict() raises if they are missing rather than assuming a default.

Scope columns
-------------
    IS_CONSTRUCTION         do we erect it                     (68.4% of archive)
    IS_INSULATION           is insulation supplied             (23.0%)
    IS_INSULATION_ERECTION  do we install the insulation       (21.8%)
    IS_FREIGHT              do we ship it                      (82.2%)
    IS_TAXABLE              does sales tax apply to this customer

Material and Fabrication carry no flag. They are present on 6,679 of 6,679 rows —
they are processes, not options.

IS_INSULATION_ERECTION implies IS_INSULATION. The archive has 1,456 rows with
both, 83 with material only (customer installs), and zero with erection alone.
The rule is enforced.

IS_TAXABLE is separate from state. Geography sets the rate (near-constant within a
state, median std 0.008); the customer's exemption status sets whether it applies
at all. Taxed share runs 0% in Oregon to 91% in New Jersey.

Accounting identity, verified on 100% of rows
---------------------------------------------
    Total Price    = Material + Fabrication + Construction
                   + InsulMaterial + InsulConstruction + Freight   (tax-EXCLUSIVE)
    Proposal Total = the five above, ex-Freight, + Tax             (tax-INCLUSIVE)

Architecture
------------
Six component regressors, each log(component) over rows where that component is
present. Each sits on a ridge log-log backbone so it can extrapolate past the
largest tank seen; the tree learns only the residual. Final estimate blends 70%
component-sum with 30% a direct whole-price model. Conformal bands from a grouped
holdout. Tax is a state-rate lookup gated by IS_TAXABLE.

No classifiers. One encoder. Single-stage prediction.
"""
import sys, json, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupShuffleSplit
import joblib
from pathlib import Path

HERE = Path(__file__).parent
MODEL_PATH = HERE / "tbt_pricing_bundle.joblib"
VERSION = "v4.2"
EPOCH = pd.Timestamp("2024-01-01")

# Always present — processes, not options. No scope flag.
ALWAYS = ["Material Price", "Fabrication Price"]
# Optional — each governed by exactly one scope flag.
OPTIONAL = {
    "Construction Price": "IS_CONSTRUCTION",
    "Insulation Material Price": "IS_INSULATION",
    "Insulation Construction Price": "IS_INSULATION_ERECTION",
    "Freight Price": "IS_FREIGHT",
}
COMPS = ALWAYS + list(OPTIONAL)
SCOPE = list(dict.fromkeys(OPTIONAL.values())) + ["IS_TAXABLE"]

# Never inputs: arithmetic identities, or pricing-policy knobs rather than tank
# attributes. Feeding any of these in produces a model that looks perfect and
# can only score a quote that has already been built.
BANNED = COMPS + ["Total Tax", "Proposal Total", "Total Price", "Margin (%)",
                  "Contingency (%)", "Insulation Margin (%)",
                  "Insulation Contingency (%)", "Commission (%)"]

# Free-text mining of Tank Name. Waste-water and silo tanks are engineered-to-order
# and the process name is the best available proxy for appendages we cannot see: a
# digester has mixers, covers and gas handling; an equalization basin is a bare
# shell. Median $/sq-ft runs 1.35x base for digesters against 0.68x for multi-zone
# configurations — a 2x spread that no other column captures.
TN_GROUPS = {
    "tn_digester":  ["digester", "anaerobic", "aerobic", "biogas"],
    "tn_bio":       ["mbbr", "sbr", "cmas", "biomass", "bioreactor", "activated", "moving bed"],
    "tn_sludge":    ["sludge", "slurry", "thickener", "thickening", "biosolid"],
    "tn_reactor":   ["reactor", "reaeration", "oxidation", "contact"],
    "tn_clarifier": ["clarifier", "settling", "sedimentation", "launder"],
    "tn_eq":        ["equalization", "equalisation", " eq ", "flow equal", "detention", "retention"],
    "tn_aeration":  ["aeration", "aerated", "diffuser", "blower"],
    "tn_leachate":  ["leachate", "landfill"],
    "tn_fire":      ["fire", "fp tank", "nfpa", "sprinkler"],
    "tn_potable":   ["potable", "drinking", "finished water"],
    "tn_process":   ["process", "chemical", "acid", "caustic", "brine", "glycol", "oil", "fuel", "diesel"],
    "tn_silo":      ["silo", "lime", "carbon", "bulk"],
    "tn_backwash":  ["backwash", "wash water", "washwater", "rinse"],
    "tn_coating":   ["epoxy", "coated", "coating", "lined", "liner", "glass", "galvan"],
    "tn_dome":      ["dome", "geodesic", "cover", "membrane"],
    "tn_rafter":    ["eccs", "hdg", "rafter", "ext."],
    "tn_config":    ["dual", "combo", "hybrid", "zone", "multi", "two-stage", "stage"],
    "tn_optional":  ["optional", "option", "opt.", "alternate", "alt "],
}
TN_FEATS = list(TN_GROUPS) + ["tn_len", "tn_words", "tn_generic", "tn_has_num", "tn_cap_k"]

NUM = ["Diameter (ft)", "Height (ft)", "Freeboard (in)", "cap_gal", "vol_cf",
       "shell_area", "floor_area", "total_area", "aspect", "hoop", "steel_proxy",
       "log_vol", "Quantity", "mi", "Miles to Site (From TBT)",
       "Miles to Site (From GT)", "Ss", "S1", "seismic", "months",
       "t_x_area", "t_x_dia", "t_sq"] + SCOPE + TN_FEATS
CAT = ["Material", "Use Type", "Deck Style", "Floor Style", "Wage Type",
       "Country", "State", "Bid Type", "Sales Manager"]

RP = dict(learning_rate=0.03, max_iter=1200, min_samples_leaf=15,
          l2_regularization=1.0, random_state=0)

# GBM variants averaged in log space. Two variants capture most of the ensemble
# gain at half the training cost; append the commented-out pair below if you want
# the last ~0.05 points and do not mind an 8-minute retrain.
ENSEMBLE = [
    RP,
    dict(learning_rate=0.05, max_iter=700, min_samples_leaf=25,
         l2_regularization=0.3, random_state=1),
    # dict(learning_rate=0.02, max_iter=1800, min_samples_leaf=10,
    #      l2_regularization=3.0, random_state=2),
    # dict(learning_rate=0.04, max_iter=900, min_samples_leaf=20,
    #      l2_regularization=1.0, max_leaf_nodes=63, random_state=3),
]
BLEND = 0.70
HALFLIFE = 1.0
PSF_LO, PSF_HI = 20, 250


class ScopeError(ValueError):
    """Raised when scope is missing or internally inconsistent. Deliberately loud:
    guessing is the failure mode v4 exists to remove."""


# ----------------------------------------------------------------- features
def engineer(df):
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
    df["shell_area"] = np.pi * D * H
    df["floor_area"] = np.pi * (D / 2) ** 2
    df["total_area"] = df["shell_area"] + 2.02 * df["floor_area"]
    df["aspect"] = H / D.replace(0, np.nan)
    df["hoop"] = D * H
    df["steel_proxy"] = df["shell_area"] * np.sqrt(df["hoop"])
    df["log_vol"] = np.log1p(df["vol_cf"])
    df["seismic"] = df.get("Ss", np.nan) * df.get("S1", np.nan)

    # Drift is size-dependent: over the archive, the largest quartile of tanks rose
    # 22% per sq-ft while the second-largest fell 4%. A lone time feature cannot
    # express that, so time is allowed to interact with size.
    df["t_x_area"] = df["months"] * np.log1p(df["shell_area"])
    df["t_x_dia"] = df["months"] * D
    df["t_sq"] = df["months"] ** 2

    # --- Tank Name text features ---
    t = (df["Tank Name"].astype(str).str.lower() if "Tank Name" in df
         else pd.Series("", index=df.index))
    for feat, keys in TN_GROUPS.items():
        df[feat] = t.apply(lambda s, k=keys: int(any(x in s for x in k))).values
    # A bespoke tank gets a long descriptive name; a commodity one is "Tank 1".
    df["tn_len"] = t.str.len().values
    df["tn_words"] = t.str.split().str.len().fillna(0).values
    df["tn_generic"] = t.str.match(r"^\s*tank\s*\d*\s*$").astype(int).values
    df["tn_has_num"] = t.str.contains(r"\d").astype(int).values
    df["tn_cap_k"] = pd.to_numeric(
        t.str.extract(r"(\d+(?:\.\d+)?)\s*k\b", expand=False),
        errors="coerce").fillna(0).values

    mc = [c for c in ["Miles to Site (From TBT)", "Miles to Site (From GT)"] if c in df]
    df["mi"] = df[mc].min(axis=1) if mc else np.nan
    for c in NUM + CAT:
        if c not in df:
            df[c] = np.nan
    return df


def backbone(df):
    """log D and log H span all the geometry, so this basis is identified. A power
    law in it extrapolates; a tree clips at its largest leaf."""
    lD = np.log(df["Diameter (ft)"].astype(float).clip(lower=0.1).values)
    lH = np.log(df["Height (ft)"].astype(float).clip(lower=0.1).values)
    return np.column_stack([lD, lH, lD * lH, df["months"].values / 12.0,
                            np.ones(len(df))])


def check_scope(df):
    """Validate the scope columns. Raises rather than repairing — a silently
    corrected scope is the same failure as a guessed one."""
    missing = [c for c in SCOPE if c not in df.columns]
    if missing:
        raise ScopeError(
            f"missing scope columns: {', '.join(missing)}. "
            "Run prepare_data.py on the archive, or supply them per quote. "
            "v4 does not infer scope.")
    bad = df[SCOPE].isna().any(axis=1)
    if bad.any():
        raise ScopeError(f"{int(bad.sum())} rows have blank scope values. "
                         "Every quote needs an explicit yes/no on each.")
    viol = (df["IS_INSULATION_ERECTION"] == 1) & (df["IS_INSULATION"] == 0)
    if viol.any():
        raise ScopeError(
            f"{int(viol.sum())} rows set IS_INSULATION_ERECTION without "
            "IS_INSULATION. We cannot install insulation we are not supplying.")


def load_training(path):
    df = pd.read_csv(path, low_memory=False)
    for c in SCOPE:
        if c in df.columns:
            df[c] = _as01(df[c])
    df = engineer(df)
    n0 = len(df)
    df = df[(df["Diameter (ft)"] > 0) & (df["Height (ft)"] > 0) & (df["Total Price"] > 0)]
    df = df[df["Status"] != "Unfinished"] if "Status" in df else df
    check_scope(df)

    # Price columns are ALREADY per-tank. Do not divide by Quantity — doing so
    # corrupts the 15% of rows with multi-tank orders, which are 38% of the
    # top-5%-by-value quotes. The tell: price/qty/area falls as exactly 1/qty.
    for c in COMPS:
        df[c + "_u"] = df[c]
    df["total_u"] = df["Total Price"]

    # Scope must agree with the pricing actually on the quote.
    for comp, flag in OPTIONAL.items():
        mism = (df[comp] > 0) != (df[flag] == 1)
        if mism.any():
            raise ScopeError(
                f"{int(mism.sum())} rows where {flag} disagrees with {comp}. "
                "Re-run prepare_data.py.")

    df["psf"] = df["Total Price"] / df["shell_area"]
    df["plausible"] = df["psf"].between(PSF_LO, PSF_HI)

    ident = (df[[c + "_u" for c in COMPS]].sum(axis=1) - df["total_u"]).abs()
    print(f"  {n0} rows -> {len(df)} usable "
          f"({(~df['plausible']).sum()} implausible, excluded from fitting)")
    print(f"  accounting identity holds on {(ident < .05).mean()*100:.1f}%")
    print("  scope: " + ", ".join(
        f"{f.replace('IS_','')} {df[f].mean()*100:.0f}%" for f in SCOPE))
    return df.reset_index(drop=True)


def _as01(s):
    """Accept Yes/No, TRUE/FALSE, Y/N, 1/0 — write 1/0."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (s.fillna(0) > 0).astype(int)
    m = s.astype(str).str.strip().str.upper()
    out = m.map({"YES": 1, "Y": 1, "TRUE": 1, "T": 1, "1": 1,
                 "NO": 0, "N": 0, "FALSE": 0, "F": 0, "0": 0})
    return out


def encode(df, enc=None):
    X = df[NUM + CAT].copy()
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
    """Effective rate by state, from rows where tax was actually charged. The rate
    is geography; whether it applies is IS_TAXABLE."""
    s = df[df["Total Tax"] > 0].copy()
    s["rate"] = s["Total Tax"] / (s["Proposal Total"] - s["Total Tax"]).clip(lower=1)
    t = s.groupby("State")["rate"].agg(["median", "size"])
    return {k: float(v) for k, v in t[t["size"] >= 10]["median"].items()}


# ------------------------------------------------------------------ fitting
class Stage:
    """Ridge log-log backbone + an ensemble of gradient-boosted residual models.

    The backbone is what lets the model extrapolate: a tree predicts a constant
    beyond its largest leaf, so without it a tank bigger than anything in training
    gets priced at the biggest thing we have seen. A power law in log D and log H
    keeps rising. The tree then learns only what the power law missed.
    """

    def __init__(self, ci, params=None):
        self.ci = ci
        self.params = params or ENSEMBLE

    def fit(self, X, B, y, w=None):
        self.lin = Ridge(alpha=1.0).fit(B, y)
        resid = y - self.lin.predict(B)
        self.gbms = [HistGradientBoostingRegressor(categorical_features=self.ci, **p)
                     .fit(X, resid, sample_weight=w) for p in self.params]
        return self

    def predict(self, X, B):
        # Mean in log space = geometric mean of prices, which is the right average
        # for a multiplicative target.
        return self.lin.predict(B) + np.mean([g.predict(X) for g in self.gbms], axis=0)


def fit_models(df, X, B, ci, idx, w):
    regs = {}
    for c in COMPS:
        m = idx & (df[c + "_u"].values > 0)
        regs[c] = Stage(ci).fit(X[m], B[m], np.log(df[c + "_u"].values[m]), w[m])
    direct = Stage(ci).fit(X[idx], B[idx], np.log(df["total_u"].values[idx]), w[idx])
    return regs, direct


def combine(regs, direct, df, X, B, te):
    comp = np.zeros(int(te.sum()))
    for c in COMPS:
        comp += np.exp(regs[c].predict(X[te], B[te])) * (df[c + "_u"].values[te] > 0)
    return BLEND * comp + (1 - BLEND) * np.exp(direct.predict(X[te], B[te]))


# ---------------------------------------------------------------- validate
def validate(df, n_folds=6):
    """Rolling origin. A random split leaks quote revisions — 6,892 rows are only
    3,018 quotes — and reports roughly 3% instead of 8%."""
    X, _ = encode(df); ci = _ci(X); B = backbone(df); w = recency_weights(df)
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
        regs, direct = fit_models(df, X, B, ci, tr, w)
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
    for lab, sel in [("mean", r), ("last 3", r[-3:])]:
        print(f"  {lab:<9}{'':>6}{sel[:,0].mean():>8.1f}%{sel[:,1].mean():>7.1f}%"
              f"{sel[:,2].mean():>7.1f}%{sel[:,3].mean():>8.1f}%{sel[:,4].mean():>8.1f}%")
    return r


# ------------------------------------------------------------------- train
def train(path, skip_validate=False):
    print("Loading…")
    df = load_training(path)
    if not skip_validate:
        validate(df)

    X, enc = encode(df); ci = _ci(X); B = backbone(df); w = recency_weights(df)
    pl = df["plausible"].values
    g = df["Quote #"].values if "Quote #" in df else np.arange(len(df))
    act = df["Total Price"].values

    print("\n  calibrating prediction bands…")
    tr_i, cal_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=0)
                       .split(X, act, g))
    tr = np.zeros(len(df), bool); tr[tr_i] = True; tr &= pl
    cal = np.zeros(len(df), bool); cal[cal_i] = True; cal &= pl
    regs, direct = fit_models(df, X, B, ci, tr, w)
    pred = combine(regs, direct, df, X, B, cal)
    resid = np.abs(np.log(act[cal]) - np.log(pred))
    bands = {int((1 - a) * 100): float(np.exp(np.quantile(resid, 1 - a)))
             for a in (0.20, 0.10)}
    print(f"     80% band = /x {bands[80]:.2f}      90% band = /x {bands[90]:.2f}")

    print("\nFitting final models on all plausible rows…")
    regs, direct = fit_models(df, X, B, ci, pl, w)

    lim = {c: (float(df.loc[pl, c].quantile(.005)), float(df.loc[pl, c].quantile(.995)))
           for c in ["Diameter (ft)", "Height (ft)"]}
    lim["shell_area_max"] = float(df.loc[pl, "shell_area"].max())

    joblib.dump({"regs": regs, "direct": direct, "encoder": enc, "bands": bands,
                 "tax": tax_table(df), "blend": BLEND, "limits": lim,
                 "big_threshold": float(df.loc[pl, "Total Price"].quantile(.95)),
                 "rows": int(pl.sum()), "version": VERSION,
                 "trained": str(pd.Timestamp.today().date()),
                 "data_through": str(pd.to_datetime(df["Due Date"],
                                                    errors="coerce").max().date())},
                MODEL_PATH)
    n = len(regs) * 2 + 2
    print(f"Saved -> {MODEL_PATH}  ({n} fitted estimators, no classifiers)")


# ----------------------------------------------------------------- predict
_B = None

ALIAS = {"diameter": "Diameter (ft)", "height": "Height (ft)",
         "freeboard": "Freeboard (in)", "material": "Material",
         "use_type": "Use Type", "deck": "Deck Style", "floor": "Floor Style",
         "wage": "Wage Type", "country": "Country", "state": "State",
         "bid_type": "Bid Type", "sales_manager": "Sales Manager",
         "miles": "Miles to Site (From TBT)", "quantity": "Quantity",
         "ss": "Ss", "s1": "S1", "due_date": "Due Date",
         "tank_name": "Tank Name", "name": "Tank Name"}


def predict(construction=None, insulation=None, insulation_erection=None,
            freight=None, taxable=None, **kw):
    """Price one tank.

    Scope is REQUIRED. construction, insulation, freight and taxable must each be
    True or False — the model prices scope, it does not guess it. Passing None
    raises ScopeError.

    insulation_erection defaults to the value of insulation (if we supply it we
    usually install it; 1,456 of 1,539 archive cases). Pass False explicitly for
    supply-only jobs.

    tank_name is optional but worth passing. The process descriptor carries real
    signal on engineered-to-order work: median $/sq-ft runs 1.35x base for
    digesters and 0.68x for multi-zone configurations. A generic "Tank 1" is
    itself informative - it usually means a commodity shell.
    """
    global _B
    if _B is None:
        _B = joblib.load(MODEL_PATH)
    b = _B

    if insulation_erection is None:
        insulation_erection = insulation
    given = {"construction": construction, "insulation": insulation,
             "freight": freight, "taxable": taxable}
    absent = [k for k, v in given.items() if v is None]
    if absent:
        raise ScopeError(
            "scope not supplied: " + ", ".join(absent) + ". "
            "Each must be True or False. v4 does not infer scope — whether a "
            "quote includes erection, insulation, freight or tax is a commercial "
            "decision, not a property of the tank.")
    if insulation_erection and not insulation:
        raise ScopeError("insulation_erection=True requires insulation=True. "
                         "We cannot install insulation we are not supplying.")

    row = {ALIAS.get(k, k): v for k, v in kw.items()}
    row.setdefault("Quantity", 1)
    row.setdefault("Country", "US")
    row.setdefault("Bid Type", "Firm")
    row.setdefault("Due Date", pd.Timestamp.today().strftime("%m/%d/%Y"))
    row["IS_CONSTRUCTION"] = int(bool(construction))
    row["IS_INSULATION"] = int(bool(insulation))
    row["IS_INSULATION_ERECTION"] = int(bool(insulation_erection))
    row["IS_FREIGHT"] = int(bool(freight))
    row["IS_TAXABLE"] = int(bool(taxable))

    d = engineer(pd.DataFrame([row]))
    X, _ = encode(d, b["encoder"])
    B = backbone(d)

    present = {"Material Price": True, "Fabrication Price": True,
               "Construction Price": bool(construction),
               "Insulation Material Price": bool(insulation),
               "Insulation Construction Price": bool(insulation_erection),
               "Freight Price": bool(freight)}
    parts, total = {}, 0.0
    for c in COMPS:
        v = float(np.exp(b["regs"][c].predict(X, B)[0])) * present[c]
        parts[c] = round(v, 0); total += v
    direct = float(np.exp(b["direct"].predict(X, B)[0]))
    est = b["blend"] * total + (1 - b["blend"]) * direct

    qty = float(row.get("Quantity", 1) or 1)
    rate = b["tax"].get(str(row.get("State", "")), 0.0) if taxable else 0.0
    ex_frt = total - parts["Freight Price"]

    warn = []
    for c, label in [("Diameter (ft)", "diameter"), ("Height (ft)", "height")]:
        lo, hi = b["limits"][c]
        if not lo <= float(row[c]) <= hi:
            warn.append(f"{label} outside the fitted range ({lo:.0f}-{hi:.0f} ft)")
    if float(d["shell_area"].iloc[0]) > b["limits"]["shell_area_max"]:
        warn.append("larger than anything in the training data — extrapolating")
    if est >= b["big_threshold"]:
        warn.append("top-5% by value: the model underprices this band by roughly "
                    "13-17% in aggregate — treat as a floor, not an estimate")
    psf = est / float(d["shell_area"].iloc[0])
    if not PSF_LO <= psf <= PSF_HI:
        warn.append(f"implied ${psf:.0f}/sq-ft is outside the plausible range")
    if int(d["tn_generic"].iloc[0]) == 1 or not str(row.get("Tank Name", "")).strip():
        warn.append("no descriptive tank name supplied — process keywords "
                    "(digester, MBBR, equalization, silo) materially improve the "
                    "estimate on engineered-to-order tanks")
    if taxable and rate == 0.0:
        warn.append(f"marked taxable but no rate on file for state "
                    f"'{row.get('State')}' — tax shown as zero")

    return {"unit_price": round(est, 0), "total": round(est * qty, 0),
            "p80_low": round(est * qty / b["bands"][80], 0),
            "p80_high": round(est * qty * b["bands"][80], 0),
            "p90_low": round(est * qty / b["bands"][90], 0),
            "p90_high": round(est * qty * b["bands"][90], 0),
            "components": {k: round(v, 0) for k, v in parts.items()},
            "scope": {"construction": bool(construction),
                      "insulation": bool(insulation),
                      "insulation_erection": bool(insulation_erection),
                      "freight": bool(freight), "taxable": bool(taxable)},
            "tax_rate": round(rate, 4),
            "est_tax": round(ex_frt * rate, 0),
            "proposal_total": round(ex_frt * (1 + rate), 0),
            "psf": round(psf, 1),
            "direct_model_says": round(direct, 0),
            "component_sum_says": round(total, 0),
            "warnings": warn,
            "model_version": b.get("version", VERSION),
            "model_trained": b["trained"], "data_through": b["data_through"]}


if __name__ == "__main__":
    # Route through the module so Stage pickles as tbt_model.Stage rather than
    # __main__.Stage, which would fail to unpickle from any importing process.
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
                  "state", "bid-type", "sales-manager", "due-date", "tank-name"]:
            ap.add_argument(f"--{f}")
        ap.add_argument("--quantity", type=int, default=1)
        for f in ["construction", "insulation", "insulation-erection",
                  "freight", "taxable"]:
            ap.add_argument(f"--{f}", choices=["yes", "no"], required=(f != "insulation-erection"))
        a = vars(ap.parse_args(sys.argv[2:]))
        sc = {}
        for f in ["construction", "insulation", "insulation_erection",
                  "freight", "taxable"]:
            v = a.pop(f.replace("_", "-").replace("-", "_"), None)
            sc[f] = None if v is None else (v == "yes")
        kw = {k.replace("-", "_"): v for k, v in a.items() if v is not None}
        try:
            print(json.dumps(_M.predict(**sc, **kw), indent=2))
        except ScopeError as e:
            print(f"SCOPE ERROR: {e}", file=sys.stderr)
            sys.exit(2)
