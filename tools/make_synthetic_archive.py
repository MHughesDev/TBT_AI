"""Generate a synthetic TBT-shaped archive.

WHY THIS EXISTS
---------------
The real archive is not in this repository (it is gitignored: it holds customer
names and real pricing). Without it no real accuracy number can be produced.
This generator exists so the PIPELINE can be exercised end to end and the
harness proven, not so an accuracy number can be invented.

WHAT IT IS CALIBRATED TO
------------------------
Every structural fact documented in the field notes: row/quote counts, the
revision and multi-tank structure, price and diameter ranges, component
presence shares, the accounting identities, size-dependent drift, the
superlinear large-tank effect, the per-segment variance ordering, the
Argentina bias, and the ~173 implausible partial quotes.

WHAT IT IS **NOT**
------------------
It is not the real data-generating process. Two consequences that must be
stated wherever its numbers are quoted:

1. The MAPE measured on it is a property of the noise level chosen here. It
   is NOT a prediction of real-world accuracy.
2. The cost model below is deliberately NOT the model's physics term. It uses
   different thickness constants, a crane-class STEP function and a rigging
   premium that `features.steel_lb_est` does not contain. So the OPEN-2
   experiment is an approximation test rather than a tautology -- but it is
   still a test against a process this file chose, so its outcome is not
   evidence about the real archive either.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

RNG_DEFAULT = 20260916

# Calibrated so this archive's irreducible floor is 4.60% mean APE, matching the
# documented estimate for the real archive (the residual spread among quotes
# identical in specification, state, wage type and year). That makes a MAPE
# measured here comparable IN SCALE to the real 8.5% baseline -- it does not
# make it a prediction of real-world accuracy.
DEFAULT_NOISE = 0.55

# ------------------------------------------------------------------ taxonomy
USE_TYPES = [
    "Fire Protection Storage Tank", "Potable Water Storage Tank",
    "Waste Water Storage Tank", "Industrial Storage Tank", "Industrial Silo",
    "Sewage Storage Tank", "Sludge Storage Tank", "Process Water Tank",
    "Chemical Storage Tank", "Raw Water Storage Tank",
    "Leachate Storage Tank", "Digester Tank", "Oil Storage Tank",
]
USE_WEIGHTS = np.array([.31, .11, .21, .09, .035, .04, .035, .04, .03, .03, .015, .02, .02])

DECK_STYLES = [
    "2:12 Roof (Comp/Tension Ring - 2 inch Rise to 12 inch Run)", "Open-Top",
    "Flat Roof", "Aluminum Geodesic Dome", "Aluminum Flat Cover",
    "3:12 Roof (Comp/Tension Ring)", "Cone Roof", "Column Supported Roof",
    "Ext. Rafter Roof", "Membrane Cover", "Fixed Roof w/ Rafters",
    "Self-Supported Cone", "Umbrella Roof", "Dome Roof (Steel)",
    "Knuckle Roof", "Suspended Deck", "Floating Cover", "Hinged Cover",
    "No Roof Supplied",
]
FLOOR_STYLES = [
    "Flat Steel Floor", "Embedded Ring", "Cone Down Floor", "Cone Up Floor",
    "Concrete Floor (By Others)", "Chime Ring", "Sloped Steel Floor",
    "Annular Ring Only", "Double Bottom", "Grade Supported",
]
MATERIALS = ["CS", "304SS", "316SS", "Aluminum"]
MATERIAL_W = np.array([.86, .085, .035, .02])
MATERIAL_MULT = {"CS": 1.0, "304SS": 2.15, "316SS": 2.75, "Aluminum": 1.55}

US_STATES = ["TX", "CA", "FL", "MO", "IA", "NJ", "PA", "OH", "GA", "NC", "IL",
             "WI", "TN", "AL", "LA", "OR", "WA", "AZ", "CO", "MN", "MI", "IN",
             "SC", "VA", "NY", "AR", "OK", "KS", "NE", "UT"]
STATE_TAX = {"TX": .0530, "CA": .0725, "FL": .0600, "MO": .0423, "IA": .0600,
             "NJ": .0663, "PA": .0382, "OH": .0575, "GA": .0482, "NC": .0458,
             "IL": .0625, "WI": .0332, "TN": .0612, "AL": .0621, "LA": .0626,
             "OR": .0000, "WA": .0650, "AZ": .0560, "CO": .0290, "MN": .0688,
             "MI": .0600, "IN": .0700, "SC": .0600, "VA": .0530, "NY": .0400,
             "AR": .0650, "OK": .0450, "KS": .0650, "NE": .0550, "UT": .0485}
# customer exemption share by state -- SPEC 4.6 / field notes: geography sets
# the rate, the customer sets whether it applies
STATE_TAXABLE_SHARE = {"OR": .00, "AR": .18, "TX": .38, "IL": .77, "NJ": .91}

COUNTRIES = ["US", "MX", "CA", "PE", "CL", "CO", "AR", "BR", "EC", "GT",
             "PA", "DO", "CR", "HN", "NI", "SV", "BO", "PY", "UY", "VE",
             "JM", "TT", "BS", "BB", "GY", "SR", "BZ", "HT", "CU", "PR",
             "AU", "NZ", "PH"]
COUNTRY_W = np.array([.745, .088, .022, .021, .019, .017, .016, .010, .008, .006,
                      .005, .004, .004, .003, .003, .003, .002, .002, .002, .002,
                      .002, .002, .002, .001, .001, .001, .001, .001, .001, .001,
                      .001, .001, .001])
COUNTRY_W = COUNTRY_W / COUNTRY_W.sum()

WAGE_TYPES = ["Non-Union / Non-Prevailing", "Prevailing Wage", "Union",
              "No Erection Included"]
SALES_MANAGERS = [f"SM{i:02d}" for i in range(1, 13)]
BID_TYPES = ["Firm", "Budget", "Alternate"]

NAME_BITS = {
    "Fire Protection Storage Tank": ["Fire Water Tank", "FP Storage Tank", "Tank 1",
                                     "NFPA-22 Tank", "Fire Protection Tank"],
    "Waste Water Storage Tank": ["Primary Anaerobic Digester - Hybrid",
                                 "MBBR Reaeration Reactor", "Flow Equalization Basin",
                                 "Sludge Holding Tank", "Aerobic Digester",
                                 "Backwash Equalization Tank", "Dual Zone Reactor"],
    "Industrial Silo": ["Lime Silo", "Fly Ash Silo", "Cement Storage Silo"],
    "Digester Tank": ["Anaerobic Digester No. 2", "Primary Digester"],
    "Leachate Storage Tank": ["Leachate Storage - Landfill Cell 4"],
    "Chemical Storage Tank": ["Acidified Waste Slurry Tank", "Process Chemical Tank"],
}
GENERIC_NAMES = ["Tank 1", "Tank 2", "T-101", "Tank A", "TK-1", ""]


def _pick(rng, items, weights=None, size=None):
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    idx = rng.choice(len(items), size=size, p=weights)
    if size is None:
        return items[idx]
    return np.array([items[i] for i in idx], dtype=object)


def true_cost_driver(D, H, rng):
    """The synthetic cost driver. DELIBERATELY NOT the model's physics term.

    Uses different thickness constants, an explicit crane-class STEP and a
    rigging premium, so `features.steel_lb_est` is an approximation of this
    rather than a restatement of it.
    """
    # thickness from a different allowable stress and joint efficiency
    H_liq = np.maximum(H - 0.5, 1.0)
    t_hoop = 2.6 * D * H_liq * 1.0 / (23000.0 * 0.80)       # not 20000 / 0.85
    t_min = np.where(D < 40, 0.1875, np.where(D < 100, 0.25,
             np.where(D < 170, 0.3125, 0.4375)))            # different bands
    t = np.maximum(t_hoop, t_min)
    shell_lb = np.pi * D * H * t * 41.8                     # not 40.8
    floor_lb = np.pi * (D / 2) ** 2 * 0.28 * 41.8
    steel = shell_lb + floor_lb

    # crane class steps -- a genuine discontinuity no smooth term reproduces
    crane = np.where(H < 24, 1.00, np.where(H < 40, 1.06,
             np.where(H < 56, 1.15, 1.28)))
    # specialised rigging premium on the very large end
    rigging = 1.0 + 0.11 * np.clip((D - 110.0) / 90.0, 0.0, 1.0) ** 1.5
    return steel, crane * rigging


def generate(n_quotes=3018, target_rows=6892, seed=RNG_DEFAULT,
             noise_scale=DEFAULT_NOISE) -> pd.DataFrame:
    """noise_scale=0 draws the SAME random numbers but zeroes the irreducible
    noise, giving the noiseless price for the identical set of tanks. The gap
    between the two is this archive's exact irreducible floor: the best any
    model could possibly do here. It is what makes a measured MAPE on synthetic
    data interpretable."""
    rng = np.random.default_rng(seed)

    # ---- quote skeleton: revisions and multi-tank lines -------------------
    # 855/3018 = 28.3% of quotes carry >1 revision; 870/3018 = 28.8% carry >1
    # tank; and 6892/3018 = 2.28 rows per quote. Matching all three needs a
    # heavy tail among the multi-row quotes, not a thin binomial.
    multi_rev = rng.random(n_quotes) < 0.283
    n_rev = np.where(multi_rev, 2 + rng.poisson(0.80, n_quotes), 1)
    multi_tank = rng.random(n_quotes) < 0.288
    n_tank = np.where(multi_tank, 2 + rng.poisson(0.78, n_quotes), 1)
    rows_per_quote = n_rev * n_tank
    # trim to the target row count
    order = rng.permutation(n_quotes)
    cum = np.cumsum(rows_per_quote[order])
    keep = order[cum <= target_rows]

    recs = []
    start = pd.Timestamp("2023-12-01")
    span_days = (pd.Timestamp("2026-07-31") - start).days

    for qi in keep:
        qnum = f"Q{10000 + int(qi)}"
        base_day = int(rng.integers(0, span_days))
        # quote-level attributes shared by all its rows
        country = _pick(rng, COUNTRIES, COUNTRY_W)
        state = _pick(rng, US_STATES) if country == "US" else ""
        use = _pick(rng, USE_TYPES, USE_WEIGHTS)
        mat = _pick(rng, MATERIALS, MATERIAL_W)
        sm = _pick(rng, SALES_MANAGERS)
        bid = _pick(rng, BID_TYPES, np.array([.74, .21, .05]))
        wage = _pick(rng, WAGE_TYPES, np.array([.62, .07, .11, .20]))
        miles = float(np.clip(rng.lognormal(5.6, 0.9), 5, 4200))

        # scope: commercial decisions, fixed per quote
        constr = wage != "No Erection Included" and rng.random() < 0.855
        insul = rng.random() < 0.226
        insul_er = insul and rng.random() < 0.947
        freight = rng.random() < (0.93 if country == "US" else 0.52)
        if country == "US":
            share = STATE_TAXABLE_SHARE.get(state, 0.53)
        else:
            share = 0.02
        taxable = rng.random() < share

        for ti in range(int(n_tank[qi])):
            # tank geometry: log-normal diameter, aspect-driven height
            D = float(np.clip(rng.lognormal(3.45, 0.52), 3, 196))
            aspect = float(np.clip(rng.lognormal(-0.18, 0.42), 0.15, 3.2))
            H = float(np.clip(D * aspect, 6, 72))
            deck = _pick(rng, DECK_STYLES)
            floor = _pick(rng, FLOOR_STYLES)
            fb = float(rng.choice([0, 6, 12, 18, 24]))
            qty = int(rng.choice([1, 2, 3, 4, 5, 6],
                                 p=[.85, .075, .035, .02, .012, .008]))
            if use in NAME_BITS and rng.random() < 0.72:
                tname = str(rng.choice(NAME_BITS[use]))
            else:
                tname = str(rng.choice(GENERIC_NAMES))
            if int(n_tank[qi]) > 1:
                # one row per tank LINE: the name distinguishes the lines
                tname = (f"{tname} No. {ti + 1}" if tname else f"Tank {ti + 1}")
            ss = round(float(np.clip(rng.lognormal(-2.1, 0.75), 0.02, 2.5)), 3)
            s1 = round(ss * float(np.clip(rng.normal(0.55, 0.12), 0.2, 0.95)), 3)

            for ri in range(int(n_rev[qi])):
                due = start + pd.Timedelta(days=base_day + 21 * ri)
                if due > pd.Timestamp("2026-07-31"):
                    due = pd.Timestamp("2026-07-31")
                recs.append(dict(
                    quote=qnum, rev=ri, tank=ti, due=due, country=country,
                    state=state, use=use, mat=mat, sm=sm, bid=bid, wage=wage,
                    miles=miles, constr=constr, insul=insul, insul_er=insul_er,
                    freight=freight, taxable=taxable, D=D, H=H, deck=deck,
                    floor=floor, fb=fb, qty=qty, tname=tname, ss=ss, s1=s1,
                ))
    df = pd.DataFrame(recs).reset_index(drop=True)

    n = len(df)
    D = df["D"].values
    H = df["H"].values
    steel, size_mult = true_cost_driver(D, H, rng)
    shell_area = np.pi * D * H

    # ---- price structure --------------------------------------------------
    t_years = (pd.to_datetime(df["due"]) - pd.Timestamp("2024-01-01")).dt.days.values / 365.25

    # size-dependent drift: the largest quartile rises, the second-largest falls
    size_rank = pd.Series(shell_area).rank(pct=True).values
    drift = 0.049 * t_years + 0.034 * t_years * (size_rank - 0.62)

    mat_mult = np.array([MATERIAL_MULT[m] for m in df["mat"]])
    use_mult = np.where(df["use"].values == "Fire Protection Storage Tank", 0.92,
                np.where(df["use"].values == "Waste Water Storage Tank", 1.14,
                 np.where(df["use"].values == "Industrial Silo", 1.07, 1.0)))
    deck_mult = np.where(pd.Series(df["deck"]).str.contains("Geodesic|Dome", regex=True).values, 1.24,
                 np.where(df["deck"].values == "Open-Top", 0.88, 1.0))
    wage_mult = np.where(df["wage"].values == "Prevailing Wage", 1.17,
                 np.where(df["wage"].values == "Union", 1.09, 1.0))
    seis_mult = 1.0 + 0.16 * df["ss"].values
    name_mult = np.where(pd.Series(df["tname"]).str.contains("Digester|Anaerobic", regex=True).values, 1.10,
                 np.where(pd.Series(df["tname"]).str.contains("Backwash").values, 1.16,
                  np.where(pd.Series(df["tname"]).str.contains("Equalization").values, 0.94, 1.0)))

    # ---- irreducible noise, per segment ----------------------------------
    # Ordering is taken from the documented out-of-fold spreads: Fire tightest,
    # Waste/Industrial widest, Canada widest of all, Prevailing Wage wide.
    sigma = np.where(df["use"].values == "Fire Protection Storage Tank", 0.055,
             np.where(pd.Series(df["use"]).str.contains("Waste|Sewage|Sludge|Digester", regex=True).values, 0.122,
              np.where(pd.Series(df["use"]).str.contains("Industrial|Silo|Chemical|Process|Oil", regex=True).values, 0.132,
               np.where(pd.Series(df["use"]).str.contains("Potable|Raw Water", regex=True).values, 0.075, 0.100))))
    sigma = sigma * np.where(df["wage"].values == "Prevailing Wage", 2.0, 1.0)
    sigma = sigma * np.where(df["country"].values == "CA", 2.7,
                      np.where(df["country"].values == "AR", 2.5, 1.0))
    noise = rng.normal(0.0, 1.0, n) * sigma * noise_scale
    # Argentina is the one genuine bias in the data
    ar_bias = np.where(df["country"].values == "AR", np.log(1.0 - 0.167), 0.0)

    # base material rate per lb of steel, in sell dollars
    base_rate = 1.80
    material = (base_rate * steel * mat_mult * use_mult * deck_mult
                * seis_mult * name_mult * np.exp(drift + ar_bias + noise))
    fabrication = (material / mat_mult ** 0.55) * np.exp(
        rng.normal(-0.06, 0.09, n) * noise_scale) * 1.02
    construction = (shell_area * 24.0 * size_mult * wage_mult
                    * np.exp(drift * 1.15 + rng.normal(0, 0.16, n) * noise_scale)
                    * np.where(df["use"].values == "Waste Water Storage Tank", 1.18, 1.0))
    insul_mat = shell_area * 11.5 * np.exp(drift + rng.normal(0, 0.17, n) * noise_scale)
    insul_con = shell_area * 9.2 * wage_mult * np.exp(drift + rng.normal(0, 0.19, n) * noise_scale)
    freight_p = (240.0 + df["miles"].values * 2.35 * np.sqrt(steel / 30000.0)
                 ) * np.exp(drift * 0.8 + rng.normal(0, 0.22, n) * noise_scale)

    material = np.maximum(material, 1200.0)
    fabrication = np.maximum(fabrication, 1000.0)

    constr_on = df["constr"].values.astype(bool)
    insul_on = df["insul"].values.astype(bool)
    insul_er_on = df["insul_er"].values.astype(bool)
    frt_on = df["freight"].values.astype(bool)
    tax_on = df["taxable"].values.astype(bool)

    construction = np.where(constr_on, construction, 0.0)
    insul_mat = np.where(insul_on, insul_mat, 0.0)
    insul_con = np.where(insul_er_on, insul_con, 0.0)
    freight_p = np.where(frt_on, freight_p, 0.0)

    rate = np.array([STATE_TAX.get(s, 0.0) if c == "US" else 0.0
                     for s, c in zip(df["state"], df["country"])])
    ex_freight = material + fabrication + construction + insul_mat + insul_con
    tax = np.where(tax_on, ex_freight * rate, 0.0)

    total_price = ex_freight + freight_p          # tax-EXCLUSIVE, freight-INCLUSIVE
    proposal_total = ex_freight + tax             # tax-INCLUSIVE, freight-EXCLUSIVE

    margin_band = rng.choice([16, 18, 20, 22, 26], n, p=[.011, .374, .401, .162, .052])

    out = pd.DataFrame({
        "Quote #": df["quote"], "Revision #": df["rev"],
        "Project Name": ["" if rng.random() < 0.85 else f"Project {i}" for i in range(n)],
        "Bid Type": df["bid"],
        "Status": _pick(rng, ["Deliver To Customer", "Revised", "Sales", "Won",
                              "Lost", "Bid Review"],
                        np.array([.478, .338, .107, .0245, .0205, .0015]), size=n),
        "Tank Name": df["tname"],
        "Due Date": pd.to_datetime(df["due"]).dt.strftime("%m/%d/%Y"),
        "Company Name": [f"Customer {int(h)%640:03d}" for h in
                         pd.util.hash_pandas_object(df["quote"], index=False).values],
        "Customer Name": "", "Country": df["country"], "State": df["state"],
        "City": "", "Sales Manager": df["sm"],
        "Sales Rep": ["" if rng.random() < 0.73 else "Rep" for _ in range(n)],
        "Commission (%)": np.round(rng.uniform(0, 5, n), 2),
        "Wage Type": df["wage"],
        "Miles to Site (From TBT)": np.round(df["miles"].values, 0),
        "Miles to Site (From GT)": np.round(df["miles"].values * rng.uniform(0.7, 1.4, n), 0),
        "Ss": df["ss"], "S1": df["s1"], "Quantity": df["qty"],
        "Material": df["mat"], "Use Type": df["use"],
        "Deck Style": df["deck"], "Floor Style": df["floor"],
        "Diameter (ft)": np.round(D, 1), "Height (ft)": np.round(H, 1),
        "Freeboard (in)": df["fb"],
        "Usable Capacity": np.round(np.pi * (D / 2) ** 2 * (H - df["fb"].values / 12) * 7.48, 0),
        "Margin (%)": margin_band, "Contingency (%)": np.round(rng.uniform(0, 4, n), 1),
        "Insulation Margin (%)": np.where(insul_on, margin_band, 0),
        "Insulation Contingency (%)": 0.0,
        "Material Price": np.round(material, 2),
        "Fabrication Price": np.round(fabrication, 2),
        "Construction Price": np.round(construction, 2),
        "Insulation Material Price": np.round(insul_mat, 2),
        "Insulation Construction Price": np.round(insul_con, 2),
        "Total Tax": np.round(tax, 2),
        "Proposal Total": 0.0, "Freight Price": np.round(freight_p, 2),
        "Total Price": 0.0,
    })

    # rebuild the identities from the ROUNDED components so they hold exactly
    comps = ["Material Price", "Fabrication Price", "Construction Price",
             "Insulation Material Price", "Insulation Construction Price"]
    out["Total Price"] = out[comps].sum(axis=1) + out["Freight Price"]
    out["Proposal Total"] = out[comps].sum(axis=1) + out["Total Tax"]

    # ---- ~173 partial quotes with scope lines not yet filled in -----------
    n_bad = 205
    idx = rng.choice(n, n_bad, replace=False)
    half = n_bad // 2
    # low: a quote with most lines still blank
    lo = idx[:half]
    out.loc[lo, "Construction Price"] = 0.0
    out.loc[lo, "Freight Price"] = 0.0
    out.loc[lo, "Material Price"] = np.round(out.loc[lo, "Material Price"] * 0.10, 2)
    out.loc[lo, "Fabrication Price"] = np.round(out.loc[lo, "Fabrication Price"] * 0.10, 2)
    # high: a line item double-counted
    hi = idx[half:]
    out.loc[hi, "Material Price"] = np.round(out.loc[hi, "Material Price"] * 4.2, 2)
    out.loc[idx, "Total Price"] = out.loc[idx, comps].sum(axis=1) + out.loc[idx, "Freight Price"]
    out.loc[idx, "Proposal Total"] = out.loc[idx, comps].sum(axis=1) + out.loc[idx, "Total Tax"]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=RNG_DEFAULT)
    ap.add_argument("--rows", type=int, default=6892)
    ap.add_argument("--quotes", type=int, default=3018)
    ap.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE)
    a = ap.parse_args()
    df = generate(n_quotes=a.quotes, target_rows=a.rows, seed=a.seed,
                  noise_scale=a.noise_scale)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows, {df['Quote #'].nunique()} quotes")


if __name__ == "__main__":
    main()
