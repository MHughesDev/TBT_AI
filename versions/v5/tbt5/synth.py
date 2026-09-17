"""A synthetic archive with the real one's schema and shape.

Why this exists
---------------
The real archive is customer data and is not in the repository — `*.csv` is
gitignored on purpose. Without it nothing in this package can be run, let alone
measured, and "here is some code that has never executed" is not a deliverable.

So this module manufactures an archive with the documented schema, the documented
row and quote counts, the documented scope mix and the documented date range,
priced by a generative cost process.

What a number measured on this data does and does not mean
----------------------------------------------------------
It is a test of the plumbing: that every stage runs, that the causal machinery is
actually causal, that the metrics compute, that a model trained on quarter N can
score quarter N+1.

It is NOT an accuracy result, and no number produced from it should ever be
quoted as one. The synthetic process is a guess at how tank pricing works, and a
model measured against a guess is measuring the guess.

The one thing deliberately built in is the *structure* v4 identified and could
not fit: costs here are assembled course by course, so steel intensity rises with
tank size the way it does in the archive. The generator computes that with
different constants, different plate widths and different rounding from
physics.py, so recovering it is a genuine inference rather than inverting an
identity the model was handed.
"""
import numpy as np
import pandas as pd

MATERIALS = {"CS": 1.00, "304SS": 2.35, "316SS": 3.05, "A36": 1.02, "HDG": 1.18}
MATERIAL_P = [0.80, 0.09, 0.03, 0.05, 0.03]

USE_TYPES = [
    "Fire Protection Storage Tank", "Waste Water Storage Tank",
    "Industrial Storage Tank", "Potable Water Storage Tank",
    "Process Water Storage Tank", "Agricultural Storage Tank",
    "Leachate Storage Tank", "Bulk Storage Silo",
]
USE_P = [0.44, 0.21, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02]

DECKS = ["2:12 Roof (Comp/Tension Ring - 2 inch Rise to 12 inch Run)",
         "Flat Roof", "Open-Top", "Geodesic Dome", "Column-Supported Cone"]
DECK_P = [0.46, 0.20, 0.22, 0.06, 0.06]
FLOORS = ["Flat Steel Floor", "Embedded Ring", "Concrete Floor", "Cone Down"]
FLOOR_P = [0.55, 0.22, 0.18, 0.05]
WAGES = ["Non-Union / Non-Prevailing", "Prevailing Wage", "Union"]
WAGE_P = [0.62, 0.26, 0.12]
WAGE_MULT = {"Non-Union / Non-Prevailing": 1.00, "Prevailing Wage": 1.34, "Union": 1.46}

STATES = {
    "TX": (0.0825, 0.38), "CA": (0.0875, 0.55), "MO": (0.0723, 0.41),
    "IA": (0.0700, 0.33), "NJ": (0.0663, 0.91), "OR": (0.0000, 0.00),
    "FL": (0.0700, 0.47), "OH": (0.0725, 0.44), "NY": (0.0800, 0.62),
    "WA": (0.0920, 0.51), "CO": (0.0640, 0.36), "GA": (0.0730, 0.40),
    "IL": (0.0825, 0.49), "PA": (0.0634, 0.43), "AZ": (0.0840, 0.39),
}
FOREIGN = {"MX": 0.42, "PE": 0.10, "CL": 0.10, "CO": 0.09, "AR": 0.08, "CA": 0.21}

NAME_BITS = [
    "Primary Anaerobic Digester", "Flow Equalization Basin", "MBBR Reactor",
    "Sludge Holding Tank", "Secondary Clarifier", "Aeration Basin",
    "Leachate Storage", "Fire Protection Tank", "Potable Water Standpipe",
    "Process Water Tank", "Lime Silo", "Backwash Tank", "Tank 1", "Tank 2",
    "Tank A", "Dual-Zone SBR - Hybrid", "Epoxy Lined Storage",
    "Geodesic Dome Cover", "Reaeration Contact Tank", "Thickener",
]

# Generator's own plate rules — deliberately not physics.py's.
_GEN_PLATE_WIDTH = 7.5
_GEN_STRESS = 21500.0
_GEN_MIN_T = [(45.0, 0.1875), (110.0, 0.25), (185.0, 0.3125), (1e9, 0.375)]


def _gen_shell_lb(D, H, rng):
    """Course-by-course steel weight under the generator's own design rule."""
    n = len(D)
    out = np.zeros(n)
    t_bot = np.zeros(n)
    for i in range(n):
        d, h = D[i], H[i]
        tmin = next(t for lim, t in _GEN_MIN_T if d < lim)
        z, lb = 0.0, 0.0
        first = None
        while z < h:
            w = min(_GEN_PLATE_WIDTH, h - z)
            head = max(h - z - 1.0, 0.0)
            t = max(2.6 * d * head / (_GEN_STRESS * 0.88), tmin)
            t = np.ceil(t * 32.0) / 32.0          # generator buys in 1/32"
            lb += np.pi * d * w * t * 40.833
            if first is None:
                first = t
            z += _GEN_PLATE_WIDTH
        out[i] = lb
        t_bot[i] = first or tmin
    return out, t_bot


def generate(n_quotes=3018, seed=0):
    """Build an archive-shaped frame. ~6,900 rows across ~3,000 quotes."""
    rng = np.random.default_rng(seed)

    # --- tank geometry: lognormal, with a heavy tail of big jobs -------------
    D = np.clip(np.exp(rng.normal(3.45, 0.42, n_quotes)), 8, 220).round(0)
    H = np.clip(np.exp(rng.normal(3.30, 0.34, n_quotes)), 8, 75).round(0)
    # Tall tanks tend to be narrow; break the independence.
    H = np.clip(H * (1.0 + 0.18 * (np.log(D) - 3.45) * -1), 8, 80).round(0)

    material = rng.choice(list(MATERIALS), n_quotes, p=MATERIAL_P)
    use = rng.choice(USE_TYPES, n_quotes, p=USE_P)
    deck = rng.choice(DECKS, n_quotes, p=DECK_P)
    floor = rng.choice(FLOORS, n_quotes, p=FLOOR_P)
    wage = rng.choice(WAGES, n_quotes, p=WAGE_P)
    name = rng.choice(NAME_BITS, n_quotes)

    is_us = rng.random(n_quotes) < 0.86
    st = rng.choice(list(STATES), n_quotes, p=_norm([1.0] * len(STATES)))
    country = np.where(is_us, "US", rng.choice(list(FOREIGN), n_quotes,
                                               p=_norm(list(FOREIGN.values()))))
    state = np.where(is_us, st, "")

    days = rng.integers(0, 1096, n_quotes)
    due = pd.Timestamp("2023-12-01") + pd.to_timedelta(days, unit="D")
    years = days / 365.25

    miles = np.clip(rng.gamma(2.2, 260, n_quotes), 5, 4200).round(0)
    Ss = np.clip(rng.gamma(1.6, 0.16, n_quotes), 0.02, 2.4).round(3)
    S1 = np.clip(Ss * rng.uniform(0.28, 0.62, n_quotes), 0.01, 1.2).round(3)
    freeboard = rng.choice([0, 6, 12, 18, 24], n_quotes, p=[.14, .52, .18, .09, .07])
    qty = rng.choice([1, 2, 3, 4, 5, 6], n_quotes, p=[.85, .08, .03, .02, .01, .01])

    # --- scope: commercial decisions, only loosely tied to the tank ----------
    is_con = (rng.random(n_quotes) < 0.684).astype(int)
    is_ins = (rng.random(n_quotes) < 0.230).astype(int)
    is_ins_e = (is_ins & (rng.random(n_quotes) < 0.946)).astype(int)
    is_frt = (rng.random(n_quotes) < 0.822).astype(int)
    # Foreign buyers usually arrange their own export shipping.
    is_frt = np.where(country == "MX", (rng.random(n_quotes) < 0.43).astype(int), is_frt)
    is_frt = np.where(np.isin(country, ["PE", "CL", "CO"]),
                      (rng.random(n_quotes) < 0.12).astype(int), is_frt)
    taxed_p = np.array([STATES[s][1] if s in STATES else 0.0 for s in state])
    is_tax = ((rng.random(n_quotes) < taxed_p) & is_us).astype(int)

    # --- cost build-up ------------------------------------------------------
    shell_lb, t_bot = _gen_shell_lb(D, H, rng)
    floor_area = np.pi * (D / 2) ** 2
    shell_area = np.pi * D * H
    open_top = deck == "Open-Top"
    roof_lb = np.where(open_top, 0.0, floor_area * 0.22 * 40.833 * 1.9)
    steel_lb = shell_lb + floor_area * 0.25 * 40.833 + roof_lb

    esc = 1.0 + 0.052 * years                      # ~5% a year on steel
    mat_rate = np.array([MATERIALS[m] for m in material]) * 1.42 * esc
    material_price = steel_lb * mat_rate * rng.lognormal(0, 0.11, n_quotes)

    # Fabrication tracks weld length and plate handling, not weight.
    courses = np.ceil(H / _GEN_PLATE_WIDTH)
    weld_ft = np.pi * D * courses + H * np.ceil(np.pi * D / 8.0)
    fab_hours = weld_ft * 0.085 + steel_lb * 0.0022
    fab_rate = 96.0 * (1.0 + 0.038 * years)
    fab_price = fab_hours * fab_rate * rng.lognormal(0, 0.13, n_quotes)

    # Erection: crane class steps with height, wage regime scales labour.
    crane = 1.0 + 0.30 * (H > 32) + 0.42 * (H > 52) + 0.55 * (H > 70)
    wmul = np.array([WAGE_MULT[w] for w in wage])
    seis = 1.0 + 0.13 * Ss
    con_price = (shell_area * 9.4 * crane * wmul * seis
                 * (1.0 + 0.041 * years) * rng.lognormal(0, 0.17, n_quotes)) * is_con

    ins_area = shell_area + np.where(open_top, 0.0, floor_area)
    ins_mat = ins_area * 11.8 * (1.0 + 0.030 * years) * rng.lognormal(0, 0.15, n_quotes) * is_ins
    ins_con = ins_area * 8.6 * wmul * rng.lognormal(0, 0.18, n_quotes) * is_ins_e

    frt = (steel_lb / 2000.0 * miles * 0.185 + 1750.0) \
        * rng.lognormal(0, 0.21, n_quotes) * is_frt

    # Process descriptor carries appendages that are in no column.
    bump = np.ones(n_quotes)
    for kw, mult in [("Digester", 1.35), ("SBR", 1.22), ("MBBR", 1.18),
                     ("Clarifier", 1.14), ("Equalization", 0.86),
                     ("Silo", 0.92), ("Tank ", 0.97), ("Dome", 1.16)]:
        bump = np.where(np.char.find(name.astype(str), kw) >= 0, mult, bump)
    material_price *= bump
    fab_price *= bump

    comps = np.column_stack([material_price, fab_price, con_price,
                             ins_mat, ins_con, frt])
    total = comps.sum(axis=1)
    rate = np.array([STATES[s][0] if s in STATES else 0.0 for s in state])
    tax = (total - frt) * rate * is_tax
    proposal = (total - frt) + tax

    q = pd.DataFrame({
        "Quote #": [f"Q{100000 + i}" for i in range(n_quotes)],
        "Tank Name": name,
        "Due Date": due.strftime("%m/%d/%Y"),
        "Country": country, "State": state,
        "Diameter (ft)": D, "Height (ft)": H, "Freeboard (in)": freeboard,
        "Quantity": qty, "Material": material, "Use Type": use,
        "Deck Style": deck, "Floor Style": floor, "Wage Type": wage,
        "Bid Type": rng.choice(["Firm", "Budget"], n_quotes, p=[0.71, 0.29]),
        "Sales Manager": rng.choice([f"SM{i}" for i in range(9)], n_quotes),
        "Miles to Site (From TBT)": miles,
        "Ss": Ss, "S1": S1,
        "Status": rng.choice(["Deliver To Customer", "Revised", "Sales", "Won", "Lost"],
                             n_quotes, p=[0.485, 0.354, 0.114, 0.026, 0.021]),
        "Material Price": material_price.round(2),
        "Fabrication Price": fab_price.round(2),
        "Construction Price": con_price.round(2),
        "Insulation Material Price": ins_mat.round(2),
        "Insulation Construction Price": ins_con.round(2),
        "Freight Price": frt.round(2),
        "Total Price": total.round(2),
        "Total Tax": tax.round(2),
        "Proposal Total": proposal.round(2),
        "IS_CONSTRUCTION": np.where(is_con == 1, "Yes", "No"),
        "IS_INSULATION": np.where(is_ins == 1, "Yes", "No"),
        "IS_INSULATION_ERECTION": np.where(is_ins_e == 1, "Yes", "No"),
        "IS_FREIGHT": np.where(is_frt == 1, "Yes", "No"),
        "IS_TAXABLE": np.where(is_tax == 1, "Yes", "No"),
    })

    rows = [q]
    # Revisions: a re-quote of the same job, days later, at a nudged price. These
    # are the near-duplicates that make a random split report ~3% and mean
    # nothing — the reason every split in this package is by time and by quote.
    n_rev = int(round(n_quotes * 2.283)) - n_quotes
    idx = rng.choice(n_quotes, n_rev, replace=True)
    r = q.iloc[idx].copy().reset_index(drop=True)
    r["Due Date"] = (pd.to_datetime(r["Due Date"])
                     + pd.to_timedelta(rng.integers(3, 70, n_rev), unit="D")
                     ).dt.strftime("%m/%d/%Y")
    r["Status"] = "Revised"
    nudge = rng.lognormal(0, 0.045, n_rev)
    for c in ["Material Price", "Fabrication Price", "Construction Price",
              "Insulation Material Price", "Insulation Construction Price",
              "Freight Price", "Total Price", "Total Tax", "Proposal Total"]:
        r[c] = (r[c].values * nudge).round(2)
    rows.append(r)

    out = pd.concat(rows, ignore_index=True)

    # A slice of partial quotes with scope lines missing. The archive has 173 of
    # these; the $/sq-ft gate is what catches them.
    n_bad = int(round(len(out) * 0.025))
    bad = rng.choice(len(out), n_bad, replace=False)
    out.loc[bad, "Total Price"] = out.loc[bad, "Total Price"].values * rng.uniform(
        0.04, 0.16, n_bad)

    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _norm(v):
    v = np.asarray(v, dtype=float)
    return v / v.sum()


if __name__ == "__main__":
    import sys
    dst = sys.argv[1] if len(sys.argv) > 1 else "synthetic_archive.csv"
    df = generate()
    df.to_csv(dst, index=False)
    print(f"wrote {dst}: {len(df)} rows, {df['Quote #'].nunique()} quotes")
