"""
xlwings bridge — v3.

Setup, once, on the ThinkPad:
    pip install pandas scikit-learn joblib openpyxl xlwings
    xlwings addin install
    # keep this file, tbt_model.py and tbt_pricing_model.joblib beside the
    # workbook, then: Excel ribbon -> xlwings -> Import Functions

Pass the three scope flags whenever the estimator knows the scope — it is worth
about 2 points of accuracy. Leave them blank and the model infers scope itself.

All functions are non-volatile, so they recalc only when their own inputs change.
For more than a few dozen rows use batch_score.py; Excel will crawl if it fires
inference on every recalculation.

TBT_FLAG is the one to deploy first. It compares a human-built quote against the
band rather than replacing the estimator, which is the right level of trust for a
model with ~9% mean error.
"""
import xlwings as xw
from tbt_model import predict


def _flag(v):
    """Excel blank -> None (let the model infer). TRUE/FALSE/1/0 -> bool."""
    if v is None or v == "" or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, str):
        return v.strip().upper() in ("TRUE", "YES", "Y", "1")
    return bool(v)


def _call(diameter, height, material, use_type, state, wage, floor, deck,
          miles, quantity, constr, insul, freight, bid_type):
    return predict(
        diameter=diameter, height=height, material=material, use_type=use_type,
        state=state, wage=wage, floor=floor, deck=deck, miles=miles,
        quantity=quantity, bid_type=bid_type,
        has_construction=_flag(constr), has_insulation=_flag(insul),
        has_freight=_flag(freight))


@xw.func
@xw.arg("diameter", numbers=float)
@xw.arg("height", numbers=float)
def TBT_PRICE(diameter, height, material="CS", use_type="Water Storage Tank",
              state="MO", wage="Non-Union / Non-Prevailing",
              floor="Flat Steel Floor", deck="", miles=200, quantity=1,
              constr=None, insul=None, freight=None, bid_type="Firm"):
    """Extended total, tax-exclusive, matching the Total Price column.
    Note the price columns are per-tank; this multiplies by quantity for you."""
    try:
        return _call(diameter, height, material, use_type, state, wage, floor,
                     deck, miles, quantity, constr, insul, freight, bid_type)["total"]
    except Exception as e:
        return f"#ERR {e}"


@xw.func
def TBT_PRICE_PER_TANK(diameter, height, material="CS", use_type="Water Storage Tank",
                       state="MO", wage="Non-Union / Non-Prevailing",
                       floor="Flat Steel Floor", deck="", miles=200,
                       constr=None, insul=None, freight=None, bid_type="Firm"):
    """Single-tank price, ignoring order quantity."""
    try:
        return _call(diameter, height, material, use_type, state, wage, floor,
                     deck, miles, 1, constr, insul, freight, bid_type)["unit_price"]
    except Exception as e:
        return f"#ERR {e}"


@xw.func
@xw.ret(expand="table")
def TBT_BREAKDOWN(diameter, height, material="CS", use_type="Water Storage Tank",
                  state="MO", wage="Non-Union / Non-Prevailing",
                  floor="Flat Steel Floor", deck="", miles=200, quantity=1,
                  constr=None, insul=None, freight=None, bid_type="Firm"):
    """Spills the six components plus totals, bands and any warnings. This is what
    makes the estimate auditable — if the number looks wrong, it shows which piece
    is wrong, which is the whole point of decomposing the model."""
    r = _call(diameter, height, material, use_type, state, wage, floor, deck,
              miles, quantity, constr, insul, freight, bid_type)
    c = r["components"]
    out = [[k.replace(" Price", ""), c[k]] for k in c]
    out += [["--", ""],
            ["Price per tank", r["unit_price"]],
            ["Extended total (ex tax)", r["total"]],
            ["Implied $/sq-ft", r["psf"]],
            ["Est. tax", r["est_tax"]],
            ["Proposal Total", r["proposal_total"]],
            ["80% low", r["p80_low"]],
            ["80% high", r["p80_high"]]]
    for w in r["warnings"]:
        out.append(["WARNING", w])
    return out


@xw.func
@xw.ret(expand="table")
def TBT_RANGE(diameter, height, material="CS", use_type="Water Storage Tank",
              state="MO", wage="Non-Union / Non-Prevailing",
              floor="Flat Steel Floor", deck="", miles=200, quantity=1,
              constr=None, insul=None, freight=None, bid_type="Firm"):
    """80% conformal band: [low, estimate, high]."""
    r = _call(diameter, height, material, use_type, state, wage, floor, deck,
              miles, quantity, constr, insul, freight, bid_type)
    return [[r["p80_low"], r["total"], r["p80_high"]]]


@xw.func
def TBT_FLAG(quoted_price, diameter, height, material="CS",
             use_type="Water Storage Tank", state="MO",
             wage="Non-Union / Non-Prevailing", floor="Flat Steel Floor",
             deck="", miles=200, quantity=1,
             constr=None, insul=None, freight=None, bid_type="Firm"):
    """Deploy this one first. Returns OK / LOW / HIGH against the 80% band with the
    gap. Catches transposed dimensions, missing scope lines and forgotten stainless
    premiums, without pretending to set prices."""
    try:
        r = _call(diameter, height, material, use_type, state, wage, floor, deck,
                  miles, quantity, constr, insul, freight, bid_type)
        gap = (quoted_price - r["total"]) / r["total"]
        tag = ("LOW" if quoted_price < r["p80_low"] else
               "HIGH" if quoted_price > r["p80_high"] else "OK")
        suffix = " (large tank — model reads low)" if any(
            "top-5%" in w for w in r["warnings"]) else ""
        return f"{tag} {gap:+.0%}{suffix}"
    except Exception as e:
        return f"#ERR {e}"


@xw.func
def TBT_WORST_COMPONENT(quoted_material, quoted_fab, quoted_constr,
                        diameter, height, material="CS",
                        use_type="Water Storage Tank", state="MO",
                        wage="Non-Union / Non-Prevailing",
                        floor="Flat Steel Floor", deck="", miles=200):
    """Given the estimator's own component figures, names the line that deviates
    most from the model. Turns 'this quote looks off' into 'check the construction
    line' — the difference between a tool people use and one they ignore."""
    try:
        r = _call(diameter, height, material, use_type, state, wage, floor, deck,
                  miles, 1, True, None, None, "Firm")
        c = r["components"]
        pairs = [("Material", quoted_material, c["Material Price"]),
                 ("Fabrication", quoted_fab, c["Fabrication Price"]),
                 ("Construction", quoted_constr, c["Construction Price"])]
        worst, gap = None, 0.0
        for name, q, p in pairs:
            if not q or not p:
                continue
            g = (q - p) / p
            if abs(g) > abs(gap):
                worst, gap = name, g
        return f"{worst} {gap:+.0%}" if worst else "n/a"
    except Exception as e:
        return f"#ERR {e}"


@xw.func
def TBT_MODEL_INFO():
    """Returns when the model was trained and how current its data is. Worth
    putting in a corner of the sheet so nobody quotes off a stale model."""
    try:
        r = predict(diameter=30, height=30, material="CS", state="MO")
        return f"trained {r['model_trained']}, data through {r['data_through']}"
    except Exception as e:
        return f"#ERR {e}"
