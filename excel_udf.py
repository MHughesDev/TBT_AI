"""
xlwings bridge — v4.

Setup, once:
    pip install -r requirements.txt
    xlwings addin install
    # keep tbt_model.py + tbt_pricing_bundle.joblib + this file beside the
    # workbook, then: Excel ribbon -> xlwings -> Import Functions

Scope is required on every call: construction, insulation, freight, taxable.
Leave one blank and the function returns #SCOPE rather than a number. That is
deliberate — v4 prices scope, it does not guess it, and a silent default is the
same failure as a wrong guess.

Put the four scope answers in their own columns on the quote sheet, as Yes/No
data-validation dropdowns. They are the cheapest accuracy you will ever buy.

All functions are non-volatile. For more than a few dozen rows use batch_score.py;
Excel will crawl if it fires inference on every recalculation.
"""
import xlwings as xw
from tbt_model import predict, ScopeError

_TRUE = ("TRUE", "YES", "Y", "1")
_FALSE = ("FALSE", "NO", "N", "0")


def _yn(v):
    """Yes/No/TRUE/FALSE/1/0 -> bool. Blank -> None, which will raise downstream."""
    if v is None or v == "" or (isinstance(v, float) and v != v):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    t = str(v).strip().upper()
    return True if t in _TRUE else False if t in _FALSE else None


def _call(diameter, height, material, use_type, state, wage, floor, deck, miles,
          quantity, bid_type, constr, insul, insul_erect, freight, taxable):
    return predict(
        diameter=diameter, height=height, material=material, use_type=use_type,
        state=state, wage=wage, floor=floor, deck=deck, miles=miles,
        quantity=quantity, bid_type=bid_type,
        construction=_yn(constr), insulation=_yn(insul),
        insulation_erection=_yn(insul_erect), freight=_yn(freight),
        taxable=_yn(taxable))


@xw.func
@xw.arg("diameter", numbers=float)
@xw.arg("height", numbers=float)
def TBT_PRICE(diameter, height, constr, insul, freight, taxable,
              material="CS", use_type="Water Storage Tank", state="MO",
              wage="Non-Union / Non-Prevailing", floor="Flat Steel Floor",
              deck="", miles=200, quantity=1, bid_type="Firm", insul_erect=None):
    """Extended total, tax-exclusive, matching the Total Price column.
    Scope arguments come first because they are required."""
    try:
        return _call(diameter, height, material, use_type, state, wage, floor,
                     deck, miles, quantity, bid_type, constr, insul,
                     insul_erect, freight, taxable)["total"]
    except ScopeError as e:
        return f"#SCOPE {e}"
    except Exception as e:
        return f"#ERR {e}"


@xw.func
@xw.ret(expand="table")
def TBT_BREAKDOWN(diameter, height, constr, insul, freight, taxable,
                  material="CS", use_type="Water Storage Tank", state="MO",
                  wage="Non-Union / Non-Prevailing", floor="Flat Steel Floor",
                  deck="", miles=200, quantity=1, bid_type="Firm", insul_erect=None):
    """Spills the six components, totals, bands and warnings. This is what makes
    the estimate auditable — if the number looks wrong it shows which piece is
    wrong, which is the whole point of pricing components separately."""
    try:
        r = _call(diameter, height, material, use_type, state, wage, floor, deck,
                  miles, quantity, bid_type, constr, insul, insul_erect,
                  freight, taxable)
    except ScopeError as e:
        return [["SCOPE ERROR", str(e)]]
    c = r["components"]
    out = [[k.replace(" Price", ""), c[k]] for k in c]
    out += [["--", ""],
            ["Price per tank", r["unit_price"]],
            ["Extended total (ex tax)", r["total"]],
            ["Implied $/sq-ft", r["psf"]],
            ["Tax rate", r["tax_rate"]],
            ["Est. tax", r["est_tax"]],
            ["Proposal Total", r["proposal_total"]],
            ["80% low", r["p80_low"]],
            ["80% high", r["p80_high"]]]
    for w in r["warnings"]:
        out.append(["WARNING", w])
    return out


@xw.func
@xw.ret(expand="table")
def TBT_RANGE(diameter, height, constr, insul, freight, taxable,
              material="CS", use_type="Water Storage Tank", state="MO",
              wage="Non-Union / Non-Prevailing", floor="Flat Steel Floor",
              deck="", miles=200, quantity=1, bid_type="Firm", insul_erect=None):
    """80% conformal band: [low, estimate, high]."""
    try:
        r = _call(diameter, height, material, use_type, state, wage, floor, deck,
                  miles, quantity, bid_type, constr, insul, insul_erect,
                  freight, taxable)
        return [[r["p80_low"], r["total"], r["p80_high"]]]
    except ScopeError as e:
        return [[f"#SCOPE {e}", "", ""]]


@xw.func
def TBT_FLAG(quoted_price, diameter, height, constr, insul, freight, taxable,
             material="CS", use_type="Water Storage Tank", state="MO",
             wage="Non-Union / Non-Prevailing", floor="Flat Steel Floor",
             deck="", miles=200, quantity=1, bid_type="Firm", insul_erect=None):
    """Deploy this one first. Returns OK / LOW / HIGH against the 80% band with the
    gap. Catches transposed dimensions, missing scope lines and forgotten stainless
    premiums, without pretending to set prices."""
    try:
        r = _call(diameter, height, material, use_type, state, wage, floor, deck,
                  miles, quantity, bid_type, constr, insul, insul_erect,
                  freight, taxable)
        gap = (quoted_price - r["total"]) / r["total"]
        tag = ("LOW" if quoted_price < r["p80_low"] else
               "HIGH" if quoted_price > r["p80_high"] else "OK")
        note = " (large tank — model reads low)" if any(
            "top-5%" in w for w in r["warnings"]) else ""
        return f"{tag} {gap:+.0%}{note}"
    except ScopeError as e:
        return f"#SCOPE {e}"
    except Exception as e:
        return f"#ERR {e}"


@xw.func
def TBT_WORST_COMPONENT(quoted_material, quoted_fab, quoted_constr,
                        diameter, height, constr, insul, freight, taxable,
                        material="CS", use_type="Water Storage Tank", state="MO",
                        wage="Non-Union / Non-Prevailing",
                        floor="Flat Steel Floor", deck="", miles=200):
    """Given the estimator's own component figures, names the line that deviates
    most from the model. Turns 'this quote looks off' into 'check the construction
    line' — the practical payoff of pricing components separately."""
    try:
        r = _call(diameter, height, material, use_type, state, wage, floor, deck,
                  miles, 1, "Firm", constr, insul, None, freight, taxable)
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
    except ScopeError as e:
        return f"#SCOPE {e}"
    except Exception as e:
        return f"#ERR {e}"


@xw.func
def TBT_MODEL_INFO():
    """Version, training date and data currency. Worth parking in a corner of the
    sheet so nobody prices off a stale model without noticing."""
    try:
        r = predict(diameter=30, height=30, material="CS", state="MO",
                    construction=True, insulation=False, freight=True, taxable=False)
        return (f"{r['model_version']} — trained {r['model_trained']}, "
                f"data through {r['data_through']}")
    except Exception as e:
        return f"#ERR {e}"
