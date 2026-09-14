"""Single-quote prediction — the interface an estimator or a spreadsheet calls.

Scope is required and is never defaulted. Omit any of the four answers and this
raises ScopeError rather than returning a number. That is not friction for its
own sake: whether a quote includes erection, insulation, freight or tax is a
commercial decision the estimator already knows, it is not a property of the
tank, and a scope guessed wrong changes the price with nothing visible on the
sheet to show it. A `#SCOPE` error costs one question and stops a wrong number
going out the door.
"""
import numpy as np
import pandas as pd

from . import features
from .config import BUNDLE, COMPONENTS, PSF_LO, PSF_HI, VERSION
from .data import ScopeError
from .model import TBT5

ALIAS = {
    "diameter": "Diameter (ft)", "height": "Height (ft)",
    "freeboard": "Freeboard (in)", "material": "Material",
    "use_type": "Use Type", "deck": "Deck Style", "floor": "Floor Style",
    "wage": "Wage Type", "country": "Country", "state": "State",
    "bid_type": "Bid Type", "sales_manager": "Sales Manager",
    "miles": "Miles to Site (From TBT)", "quantity": "Quantity",
    "ss": "Ss", "s1": "S1", "due_date": "Due Date",
    "tank_name": "Tank Name", "name": "Tank Name",
}

_CACHE = {}


def _bundle(path=BUNDLE):
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = TBT5.load(path)
    return _CACHE[key]


def predict(construction=None, insulation=None, insulation_erection=None,
            freight=None, taxable=None, objective=None, bundle=BUNDLE, **kw):
    """Price one tank.

    construction, insulation, freight and taxable must each be True or False.
    insulation_erection defaults to insulation — if we supply it we usually
    install it, 1,456 of 1,539 archive cases — so pass False explicitly for
    supply-only jobs.

    tank_name is optional and worth supplying. The process descriptor is the only
    proxy available for appendages that appear in no column: a digester has
    mixers, covers and gas handling; an equalization basin is a bare shell.

    objective selects the decision rule and defaults to the one the bundle was
    calibrated with:
        "mape"     minimise per-quote percentage error   (the default)
        "median"   the conditional median
        "unbiased" the conditional mean — use this, and only this, when the
                   predictions are going to be summed, as in valuing a backlog
    """
    # Scope is validated before the bundle is touched. A missing or unreadable
    # model must not mask a missing scope answer: the caller needs to be told
    # which of the two is wrong, and the scope check does not need a fitted
    # model to run.
    if insulation_erection is None:
        insulation_erection = insulation
    given = {"construction": construction, "insulation": insulation,
             "freight": freight, "taxable": taxable}
    absent = [k for k, v in given.items() if v is None]
    if absent:
        raise ScopeError(
            "scope not supplied: " + ", ".join(absent) + ". Each must be True or "
            "False. v5 does not infer scope — whether a quote includes erection, "
            "insulation, freight or tax is a commercial decision, not a property "
            "of the tank.")
    if insulation_erection and not insulation:
        raise ScopeError("insulation_erection=True requires insulation=True. We "
                         "cannot install insulation we are not supplying.")

    m = _bundle(bundle)

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
    for c in ("Diameter (ft)", "Height (ft)"):
        if row.get(c) is None:
            raise ValueError(f"{c} is required")

    d = features.build(pd.DataFrame([row]))
    d["plausible"] = True
    out = m.estimate(d, objective=objective)

    unit = float(out["estimate"][0])
    sigma = float(out["sigma"][0])
    qty = float(row.get("Quantity", 1) or 1)
    parts = {c: float(v[0]) for c, v in out["components"].items()}
    comp_sum = sum(parts.values())
    freight_part = parts.get("Freight Price", 0.0)

    rate = m.tax_.get(str(row.get("State", "")), 0.0) if taxable else 0.0
    ex_freight = unit - (freight_part if comp_sum > 0 else 0.0)

    area = float(d["shell_area"].iloc[0])
    psf = unit / area if area > 0 else np.nan

    warn = []
    for c, label in [("Diameter (ft)", "diameter"), ("Height (ft)", "height")]:
        lo, hi = m.limits_[c]
        if not lo <= float(row[c]) <= hi:
            warn.append(f"{label} is outside the fitted range ({lo:.0f}-{hi:.0f} ft)")
    if area > m.limits_["shell_area_max"]:
        warn.append("larger than anything in the training data — extrapolating "
                    "on the parametric backbone")
    if unit >= m.limits_["big_threshold"]:
        warn.append("top-5% by value: check this one against an estimator. v4 "
                    "underpriced this band by 13-17% in aggregate; v5's physics "
                    "features target that directly but it has not been measured "
                    "on the real archive")
    if not PSF_LO <= psf <= PSF_HI:
        warn.append(f"implied ${psf:.0f}/sq-ft is outside the plausible range "
                    f"({PSF_LO:.0f}-{PSF_HI:.0f})")
    if int(d["tn_generic"].iloc[0]) == 1 or not str(row.get("Tank Name", "")).strip():
        warn.append("no descriptive tank name supplied — process keywords "
                    "(digester, MBBR, equalization, silo) materially improve the "
                    "estimate on engineered-to-order tanks")
    if taxable and rate == 0.0:
        warn.append(f"marked taxable but no rate on file for state "
                    f"'{row.get('State')}' — tax shown as zero")
    # Ask the learner, not the input frame. features.build() creates cmp_logpsf
    # as NaN like every other absent numeric column; the real values are attached
    # inside the model, so testing the input frame reported "no comparables" on
    # every single call.
    cmp_pred = out["learners"].get("comparables")
    if cmp_pred is not None and not np.isfinite(cmp_pred[0]):
        warn.append("no close comparables in the archive — this spec is unlike "
                    "anything quoted before")
    if sigma > 0.30:
        warn.append(f"wide conditional spread (sigma {sigma:.2f}) — the model is "
                    "unsure about this combination of specs")

    obj = objective or m.objective
    if obj == "mape":
        warn.append("objective=mape leans low by design; use objective='unbiased' "
                    "if these numbers are going to be summed")

    return {
        "unit_price": round(unit, 0),
        "total": round(unit * qty, 0),
        "p80_low": round(unit * qty / m.bands_[80], 0),
        "p80_high": round(unit * qty * m.bands_[80], 0),
        "p90_low": round(unit * qty / m.bands_[90], 0),
        "p90_high": round(unit * qty * m.bands_[90], 0),
        "components": {k: round(v, 0) for k, v in parts.items()},
        "learners": {k: round(float(v[0]), 0) for k, v in out["learners"].items()
                     if np.isfinite(v[0])},
        "blend_weights": m.blend_,
        "scope": {"construction": bool(construction), "insulation": bool(insulation),
                  "insulation_erection": bool(insulation_erection),
                  "freight": bool(freight), "taxable": bool(taxable)},
        "tax_rate": round(rate, 4),
        "est_tax": round(ex_freight * rate, 0),
        "proposal_total": round(ex_freight * (1 + rate), 0),
        "psf": round(psf, 1),
        "sigma": round(sigma, 3),
        "objective": obj,
        "warnings": warn,
        "model_version": m.meta_.get("version", VERSION),
        "model_trained": m.meta_.get("trained"),
        "data_through": m.meta_.get("data_through"),
    }


def flag(quoted, tolerance=0.15, **kw):
    """OK / LOW / HIGH against a quoted price — the output to deploy first.

    At single-digit mean error the model cannot set prices, but it reliably
    catches a transposed dimension, a missing scope line or a forgotten stainless
    premium. Real money at almost no risk, and it builds the track record you
    would need before trusting anything further.
    """
    r = predict(**kw)
    est = r["unit_price"]
    if not est:
        return "#ERR"
    gap = (float(quoted) - est) / est
    tag = "OK" if abs(gap) <= tolerance else ("LOW" if gap < 0 else "HIGH")
    return f"{tag} {gap * 100:+.0f}%"
