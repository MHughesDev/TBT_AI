"""The public scoring API and every guard rail.

SPEC 2.9 (edges of competence), 6.5 (negative acceptance), 7.4 (tiers),
7.5 (signatures).

The governing rule: the failure mode outside competence must be VISIBLE on the
sheet, never a plausible-looking number.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from . import config as C
from . import conformal as CF
from . import features as F
from . import model as M
from .contract import (Check, Estimate, ModelInfo, QuoteInput, RefusalError,
                       TOKEN_MODEL, TOKEN_QTY, TOKEN_RANGE, TOKEN_SCOPE,
                       TOKEN_SCOPE_NEST, TOKEN_SCOPE_WAGE,
                       TOKEN_UNKNOWN_MATERIAL, TOKEN_UNKNOWN_USETYPE,
                       WARN_AR_BIAS, WARN_BIG, WARN_EXTRAP, WARN_GEO,
                       WARN_STALE, WARN_STALE_HARD, WARN_TAXRATE, WARN_UNSEEN,
                       WARN_WAGE)

_TRUE = {"yes", "y", "true", "t", "1", 1, 1.0, True}
_FALSE = {"no", "n", "false", "f", "0", 0, 0.0, False}


def as_bool(v, field: str) -> bool:
    """Strict. A blank is a refusal, never a default. SPEC 2.4 / 8.1 R1."""
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        raise RefusalError(TOKEN_SCOPE, f"{field} is blank")
    k = v.strip().lower() if isinstance(v, str) else v
    if k in _TRUE:
        return True
    if k in _FALSE:
        return False
    raise RefusalError(TOKEN_SCOPE, f"{field}={v!r} is not yes/no")


def validate_input(q: QuoteInput, bundle: M.Bundle) -> list[str]:
    """Run every refusal check, then return the warning codes. SPEC 2.9."""
    warnings: list[str] = []

    # --- scope (tier 1) -----------------------------------------------------
    scope = {}
    for name in ("construction", "insulation", "insulation_erection", "freight",
                 "taxable"):
        scope[name] = as_bool(getattr(q, name), name)
    if scope["insulation_erection"] and not scope["insulation"]:
        raise RefusalError(TOKEN_SCOPE_NEST,
                           "insulation_erection=True requires insulation=True")
    if (q.wage_type is not None
            and str(q.wage_type).strip() == F.NO_ERECTION
            and scope["construction"]):
        raise RefusalError(TOKEN_SCOPE_WAGE,
                           "wage_type says no erection but construction=True")

    # --- quantity -----------------------------------------------------------
    try:
        qty = float(q.quantity)
    except (TypeError, ValueError):
        raise RefusalError(TOKEN_QTY, f"quantity={q.quantity!r}")
    if not np.isfinite(qty) or qty < 1 or abs(qty - round(qty)) > 1e-9:
        raise RefusalError(TOKEN_QTY, f"quantity={q.quantity!r}")

    # --- geometry -----------------------------------------------------------
    try:
        D, H = float(q.diameter_ft), float(q.height_ft)
    except (TypeError, ValueError):
        raise RefusalError(TOKEN_RANGE, "diameter/height not numeric")
    if not (np.isfinite(D) and np.isfinite(H)) or D <= 0 or H <= 0:
        raise RefusalError(TOKEN_RANGE, f"diameter={q.diameter_ft}, height={q.height_ft}")

    (dlo, dhi) = bundle.envelope.get("D", (0.0, np.inf))
    (hlo, hhi) = bundle.envelope.get("H", (0.0, np.inf))
    hard = lambda lo, hi: (0.5 * lo, 1.5 * hi)
    for val, (lo, hi), nm in ((D, (dlo, dhi), "diameter"), (H, (hlo, hhi), "height")):
        h_lo, h_hi = hard(lo, hi)
        if val < h_lo or val > h_hi:
            raise RefusalError(TOKEN_RANGE,
                               f"{nm}={val} outside [{h_lo:.1f}, {h_hi:.1f}]")
        if val < lo or val > hi:
            warnings.append(WARN_EXTRAP)

    # --- vocabulary ---------------------------------------------------------
    mat_vocab = bundle.encoder.vocab.get("Material", [])
    if mat_vocab and str(q.material).strip() not in mat_vocab:
        raise RefusalError(TOKEN_UNKNOWN_MATERIAL, str(q.material))
    use_vocab = bundle.encoder.vocab.get("Use Type", [])
    if use_vocab and str(q.use_type).strip() not in use_vocab:
        raise RefusalError(TOKEN_UNKNOWN_USETYPE, str(q.use_type))

    # --- degrade gracefully (tier 3) ---------------------------------------
    for field_name, col in (("deck_style", "Deck Style"), ("floor_style", "Floor Style"),
                            ("bid_type", "Bid Type"), ("sales_manager", "Sales Manager")):
        v = getattr(q, field_name)
        if v is None:
            continue
        vocab = bundle.encoder.vocab.get(col, [])
        if vocab and str(v).strip() and str(v).strip() not in vocab:
            warnings.append(f"{WARN_UNSEEN}:{col}")

    for field_name, col in (("state", "State"), ("country", "Country")):
        v = getattr(q, field_name)
        if v is None or not str(v).strip():
            continue
        vocab = bundle.encoder.vocab.get(col, [])
        if vocab and str(v).strip() not in vocab:
            warnings.append(WARN_GEO)

    if str(q.country or "").strip().upper() in {"AR", "ARGENTINA"}:
        warnings.append(WARN_AR_BIAS)

    if scope["construction"] and (q.wage_type is None or not str(q.wage_type).strip()):
        warnings.append(WARN_WAGE)

    # --- staleness ----------------------------------------------------------
    if bundle.trained_through is not None:
        age = (date.today() - bundle.trained_through).days
        if age > C.STALE_HARD_DAYS:
            warnings.append(WARN_STALE_HARD)
        elif age > C.STALE_DAYS:
            warnings.append(WARN_STALE)

    return list(dict.fromkeys(warnings))


def tier_of(half_width: float, warnings: list[str]) -> str:
    """SPEC 7.4. Derived, not fitted."""
    serious = {WARN_EXTRAP, WARN_BIG, WARN_GEO, WARN_AR_BIAS, WARN_STALE_HARD}
    has_serious = any(w.split(":")[0] in serious for w in warnings)
    if not np.isfinite(half_width):
        return "C"
    if half_width <= 0.12 and not warnings:
        return "A"
    if half_width <= 0.20 and not has_serious:
        return "B"
    return "C"


def _require_bundle(bundle) -> M.Bundle:
    if bundle is None:
        raise RefusalError(TOKEN_MODEL, "no bundle loaded")
    if not isinstance(bundle, M.Bundle):
        raise RefusalError(TOKEN_MODEL, "bundle is not a Bundle")
    import sklearn
    want = ".".join(bundle.sklearn_version.split(".")[:2]) if bundle.sklearn_version else ""
    have = ".".join(sklearn.__version__.split(".")[:2])
    if want and want != have:
        raise RefusalError(TOKEN_MODEL,
                           f"bundle built on scikit-learn {bundle.sklearn_version}, "
                           f"running {sklearn.__version__}")
    return bundle


def estimate(q: QuoteInput, bundle=None) -> Estimate:
    """Score one tank. Raises RefusalError rather than returning a bad number."""
    b = _require_bundle(bundle)
    warnings = validate_input(q, b)

    row = q.to_row()
    if row["Due Date"] is None:
        row["Due Date"] = date.today()
    df = pd.DataFrame([row])

    preds = M.predict_frame(b, df)
    point = float(preds["point"].iloc[0])

    is_big = point >= b.big_threshold
    if is_big:
        warnings.append(WARN_BIG)
    fam = F.use_family_of(q.use_type)
    group = CF.group_of(fam, is_big)

    smear = b.smear.get("top5" if is_big else "rest", 1.0)
    book = point * smear

    (lo80, hi80), (lo90, hi90) = CF.bands_for(b.conformal, group, point)
    half_width = (hi80 / point - 1.0) if np.isfinite(hi80) and point > 0 else float("nan")

    comps = {}
    raw_sum = float(preds["components_sum"].iloc[0])
    scale = (point / raw_sum) if raw_sum > 0 else 1.0
    for comp in C.COMPONENTS:
        key = "est_" + comp.replace(" Price", "")
        comps[comp] = float(preds[key].iloc[0]) * scale

    taxable = as_bool(q.taxable, "taxable")
    has_rate = bool(preds["has_tax_rate"].iloc[0])
    if taxable and not has_rate:
        warnings.append(WARN_TAXRATE)
        tax = None
        proposal = None
    else:
        tax = float(preds["est_tax"].iloc[0]) * scale
        proposal = float(sum(v for k, v in comps.items() if k != "Freight Price")) + tax

    tier = tier_of(half_width, warnings)

    return Estimate(
        point=point, book=book,
        band80=(lo80, hi80), band90=(lo90, hi90),
        components=comps, tax=tax, proposal_total=proposal,
        extended_point=point * float(q.quantity),
        tier=tier, group=group,
        warnings=list(dict.fromkeys(warnings)),
        bundle_id=b.bundle_id, trained_through=b.trained_through,
    )


def check(q: QuoteInput, quoted_total: float,
          quoted_components: dict | None = None, bundle=None) -> Check:
    """Phase 1 surface. SPEC 7.3.

    The gap is expressed as a percentage of the ESTIMATOR's own number, and the
    model's price is not part of the return: it is not on the sheet, and not
    being on the sheet is what matters for anchoring (SPEC 8.1 R14).
    """
    est = estimate(q, bundle=bundle)
    qt = float(quoted_total)
    if qt <= 0:
        raise RefusalError(TOKEN_RANGE, "quoted_total must be positive")

    lo, hi = est.band80
    if np.isfinite(lo) and qt < lo:
        flag = "LOW"
    elif np.isfinite(hi) and qt > hi:
        flag = "HIGH"
    else:
        flag = "OK"
    gap = (qt - est.point) / qt * 100.0

    reasons: list[tuple[str, float]] = []
    if quoted_components:
        for comp, model_v in est.components.items():
            given = quoted_components.get(comp)
            if given is None or float(given) <= 0:
                continue
            reasons.append((comp.replace(" Price", ""),
                            (float(given) - model_v) / float(given) * 100.0))
        reasons.sort(key=lambda t: -abs(t[1]))
        reasons = reasons[:2]

    return Check(flag=flag, gap_pct=gap, reasons=reasons, tier=est.tier,
                 warnings=est.warnings)


def estimate_many(df: pd.DataFrame, bundle=None) -> pd.DataFrame:
    """Vectorised scoring. Refusals become an `error` column, never an
    exception, so one bad row cannot fail a batch. SPEC 7.5."""
    b = _require_bundle(bundle)
    d = df.copy()

    errors = pd.Series([None] * len(d), index=d.index, dtype=object)
    warn_col = pd.Series([""] * len(d), index=d.index, dtype=object)

    # scope must be present and valid on every row
    for flag, field_name in (("IS_CONSTRUCTION", "construction"),
                             ("IS_INSULATION", "insulation"),
                             ("IS_INSULATION_ERECTION", "insulation_erection"),
                             ("IS_FREIGHT", "freight"),
                             ("IS_TAXABLE", "taxable")):
        if flag not in d.columns:
            d[flag] = None
        vals, errs = [], []
        for v in d[flag]:
            try:
                vals.append(int(as_bool(v, field_name)))
                errs.append(None)
            except RefusalError as e:
                vals.append(0)
                errs.append(e.code)
        d[flag] = vals
        errors = errors.where(errors.notna(), pd.Series(errs, index=d.index))

    nest = (d["IS_INSULATION_ERECTION"] == 1) & (d["IS_INSULATION"] == 0)
    errors = errors.where(~(nest & errors.isna()), TOKEN_SCOPE_NEST)

    D = pd.to_numeric(d.get("Diameter (ft)"), errors="coerce")
    H = pd.to_numeric(d.get("Height (ft)"), errors="coerce")
    bad_geom = ~((D > 0) & (H > 0))
    (dlo, dhi) = b.envelope.get("D", (0.0, np.inf))
    (hlo, hhi) = b.envelope.get("H", (0.0, np.inf))
    outside = (D < 0.5 * dlo) | (D > 1.5 * dhi) | (H < 0.5 * hlo) | (H > 1.5 * hhi)
    errors = errors.where(~((bad_geom | outside) & errors.isna()), TOKEN_RANGE)
    extrap = ((D < dlo) | (D > dhi) | (H < hlo) | (H > hhi)).fillna(False)

    qty = pd.to_numeric(d.get("Quantity", 1), errors="coerce").fillna(1)
    bad_qty = (qty < 1) | ((qty - qty.round()).abs() > 1e-9)
    errors = errors.where(~(bad_qty & errors.isna()), TOKEN_QTY)

    mat_vocab = b.encoder.vocab.get("Material", [])
    if mat_vocab and "Material" in d.columns:
        bad = ~d["Material"].astype(object).map(
            lambda v: str(v).strip() in mat_vocab if v is not None else False)
        errors = errors.where(~(bad & errors.isna()), TOKEN_UNKNOWN_MATERIAL)
    use_vocab = b.encoder.vocab.get("Use Type", [])
    if use_vocab and "Use Type" in d.columns:
        bad = ~d["Use Type"].astype(object).map(
            lambda v: str(v).strip() in use_vocab if v is not None else False)
        errors = errors.where(~(bad & errors.isna()), TOKEN_UNKNOWN_USETYPE)

    ok = errors.isna()
    out = pd.DataFrame(index=d.index)
    out["error"] = errors

    if ok.any():
        preds = M.predict_frame(b, d.loc[ok])
        eng = F.engineer(d.loc[ok], use_family_map=b.use_family_map)
        point = preds["point"].values
        is_big = point >= b.big_threshold
        fam = eng["use_family"].values
        groups = np.array([CF.group_of(f, g) for f, g in zip(fam, is_big)])
        smear = np.array([b.smear.get("top5" if g else "rest", 1.0) for g in is_big])

        lo = np.empty(len(point)); hi = np.empty(len(point))
        for i, g in enumerate(groups):
            (l, h), _ = CF.bands_for(b.conformal, g, point[i])
            lo[i], hi[i] = l, h
        hw = np.where(point > 0, hi / point - 1.0, np.nan)

        wl = []
        idx = list(d.index[ok])
        for i, ix in enumerate(idx):
            w = []
            if bool(extrap.get(ix, False)):
                w.append(WARN_EXTRAP)
            if is_big[i]:
                w.append(WARN_BIG)
            if not bool(preds["has_tax_rate"].iloc[i]) and d.loc[ix, "IS_TAXABLE"] == 1:
                w.append(WARN_TAXRATE)
            wl.append(",".join(w))

        out.loc[ok, "point"] = point
        out.loc[ok, "book"] = point * smear
        out.loc[ok, "band80_lo"] = lo
        out.loc[ok, "band80_hi"] = hi
        out.loc[ok, "group"] = groups
        out.loc[ok, "extended_point"] = point * qty[ok].values
        out.loc[ok, "tier"] = [tier_of(h, w.split(",") if w else [])
                               for h, w in zip(hw, wl)]
        out.loc[ok, "warnings"] = wl
        raw_sum = preds["components_sum"].values
        scale = np.where(raw_sum > 0, point / np.where(raw_sum > 0, raw_sum, 1.0), 1.0)
        for comp in C.COMPONENTS:
            key = "est_" + comp.replace(" Price", "")
            out.loc[ok, key] = preds[key].values * scale
        out.loc[ok, "est_tax"] = preds["est_tax"].values * scale
        out.loc[ok, "implied_psf"] = point / eng["shell_area"].values

    return out


def model_info(bundle, last_retrain_status: str = "ok", ledger=None) -> ModelInfo:
    b = _require_bundle(bundle)
    age = (date.today() - b.trained_through).days if b.trained_through else 9999
    alarms = list(b.smear.get("alarm", []))
    if age > C.STALE_HARD_DAYS:
        alarms.append(WARN_STALE_HARD)
    elif age > C.STALE_DAYS:
        alarms.append(WARN_STALE)
    return ModelInfo(
        bundle_id=b.bundle_id, trained_through=b.trained_through, age_days=age,
        last_retrain_status=last_retrain_status,
        s_rest=float(b.smear.get("rest", 1.0)), s_top5=float(b.smear.get("top5", 1.0)),
        n_train_rows=b.n_train_rows, bands=b.conformal, alarms=alarms,
    )
