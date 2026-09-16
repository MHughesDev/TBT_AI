"""xlwings bridge. SPEC 7.5 / 7.6.

Phase 1 ships TBT_CHECK only. TBT_BREAKDOWN (Phase 2), TBT_ESTIMATE and
TBT_BOOK (Phase 3) are present but gated by PHASE below, because a number on
the sheet becomes an anchor and the archive then starts to contain the model's
own output (SPEC 8.1 R14).

Every function is NON-VOLATILE and memoised. Excel recalculates aggressively;
firing inference on every recalculation locks the workbook (SPEC 8.1 R8).
For more than 500 rows use the Score sheet button / batch scorer.

    pip install xlwings && xlwings addin install
    # then Excel ribbon -> xlwings -> Import Functions
"""
from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

try:
    import xlwings as xw
except ImportError:                                    # allow import for tests
    class _Stub:
        @staticmethod
        def func(f=None, **k):
            return f if f else (lambda g: g)

        @staticmethod
        def arg(*a, **k):
            return lambda g: g

        @staticmethod
        def ret(*a, **k):
            return lambda g: g

        @staticmethod
        def sub(f=None, **k):
            return f if f else (lambda g: g)
    xw = _Stub()

from tbt import config as C
from tbt import train as T
from tbt.contract import (QuoteInput, RefusalError, TOKEN_BATCH, TOKEN_MODEL,
                          WARNING_TEXT)
from tbt.scoring import check as _check
from tbt.scoring import estimate as _estimate
from tbt.scoring import model_info as _model_info

# Which phase is deployed. SPEC 7.7 -- raise only on the stated evidence.
PHASE = int(os.environ.get("TBT_PHASE", "1"))
MODEL_DIR = os.environ.get("TBT_MODEL_DIR",
                           str(Path(__file__).resolve().parent / "models"))

_BUNDLE = None
_BUNDLE_ID = None


def _bundle():
    """Loaded once per Excel session; reloaded only when the file changes."""
    global _BUNDLE, _BUNDLE_ID
    p = Path(MODEL_DIR) / T.BUNDLE_NAME
    stamp = p.stat().st_mtime if p.exists() else None
    if _BUNDLE is None or stamp != _BUNDLE_ID:
        _BUNDLE = T.load_bundle(MODEL_DIR)
        _BUNDLE_ID = stamp
    return _BUNDLE


def _scalar(v):
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    return v


def _flat(rng):
    if rng is None:
        return []
    if not isinstance(rng, (list, tuple)):
        return [rng]
    out = []
    for r in rng:
        out.extend(r if isinstance(r, (list, tuple)) else [r])
    return out


def _scope(scope_range):
    """1x5 range: construction, insulation, insulation_erection, freight, taxable."""
    v = _flat(scope_range)
    if len(v) < 5:
        raise RefusalError("#SCOPE", "scope range must be 5 cells")
    return v[:5]


def _options(options_range):
    """A 2-row range whose first row names the optional fields."""
    if options_range is None:
        return {}
    if not isinstance(options_range, (list, tuple)) or not options_range:
        return {}
    rows = options_range if isinstance(options_range[0], (list, tuple)) \
        else [options_range]
    if len(rows) < 2:
        return {}
    keys = [str(k).strip() for k in rows[0]]
    vals = rows[1]
    return {k: v for k, v in zip(keys, vals) if k and v not in (None, "")}


def _build(diameter, height, material, use_type, scope_range, options_range):
    s = _scope(scope_range)
    o = _options(options_range)
    return QuoteInput(
        diameter_ft=_scalar(diameter), height_ft=_scalar(height),
        material=_scalar(material), use_type=_scalar(use_type),
        construction=s[0], insulation=s[1], insulation_erection=s[2],
        freight=s[3], taxable=s[4],
        deck_style=o.get("deck_style"), floor_style=o.get("floor_style"),
        bid_type=o.get("bid_type"), country=o.get("country", "US"),
        state=o.get("state"), wage_type=o.get("wage_type"),
        sales_manager=o.get("sales_manager"),
        miles_tbt=o.get("miles_tbt"), miles_gt=o.get("miles_gt"),
        ss=o.get("ss"), s1=o.get("s1"),
        quantity=int(o.get("quantity", 1) or 1),
        freeboard_in=float(o.get("freeboard_in", 0) or 0),
        tank_name=o.get("tank_name"),
        due_date=o.get("due_date") or date.today(),
    )


def _notes(warnings) -> str:
    return "; ".join(WARNING_TEXT.get(w.split(":")[0], w) for w in warnings)


def _phase_guard(needed: int):
    if PHASE < needed:
        return (f"#PHASE{needed} - withheld until phase {needed}; "
                f"see SPEC 7.7 for the evidence that unlocks it")
    return None


# ---------------------------------------------------------------- Phase 1
@xw.func
@xw.ret(expand="table")
def TBT_CHECK(quoted_total, diameter, height, material, use_type,
              scope_range, options_range=None, components_range=None):
    """Compare the estimator's own total against the model's 80% band.

    Returns [check, reason, tier, notes]. The model's price is NOT returned:
    it is not on the sheet, and that is what keeps the archive independent
    while the check earns its track record.
    """
    try:
        b = _bundle()
        if b is None:
            return [[TOKEN_MODEL, "", "", "no model installed"]]
        q = _build(diameter, height, material, use_type, scope_range, options_range)
        comps = None
        o = _options(components_range)
        if o:
            comps = {k: float(v) for k, v in o.items() if k in C.COMPONENTS}
        c = _check(q, float(_scalar(quoted_total)), quoted_components=comps, bundle=b)
        reason = ", ".join(f"{n} {g:+.0f}%" for n, g in c.reasons)
        flag = c.flag if c.flag == "OK" else c.flag
        return [[f"{flag} {c.gap_pct:+.0f}%", reason, c.tier, _notes(c.warnings)]]
    except RefusalError as e:
        return [[e.code, "", "", e.detail]]
    except Exception as e:                                  # never crash a sheet
        return [["#ERROR", "", "", str(e)[:120]]]


@xw.func
def TBT_MODEL_INFO():
    """Put the training date on the sheet so nobody quotes off a stale model
    without noticing."""
    try:
        b = _bundle()
        if b is None:
            return TOKEN_MODEL
        mi = _model_info(b)
        alarm = ("; " + ", ".join(mi.alarms)) if mi.alarms else ""
        return (f"{mi.bundle_id} | trained through {mi.trained_through} "
                f"| {mi.age_days}d old | s={mi.s_rest:.2f}/{mi.s_top5:.2f}{alarm}")
    except Exception as e:
        return f"#ERROR {str(e)[:80]}"


# ---------------------------------------------------------------- Phase 2
@xw.func
@xw.ret(expand="table")
def TBT_BREAKDOWN(diameter, height, material, use_type, scope_range,
                  options_range=None):
    """Six components, tax and the 80% band. Phase 2."""
    g = _phase_guard(2)
    if g:
        return [[g] + [""] * 8]
    try:
        b = _bundle()
        if b is None:
            return [[TOKEN_MODEL] + [""] * 8]
        e = _estimate(_build(diameter, height, material, use_type,
                             scope_range, options_range), bundle=b)
        return [[round(e.components[c]) for c in C.COMPONENTS]
                + [round(e.tax) if e.tax is not None else "#TAXRATE",
                   round(e.band80[0]), round(e.band80[1])]]
    except RefusalError as ex:
        return [[ex.code] + [""] * 8]
    except Exception as ex:
        return [["#ERROR " + str(ex)[:80]] + [""] * 8]


# ---------------------------------------------------------------- Phase 3
@xw.func
@xw.ret(expand="table")
def TBT_ESTIMATE(diameter, height, material, use_type, scope_range,
                 options_range=None):
    """Point estimate, band, tier and notes. Phase 3."""
    g = _phase_guard(3)
    if g:
        return [[g, "", "", "", ""]]
    try:
        b = _bundle()
        if b is None:
            return [[TOKEN_MODEL, "", "", "", ""]]
        e = _estimate(_build(diameter, height, material, use_type,
                             scope_range, options_range), bundle=b)
        return [[round(e.point), round(e.band80[0]), round(e.band80[1]),
                 e.tier, _notes(e.warnings)]]
    except RefusalError as ex:
        return [[ex.code, "", "", "", ex.detail]]
    except Exception as ex:
        return [["#ERROR", "", "", "", str(ex)[:100]]]


@xw.func
def TBT_BOOK(point_range):
    """Sum a column of point estimates into a portfolio figure.

    Uses the retransformation factors, because the sum of conditional medians
    is NOT the expected sum (SPEC 1.3).
    """
    g = _phase_guard(3)
    if g:
        return g
    try:
        b = _bundle()
        if b is None:
            return TOKEN_MODEL
        vals = [float(v) for v in _flat(point_range)
                if isinstance(v, (int, float)) and v > 0]
        if not vals:
            return 0.0
        tot = sum(v * (b.smear.get("top5", 1.0) if v >= b.big_threshold
                       else b.smear.get("rest", 1.0)) for v in vals)
        return round(tot)
    except Exception as e:
        return f"#ERROR {str(e)[:80]}"


# ---------------------------------------------------------------- ribbon
@xw.sub
def score_sheet():
    """Ribbon button: score the active table in ONE call and write VALUES.

    This is the recommended surface. The UDFs above are for single-row
    what-ifs; dragging one down thousands of rows is what makes Excel crawl.
    """
    import pandas as pd
    from tbt.scoring import estimate_many
    wb = xw.Book.caller()
    sht = wb.sheets.active
    df = sht.range("A1").options(pd.DataFrame, expand="table", index=False).value
    if df is None or len(df) == 0:
        return
    out = estimate_many(df, bundle=_bundle())
    cols = ["error", "point", "band80_lo", "band80_hi", "tier", "warnings"]
    start = sht.range("A1").end("right").offset(0, 1)
    start.value = [cols] + out[cols].fillna("").values.tolist()


def row_cap_guard(n_rows: int):
    """SPEC 6.6: a UDF called over more than 500 rows returns #BATCH and
    points at the batch scorer."""
    return TOKEN_BATCH if n_rows > C.UDF_ROW_CAP else None
