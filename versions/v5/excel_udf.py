"""Excel bridge for v5, via xlwings.

    xlwings addin install        # then: Excel ribbon -> xlwings -> Import Functions

Keep this file, `tbt5/` and `tbt5_bundle.joblib` beside the workbook.

Add four Yes/No dropdown columns for scope and then:

    =TBT5_FLAG(H2, B2, C2, D2, E2, F2, G2, I2, J2, K2)   -> "OK +3%" / "LOW -22%"
    =TBT5_PRICE(B2, C2, D2, E2, F2, G2, I2, J2, K2)      -> a single number
    =TBT5_BREAKDOWN(...)                                  -> components and warnings
    =TBT5_INFO()                                          -> how stale the model is

Blank scope returns `#SCOPE`, never a guess.

Deploy TBT5_FLAG before TBT5_PRICE. At single-digit mean error the model cannot
set prices, but it reliably catches a transposed dimension, a missing scope line
or a forgotten stainless premium — real money at almost no risk, and it builds
the track record you would need before trusting it further.

Past a few dozen rows use `python cli.py score` instead. Excel fires inference on
every recalculation and will crawl.

Python-in-Excel (`=PY()`) cannot be used for this: it runs in an Azure sandbox
with no filesystem or network access, so it cannot reach a local model. This was
checked; xlwings is the right bridge.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import xlwings as xw  # noqa: E402

from tbt5.api import predict  # noqa: E402
from tbt5.data import ScopeError  # noqa: E402
from tbt5.model import TBT5  # noqa: E402


def _yn(v):
    """Yes/No cell -> bool, or None when blank. None is what raises #SCOPE."""
    if v is None:
        return None
    s = str(v).strip().upper()
    if s in ("YES", "Y", "TRUE", "T", "1"):
        return True
    if s in ("NO", "N", "FALSE", "F", "0"):
        return False
    return None


def _call(diameter, height, material, state, construction, insulation, freight,
          taxable, tank_name=None, **kw):
    return predict(
        diameter=float(diameter), height=float(height),
        material=material, state=state, tank_name=tank_name,
        construction=_yn(construction), insulation=_yn(insulation),
        freight=_yn(freight), taxable=_yn(taxable), **kw)


@xw.func
def TBT5_PRICE(diameter, height, material, state, construction, insulation,
               freight, taxable, tank_name=None):
    """Unit price. Returns #SCOPE if any scope answer is blank."""
    try:
        return _call(diameter, height, material, state, construction,
                     insulation, freight, taxable, tank_name)["unit_price"]
    except ScopeError:
        return "#SCOPE"
    except Exception as e:                        # noqa: BLE001
        return f"#ERR {e}"


@xw.func
def TBT5_FLAG(quoted, diameter, height, material, state, construction,
              insulation, freight, taxable, tank_name=None, tolerance=0.15):
    """OK / LOW / HIGH against the quoted price. Deploy this one first."""
    try:
        r = _call(diameter, height, material, state, construction, insulation,
                  freight, taxable, tank_name)
        est = r["unit_price"]
        gap = (float(quoted) - est) / est
        tag = "OK" if abs(gap) <= float(tolerance) else ("LOW" if gap < 0 else "HIGH")
        return f"{tag} {gap * 100:+.0f}%"
    except ScopeError:
        return "#SCOPE"
    except Exception as e:                        # noqa: BLE001
        return f"#ERR {e}"


@xw.func
@xw.ret(expand="table")
def TBT5_BREAKDOWN(diameter, height, material, state, construction, insulation,
                   freight, taxable, tank_name=None):
    """Components, band, spread and every warning, as a two-column block."""
    try:
        r = _call(diameter, height, material, state, construction, insulation,
                  freight, taxable, tank_name)
    except ScopeError as e:
        return [["#SCOPE", str(e)]]
    except Exception as e:                        # noqa: BLE001
        return [["#ERR", str(e)]]

    rows = [[k.replace(" Price", ""), v] for k, v in r["components"].items()]
    rows += [
        ["— estimate —", r["unit_price"]],
        ["80% range", f"{r['p80_low']:,.0f} to {r['p80_high']:,.0f}"],
        ["$/sq-ft", r["psf"]],
        ["spread (sigma)", r["sigma"]],
        ["objective", r["objective"]],
        ["est. tax", r["est_tax"]],
        ["proposal total", r["proposal_total"]],
    ]
    rows += [["warning", w] for w in r["warnings"]]
    return rows


@xw.func
def TBT5_INFO():
    """Version, training date and how far the data runs. Check it is not stale —
    retrain monthly; the cadence is worth more than any modelling change."""
    try:
        m = TBT5.load()
        return (f"{m.meta_['version']} | trained {m.meta_['trained']} | "
                f"data through {m.meta_['data_through']} | "
                f"{m.meta_['rows']} rows | objective {m.meta_['objective']}")
    except Exception as e:                        # noqa: BLE001
        return f"#ERR {e}"
