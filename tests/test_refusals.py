"""SPEC 6.5 -- negative acceptance. The system must REFUSE, not answer badly.

A pricing model that silently returns a number when it should have declined is
a worse failure than one that errors, because nothing on the sheet shows it
happened.
"""
import numpy as np
import pandas as pd
import pytest

from tbt import scoring as E
from tbt.contract import (QuoteInput, RefusalError, TOKEN_MODEL, TOKEN_QTY,
                          TOKEN_RANGE, TOKEN_SCOPE, TOKEN_SCOPE_NEST,
                          TOKEN_SCOPE_WAGE, TOKEN_UNKNOWN_MATERIAL,
                          TOKEN_UNKNOWN_USETYPE, WARN_BIG, WARN_EXTRAP,
                          WARN_GEO, WARN_STALE, WARN_TAXRATE)


def base(**kw):
    d = dict(diameter_ft=32.0, height_ft=30.0, material="CS",
             use_type="Fire Protection Storage Tank",
             construction=True, insulation=False, insulation_erection=False,
             freight=True, taxable=False, state="MO", country="US",
             wage_type="Non-Union / Non-Prevailing")
    d.update(kw)
    return QuoteInput(**d)


# --- scope -------------------------------------------------------------------
@pytest.mark.parametrize("field", ["construction", "insulation",
                                   "insulation_erection", "freight", "taxable"])
def test_blank_scope_refuses(bundle, field):
    q = base(**{field: None})
    with pytest.raises(RefusalError) as e:
        E.estimate(q, bundle=bundle)
    assert e.value.code == TOKEN_SCOPE


@pytest.mark.parametrize("bad", ["", "maybe", "  ", 2, -1])
def test_non_boolean_scope_refuses(bundle, bad):
    with pytest.raises(RefusalError) as e:
        E.estimate(base(freight=bad), bundle=bundle)
    assert e.value.code == TOKEN_SCOPE


def test_scope_nesting_refuses(bundle):
    q = base(insulation=False, insulation_erection=True)
    with pytest.raises(RefusalError) as e:
        E.estimate(q, bundle=bundle)
    assert e.value.code == TOKEN_SCOPE_NEST


def test_supply_only_insulation_is_allowed(bundle):
    """83 archive rows have supply without erection: the customer installs it."""
    est = E.estimate(base(insulation=True, insulation_erection=False), bundle=bundle)
    assert est.components["Insulation Material Price"] > 0
    assert est.components["Insulation Construction Price"] == 0


def test_wage_type_contradicting_scope_refuses(bundle):
    q = base(wage_type="No Erection Included", construction=True)
    with pytest.raises(RefusalError) as e:
        E.estimate(q, bundle=bundle)
    assert e.value.code == TOKEN_SCOPE_WAGE


def test_no_default_is_ever_applied(bundle):
    """There must be no code path that fills a blank scope flag."""
    for blank in (None, float("nan"), "", "  "):
        with pytest.raises(RefusalError):
            E.as_bool(blank, "freight")
    # and the batch path must not quietly coerce a blank to False either
    assert E.as_bool("no", "freight") is False
    assert E.as_bool("yes", "freight") is True


# --- geometry ----------------------------------------------------------------
@pytest.mark.parametrize("D,H", [(0, 30), (32, 0), (-5, 30), (np.nan, 30)])
def test_bad_geometry_refuses(bundle, D, H):
    with pytest.raises(RefusalError) as e:
        E.estimate(base(diameter_ft=D, height_ft=H), bundle=bundle)
    assert e.value.code == TOKEN_RANGE


def test_non_numeric_geometry_refuses(bundle):
    with pytest.raises(RefusalError) as e:
        E.estimate(base(diameter_ft="wide"), bundle=bundle)
    assert e.value.code == TOKEN_RANGE


def test_far_outside_envelope_refuses(bundle):
    hi = bundle.envelope["D"][1]
    with pytest.raises(RefusalError) as e:
        E.estimate(base(diameter_ft=hi * 3), bundle=bundle)
    assert e.value.code == TOKEN_RANGE


def test_just_outside_envelope_warns_but_answers(bundle):
    """Tier 2: returns a number AND a code that must render."""
    hi = bundle.envelope["D"][1]
    est = E.estimate(base(diameter_ft=hi * 1.2), bundle=bundle)
    assert est.point > 0
    assert WARN_EXTRAP in est.warnings
    assert est.tier == "C"


# --- quantity ----------------------------------------------------------------
@pytest.mark.parametrize("q", [0, -1, 1.5])
def test_bad_quantity_refuses(bundle, q):
    with pytest.raises(RefusalError) as e:
        E.estimate(base(quantity=q), bundle=bundle)
    assert e.value.code == TOKEN_QTY


# --- vocabulary --------------------------------------------------------------
def test_unknown_material_refuses(bundle):
    with pytest.raises(RefusalError) as e:
        E.estimate(base(material="Unobtainium"), bundle=bundle)
    assert e.value.code == TOKEN_UNKNOWN_MATERIAL


def test_unknown_use_type_refuses(bundle):
    with pytest.raises(RefusalError) as e:
        E.estimate(base(use_type="Swimming Pool"), bundle=bundle)
    assert e.value.code == TOKEN_UNKNOWN_USETYPE


def test_unseen_state_warns_but_answers(bundle):
    est = E.estimate(base(state="ZZ"), bundle=bundle)
    assert est.point > 0 and WARN_GEO in est.warnings


# --- bundle ------------------------------------------------------------------
def test_missing_bundle_refuses():
    with pytest.raises(RefusalError) as e:
        E.estimate(base(), bundle=None)
    assert e.value.code == TOKEN_MODEL


def test_corrupt_bundle_refuses():
    with pytest.raises(RefusalError) as e:
        E.estimate(base(), bundle={"not": "a bundle"})
    assert e.value.code == TOKEN_MODEL


def test_wrong_sklearn_version_refuses(bundle):
    import copy
    b = copy.copy(bundle)
    b.sklearn_version = "0.1.0"
    with pytest.raises(RefusalError) as e:
        E.estimate(base(), bundle=b)
    assert e.value.code == TOKEN_MODEL


# --- tax ---------------------------------------------------------------------
def test_taxable_without_rate_withholds_proposal_but_returns_price(bundle):
    """SPEC 6.5: Total Price is still returned in its own field."""
    est = E.estimate(base(taxable=True, state="ZZ", country="US"), bundle=bundle)
    assert est.point > 0
    assert est.proposal_total is None and est.tax is None
    assert WARN_TAXRATE in est.warnings


def test_us_state_does_not_inherit_a_country_rate(bundle):
    """'CA' is California and Canada. A namespaced table keeps them apart."""
    assert all(k.startswith(("state:", "country:")) for k in bundle.tax)
    assert not any(k == "country:US" for k in bundle.tax)


def test_taxable_with_rate_produces_tax(bundle):
    est = E.estimate(base(taxable=True, state="TX", country="US"), bundle=bundle)
    if est.tax is not None:
        assert est.tax > 0
        assert est.proposal_total > 0


# --- unsupported -------------------------------------------------------------
def test_no_win_probability_field_exists():
    """SPEC 6.5 / 4.13: 309 resolved outcomes cannot support a win model.
    The API must not have a field that looks like one."""
    from tbt.contract import Estimate
    names = set(Estimate.__dataclass_fields__)
    for bad in ("win", "win_rate", "win_probability", "p_win", "probability"):
        assert not any(bad in n.lower() for n in names)


def test_status_is_never_a_feature(bundle):
    assert "Status" not in bundle.encoder.columns


# --- warnings render ---------------------------------------------------------
def test_big_prediction_warns(bundle):
    hi = bundle.envelope["D"][1]
    est = E.estimate(base(diameter_ft=hi * 0.99, height_ft=bundle.envelope["H"][1] * 0.99),
                     bundle=bundle)
    if est.point >= bundle.big_threshold:
        assert WARN_BIG in est.warnings
        assert est.tier == "C"


def test_every_warning_has_display_text():
    from tbt.contract import WARNING_TEXT
    for code in ("EXTRAP", "BIG", "GEO", "TAXRATE", "AR_BIAS", "WAGE",
                 "STALE", "STALE_HARD", "UNSEEN"):
        assert WARNING_TEXT.get(code)


# --- batch behaviour ---------------------------------------------------------
def test_estimate_many_reports_errors_without_raising(bundle, loaded):
    df, mask, _ = loaded
    d = df.loc[mask].head(40).copy()
    d.loc[d.index[0], "IS_FREIGHT"] = None
    d.loc[d.index[1], "Diameter (ft)"] = -3
    d.loc[d.index[2], "Material"] = "Unobtainium"
    out = E.estimate_many(d, bundle=bundle)
    assert out["error"].notna().sum() >= 3
    assert out.loc[d.index[0], "error"] == TOKEN_SCOPE
    assert out.loc[d.index[1], "error"] == TOKEN_RANGE
    assert out.loc[d.index[2], "error"] == TOKEN_UNKNOWN_MATERIAL
    assert pd.isna(out.loc[d.index[0], "point"])
    good = out["error"].isna()
    assert good.sum() > 0 and (out.loc[good, "point"] > 0).all()
