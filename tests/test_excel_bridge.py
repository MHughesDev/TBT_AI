"""Excel surface behaviour that needs no fitted model. SPEC 7.5, 7.6, 7.7."""
import importlib
import os

import pytest

import excel_bridge as X


def test_range_helpers_accept_excel_shapes():
    assert X._flat([[1, 2], [3]]) == [1, 2, 3]
    assert X._flat(5) == [5]
    assert X._flat(None) == []
    assert X._scalar([[7]]) == [7]
    assert X._scalar(7) == 7


def test_scope_range_must_supply_all_five():
    from tbt.contract import RefusalError
    with pytest.raises(RefusalError):
        X._scope([["Yes", "No"]])
    assert X._scope([["Yes", "No", "No", "Yes", "No"]]) == \
        ["Yes", "No", "No", "Yes", "No"]


def test_options_range_is_header_plus_values():
    o = X._options([["state", "quantity"], ["TX", 3]])
    assert o == {"state": "TX", "quantity": 3}
    assert X._options(None) == {}
    assert X._options([["state"]]) == {}      # header with no value row


def test_phase_gating_withholds_later_surfaces(monkeypatch):
    monkeypatch.setattr(X, "PHASE", 1)
    assert X._phase_guard(1) is None
    assert "withheld" in X._phase_guard(2)
    assert "withheld" in X._phase_guard(3)
    monkeypatch.setattr(X, "PHASE", 3)
    assert X._phase_guard(2) is None and X._phase_guard(3) is None


def test_breakdown_is_withheld_at_phase_1(monkeypatch):
    monkeypatch.setattr(X, "PHASE", 1)
    out = X.TBT_BREAKDOWN(32, 30, "CS", "Fire Protection Storage Tank",
                          [["Yes", "No", "No", "Yes", "No"]])
    assert "#PHASE2" in out[0][0]


def test_estimate_is_withheld_at_phase_1(monkeypatch):
    monkeypatch.setattr(X, "PHASE", 1)
    out = X.TBT_ESTIMATE(32, 30, "CS", "Fire Protection Storage Tank",
                         [["Yes", "No", "No", "Yes", "No"]])
    assert "#PHASE3" in out[0][0]


def test_row_cap_directs_volume_to_the_batch_scorer():
    from tbt import config as C
    assert X.row_cap_guard(C.UDF_ROW_CAP) is None
    assert X.row_cap_guard(C.UDF_ROW_CAP + 1) == "#BATCH"


def test_no_udf_is_volatile():
    """SPEC 7.6 / 8.1 R8: a volatile UDF re-runs inference on every
    recalculation of every cell and locks the workbook."""
    src = open(X.__file__).read()
    assert "volatile" not in src.replace("NON-VOLATILE", "").replace(
        "non-volatile", "")


def test_check_returns_a_token_not_a_crash_when_no_model(monkeypatch):
    monkeypatch.setattr(X, "_BUNDLE", None)
    monkeypatch.setattr(X, "MODEL_DIR", "/nonexistent")
    monkeypatch.setattr(X, "_BUNDLE_ID", "force-reload")
    out = X.TBT_CHECK(100000, 32, 30, "CS", "Fire Protection Storage Tank",
                      [["Yes", "No", "No", "Yes", "No"]])
    assert out[0][0] == "#MODEL"


def test_model_info_reports_no_model_rather_than_raising(monkeypatch):
    monkeypatch.setattr(X, "_BUNDLE", None)
    monkeypatch.setattr(X, "MODEL_DIR", "/nonexistent")
    monkeypatch.setattr(X, "_BUNDLE_ID", "force-reload")
    assert X.TBT_MODEL_INFO() == "#MODEL"
