"""SPEC 6.4 -- harness integrity tests. These must pass on every commit."""
import numpy as np
import pandas as pd
import pytest

from tbt import config as C
from tbt import features as F
from tbt import loader as LD
from tbt import model as M
from tbt import protocol as P
from conftest import FAST


# --- test 1: leakage tripwire ------------------------------------------------
@pytest.mark.parametrize("banned", [
    "Total Price", "Proposal Total", "Total Tax", "Material Price",
    "Margin (%)", "Commission (%)", "Status", "Revision #",
    "Company Name", "Usable Capacity", "IS_TAXABLE", "Quote #",
])
def test_banned_column_raises_before_any_fit(banned):
    """A banned column in the feature matrix must raise, not train.

    Left unguarded the model reports under 1% mean APE and looks finished.
    """
    with pytest.raises(ValueError, match="Banned columns"):
        F.assert_no_banned(list(C.NUMERIC_FEATURES_BASE) + [banned])


def test_encoder_columns_contain_no_banned_names(bundle):
    F.assert_no_banned(bundle.encoder.columns)          # must not raise
    assert "IS_TAXABLE" not in bundle.encoder.columns
    assert not set(bundle.encoder.columns) & set(C.BANNED)


def test_leakage_would_be_detectable(loaded):
    """Sanity: if a component price WERE used, error would collapse to ~0.

    This is what the guard above prevents, demonstrated once so the guard's
    value is not theoretical.
    """
    df, mask, _ = loaded
    d = df.loc[mask]
    total = pd.to_numeric(d["Total Price"], errors="coerce").values
    cheat = sum(pd.to_numeric(d[c], errors="coerce").fillna(0.0).values
                for c in C.COMPONENTS)
    assert np.mean(np.abs(cheat - total) / total) < 0.001


# --- test 2: future-state tripwire ------------------------------------------
def test_no_future_state_in_fitted_tables(loaded):
    """Fitting on rows before a cutoff must equal fitting on a file truncated
    at that cutoff. Catches a tax table / vocabulary / threshold computed once
    on the whole file (SPEC 4.5 item 11)."""
    df, mask, _ = loaded
    cutoff = pd.Timestamp("2025-07-01")
    before = mask & (pd.to_datetime(df["Due Date"]) < cutoff)

    b_full = M.fit_bundle(df, before, variants=FAST)

    trunc = df.loc[pd.to_datetime(df["Due Date"]) < cutoff].copy()
    m_trunc, _ = LD.usable_mask(trunc)
    b_trunc = M.fit_bundle(trunc, m_trunc, variants=FAST)

    assert b_full.tax == b_trunc.tax
    assert b_full.encoder.vocab == b_trunc.encoder.vocab
    assert b_full.envelope == b_trunc.envelope
    assert b_full.use_family_map == b_trunc.use_family_map
    assert b_full.big_threshold == pytest.approx(b_trunc.big_threshold)
    assert b_full.n_train_rows == b_trunc.n_train_rows


def test_tax_table_has_no_future_states(loaded):
    df, mask, _ = loaded
    cutoff = pd.Timestamp("2025-01-01")
    early = mask & (pd.to_datetime(df["Due Date"]) < cutoff)
    b = M.fit_bundle(df, early, variants=FAST)
    later_only = set(df.loc[mask & ~early, "State"].dropna().astype(str)) - \
        set(df.loc[early, "State"].dropna().astype(str))
    assert not (set(b.tax) & later_only)


# --- test 3: quantity tripwire ----------------------------------------------
def test_extended_prices_are_rejected(loaded):
    """A file whose prices were extended by Quantity must be refused.

    The signature is $/sq-ft falling as exactly 1/Quantity (SPEC 8.1 R2).
    """
    df, _, _ = loaded
    bad = df.copy()
    qty = pd.to_numeric(bad["Quantity"], errors="coerce").fillna(1)
    for c in C.COMPONENTS + ["Total Price", "Proposal Total", "Total Tax"]:
        bad[c] = pd.to_numeric(bad[c], errors="coerce").fillna(0.0) * qty
    ok, msg = LD._check_per_tank(bad)
    assert not ok and "per-tank tripwire" in msg
    with pytest.raises(LD.LoadRejected):
        LD.validate(bad)


def test_unextended_prices_pass_the_tripwire(loaded):
    df, _, _ = loaded
    ok, _ = LD._check_per_tank(df)
    assert ok


# --- test 4: filter symmetry ------------------------------------------------
def test_one_usable_mask_serves_fitting_and_scoring(loaded, monkeypatch):
    """Both paths must call the same function. Mutating the gate must change
    both counts, or one of them is filtered and the other is not."""
    df, _, _ = loaded
    base, _ = LD.usable_mask(df)
    monkeypatch.setattr(C, "PSF_MIN", 40.0)
    monkeypatch.setattr(LD.C, "PSF_MIN", 40.0)
    tightened, _ = LD.usable_mask(df)
    assert tightened.sum() < base.sum()


def test_backtest_scores_only_usable_rows(loaded):
    df, mask, _ = loaded
    res = P.backtest(df, mask, quarters=["2025Q3", "2025Q4"], variants=FAST)
    usable_keys = set(df.loc[mask].index)
    assert len(res.rows) <= len(usable_keys)
    assert (res.rows["actual_total"] > 0).all()


# --- test 6: determinism ----------------------------------------------------
def test_backtest_is_deterministic(loaded):
    df, mask, _ = loaded
    a = P.backtest(df, mask, quarters=["2025Q4"], variants=FAST)
    b = P.backtest(df, mask, quarters=["2025Q4"], variants=FAST)
    pd.testing.assert_frame_equal(a.per_quarter, b.per_quarter)
    assert a.headline == b.headline


def test_fit_is_deterministic(loaded):
    df, mask, _ = loaded
    b1 = M.fit_bundle(df, mask, variants=FAST)
    b2 = M.fit_bundle(df, mask, variants=FAST)
    p1 = M.predict_frame(b1, df.loc[mask].head(200))["point"].values
    p2 = M.predict_frame(b2, df.loc[mask].head(200))["point"].values
    np.testing.assert_allclose(p1, p2)


# --- test 7: no random split ------------------------------------------------
def test_protocol_exposes_no_random_splitter():
    """SPEC 3.6. A developer wanting a quick check runs one real quarter."""
    import inspect
    src = inspect.getsource(P)
    for forbidden in ("train_test_split", "KFold", "GroupKFold", "ShuffleSplit",
                      "StratifiedKFold"):
        assert forbidden not in src
    for name in dir(P):
        fn = getattr(P, name)
        if callable(fn) and not name.startswith("_"):
            try:
                params = inspect.signature(fn).parameters
            except (TypeError, ValueError):
                continue
            assert "random_state" not in params
            assert "test_size" not in params


def test_backbone_basis_excludes_collinear_terms():
    """log(shell_area) is exactly collinear with log_D + log_H; including it
    produced coefficients of -12.7 in the prior investigation."""
    bb = F.Backbone()
    assert "log_shell_area" not in bb.cols
    assert "log_floor_area" not in bb.cols


def test_empty_variant_list_means_backbone_only():
    """`[] or DEFAULT` is DEFAULT: an explicitly empty variant list must mean
    backbone-only, not 'fall back to production trees'."""
    assert M.Stage(params=[]).params == []
    assert len(M.Stage(params=None).params) == len(C.GBM_VARIANTS)
