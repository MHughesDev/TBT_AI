"""Tests that guard the things that have historically gone wrong.

Run with:  python -m pytest tests -q      (from versions/v5)

Every test here corresponds to a bug that was actually shipped in some version of
this project, or to an invariant whose violation would be silent. Accuracy is not
tested — it cannot be, without the archive.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tbt5 import comparables as cmpmod, data, decision, features, physics, synth
from tbt5.config import GROUP_COL, SCOPE, TARGET
from tbt5.data import ScopeError


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    df = synth.generate(n_quotes=900, seed=3)
    p = tmp_path_factory.mktemp("d") / "a.csv"
    df.to_csv(p, index=False)
    return data.load(p, verbose=False)


# ------------------------------------------------------------------- physics
def test_minimum_thickness_is_a_step_function():
    t = physics.min_shell_thickness(np.array([20.0, 49.9, 50.1, 130.0, 250.0]))
    assert t[0] == t[1] == pytest.approx(3 / 16)
    assert t[2] == pytest.approx(4 / 16)
    assert t[3] == pytest.approx(5 / 16)
    assert t[4] == pytest.approx(6 / 16)


def test_steel_intensity_rises_with_size():
    """The whole thesis of v5. If this ever goes flat, the physics layer is
    contributing nothing that shell area was not already saying."""
    D = np.array([20.0, 40.0, 90.0, 150.0])
    H = np.array([16.0, 32.0, 50.0, 60.0])
    f = physics.compute(D, H)
    lb_per_sqft = f["ph_lb_per_sqft"]
    assert np.all(np.diff(lb_per_sqft) > 0)
    assert lb_per_sqft[-1] / lb_per_sqft[0] > 2.5


def test_small_tanks_are_minimum_governed():
    f = physics.compute(np.array([20.0]), np.array([16.0]))
    assert f["ph_min_governed_frac"][0] == pytest.approx(1.0)
    f = physics.compute(np.array([150.0]), np.array([60.0]))
    assert f["ph_min_governed_frac"][0] < 0.5


def test_course_count_matches_height():
    f = physics.compute(np.array([40.0, 40.0]), np.array([16.0, 33.0]))
    assert f["ph_n_courses"].tolist() == [2.0, 5.0]


# --------------------------------------------------------------------- scope
def test_missing_scope_column_raises(archive):
    d = archive.drop(columns=["IS_FREIGHT"])
    with pytest.raises(ScopeError, match="missing scope"):
        data.check_scope(d)


def test_blank_scope_raises(archive):
    d = archive.copy()
    d.loc[d.index[:3], "IS_TAXABLE"] = np.nan
    with pytest.raises(ScopeError, match="blank"):
        data.check_scope(d)


def test_insulation_erection_implies_supply(archive):
    d = archive.copy()
    d.loc[d.index[0], "IS_INSULATION"] = 0
    d.loc[d.index[0], "IS_INSULATION_ERECTION"] = 1
    with pytest.raises(ScopeError, match="IS_INSULATION"):
        data.check_scope(d)


def test_scope_is_never_defaulted():
    """v3 inferred scope with classifiers and it was removed on principle. This
    test exists so that reintroducing a default fails loudly."""
    from tbt5.api import predict
    with pytest.raises(ScopeError, match="scope not supplied"):
        predict(diameter=40, height=36, material="CS", state="TX",
                construction=True, insulation=True)   # freight, taxable missing


# ---------------------------------------------------------------------- data
def test_prices_are_not_divided_by_quantity(archive):
    """v1 and v2 both divided the price columns by Quantity. They are already
    per-tank, and dividing corrupted 15% of rows — 38% of the top-5% by value.
    The tell is $/sq-ft falling as exactly 1/Quantity across order sizes."""
    d = archive[archive["plausible"]]
    med = d.groupby("Quantity")["psf"].median()
    multi = med[med.index > 1]
    if len(multi) >= 2:
        assert (multi / med.loc[1]).min() > 0.5


def test_implausible_rows_are_flagged_not_dropped(archive):
    """They must be excluded from FITTING, not merely from scoring. v2 filtered
    evaluation only, which measures the model more kindly without improving it."""
    assert (~archive["plausible"]).sum() > 0
    assert archive["plausible"].dtype == bool


def test_dedup_weighting_equalises_quotes(archive):
    w = data.weights(archive, dedup=True)
    per_quote = pd.Series(w).groupby(archive[GROUP_COL].values).sum()
    spread = per_quote.max() / per_quote.min()
    raw = data.weights(archive, dedup=False)
    raw_spread = (pd.Series(raw).groupby(archive[GROUP_COL].values).sum().max()
                  / pd.Series(raw).groupby(archive[GROUP_COL].values).sum().min())
    assert spread < raw_spread


def test_banned_columns_are_not_features():
    from tbt5.config import BANNED
    used = set(features.NUMERIC) | set(features.CATEGORICAL)
    assert not (used & set(BANNED)), "a price column leaked into the feature list"


# -------------------------------------------------------------- comparables
def test_comparables_never_see_the_future(archive):
    """The causal build must not use a row dated on or after the row it prices."""
    d = archive[archive["plausible"]].reset_index(drop=True)
    c = cmpmod.Comparables()
    feats = c.fit_transform_causal(d)
    first_month = pd.to_datetime(d["Due Date"]).dt.to_period("M").min()
    early = pd.to_datetime(d["Due Date"]).dt.to_period("M") == first_month
    # Nothing precedes the first month, so nothing in it can have comparables.
    assert not np.isfinite(feats["cmp_logpsf"][early.values]).any()
    assert np.isfinite(feats["cmp_logpsf"]).any(), "no comparables found at all"


def test_comparables_exclude_same_quote(archive):
    """A revision finding its own parent reports ~3% error and predicts nothing."""
    d = archive[archive["plausible"]].reset_index(drop=True)
    dup = d[d.duplicated(GROUP_COL, keep=False)]
    if len(dup) < 20:
        pytest.skip("no revisions in this sample")
    c = cmpmod.Comparables().fit(d)
    # Query a revision against the full reference set: if self-matching were
    # allowed its comparable would be almost exactly its own price.
    q = dup.iloc[[0]]
    out = c.transform(q)
    if np.isfinite(out["cmp_logprice"][0]):
        ratio = np.exp(out["cmp_logprice"][0]) / q[TARGET].values[0]
        assert not (0.985 < ratio < 1.015), "comparable is suspiciously exact"


# ------------------------------------------------------------------ decision
def test_mape_optimum_leans_low():
    """The minimiser of E|z-Y|/Y sits below the median whenever the conditional
    distribution has spread, because APE punishes overprediction harder."""
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    from scipy.stats import norm
    for sigma in (0.05, 0.15, 0.30):
        logq = (norm.ppf(levels) * sigma)[None, :]
        z = decision.mape_optimal(logq, levels)[0]
        assert z < 0, f"sigma={sigma} gave a non-negative shift"
    # and it shrinks harder when the model is less sure
    z_small = decision.mape_optimal((norm.ppf(levels) * 0.05)[None, :], levels)[0]
    z_big = decision.mape_optimal((norm.ppf(levels) * 0.30)[None, :], levels)[0]
    assert z_big < z_small


def test_mape_optimum_matches_the_closed_form():
    """For a lognormal the answer is exp(mu - sigma^2); the numerical route
    should land on it."""
    from scipy.stats import norm
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    sigma = 0.2
    logq = (norm.ppf(levels) * sigma)[None, :]
    assert decision.mape_optimal(logq, levels)[0] == pytest.approx(-sigma ** 2, abs=0.02)


def test_mape_optimum_beats_the_median_on_mape():
    """The claim, tested directly: simulate, and check the shrunk estimate really
    does have lower mean APE than the median estimate."""
    rng = np.random.default_rng(0)
    sigma = 0.25
    y = np.exp(rng.normal(0, sigma, 200_000))
    med, opt = 1.0, np.exp(-sigma ** 2)
    assert np.mean(np.abs(opt - y) / y) < np.mean(np.abs(med - y) / y)


def test_unbiased_objective_is_unbiased():
    rng = np.random.default_rng(1)
    sigma = 0.25
    y = np.exp(rng.normal(0, sigma, 200_000))
    est = decision.point_estimate(np.zeros(1), objective="unbiased",
                                  sigma=np.array([sigma]))[0]
    assert np.exp(est) == pytest.approx(y.mean(), rel=0.02)


def test_global_shift_finds_the_optimum():
    rng = np.random.default_rng(2)
    y = np.exp(rng.normal(0, 0.2, 20_000))
    c = decision.fit_global_shift(np.ones_like(y), y)
    assert c == pytest.approx(-0.04, abs=0.02)


# ------------------------------------------------------------------ pipeline
def test_end_to_end_fit_and_score(archive):
    """The whole pipeline, small enough to run in a test."""
    from tbt5.model import TBT5
    q = data.quarters(archive)
    qs = sorted(q.dropna().unique())
    tr = archive[(q < qs[-1]).values & archive["plausible"].values]
    te = archive[(q == qs[-1]).values & archive["plausible"].values]
    if len(tr) < 300 or len(te) < 20:
        pytest.skip("synthetic sample too small")
    m = TBT5().fit(tr, verbose=False)
    out = m.estimate(te)
    assert np.isfinite(out["estimate"]).all()
    assert (out["estimate"] > 0).all()
    # Sanity only. This is synthetic data; it is not an accuracy claim.
    from tbt5.evaluate import metrics
    assert metrics(out["estimate"], te[TARGET].values)["mean"] < 60


def test_objective_ordering(archive):
    """mape <= median <= unbiased, row by row, by construction."""
    from tbt5.model import TBT5
    q = data.quarters(archive)
    qs = sorted(q.dropna().unique())
    tr = archive[(q < qs[-1]).values & archive["plausible"].values]
    te = archive[(q == qs[-1]).values & archive["plausible"].values]
    if len(tr) < 300 or len(te) < 20:
        pytest.skip("synthetic sample too small")
    m = TBT5().fit(tr, verbose=False)
    a = m.estimate(te, objective="mape")["estimate"]
    b = m.estimate(te, objective="median")["estimate"]
    c = m.estimate(te, objective="unbiased")["estimate"]
    assert np.all(a <= b * 1.0001)
    assert np.all(b <= c * 1.0001)
