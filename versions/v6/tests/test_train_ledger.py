"""SPEC 6.4 test 5 (ledger ordering) and SPEC 6.6 (operational acceptance)."""
import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import FAST
from tbt import conformal as CF
from tbt import ledger as L
from tbt import loader as LD
from tbt import model as M
from tbt import retrain as T


@pytest.fixture
def split_archive(archive_csv, tmp_path):
    """An 'old' file and a 'new' file that adds a later month."""
    df = pd.read_csv(archive_csv)
    due = pd.to_datetime(df["Due Date"], errors="coerce")
    cutoff = due.quantile(0.80)
    old = df.loc[due < cutoff]
    p_old, p_new = tmp_path / "old.csv", tmp_path / "new.csv"
    old.to_csv(p_old, index=False)
    df.to_csv(p_new, index=False)
    return p_old, p_new


def _train(archive, out, **kw):
    import tbt.config as C
    orig = C.GBM_VARIANTS
    C.GBM_VARIANTS = FAST
    try:
        return T.train(archive, out, verbose=False, **kw)
    finally:
        C.GBM_VARIANTS = orig


# --- SPEC 6.4 test 5 ---------------------------------------------------------
def test_ledger_rows_are_scored_by_the_previous_bundle(split_archive, tmp_path):
    """LOAD-BEARING (SPEC 2.7 / 8.1 R9): the retrain job must score the newly
    arrived rows with the bundle that has NOT seen them, and only then refit.
    """
    p_old, p_new = split_archive
    out = tmp_path / "models"

    r1 = _train(p_old, out)
    assert r1.status == "ok"
    first_id = r1.bundle_id

    r2 = _train(p_new, out)
    assert r2.status == "ok"
    assert r2.ledger_appended > 0, "second retrain appended no forward predictions"

    led = L.read(out / T.LEDGER_NAME)
    assert set(led["bundle_id"].unique()) == {first_id}, (
        "ledger rows must carry the PREVIOUS bundle id; if they carry the new "
        "one the job retrained before scoring and the residuals are in-sample")
    assert r2.bundle_id != first_id


def test_ledger_rows_are_all_after_the_scoring_bundles_cutoff(split_archive, tmp_path):
    p_old, p_new = split_archive
    out = tmp_path / "models"
    r1 = _train(p_old, out)
    _train(p_new, out)
    led = L.read(out / T.LEDGER_NAME)
    assert (pd.to_datetime(led["due_date"]) > pd.Timestamp(r1.trained_through)).all()


def test_in_sample_residuals_would_collapse_the_bands(loaded):
    """Why the ordering matters, demonstrated: residuals from rows the bundle
    was fitted on give a far tighter band than honest forward residuals."""
    df, mask, _ = loaded
    cutoff = pd.Timestamp("2025-10-01")
    tr = mask & (pd.to_datetime(df["Due Date"]) < cutoff)
    te = mask & (pd.to_datetime(df["Due Date"]) >= cutoff)
    b = M.fit_bundle(df, tr, variants=FAST)

    def band(rows):
        pr = M.predict_frame(b, df.loc[rows])
        led = pd.DataFrame({
            "actual_total": pd.to_numeric(df.loc[rows, "Total Price"], errors="coerce"),
            "point": pr["point"].values, "group": "ALL|std",
            "due_date": df.loc[rows, "Due Date"].values})
        return CF.fit_conformal(led, min_group=10)["ALL"]["q80"]

    in_sample = band(tr)
    forward = band(te)
    assert in_sample < forward, (
        "in-sample residuals must be tighter; if not, this demonstration is "
        "broken, not the finding")


# --- SPEC 6.6 operational ----------------------------------------------------
def test_rejected_archive_leaves_the_previous_bundle_deployed(split_archive, tmp_path):
    p_old, _ = split_archive
    out = tmp_path / "models"
    r1 = _train(p_old, out)
    good = joblib.load(out / T.BUNDLE_NAME)

    corrupt = pd.read_csv(p_old)
    corrupt["Total Price"] = corrupt["Total Price"] * 3.0      # breaks the identity
    p_bad = tmp_path / "bad.csv"
    corrupt.to_csv(p_bad, index=False)

    r2 = _train(p_bad, out)
    assert r2.status == "rejected" and r2.failures
    still = joblib.load(out / T.BUNDLE_NAME)
    assert still.bundle_id == good.bundle_id == r1.bundle_id
    status = json.loads((out / T.STATUS_NAME).read_text())
    assert status["status"] == "rejected" and status["failures"]


def test_previous_bundle_is_kept_for_rollback(split_archive, tmp_path):
    p_old, p_new = split_archive
    out = tmp_path / "models"
    r1 = _train(p_old, out)
    _train(p_new, out)
    assert (out / T.PREVIOUS_NAME).exists()
    assert joblib.load(out / T.PREVIOUS_NAME).bundle_id == r1.bundle_id


def test_batch_warns_when_scoring_rows_the_bundle_has_seen(loaded, bundle, tmp_path):
    """SPEC 8.1 R17: scoring the training archive with its own bundle is the
    obvious smoke test and reports a meaninglessly low error."""
    from tbt import batch as B
    df, mask, _ = loaded
    p = tmp_path / "seen.csv"
    df.loc[mask].head(50).to_csv(p, index=False)
    _, banner = B.score_csv(p, tmp_path / "out.csv", bundle)
    assert banner is not None and "in-sample" in banner


def test_no_network_calls_in_the_package():
    """SPEC 6.6: no data leaves the machine."""
    import pkgutil
    import tbt
    bad = ("requests", "urllib.request", "httpx", "socket.create_connection",
           "boto3", "aiohttp")
    for mod in pkgutil.iter_modules(tbt.__path__):
        src = (Path(tbt.__path__[0]) / f"{mod.name}.py").read_text()
        for b in bad:
            assert b not in src, f"{mod.name}.py references {b}"


def test_inference_is_fast_enough(loaded, bundle):
    """SPEC 6.6: a few thousand rows in seconds."""
    import time
    df, mask, _ = loaded
    rows = df.loc[mask]
    rows = pd.concat([rows] * max(1, 2000 // max(len(rows), 1) + 1)).head(2000)
    t0 = time.time()
    M.predict_frame(bundle, rows)
    assert time.time() - t0 < 10.0


# --- conformal + tiers -------------------------------------------------------
def test_conformal_falls_back_from_group_to_family_to_global():
    led = pd.DataFrame({
        "actual_total": np.exp(np.random.default_rng(0).normal(12, .2, 400)),
        "point": np.exp(np.random.default_rng(1).normal(12, .2, 400)),
        "group": ["Fire|std"] * 200 + ["Waste|big"] * 200,
        "due_date": pd.Timestamp("2026-01-01"),
    })
    tbl = CF.fit_conformal(led, min_group=150)
    assert CF.lookup(tbl, "Fire|std")["source"] == "Fire|std"
    # an unseen group falls back to its family, then to the global pool
    assert CF.lookup(tbl, "Fire|big")["source"] in ("fam:Fire", "ALL")
    assert CF.lookup(tbl, "Nonexistent|std")["source"] == "ALL"


def test_smear_alarms_instead_of_applying_an_extreme_factor():
    led = pd.DataFrame({
        "actual_total": np.full(200, 200_000.0),
        "point": np.full(200, 100_000.0),          # ratio 2.0, far outside the clip
        "group": "Fire|std", "due_date": pd.Timestamp("2026-01-01"),
    })
    s = CF.fit_smear(led, big_threshold=1e9)
    assert s["alarm"], "an out-of-range factor must raise an alarm"
    assert s["rest"] <= 1.25


def test_tier_ordering_is_derived_from_band_width():
    from tbt.scoring import tier_of
    assert tier_of(0.10, []) == "A"
    assert tier_of(0.18, []) == "B"
    assert tier_of(0.30, []) == "C"
    assert tier_of(0.10, ["EXTRAP"]) == "C"     # a serious warning forces C
    assert tier_of(float("nan"), []) == "C"
