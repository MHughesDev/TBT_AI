"""End-to-end CLI paths: train -> info -> predict -> score -> audit.

These exercise the wiring, not the accuracy. Accuracy comes only from the
rolling-origin protocol.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

import make_synthetic_archive as SYN
from conftest import FAST
from tbt import cli
from tbt import config as C


@pytest.fixture(scope="module")
def small_archive(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    p = d / "archive.csv"
    SYN.generate(seed=11, target_rows=2200, n_quotes=1000).to_csv(p, index=False)
    return p


@pytest.fixture(autouse=True)
def fast_variants(monkeypatch):
    monkeypatch.setattr(C, "GBM_VARIANTS", FAST)


def test_train_then_info_then_predict_then_score_then_audit(small_archive, tmp_path,
                                                            capsys):
    models = tmp_path / "models"

    assert cli.main(["train", str(small_archive), "--out", str(models),
                     "--bootstrap-ledger"]) == 0
    assert (models / "tbt_bundle.joblib").exists()
    status = json.loads((models / "retrain_status.json").read_text())
    assert status["status"] == "ok" and status["n_train_rows"] > 500
    assert (models / "ledger.csv").exists()

    assert cli.main(["info", "--out", str(models)]) == 0
    out = capsys.readouterr().out
    assert "bundle_id" in out and "trained_through" in out

    # a complete scope block returns a number
    ok = cli.main(["predict", "--diameter", "32", "--height", "30",
                   "--material", "CS", "--use-type", "Fire Protection Storage Tank",
                   "--state", "MO", "--construction", "yes", "--insulation", "no",
                   "--insulation-erection", "no", "--freight", "yes",
                   "--taxable", "no", "--out", str(models)])
    assert ok == 0
    out = capsys.readouterr().out
    assert "point" in out and "80% band" in out

    # a missing scope answer refuses at the CLI too
    rc = cli.main(["predict", "--diameter", "32", "--height", "30",
                   "--material", "CS", "--use-type", "Fire Protection Storage Tank",
                   "--state", "MO", "--construction", "yes",
                   "--insulation-erection", "no", "--freight", "yes",
                   "--taxable", "no", "--out", str(models)])
    assert rc == 2
    assert "#SCOPE" in capsys.readouterr().out

    # batch score
    quotes = tmp_path / "quotes.csv"
    df = pd.read_csv(small_archive).head(60)
    for f in C.SCOPE_FLAGS:
        df[f] = "Yes" if f != "IS_INSULATION_ERECTION" else "Yes"
    df["IS_INSULATION"] = "Yes"
    df.to_csv(quotes, index=False)
    dest = tmp_path / "scored.xlsx"
    assert cli.main(["score", str(quotes), str(dest), "--out", str(models)]) == 0
    assert dest.exists()
    scored = pd.read_excel(dest)
    assert "point" in scored.columns and (scored["point"] > 0).sum() > 0

    # archive audit, forward-scored
    adest = tmp_path / "audit.xlsx"
    assert cli.main(["audit", str(small_archive), str(adest)]) == 0
    assert adest.exists()
    aud = pd.read_excel(adest, sheet_name="worst_first")
    assert {"point", "actual_total", "abs_pct"} <= set(aud.columns)
    assert aud["abs_pct"].is_monotonic_decreasing


def test_backtest_cli_reports_every_required_metric(small_archive, tmp_path, capsys):
    rep = tmp_path / "report.txt"
    assert cli.main(["backtest", str(small_archive), "--report", str(rep)]) == 0
    text = rep.read_text()
    for required in ("mean APE", "median APE", "p90 APE", "agg bias point",
                     "agg bias book", "top5 bias", "coverage80",
                     "PER-SEGMENT", "COVERAGE BY GROUP"):
        assert required in text, f"report is missing {required}"


def test_train_on_a_bad_archive_exits_nonzero(small_archive, tmp_path):
    bad = tmp_path / "bad.csv"
    df = pd.read_csv(small_archive)
    df["Total Price"] = df["Total Price"] * 2.0
    df.to_csv(bad, index=False)
    assert cli.main(["train", str(bad), "--out", str(tmp_path / "m2")]) == 1
