import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import make_synthetic_archive as SYN  # noqa: E402
from tbt import loader as LD          # noqa: E402
from tbt import model as M            # noqa: E402


@pytest.fixture(scope="session")
def archive_csv(tmp_path_factory):
    d = tmp_path_factory.mktemp("archive")
    p = d / "archive.csv"
    SYN.generate(seed=7, target_rows=3000, n_quotes=1400).to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def loaded(archive_csv):
    df, mask, rep = LD.load_training(archive_csv)
    return df, mask, rep


@pytest.fixture(scope="session")
def bundle(loaded):
    df, mask, _ = loaded
    return M.fit_bundle(df, mask)
