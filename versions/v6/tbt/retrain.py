"""The unattended retrain job.

SPEC 5 B11, 6.6. LOAD-BEARING ORDERING (SPEC 2.7): score the newly arrived
rows with the CURRENTLY DEPLOYED bundle and append to the ledger, THEN retrain.
Reversing it makes the ledger residuals in-sample, which collapses the conformal
bands and drifts the smearing factor to 1 (SPEC 8.1 R9).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import conformal as CF
from . import features as F
from . import ledger as L
from . import loader as LD
from . import model as M
from . import protocol as P

BUNDLE_NAME = "tbt_bundle.joblib"
PREVIOUS_NAME = "tbt_bundle.previous.joblib"
LEDGER_NAME = "ledger.csv"
STATUS_NAME = "retrain_status.json"


@dataclass
class TrainReport:
    status: str = "ok"
    bundle_id: str = ""
    trained_through: date | None = None
    n_train_rows: int = 0
    ledger_appended: int = 0
    load_report: str = ""
    failures: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_json(self) -> str:
        return json.dumps({
            "status": self.status, "bundle_id": self.bundle_id,
            "trained_through": str(self.trained_through),
            "n_train_rows": self.n_train_rows,
            "ledger_appended": self.ledger_appended,
            "failures": self.failures,
            "seconds": round(self.seconds, 1),
            "at": datetime.utcnow().isoformat(timespec="seconds"),
        }, indent=2)


def load_bundle(out_dir) -> M.Bundle | None:
    p = Path(out_dir) / BUNDLE_NAME
    if not p.exists():
        return None
    try:
        return joblib.load(p)
    except Exception:
        return None


def score_into_ledger(df: pd.DataFrame, mask: pd.Series, bundle: M.Bundle,
                      ledger_path: Path) -> int:
    """Append forward predictions for rows the deployed bundle has NOT seen.

    'Not seen' means Due Date strictly after the bundle's trained_through.
    """
    if bundle is None or bundle.trained_through is None:
        return 0
    existing = L.read(ledger_path)
    seen = set()
    if len(existing):
        seen = set(zip(existing["quote_key"].astype(str),
                       existing["due_date"].astype(str)))

    cutoff = pd.Timestamp(bundle.trained_through)
    new = df.loc[mask & (pd.to_datetime(df["Due Date"]) > cutoff)].copy()
    if len(new) == 0:
        return 0

    preds = M.predict_frame(bundle, new)
    eng = F.engineer(new, use_family_map=bundle.use_family_map)
    point = preds["point"].values
    is_big = point >= bundle.big_threshold
    groups = np.array([CF.group_of(f, g) for f, g in zip(eng["use_family"].values, is_big)])
    smear = np.array([bundle.smear.get("top5" if g else "rest", 1.0) for g in is_big])

    lo = np.empty(len(point)); hi = np.empty(len(point))
    for i, g in enumerate(groups):
        (l, h), _ = CF.bands_for(bundle.conformal, g, point[i])
        lo[i], hi[i] = l, h

    rows = pd.DataFrame({
        "quote_key": new["Quote #"].astype(str).values,
        "due_date": pd.to_datetime(new["Due Date"]).dt.date.astype(str).values,
        "bundle_id": bundle.bundle_id,
        "point": point, "book": point * smear,
        "band80_lo": lo, "band80_hi": hi, "group": groups, "tier": "",
        "actual_total": pd.to_numeric(new[C.TARGET], errors="coerce").values,
        "scored_at": datetime.utcnow().isoformat(timespec="seconds"),
        "model_shown": False,
    })
    if seen:
        keep = ~pd.Series(list(zip(rows["quote_key"], rows["due_date"]))).isin(seen).values
        rows = rows.loc[keep]
    if len(rows) == 0:
        return 0
    return L.append(ledger_path, rows)


def train(archive_csv, out_dir, *, use_drift=False, use_physics=False,
          use_sales_manager=False, bootstrap_ledger=False,
          verbose=True) -> TrainReport:
    """Validate, score-into-ledger, retrain, calibrate, write atomically."""
    import time
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / LEDGER_NAME
    rep = TrainReport()

    deployed = load_bundle(out)
    prev_usable = deployed.n_train_rows if deployed else None

    # 1. validate; on rejection the previous bundle stays deployed
    try:
        df, mask, lrep = LD.load_training(archive_csv, previous_usable=None)
        rep.load_report = lrep.summary()
    except LD.LoadRejected as e:
        rep.status = "rejected"
        rep.failures = e.failures
        rep.seconds = time.time() - t0
        (out / STATUS_NAME).write_text(rep.to_json())
        if verbose:
            print("ARCHIVE REJECTED; previous bundle left in place")
            for f in e.failures:
                print("  -", f)
        return rep

    # 2. ledger FIRST -- with the bundle that has not seen these rows
    if deployed is not None:
        rep.ledger_appended = score_into_ledger(df, mask, deployed, ledger_path)
        if verbose:
            print(f"  ledger: appended {rep.ledger_appended} forward predictions "
                  f"scored by bundle {deployed.bundle_id}")
    elif bootstrap_ledger:
        if verbose:
            print("  ledger: bootstrapping from a rolling-origin backtest")
        res = P.backtest(df, mask, use_drift=use_drift, use_physics=use_physics,
                         use_sales_manager=use_sales_manager)
        boot = res.rows.copy()
        boot["due_date"] = pd.to_datetime(boot["due_date"]).dt.date.astype(str)
        boot["tier"] = ""
        boot["scored_at"] = datetime.utcnow().isoformat(timespec="seconds")
        boot["model_shown"] = False
        rep.ledger_appended = L.append(ledger_path, boot.reindex(columns=L.LEDGER_COLUMNS))
        if verbose:
            print(f"  ledger: bootstrapped {rep.ledger_appended} rows")

    # 3. retrain on everything usable
    bundle = M.fit_bundle(df, mask, use_drift=use_drift, use_physics=use_physics,
                          use_sales_manager=use_sales_manager)

    # 4. calibrate from the forward ledger only
    led = L.read(ledger_path)
    bundle.conformal = CF.fit_conformal(led)
    bundle.smear = CF.fit_smear(led, bundle.big_threshold)
    bundle.bundle_id = bundle.fingerprint()

    # 5. write atomically, keeping the previous bundle for rollback
    tmp = out / (BUNDLE_NAME + ".tmp")
    joblib.dump(bundle, tmp, compress=3)
    if (out / BUNDLE_NAME).exists():
        shutil.copy2(out / BUNDLE_NAME, out / PREVIOUS_NAME)
    tmp.replace(out / BUNDLE_NAME)

    rep.bundle_id = bundle.bundle_id
    rep.trained_through = bundle.trained_through
    rep.n_train_rows = bundle.n_train_rows
    rep.seconds = time.time() - t0
    (out / STATUS_NAME).write_text(rep.to_json())
    if verbose:
        print(f"  bundle {bundle.bundle_id} trained through {bundle.trained_through} "
              f"on {bundle.n_train_rows} rows in {rep.seconds:.0f}s")
        if bundle.smear.get("alarm"):
            for a in bundle.smear["alarm"]:
                print("  ALARM", a)
    return rep
