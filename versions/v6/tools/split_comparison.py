"""Demonstrate what a random split reports here, and why it is a fiction.

SPEC 3.6 states the trap: the rows are only about half as many distinct quotes
as they look, revisions are near-duplicates, and a random split puts revision 2
in training and revision 3 in test.

This script exists to make that concrete ONE TIME, so a builder can see the
size of the lie rather than take it on faith. It deliberately lives in tools/
and not in the package: `tbt.protocol` exposes no random splitter at all, and a
test asserts it never will. Nothing here may be quoted as an accuracy number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tbt import config as C          # noqa: E402
from tbt import loader as LD         # noqa: E402
from tbt import model as M           # noqa: E402
from tbt import protocol as P        # noqa: E402

FAST = [dict(learning_rate=0.06, max_iter=400, min_samples_leaf=20,
             l2_regularization=0.0, max_leaf_nodes=31, random_state=1)]


def score(df, mask, train_idx, test_idx):
    tr = pd.Series(False, index=df.index)
    tr.loc[train_idx] = True
    tr &= mask
    te = df.index.isin(test_idx) & mask.values
    if tr.sum() < 300 or te.sum() < 50:
        return None
    b = M.fit_bundle(df, tr, variants=FAST)
    pr = M.predict_frame(b, df.loc[te])
    a = pd.to_numeric(df.loc[te, C.TARGET], errors="coerce").values
    p = pr["point"].values
    good = a > 0
    return np.abs(p[good] - a[good]) / a[good]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()

    df, mask, rep = LD.load_training(a.archive)
    print(rep.summary(), "\n")
    idx = np.array(df.index)

    rows = []

    # 1. random K-fold -- the fiction
    errs = []
    for tr, te in KFold(a.folds, shuffle=True, random_state=0).split(idx):
        e = score(df, mask, idx[tr], idx[te])
        if e is not None:
            errs.append(e)
    e = np.concatenate(errs)
    rows.append(("random K-fold", len(e), e.mean(), np.median(e),
                 "FICTION - revision 2 trains, revision 3 tests"))

    # 2. grouped by Quote # -- honest about duplication, still sees the future
    errs = []
    g = df["Quote #"].astype(str).values
    for tr, te in GroupKFold(a.folds).split(idx, groups=g):
        e = score(df, mask, idx[tr], idx[te])
        if e is not None:
            errs.append(e)
    e = np.concatenate(errs)
    rows.append(("GroupKFold by Quote #", len(e), e.mean(), np.median(e),
                 "still sees the future; prices drift"))

    # 3. the protocol
    res = P.backtest(df, mask, variants=FAST)
    h = res.headline
    rows.append(("rolling origin (SPEC 3)", int(res.rows.shape[0]),
                 h["mean_ape"], h["median_ape"], "THE PROTOCOL"))

    print(f"{'scheme':26s} {'n':>6s} {'mean APE':>9s} {'median':>8s}  note")
    print("-" * 86)
    for name, n, m, md, note in rows:
        print(f"{name:26s} {n:6d} {m*100:8.2f}% {md*100:7.2f}%  {note}")

    rnd, proto = rows[0][2], rows[2][2]
    print(f"\nA random split reports {rnd*100:.2f}% where the forward protocol "
          f"says {proto*100:.2f}%.")
    print(f"It understates the error by {(proto - rnd) * 100:.2f} points, "
          f"a factor of {proto / rnd:.1f}x.")
    print("\nThis is why tbt.protocol exposes no random splitter and why any "
          "number\nnot produced by the rolling-origin protocol must be "
          "rejected at review.")


if __name__ == "__main__":
    main()
