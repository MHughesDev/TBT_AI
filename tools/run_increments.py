"""Run the SPEC build sequence (B1-B9) and apply every pre-registered rule.

Each increment writes its result to the log as it finishes, so a long run can
be monitored and resumed. Decision rules are applied exactly as written in
SPEC 2.3 / 2.7 / 2.8 / 5 -- they were fixed before any of this was measured.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tbt import config as C            # noqa: E402
from tbt import loader as LD           # noqa: E402
from tbt import protocol as P          # noqa: E402

V1 = [C.GBM_VARIANTS[0]]
V2 = C.GBM_VARIANTS


def trivial_baseline(df, mask):
    """B1: median $/sq-ft by use_family x Material from the training window,
    times shell area. Runs on the real protocol so the harness is proven on a
    model that cannot cheat."""
    from tbt import features as F
    d = df.copy()
    d["_q"] = P._quarter(d["Due Date"])
    d["_usable"] = mask.values
    eng = F.engineer(d)
    d["shell_area"] = eng["shell_area"]
    d["use_family"] = eng["use_family"]

    counts = d.loc[d["_usable"], "_q"].value_counts()
    qs = sorted(q for q in d.loc[d["_usable"], "_q"].unique()
                if q >= P.FIRST_QUARTER and counts.get(q, 0) >= P.MIN_QUARTER_ROWS)
    rows = []
    for q in qs:
        tr = d["_usable"] & (d["_q"] < q)
        te = d["_usable"] & (d["_q"] == q)
        if tr.sum() < 300 or te.sum() < P.MIN_QUARTER_ROWS:
            continue
        t = d.loc[tr]
        psf = pd.to_numeric(t[C.TARGET], errors="coerce") / t["shell_area"]
        tbl = psf.groupby([t["use_family"], t["Material"].astype(str)]).median()
        glob = float(psf.median())
        s = d.loc[te]
        key = list(zip(s["use_family"], s["Material"].astype(str)))
        rate = np.array([tbl.get(k, glob) for k in key])
        rows.append(pd.DataFrame({
            "quarter": q, "point": rate * s["shell_area"].values,
            "actual_total": pd.to_numeric(s[C.TARGET], errors="coerce").values}))
    allr = pd.concat(rows, ignore_index=True)
    allr = allr[allr["actual_total"] > 0]
    head = allr[allr["quarter"] >= P.HEADLINE_START]
    per_q = head.groupby("quarter").apply(
        lambda g: np.mean(np.abs(g["point"] - g["actual_total"]) / g["actual_total"]),
        include_groups=False)
    return {"mean_ape": float(per_q.mean()),
            "median_ape": float(np.median(
                np.abs(head["point"] - head["actual_total"]) / head["actual_total"])),
            "n": int(len(head))}


def summarise(res: P.BacktestResult) -> dict:
    h, p = res.headline, res.pooled
    return {
        "mean_ape": h.get("mean_ape"), "median_ape": h.get("median_ape"),
        "p90_ape": h.get("p90_ape"),
        "agg_bias_point": h.get("agg_bias_point"),
        "agg_bias_book": h.get("agg_bias_book"),
        "coverage80": h.get("coverage80"),
        "top5_bias_point": p.get("top5_bias_point"),
        "top5_bias_book": p.get("top5_bias_book"),
        "n_top5": p.get("n_top5"),
        "new_quote_mean_ape": res.new_quote.get("mean_ape"),
        "quarters": list(res.per_quarter["quarter"]),
        "per_quarter_mean_ape": [float(x) for x in res.per_quarter["mean_ape"]],
        "config": res.config,
    }


def wins_by_quarter(a: dict, b: dict) -> int:
    """SPEC 3.5: a variant beats another only if it also wins in enough
    individual quarters, not on the headline average alone."""
    qa = dict(zip(a["quarters"], a["per_quarter_mean_ape"]))
    qb = dict(zip(b["quarters"], b["per_quarter_mean_ape"]))
    shared = sorted(set(qa) & set(qb))
    return sum(1 for q in shared if qa[q] < qb[q]), len(shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--log", default="build_log.json")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()

    logp = Path(a.log)
    log = json.loads(logp.read_text()) if logp.exists() else {"runs": {}}

    def save():
        logp.write_text(json.dumps(log, indent=2, default=str))

    def run(name, **kw):
        if name in log["runs"] and log["runs"][name].get("mean_ape") is not None:
            print(f"[{name}] cached mean={log['runs'][name]['mean_ape']*100:.2f}%")
            return log["runs"][name]
        t0 = time.time()
        print(f"[{name}] running {kw} ...", flush=True)
        res = P.backtest(df, mask, **kw)
        s = summarise(res)
        s["seconds"] = round(time.time() - t0, 1)
        log["runs"][name] = s
        save()
        print(f"[{name}] mean={s['mean_ape']*100:.2f}% median={s['median_ape']*100:.2f}% "
              f"p90={s['p90_ape']*100:.2f}% aggP={s['agg_bias_point']*100:+.2f}% "
              f"aggB={s['agg_bias_book']*100:+.2f}% top5P={(s['top5_bias_point'] or 0)*100:+.2f}% "
              f"cov={(s['coverage80'] or 0)*100:.1f}% [{s['seconds']}s]", flush=True)
        return s

    print("loading archive ...", flush=True)
    df, mask, lrep = LD.load_training(a.archive)
    print(lrep.summary(), flush=True)
    log["load"] = lrep.summary()
    save()

    sel = set(a.only) if a.only else None
    def want(n):
        return sel is None or n in sel

    # ---- B1 trivial baseline ---------------------------------------------
    if want("B1") and "B1" not in log["runs"]:
        t0 = time.time()
        b1 = trivial_baseline(df, mask)
        b1["seconds"] = round(time.time() - t0, 1)
        log["runs"]["B1"] = b1; save()
        print(f"[B1] trivial baseline mean={b1['mean_ape']*100:.2f}% "
              f"median={b1['median_ape']*100:.2f}%", flush=True)

    # ---- B2 backbone only -------------------------------------------------
    if want("B2"):
        run("B2", variants=[], recency=False, use_names=False)

    # ---- B3 reference -----------------------------------------------------
    if want("B3"):
        b3 = run("B3", variants=V1, recency=False, use_names=False)

    # ---- B4 + recency -----------------------------------------------------
    if want("B4"):
        run("B4", variants=V1, recency=True, use_names=False)

    # ---- B5 + second variant ---------------------------------------------
    if want("B5"):
        run("B5", variants=V2, recency=True, use_names=False)

    # ---- B6 + tank-name features -----------------------------------------
    if want("B6"):
        run("B6", variants=V2, recency=True, use_names=True)

    # ---- B8 open experiments ---------------------------------------------
    if want("B8"):
        run("B8_drift", variants=V2, recency=True, use_names=True, use_drift=True)
        run("B8_physics", variants=V2, recency=True, use_names=True, use_physics=True)
        run("B8_both", variants=V2, recency=True, use_names=True,
            use_drift=True, use_physics=True)
        run("B8_blend1", variants=V2, recency=True, use_names=True, blend=1.0)
        run("B8_salesmgr", variants=V2, recency=True, use_names=True,
            use_sales_manager=True)

    # ---- B9 hyperparameter search ----------------------------------------
    if want("B9"):
        for lr, it in ((0.03, 1200), (0.02, 1800)):
            for leaf in (15, 30):
                nm = f"B9_lr{lr}_it{it}_leaf{leaf}"
                v = [dict(learning_rate=lr, max_iter=it, min_samples_leaf=leaf,
                          l2_regularization=0.0, max_leaf_nodes=31, random_state=1),
                     C.GBM_VARIANTS[1]]
                run(nm, variants=v, recency=True, use_names=True)

    save()
    print("\nDONE. runs:", len(log["runs"]))


if __name__ == "__main__":
    main()
