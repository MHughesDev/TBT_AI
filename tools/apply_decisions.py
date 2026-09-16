"""Apply every pre-registered decision rule to the build log and write RESULTS.md.

The rules are quoted verbatim from SPEC and were fixed BEFORE any of these
numbers existed. That is what makes this an experiment rather than a search for
a flattering number. Where a rule rejects the more complex option, it is
recorded as rejected -- not quietly re-run with a different threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(v):
    return "n/a" if v is None else f"{v * 100:.2f}%"


def wins(a, b):
    qa = dict(zip(a["quarters"], a["per_quarter_mean_ape"]))
    qb = dict(zip(b["quarters"], b["per_quarter_mean_ape"]))
    shared = sorted(set(qa) & set(qb))
    return sum(1 for q in shared if qa[q] < qb[q]), len(shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="RESULTS.md")
    ap.add_argument("--floor", type=float, default=None,
                    help="known irreducible mean APE of the test archive")
    a = ap.parse_args()

    log = json.loads(Path(a.log).read_text())
    R = log["runs"]
    L = []
    dec = []

    def get(n):
        return R.get(n)

    # ---------------- decision rules ------------------------------------
    b6 = get("B6")

    # B4 recency: keep unless <0.1 improvement (cost is one line) -- log it
    if get("B3") and get("B4"):
        d = get("B3")["mean_ape"] - get("B4")["mean_ape"]
        dec.append(("B4 recency weighting (1-yr half-life)",
                    f"{d*100:+.2f} pts",
                    "KEPT (spec: keep anyway, cost is one line)"))

    # B5 second variant: drop if <0.05 improvement
    if get("B4") and get("B5"):
        d = get("B4")["mean_ape"] - get("B5")["mean_ape"]
        dec.append(("B5 second GBM variant", f"{d*100:+.2f} pts",
                    "KEPT" if d >= 0.0005 else
                    "DROPPED by rule (<0.05 pts; halves retrain time)"))

    # B6 name features: drop if <0.05 improvement
    if get("B5") and get("B6"):
        d = get("B5")["mean_ape"] - get("B6")["mean_ape"]
        dec.append(("B6 tank-name features", f"{d*100:+.2f} pts",
                    "KEPT" if d >= 0.0005 else
                    "DROPPED by rule (<0.05 pts; a maintenance surface)"))

    # OPEN-1 drift: take B if >=0.15 pts better AND top5 not worse by >1 pt
    if b6 and get("B8_drift"):
        v = get("B8_drift")
        d = b6["mean_ape"] - v["mean_ape"]
        t5 = (v.get("top5_bias_point") or 0) - (b6.get("top5_bias_point") or 0)
        w, n = wins(v, b6)
        ok = d >= 0.0015 and t5 >= -0.01 and w >= max(1, (n + 1) // 2)
        dec.append(("OPEN-1 size-dependent drift term (t x log D)",
                    f"{d*100:+.2f} pts; top5 {t5*100:+.2f} pts; wins {w}/{n}",
                    "ADOPTED" if ok else "REJECTED by rule -> take (A), no term"))

    # OPEN-2 physics: take B if top5 bias improves >=3 pts AND mean not worse >0.15
    if b6 and get("B8_physics"):
        v = get("B8_physics")
        d = b6["mean_ape"] - v["mean_ape"]
        base_t5 = abs(b6.get("top5_bias_point") or 0)
        new_t5 = abs(v.get("top5_bias_point") or 0)
        imp = (base_t5 - new_t5)
        ok = imp >= 0.03 and d >= -0.0015
        dec.append(("OPEN-2 steel-weight term in the backbone",
                    f"top5 |bias| {base_t5*100:.2f}% -> {new_t5*100:.2f}% "
                    f"({imp*100:+.2f} pts); mean {d*100:+.2f} pts",
                    "ADOPTED" if ok else "REJECTED by rule -> take (A), no term"))

    # OPEN-4 blend: take 0.7 unless 1.0 within 0.1 pts AND within 0.5 on top5
    if b6 and get("B8_blend1"):
        v = get("B8_blend1")
        d = v["mean_ape"] - b6["mean_ape"]
        t5 = abs(v.get("top5_bias_point") or 0) - abs(b6.get("top5_bias_point") or 0)
        ok = d <= 0.001 and t5 <= 0.005
        dec.append(("OPEN-4 blend weight w=1.0 (pure component sum)",
                    f"mean {d*100:+.2f} pts vs w=0.7; top5 {t5*100:+.2f} pts",
                    "ADOPTED w=1.0 (simpler; breakdown sums exactly)" if ok
                    else "REJECTED -> keep w=0.7"))

    # OPEN-5 sales manager: include only if >=0.15 pts AND new-quote >=0.1 pts
    if b6 and get("B8_salesmgr"):
        v = get("B8_salesmgr")
        d = b6["mean_ape"] - v["mean_ape"]
        nq = ((b6.get("new_quote_mean_ape") or 0)
              - (v.get("new_quote_mean_ape") or 0))
        ok = d >= 0.0015 and nq >= 0.001
        dec.append(("OPEN-5 Sales Manager as a feature",
                    f"{d*100:+.2f} pts; new-quote {nq*100:+.2f} pts",
                    "ADOPTED" if ok else "REJECTED by rule -> exclude"))

    # B9 search: keep best only if >=0.15 pts better than the default
    b9 = {k: v for k, v in R.items() if k.startswith("B9_")}
    if b9 and b6:
        best = min(b9.items(), key=lambda kv: kv[1]["mean_ape"])
        d = b6["mean_ape"] - best[1]["mean_ape"]
        dec.append((f"B9 hyperparameter search (best: {best[0]})",
                    f"{d*100:+.2f} pts vs default",
                    "ADOPTED" if d >= 0.0015 else
                    "REJECTED by rule -> keep the default"))

    # ---------------- write -------------------------------------------------
    L.append("# Build results\n")
    L.append("Every number below comes from the rolling-origin protocol in "
             "SPEC section 3. No number here comes from a random split, a "
             "grouped K-fold, or an in-sample fit.\n")
    if a.floor is not None:
        L.append(f"**Irreducible floor of this archive: {a.floor:.2f}% mean APE.** "
                 "No model can beat it. Distance to the floor is the only "
                 "meaningful reading of the numbers below.\n")
    L.append("## Increment results\n")
    L.append("| run | mean APE | median | p90 | agg bias (point) | agg bias (book) "
             "| top5 bias (point) | coverage80 | new-quote mean | secs |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(R):
        r = R[k]
        if r.get("mean_ape") is None:
            continue
        L.append(f"| {k} | {pct(r.get('mean_ape'))} | {pct(r.get('median_ape'))} "
                 f"| {pct(r.get('p90_ape'))} | {pct(r.get('agg_bias_point'))} "
                 f"| {pct(r.get('agg_bias_book'))} | {pct(r.get('top5_bias_point'))} "
                 f"| {pct(r.get('coverage80'))} | {pct(r.get('new_quote_mean_ape'))} "
                 f"| {r.get('seconds', '')} |")

    L.append("\n## Pre-registered decisions\n")
    L.append("| question | measured | outcome |")
    L.append("|---|---|---|")
    for q, m, o in dec:
        L.append(f"| {q} | {m} | {o} |")

    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
