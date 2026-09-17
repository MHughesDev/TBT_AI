"""Retraining cadence simulation. SPEC 3.4 (secondary) and 4.12.

The field notes claim retraining cadence was worth more than every modelling
change combined: never 9.99%, quarterly 9.28%, monthly 8.69%. That claim is
the single biggest lever in the deployment, so it gets measured rather than
assumed.

This is a month-by-month forward test, which is what the deployment actually
experiences. It is NOT the SPEC 3 protocol and its numbers are reported
separately, never as the headline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tbt import loader as LD          # noqa: E402
from tbt import protocol as P         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--out", default="cadence.json")
    ap.add_argument("--fast", action="store_true",
                    help="smaller trees; relative ordering is what matters here")
    a = ap.parse_args()

    df, mask, rep = LD.load_training(a.archive)
    print(rep.summary(), "\n", flush=True)

    kw = {}
    if a.fast:
        kw["variants"] = [dict(learning_rate=0.06, max_iter=400,
                               min_samples_leaf=20, l2_regularization=0.0,
                               max_leaf_nodes=31, random_state=1)]

    out = {}
    for cad in ("never", "quarterly", "monthly"):
        t0 = time.time()
        r = P.cadence_simulation(df, mask, cadence=cad, **kw)
        if not r:
            print(f"{cad}: no result"); continue
        r["seconds"] = round(time.time() - t0, 1)
        out[cad] = r
        print(f"{cad:10s} n={r['n']:5d} mean={r['mean_ape']*100:6.2f}% "
              f"median={r['median_ape']*100:6.2f}% p90={r['p90_ape']*100:6.2f}% "
              f"agg={r['agg_bias']*100:+6.2f}%  [{r['seconds']}s]", flush=True)

    if "never" in out and "monthly" in out:
        d = (out["never"]["mean_ape"] - out["monthly"]["mean_ape"]) * 100
        print(f"\nmonthly beats never-retrain by {d:+.2f} points of mean APE.")
        print("The field notes put this at 1.30 points and called it larger "
              "than every\nmodelling change combined. Compare, and put the "
              "retrain on a schedule either way.")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
