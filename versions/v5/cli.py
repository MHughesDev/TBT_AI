"""v5 command line.

    python cli.py selftest                       # synthetic end-to-end run
    python cli.py validate archive_prepared.csv  # rolling-origin, the real number
    python cli.py train    archive_prepared.csv
    python cli.py predict  --diameter 32 --height 30 --material CS --state MO \
                           --construction yes --insulation no --freight yes \
                           --taxable no
    python cli.py score    quotes.csv scored.csv
    python cli.py info
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tbt5 import data, evaluate, synth                      # noqa: E402
from tbt5.api import predict                                # noqa: E402
from tbt5.config import BUNDLE, SCOPE, TARGET, VERSION      # noqa: E402
from tbt5.data import ScopeError                            # noqa: E402
from tbt5.model import TBT5                                 # noqa: E402


def cmd_validate(args):
    df = data.load(args.csv)
    print(f"\nRolling origin — train on everything before quarter Q, score Q.\n")
    folds = evaluate.rolling_origin(df, n_folds=args.folds)
    if args.segments and folds:
        print("\nWorst segments on the final fold:")
        q = data.quarters(df)
        qs = sorted(q.dropna().unique())
        tr = df[(q < qs[-1]).values & df["plausible"].values]
        te = df[(q == qs[-1]).values & df["plausible"].values]
        est = TBT5().fit(tr, verbose=False).estimate(te)["estimate"]
        for col in ["Material", "Use Type", "Country", "Wage Type"]:
            print(f"\n  by {col}")
            print(evaluate.segment_report(te, est, te[TARGET].values, col)
                  .to_string(float_format=lambda v: f"{v:7.1f}"))
    return folds


def cmd_train(args):
    df = data.load(args.csv)
    if not args.fast:
        print("\nValidating before fitting (skip with --fast):")
        evaluate.rolling_origin(df, n_folds=args.folds)
    print()
    t0 = time.time()
    m = TBT5(objective=args.objective).fit(df)
    path = m.save(args.out)
    print(f"\nSaved -> {path}   ({time.time() - t0:.0f}s)")
    print(f"  {m.meta_['rows']} rows, {m.meta_['quotes']} quotes, "
          f"data through {m.meta_['data_through']}")


def cmd_predict(args):
    sc = {}
    for f in ["construction", "insulation", "insulation_erection", "freight", "taxable"]:
        v = getattr(args, f, None)
        sc[f] = None if v is None else (v == "yes")
    kw = {k: v for k, v in vars(args).items()
          if v is not None and k not in sc and k not in
          ("cmd", "func", "objective", "bundle")}
    print(json.dumps(predict(objective=args.objective, bundle=args.bundle,
                             **sc, **kw), indent=2))


def cmd_score(args):
    """Batch-score a CSV. Use this rather than a spreadsheet UDF past a few dozen
    rows — Excel fires inference on every recalculation and will crawl."""
    m = TBT5.load(args.bundle)
    raw = pd.read_csv(args.csv, low_memory=False)
    missing = [c for c in SCOPE if c not in raw.columns]
    if missing:
        raise ScopeError(f"input is missing scope columns: {', '.join(missing)}")
    from tbt5 import features
    d = features.build(raw)
    d["plausible"] = True
    out = m.estimate(d, objective=args.objective)
    raw["TBT5_Estimate"] = np.round(out["estimate"], 0)
    raw["TBT5_Sigma"] = np.round(out["sigma"], 3)
    if TARGET in raw.columns:
        a = pd.to_numeric(raw[TARGET], errors="coerce")
        raw["TBT5_Gap_%"] = np.round((raw["TBT5_Estimate"] - a) / a * 100, 1)
    raw.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(raw)} rows")


def cmd_info(args):
    m = TBT5.load(args.bundle)
    print(json.dumps({**m.meta_, "blend": m.blend_,
                      "global_shift_x": round(float(np.exp(m.shift_)), 4),
                      "bands": m.bands_}, indent=2))


def cmd_selftest(args):
    """Run the whole pipeline on a synthetic archive.

    This proves the plumbing works. It does NOT measure accuracy: the data is
    manufactured, and a model scored against a guess at how pricing works is
    measuring the guess. Real numbers need the real archive.
    """
    out = Path(args.out or (HERE / "synthetic_archive.csv"))
    print(f"Generating a synthetic archive -> {out}")
    synth.generate(seed=args.seed).to_csv(out, index=False)
    df = data.load(out)
    print("\nRolling origin on SYNTHETIC data (plumbing check, not an accuracy "
          "result):\n")
    folds = evaluate.rolling_origin(df, n_folds=args.folds)
    if not folds:
        sys.exit("selftest produced no folds — check the date range")
    print("\nOK: every stage ran and produced finite metrics.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tbt5", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="rolling-origin validation")
    v.add_argument("csv")
    v.add_argument("--folds", type=int, default=6)
    v.add_argument("--segments", action="store_true", help="per-segment breakdown")
    v.set_defaults(func=cmd_validate)

    t = sub.add_parser("train", help="fit and save a bundle")
    t.add_argument("csv")
    t.add_argument("--out", default=str(BUNDLE))
    t.add_argument("--folds", type=int, default=6)
    t.add_argument("--fast", action="store_true", help="skip validation")
    t.add_argument("--objective", default="mape",
                   choices=["mape", "median", "unbiased"])
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="price one tank")
    for f in ["diameter", "height", "freeboard", "miles", "ss", "s1"]:
        pr.add_argument(f"--{f}", type=float)
    for f in ["material", "use_type", "deck", "floor", "wage", "country",
              "state", "bid_type", "sales_manager", "due_date", "tank_name"]:
        pr.add_argument(f"--{f.replace('_', '-')}", dest=f)
    pr.add_argument("--quantity", type=int)
    for f in ["construction", "insulation", "freight", "taxable"]:
        pr.add_argument(f"--{f}", choices=["yes", "no"], required=True)
    pr.add_argument("--insulation-erection", dest="insulation_erection",
                    choices=["yes", "no"])
    pr.add_argument("--objective", default=None,
                    choices=["mape", "median", "unbiased"])
    pr.add_argument("--bundle", default=str(BUNDLE))
    pr.set_defaults(func=cmd_predict)

    s = sub.add_parser("score", help="batch-score a CSV")
    s.add_argument("csv")
    s.add_argument("out")
    s.add_argument("--bundle", default=str(BUNDLE))
    s.add_argument("--objective", default=None,
                   choices=["mape", "median", "unbiased"])
    s.set_defaults(func=cmd_score)

    i = sub.add_parser("info", help="what is in the bundle")
    i.add_argument("--bundle", default=str(BUNDLE))
    i.set_defaults(func=cmd_info)

    st = sub.add_parser("selftest", help="end-to-end run on synthetic data")
    st.add_argument("--out", default=None)
    st.add_argument("--folds", type=int, default=4)
    st.add_argument("--seed", type=int, default=0)
    st.set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except ScopeError as e:
        print(f"SCOPE ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
