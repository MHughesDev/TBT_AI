"""Command line entry point.

    python -m tbt train    <archive.csv> --out models/
    python -m tbt backtest <archive.csv> [--drift] [--physics] [--report f.txt]
    python -m tbt cadence  <archive.csv> --cadence monthly|quarterly|never
    python -m tbt predict  --diameter 32 --height 30 ... --construction yes ...
    python -m tbt score    <quotes.csv> <out.xlsx> --out models/
    python -m tbt audit    <archive.csv> <out.xlsx>
    python -m tbt info     --out models/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from . import audit as A
from . import batch as B
from . import conformal as CF
from . import loader as LD
from . import protocol as P
from . import retrain as T
from .contract import QuoteInput, RefusalError
from .scoring import estimate, model_info


def _yn(v):
    if v is None:
        return None
    return str(v).strip().lower() in ("yes", "y", "true", "t", "1")


def cmd_train(a):
    rep = T.train(a.archive, a.out, use_drift=a.drift, use_physics=a.physics,
                  use_sales_manager=a.sales_manager,
                  bootstrap_ledger=a.bootstrap_ledger)
    print(rep.to_json())
    return 0 if rep.status == "ok" else 1


def cmd_backtest(a):
    df, mask, lrep = LD.load_training(a.archive)
    print(lrep.summary()); print()
    res = P.backtest(df, mask, use_drift=a.drift, use_physics=a.physics,
                     use_sales_manager=a.sales_manager,
                     apply_smear_to_point=a.smear_on_point,
                     quarters=a.quarters, progress=print if a.verbose else None)
    text = res.report()
    print(); print(text)
    if a.report:
        Path(a.report).write_text(text + "\n\nPER-SEGMENT\n"
                                  + res.segments.to_string(index=False)
                                  + "\n\nCOVERAGE BY GROUP\n"
                                  + res.coverage_by_group.to_string(index=False))
        print(f"\nwrote {a.report}")
    return 0


def cmd_cadence(a):
    df, mask, _ = LD.load_training(a.archive)
    for c in (a.cadence or ["never", "quarterly", "monthly"]):
        r = P.cadence_simulation(df, mask, cadence=c)
        if r:
            print(f"{c:10s} n={r['n']:5d} mean={r['mean_ape']*100:6.2f}% "
                  f"median={r['median_ape']*100:6.2f}% p90={r['p90_ape']*100:6.2f}% "
                  f"agg={r['agg_bias']*100:6.2f}%")
    return 0


def cmd_predict(a):
    bundle = T.load_bundle(a.out)
    q = QuoteInput(
        diameter_ft=a.diameter, height_ft=a.height, material=a.material,
        use_type=a.use_type, construction=_yn(a.construction),
        insulation=_yn(a.insulation),
        insulation_erection=_yn(a.insulation_erection),
        freight=_yn(a.freight), taxable=_yn(a.taxable),
        state=a.state, country=a.country, deck_style=a.deck,
        floor_style=a.floor, wage_type=a.wage, quantity=a.quantity,
        freeboard_in=a.freeboard, tank_name=a.tank_name,
        miles_tbt=a.miles, due_date=date.today(),
    )
    try:
        est = estimate(q, bundle=bundle)
    except RefusalError as e:
        print(e.code, "-", e.detail)
        return 2
    print(f"point            ${est.point:,.0f}")
    print(f"book             ${est.book:,.0f}")
    print(f"80% band         ${est.band80[0]:,.0f} - ${est.band80[1]:,.0f}")
    print(f"tier             {est.tier}  (group {est.group})")
    for k, v in est.components.items():
        print(f"  {k:32s} ${v:,.0f}")
    if est.tax is not None:
        print(f"  {'Tax':32s} ${est.tax:,.0f}")
        print(f"  {'Proposal Total':32s} ${est.proposal_total:,.0f}")
    if est.warnings:
        print("warnings        ", ", ".join(est.warnings))
    return 0


def cmd_score(a):
    bundle = T.load_bundle(a.out)
    B.score_csv(a.quotes, a.dest, bundle, quoted_total_col=a.quoted_col)
    print(f"wrote {a.dest}")
    return 0


def cmd_audit(a):
    df, mask, _ = LD.load_training(a.archive)
    res = P.backtest(df, mask, use_drift=a.drift, use_physics=a.physics,
                     progress=print if a.verbose else None)
    d = A.build_audit(res.rows, df, a.dest)
    print(f"wrote {a.dest}: {len(d)} rows, "
          f"{int((d['abs_pct'] > 30).sum())} beyond +/-30%")
    return 0


def cmd_info(a):
    bundle = T.load_bundle(a.out)
    if bundle is None:
        print("#MODEL - no bundle at", a.out)
        return 1
    status = "unknown"
    sp = Path(a.out) / T.STATUS_NAME
    if sp.exists():
        status = json.loads(sp.read_text()).get("status", "unknown")
    mi = model_info(bundle, last_retrain_status=status)
    print(f"bundle_id        {mi.bundle_id}")
    print(f"trained_through  {mi.trained_through}  ({mi.age_days} days old)")
    print(f"train rows       {mi.n_train_rows}")
    print(f"last retrain     {mi.last_retrain_status}")
    print(f"smear rest/top5  {mi.s_rest:.3f} / {mi.s_top5:.3f}")
    print(f"conformal groups {len(mi.bands)}")
    if mi.alarms:
        print("ALARMS          ", ", ".join(mi.alarms))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tbt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--drift", action="store_true", help="OPEN-1 term")
        p.add_argument("--physics", action="store_true", help="OPEN-2 term")
        p.add_argument("--sales-manager", action="store_true", help="OPEN-5")
        p.add_argument("--verbose", action="store_true")

    p = sub.add_parser("train"); p.add_argument("archive"); p.add_argument("--out", default="models")
    p.add_argument("--bootstrap-ledger", action="store_true"); common(p); p.set_defaults(fn=cmd_train)

    p = sub.add_parser("backtest"); p.add_argument("archive")
    p.add_argument("--report"); p.add_argument("--quarters", nargs="*")
    p.add_argument("--smear-on-point", action="store_true", help="OPEN-3")
    common(p); p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("cadence"); p.add_argument("archive")
    p.add_argument("--cadence", nargs="*"); p.set_defaults(fn=cmd_cadence)

    p = sub.add_parser("predict")
    for n, t in [("diameter", float), ("height", float)]:
        p.add_argument(f"--{n}", type=t, required=True)
    p.add_argument("--material", default="CS")
    p.add_argument("--use-type", default="Fire Protection Storage Tank")
    for n in ("construction", "insulation", "insulation-erection", "freight", "taxable"):
        p.add_argument(f"--{n}")
    p.add_argument("--state"); p.add_argument("--country", default="US")
    p.add_argument("--deck"); p.add_argument("--floor"); p.add_argument("--wage")
    p.add_argument("--quantity", type=int, default=1)
    p.add_argument("--freeboard", type=float, default=0.0)
    p.add_argument("--tank-name"); p.add_argument("--miles", type=float)
    p.add_argument("--out", default="models"); p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("score"); p.add_argument("quotes"); p.add_argument("dest")
    p.add_argument("--out", default="models"); p.add_argument("--quoted-col")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("audit"); p.add_argument("archive"); p.add_argument("dest")
    common(p); p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("info"); p.add_argument("--out", default="models")
    p.set_defaults(fn=cmd_info)

    a = ap.parse_args(argv)
    # argparse turns --insulation-erection into insulation_erection
    if not hasattr(a, "insulation_erection") and hasattr(a, "insulation-erection"):
        a.insulation_erection = getattr(a, "insulation-erection")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
