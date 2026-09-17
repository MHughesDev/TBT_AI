"""The rolling-origin backtest and every required metric.

SPEC 3. Two people implementing this independently must produce the same
number. There is deliberately NO random-split or grouped-K-fold entry point
(SPEC 3.6): a developer wanting a quick check runs one real quarter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from . import conformal as CF
from . import features as F
from . import ledger as L
from . import model as M

HEADLINE_START = "2026Q1"
MIN_QUARTER_ROWS = 100
FIRST_QUARTER = "2024Q4"


# --------------------------------------------------------------------- metrics
def ape(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    return np.abs(pred - actual) / actual


def agg_bias(pred: np.ndarray, actual: np.ndarray) -> float:
    s = actual.sum()
    return float((pred.sum() - s) / s) if s > 0 else float("nan")


def quarter_metrics(pred_point, pred_book, actual, big_mask=None,
                    inside80=None) -> dict:
    a = np.asarray(actual, float)
    p = np.asarray(pred_point, float)
    b = np.asarray(pred_book, float)
    e = ape(p, a)
    out = {
        "n": int(len(a)),
        "mean_ape": float(np.mean(e)),
        "median_ape": float(np.median(e)),
        "p90_ape": float(np.percentile(e, 90)),
        "agg_bias_point": agg_bias(p, a),
        "agg_bias_book": agg_bias(b, a),
    }
    if big_mask is not None and np.any(big_mask):
        out["top5_bias_point"] = agg_bias(p[big_mask], a[big_mask])
        out["top5_bias_book"] = agg_bias(b[big_mask], a[big_mask])
        out["n_top5"] = int(np.sum(big_mask))
    if inside80 is not None and len(inside80):
        out["coverage80"] = float(np.mean(inside80))
    return out


def _quarter(s: pd.Series) -> pd.Series:
    return pd.PeriodIndex(pd.to_datetime(s), freq="Q").astype(str)


@dataclass
class BacktestResult:
    per_quarter: pd.DataFrame
    rows: pd.DataFrame                 # every scored row, for pooled metrics
    headline: dict = field(default_factory=dict)
    pooled: dict = field(default_factory=dict)
    segments: pd.DataFrame | None = None
    coverage_by_group: pd.DataFrame | None = None
    new_quote: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def report(self) -> str:
        lines = []
        cfg = ", ".join(f"{k}={v}" for k, v in self.config.items())
        lines.append(f"CONFIG  {cfg}")
        lines.append("")
        cols = ["quarter", "n", "mean_ape", "median_ape", "p90_ape",
                "agg_bias_point", "agg_bias_book", "coverage80"]
        have = [c for c in cols if c in self.per_quarter.columns]
        pq = self.per_quarter[have].copy()
        for c in have:
            if c not in ("quarter", "n"):
                pq[c] = (pq[c] * 100).round(2)
        lines.append(pq.to_string(index=False))
        lines.append("")
        h = self.headline
        lines.append("HEADLINE (unweighted mean across quarters from "
                     f"{HEADLINE_START})")
        lines.append(f"  mean APE      {h.get('mean_ape', float('nan'))*100:6.2f}%")
        lines.append(f"  median APE    {h.get('median_ape', float('nan'))*100:6.2f}%")
        lines.append(f"  p90 APE       {h.get('p90_ape', float('nan'))*100:6.2f}%")
        lines.append(f"  agg bias point{h.get('agg_bias_point', float('nan'))*100:6.2f}%")
        lines.append(f"  agg bias book {h.get('agg_bias_book', float('nan'))*100:6.2f}%")
        lines.append(f"  coverage80    {h.get('coverage80', float('nan'))*100:6.2f}%")
        p = self.pooled
        lines.append("POOLED over headline rows")
        lines.append(f"  mean APE      {p.get('mean_ape', float('nan'))*100:6.2f}%")
        lines.append(f"  top5 bias pt  {p.get('top5_bias_point', float('nan'))*100:6.2f}%"
                     f"   (n={p.get('n_top5', 0)})")
        lines.append(f"  top5 bias book{p.get('top5_bias_book', float('nan'))*100:6.2f}%")
        if self.new_quote:
            lines.append(f"  new-quote mean APE {self.new_quote.get('mean_ape', float('nan'))*100:6.2f}%"
                         f"   (n={self.new_quote.get('n', 0)})")
        return "\n".join(lines)


def backtest(df: pd.DataFrame, mask: pd.Series, *, use_drift=False,
             use_physics=False, use_sales_manager=False, variants=None,
             blend=C.BLEND_WEIGHT, recency=True, quarters=None,
             apply_smear_to_point=False, verbose=False, use_names=True,
             progress=None) -> BacktestResult:
    """Rolling origin by calendar quarter. SPEC 3.2.

    Everything is refitted from scratch inside the loop: the models, the tax
    table, the vocabularies, the envelope, the big threshold, the
    standardisation constants and the conformal table. Carrying any of them
    across quarters is leakage of the future (SPEC 4.5 item 11).
    """
    d = df.copy()
    d["_q"] = _quarter(d["Due Date"])
    d["_usable"] = mask.values

    all_q = sorted(q for q in d.loc[d["_usable"], "_q"].unique() if q >= FIRST_QUARTER)
    counts = d.loc[d["_usable"], "_q"].value_counts()
    all_q = [q for q in all_q if counts.get(q, 0) >= MIN_QUARTER_ROWS]
    if quarters:
        all_q = [q for q in all_q if q in set(quarters)]

    rows = []
    running_ledger = L.empty()

    for qi, q in enumerate(all_q):
        tr = d["_usable"] & (d["_q"] < q)
        te = d["_usable"] & (d["_q"] == q)
        if tr.sum() < 300 or te.sum() < MIN_QUARTER_ROWS:
            continue
        if progress:
            progress(f"  quarter {q}: train={int(tr.sum())} score={int(te.sum())}")

        bundle = M.fit_bundle(d, tr, use_drift=use_drift, use_physics=use_physics,
                              use_sales_manager=use_sales_manager,
                              variants=variants, blend=blend, recency=recency,
                              use_names=use_names)
        # conformal + smear from PRIOR quarters' forward predictions only
        bundle.conformal = CF.fit_conformal(running_ledger)
        bundle.smear = CF.fit_smear(running_ledger, bundle.big_threshold)

        test = d.loc[te]
        preds = M.predict_frame(bundle, test)

        point = preds["point"].values
        is_big = point >= bundle.big_threshold
        fam = F.engineer(test, use_family_map=bundle.use_family_map)["use_family"].values
        groups = np.array([CF.group_of(f, bg) for f, bg in zip(fam, is_big)])
        smear = np.where(is_big, bundle.smear.get("top5", 1.0),
                         bundle.smear.get("rest", 1.0))
        book = point * smear
        shown = point * smear if apply_smear_to_point else point

        lo = np.full(len(test), np.nan)
        hi = np.full(len(test), np.nan)
        for i, g in enumerate(groups):
            (l80, h80), _ = CF.bands_for(bundle.conformal, g, shown[i])
            lo[i], hi[i] = l80, h80

        actual = pd.to_numeric(test[C.TARGET], errors="coerce").values
        train_quotes = set(d.loc[tr, "Quote #"].astype(str))

        rows.append(pd.DataFrame({
            "quarter": q,
            "quote_key": test["Quote #"].astype(str).values,
            "due_date": test["Due Date"].values,
            "bundle_id": bundle.bundle_id,
            "point": shown,
            "book": book,
            "band80_lo": lo,
            "band80_hi": hi,
            "group": groups,
            "actual_total": actual,
            "is_new_quote": [str(qk) not in train_quotes
                             for qk in test["Quote #"].astype(str)],
            "use_family": fam,
            "Country": test.get("Country", pd.Series([None] * len(test))).values,
            "Wage Type": test.get("Wage Type", pd.Series([None] * len(test))).values,
            "Material": test.get("Material", pd.Series([None] * len(test))).values,
        }))
        running_ledger = pd.concat([running_ledger, rows[-1].reindex(
            columns=L.LEDGER_COLUMNS)], ignore_index=True)

    if not rows:
        raise ValueError("no quarters produced results")

    allrows = pd.concat(rows, ignore_index=True)
    allrows = allrows[allrows["actual_total"] > 0].copy()

    head = allrows[allrows["quarter"] >= HEADLINE_START].copy()
    if len(head) == 0:
        head = allrows.copy()

    # SPEC 3.4: one fixed top-5% threshold over ALL usable rows in the
    # headline window, defined on the ACTUAL value for the metric.
    T = float(np.percentile(head["actual_total"].values, 95))
    head["is_top5"] = head["actual_total"] >= T
    allrows["is_top5"] = allrows["actual_total"] >= T

    per_q = []
    for q, g in allrows.groupby("quarter"):
        inside = None
        if g["band80_lo"].notna().any():
            gg = g.dropna(subset=["band80_lo", "band80_hi"])
            inside = ((gg["actual_total"] >= gg["band80_lo"])
                      & (gg["actual_total"] <= gg["band80_hi"])).values
        m = quarter_metrics(g["point"].values, g["book"].values,
                            g["actual_total"].values,
                            big_mask=g["is_top5"].values, inside80=inside)
        m["quarter"] = q
        per_q.append(m)
    per_quarter = pd.DataFrame(per_q).sort_values("quarter").reset_index(drop=True)

    hq = per_quarter[per_quarter["quarter"] >= HEADLINE_START]
    if len(hq) == 0:
        hq = per_quarter
    headline = {c: float(hq[c].mean()) for c in
                ("mean_ape", "median_ape", "p90_ape",
                 "agg_bias_point", "agg_bias_book")
                if c in hq.columns}
    if "coverage80" in hq.columns and hq["coverage80"].notna().any():
        headline["coverage80"] = float(hq["coverage80"].mean())

    pooled = quarter_metrics(head["point"].values, head["book"].values,
                             head["actual_total"].values,
                             big_mask=head["is_top5"].values)
    pooled["top5_threshold"] = T

    nq = head[head["is_new_quote"]]
    new_quote = {}
    if len(nq) > 0:
        new_quote = quarter_metrics(nq["point"].values, nq["book"].values,
                                    nq["actual_total"].values)

    segs = []
    for col in ("use_family", "Country", "Wage Type", "Material"):
        if col not in head.columns:
            continue
        for val, g in head.groupby(head[col].astype(str)):
            if len(g) < 25:
                continue
            e = ape(g["point"].values, g["actual_total"].values)
            signed = (g["point"].values - g["actual_total"].values) / g["actual_total"].values
            segs.append({"dimension": col, "value": val, "n": len(g),
                         "mean_ape": float(np.mean(e)),
                         "median_signed": float(np.median(signed)),
                         "iqr_signed": float(np.percentile(signed, 75)
                                             - np.percentile(signed, 25))})
    segments = pd.DataFrame(segs)

    cov = []
    cv = head.dropna(subset=["band80_lo", "band80_hi"])
    for g, grp in cv.groupby("group"):
        inside = ((grp["actual_total"] >= grp["band80_lo"])
                  & (grp["actual_total"] <= grp["band80_hi"]))
        cov.append({"group": g, "n": len(grp), "coverage80": float(inside.mean()),
                    "half_width": float(np.mean(grp["band80_hi"] / grp["point"] - 1.0))})
    coverage_by_group = pd.DataFrame(cov)

    return BacktestResult(
        per_quarter=per_quarter, rows=allrows, headline=headline, pooled=pooled,
        segments=segments, coverage_by_group=coverage_by_group, new_quote=new_quote,
        config={"drift": use_drift, "physics": use_physics,
                "sales_manager": use_sales_manager, "blend": blend,
                "recency": recency, "names": use_names,
                "smear_on_point": apply_smear_to_point,
                "variants": len(C.GBM_VARIANTS if variants is None else variants)},
    )


def cadence_simulation(df: pd.DataFrame, mask: pd.Series, cadence: str = "monthly",
                       start: str = HEADLINE_START, **kw) -> dict:
    """SPEC 3.4 secondary: month-by-month forward test with the bundle refit at
    the given cadence. This is the number the deployment actually experiences."""
    d = df.copy()
    d["_m"] = pd.PeriodIndex(pd.to_datetime(d["Due Date"]), freq="M").astype(str)
    d["_q"] = _quarter(d["Due Date"])
    d["_usable"] = mask.values
    months = sorted(m for m in d.loc[d["_usable"], "_m"].unique()
                    if pd.Period(m, freq="M") >= pd.Period(start, freq="Q").start_time.to_period("M"))

    bundle = None
    last_fit = None
    errs, preds, acts = [], [], []
    for m in months:
        tr = d["_usable"] & (d["_m"] < m)
        te = d["_usable"] & (d["_m"] == m)
        if tr.sum() < 300 or te.sum() < 20:
            continue
        refit = (bundle is None)
        if cadence == "monthly":
            refit = True
        elif cadence == "quarterly":
            q = str(pd.Period(m, freq="M").asfreq("Q"))
            refit = refit or (last_fit != q)
            last_fit = q
        elif cadence == "never":
            refit = bundle is None
        if refit:
            bundle = M.fit_bundle(d, tr, **kw)
        pr = M.predict_frame(bundle, d.loc[te])
        a = pd.to_numeric(d.loc[te, C.TARGET], errors="coerce").values
        p = pr["point"].values
        good = a > 0
        errs.append(ape(p[good], a[good]))
        preds.append(p[good]); acts.append(a[good])
    if not errs:
        return {}
    e = np.concatenate(errs)
    return {"cadence": cadence, "n": int(len(e)), "mean_ape": float(np.mean(e)),
            "median_ape": float(np.median(e)), "p90_ape": float(np.percentile(e, 90)),
            "agg_bias": agg_bias(np.concatenate(preds), np.concatenate(acts))}
