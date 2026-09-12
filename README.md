# TBT Tank Pricing Model — v3

Predicts tank pricing from specifications, trained on 6,679 quotes from the TBT
archive (Dec 2023 – Dec 2026).

**Accuracy: 8.7% mean error, 6.1% median**, measured by retraining each quarter on
prior data and scoring the next — which is how it will actually be run.

---

## Start here

```bash
pip install pandas scikit-learn joblib openpyxl xlwings
python tbt_model.py train archive_08-03-26_cleaned.csv --fast
python tbt_model.py predict --diameter 32 --height 30 --material CS --state MO \
       --construction yes --insulation no --freight yes
```

Everything runs on CPU. Training takes about two minutes on a ThinkPad; no GPU,
no cloud, no data leaves the machine.

## For Excel

```bash
xlwings addin install    # then: Excel ribbon -> xlwings -> Import Functions
```

Keep `tbt_model.py`, `tbt_pricing_model.joblib` and `excel_udf.py` in the same
folder as the workbook. Then:

```
=TBT_FLAG(H2, B2, C2, D2, E2, F2)     -> "OK +3%" / "LOW -22%" / "HIGH +31%"
=TBT_BREAKDOWN(B2, C2, D2, E2, F2)    -> spills the six components and the band
=TBT_PRICE(B2, C2, D2, E2, F2)        -> a single number
```

For more than a few dozen rows use `batch_score.py` instead — Excel will crawl if
it fires inference on every recalculation.

---

## Read this before anyone quotes off it

**Deploy `TBT_FLAG` first, not `TBT_PRICE`.** At 8.7% mean error the model cannot
set prices, but it reliably catches a transposed dimension, a missing scope line or
a forgotten stainless premium. That is real money at almost no risk, and it builds
the track record you would need before trusting it further.

**It underprices the largest 5% of quotes by 13-17%.** Large tanks scale
superlinearly — thicker plate, more shell courses, bigger cranes — and nothing in
the 42 available columns explains it. Every prediction in that band carries a
warning; treat those numbers as a floor.

**Do not use it to value a backlog.** The aggregate bias runs about -4%, driven
almost entirely by the large end.

**It models what TBT quotes, not what wins.** The archive has 173 Won and 142 Lost
out of 6,892 rows. There is no win-rate model here and the data cannot support one.

**Scope flags are worth about two points of accuracy.** Pass erection, insulation
and freight as TRUE/FALSE whenever the estimator knows. Left blank, the model infers
them (AUC 0.91–0.99) but you are giving away accuracy for nothing.

**Retrain monthly.** `--fast` makes it a two-minute scheduled task.
`=TBT_MODEL_INFO()` on the sheet shows how stale the model is.

---

## Two accounting facts worth confirming with finance

Both verified on 100% of rows:

```
Total Price    = 5 components + Freight     (tax-EXCLUSIVE)
Proposal Total = 5 components + Tax         (freight-EXCLUSIVE)
```

They are not nested. Also: the component prices already include margin — they are
sell prices, not costs, so no cost model can be built from this file.

---

## The most useful output

`audit_archive.xlsx` re-scores every historical quote using a model that never saw
it, sorted worst-first, naming the component that drives each gap. Material and
Construction account for 87% of large disagreements.

Work the top 50 with an estimator. If the model is right about most of them, you
have a business case. If it is wrong, you learn which column is missing — worth
more than another tenth of a point of accuracy.

---

`DESIGN.md` has the full technical record: what was tried, what failed and why,
the three bugs found in v1/v2, and where the remaining headroom is.
