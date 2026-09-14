# TBT Tank Pricing Model — v4

Predicts tank pricing from specifications, trained on 6,679 quotes from the TBT
archive (Dec 2023 – Dec 2026).

**Accuracy: 8.5% mean absolute error, 5.9% median**, measured by retraining each
quarter on prior data and scoring the next — which is how it will actually be run.

**Retrain monthly, not quarterly.** It is the single largest lever in the whole
system — larger than every modelling change combined:

| Cadence | mean | median | p90 |
|---|---|---|---|
| Never retrain | 9.99% | 7.41% | 21.5% |
| Quarterly | 9.28% | 6.71% | 19.6% |
| **Monthly** | **8.69%** | **6.09%** | **19.3%** |

Two minutes of CPU a month buys 1.3 points. Put it on a scheduled task.

---

## The one thing to understand

**The model prices scope. It does not guess scope.**

Whether a quote includes erection, insulation, freight or sales tax is a commercial
decision the estimator already knows. It is not a property of the tank and it is
not inferable from the specs. Every call must state it:

```python
predict(diameter=40, height=36, material="CS", state="TX",
        construction=True,      # do we erect it
        insulation=True,        # is insulation supplied
        freight=True,           # do we ship it
        taxable=False)          # does sales tax apply to this customer
```

Omit any of them and you get a `ScopeError`, not a number. That is deliberate.
Earlier versions inferred scope with classifiers; a classifier that is confidently
wrong silently changes the price without changing anything visible on the sheet.

Scope matters enormously. Same tank, four scopes:

| Scope | Price |
|---|---|
| Shop only, customer collects | $213,457 |
| Erected, no insulation | $330,477 |
| Insulation supplied, customer installs | $413,190 |
| Full scope | $458,215 |

Material and Fabrication carry no flag — they are present on 6,679 of 6,679 rows.
They are processes, not options.

### Pass the tank name too

`tank_name` is optional but worth supplying. The process descriptor carries real
signal on engineered-to-order work — median $/sq-ft runs 1.35x base for digesters
against 0.68x for multi-zone configurations. A generic "Tank 1" is itself
informative: it usually means a commodity shell.

```python
predict(..., tank_name="Primary Anaerobic Digester - Hybrid")   # +3.2%
predict(..., tank_name="Flow Equalization Basin")               # -2.7%
```

---

## Start here

```bash
pip install -r requirements.txt

python prepare_data.py archive_08-03-26_cleaned.csv archive_prepared.csv
python tbt_model.py train archive_prepared.csv --fast
python tbt_model.py predict --diameter 32 --height 30 --material CS --state MO \
       --construction yes --insulation no --freight yes --taxable no
```

Everything runs on CPU. Training is about two minutes on a ThinkPad — no GPU, no
cloud, no data leaves the machine.

`prepare_data.py` backfills the five scope columns from the archive by reading
which components carry pricing. That is a one-time migration. Going forward these
should be captured at quote time as real fields: a blank price and a genuinely
excluded scope look identical after the fact, and only the estimator can tell them
apart.

## For Excel

```bash
xlwings addin install    # then: Excel ribbon -> xlwings -> Import Functions
```

Keep `tbt_model.py`, `tbt_pricing_bundle.joblib` and `excel_udf.py` beside the
workbook. Add four Yes/No dropdown columns for scope, then:

```
=TBT_FLAG(H2, B2, C2, D2, E2, F2, G2)   -> "OK +3%" / "LOW -22%" / "HIGH +31%"
=TBT_BREAKDOWN(B2, C2, D2, E2, F2, G2)  -> the six components, band and warnings
=TBT_PRICE(B2, C2, D2, E2, F2, G2)      -> a single number
```

Blank scope returns `#SCOPE`, not a guess.

For more than a few dozen rows use `batch_score.py` — Excel will crawl if it fires
inference on every recalculation.

---

## Read this before anyone quotes off it

**Deploy `TBT_FLAG` first, not `TBT_PRICE`.** At 8.7% mean error the model cannot
set prices, but it reliably catches a transposed dimension, a missing scope line or
a forgotten stainless premium. Real money at almost no risk, and it builds the
track record you would need before trusting it further.

**It underprices the largest 5% of quotes by 13–17%.** Large tanks scale
superlinearly — thicker plate, more shell courses, bigger cranes — and nothing in
the available columns explains it. Every prediction in that band carries a warning;
treat those numbers as a floor.

**Do not use it to value a backlog.** Aggregate bias runs about −4%, driven almost
entirely by the large end.

**It models what TBT quotes, not what wins.** The archive has 173 Won and 142 Lost
out of 6,892 rows. There is no win-rate model here and the data cannot support one.

**Retrain monthly.** `--fast` makes it a two-minute scheduled task, and
`=TBT_MODEL_INFO()` shows how stale the model is.

---

## Two accounting facts worth confirming with finance

Both verified on 100% of rows:

```
Total Price    = 5 components + Freight     (tax-EXCLUSIVE)
Proposal Total = 5 components + Tax         (freight-EXCLUSIVE)
```

They are not nested. Also: the component prices already include margin — they are
sell prices, not costs, so no cost model can be built from this file.

Sales tax is two separate things. Geography sets the **rate** (near-constant within
a state, median std 0.008). The customer's exemption status sets **whether it
applies** — 0% of Oregon quotes are taxed, 38% of Texas, 91% of New Jersey. That is
why `IS_TAXABLE` is an input rather than something derived from `State`.

---

## The most useful output

`audit_archive.xlsx` re-scores every historical quote with a model that never saw
it, sorted worst-first, naming the component driving each gap. Material and
Construction account for 87% of large disagreements.

Work the top 50 with an estimator. If the model is right about most of them you
have a business case. If it is wrong you learn which column is missing — worth more
than another tenth of a point of accuracy.

---

`HANDOFF.md` has open problems and prioritised next steps. `DESIGN.md` has the full
technical record: what was tried, what failed, and why.
