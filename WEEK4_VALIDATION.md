# Week 4 — Validation, covariates & gap-filling

Three deliverables, run in this order. Each reuses the aligned table from Phase 1
and the model runners from `src/forecast.py`. Nothing new to install beyond
`requirements-forecast.txt`.

---

## 1. Sliding-origin backtest — choose the winner properly

Week 3 ranked the models on ONE window. This replays them over several windows
and reports mean **and** standard deviation, so the "best model" is defensible.

```bash
python -m src.backtest --config config/field_config.yaml --horizon 14 --windows 6 --step 7
```

- Runs 4 models × 6 windows → allow **~10-15 min** (foundation weights reload,
  N-HiTS retrains each window).
- Output: `data/processed/backtest_summary.csv` (mean±std per model) and
  `backtest_per_window.csv` (every window).

**How to read it.** Pick the model with the lowest `RMSE_mean` *and* a low
`RMSE_std`. A model that wins on average but swings wildly (high std) is less
trustworthy than a slightly-worse but stable one. This is the number you quote
in the report — not the single-window Week-3 result.

---

## 2. Covariate study — does the weather help?

Feeds rain, ET₀, temperature (known-future) and NDVI (past) into Chronos-2 and
N-HiTS, and prints the univariate → +covariate improvement.

```bash
python -m src.forecast_cov --config config/field_config.yaml --horizon 14
```

- Output: `data/processed/forecast_cov_metrics.csv` + an explicit
  "covariate effect (RMSE)" summary in the console.

> **Honesty note for your report.** The backtest uses the *actual recorded*
> weather over the test window as the "future" covariate — i.e. a **perfect
> weather forecast**. So the covariate gain you see is an **optimistic upper
> bound**: it shows how much weather information *could* help. A real deployment
> would feed a weather *forecast* (with its own error), so the real-world gain is
> somewhat smaller. State this explicitly — it's exactly the kind of rigour §9
> rewards.

If `chronos+cov` / `nhits+cov` beat their univariate versions, you've confirmed
the Week-3 diagnosis: the missing ingredient was future weather, not model power.

---

## 3. Synthetic gap-filling (brief §5.2)

Tests whether Chronos-2 zero-shot imputes sensor gaps better than plain
interpolation, then produces a gap-filled series.

```bash
python -m src.gapfill --config config/field_config.yaml --n-gaps 8 --gap-len 5
```

- Punches 8 synthetic 5-day gaps into known stretches (where we know the truth),
  fills them two ways, and compares the error.
- Output: `data/processed/gapfill_eval.csv` (per-gap errors) and a gap-filled
  series `<field>_soil_gapfilled.csv` with a `method` column flagging each day as
  `observed` / `interpolation` / `chronos`.

**Likely finding.** Over short gaps (a few days) linear interpolation is hard to
beat, because soil moisture is smooth; Chronos tends to help more on **longer**
gaps where interpolation drifts. Try `--gap-len 10` to see the difference — a
nice figure/table for the report.

---

## What Week 4 gives you

| Brief item (Week 4) | Delivered by |
|---|---|
| "Forecast accuracy measured" | `backtest.py` (multi-window mean±std) |
| Covariate-informed forecasting | `forecast_cov.py` |
| "Synthetic gap-filling in place" (§5.2) | `gapfill.py` |

After running all three, tell me the numbers and we lock in the winning model +
setup for the Week-5 fusion agent.
