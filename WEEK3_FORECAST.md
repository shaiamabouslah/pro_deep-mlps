# Week 3 — Forecasting & model comparison

**Goal (brief §2, Step 2):** a short-horizon forecaster for soil moisture, with a
foundation model vs a supervised baseline, judged on MASE / sMAPE / RMSE (§9).

**What we compare** (all mandated by the brief §6, plus the required reference):

| Model | Type | Trains? |
|---|---|---|
| **Chronos-2** (Chronos-Bolt) | foundation | no — zero-shot |
| **TimesFM** | foundation | no — zero-shot |
| **N-HiTS** | supervised deep | yes, on your field |
| **seasonal-naive** | reference | no |

Input: `data/processed/<field>_aligned_daily.csv` (built in Phase 1).
Target: `soil_moisture_sensor_mean` (the real ISMN sensor series).
First pass is **univariate**; adding weather/NDVI covariates is the next step.

---

## 1. Install (once)

```bash
.venv\Scripts\activate
pip install -r requirements-forecast.txt
```

The first run downloads the model weights (Chronos-Bolt ~200 MB, TimesFM a few
hundred MB) — one-time, then cached. **CPU is fine** for ~200 days of data.

> On a modest laptop, if PyTorch/TimesFM are too heavy to install, see the
> **Google Colab** option below — it runs everything free in the cloud.

## 2. Run

```bash
python -m src.forecast --config config/field_config.yaml --horizon 14
```

`--horizon` = how many days ahead to forecast (default 14). The script holds out
the last `horizon` days as the test set, runs every model, and prints a table.

## 3. Read the result

```
=== Forecast comparison (lower is better) ===
        model   MASE  sMAPE    RMSE
      chronos  0.83  6.12   0.021
        nhits  0.95  7.44   0.028
 seasonal_naive 1.00  8.10   0.031
       timesfm  0.88  6.55   0.024

Best model by MASE: chronos
```

- **How to read MASE here.** MASE divides the *h*-step-ahead error by the
  in-sample **1-step** naive error. Soil moisture barely moves day to day, so
  that denominator is tiny and **MASE >> 1 is normal at h=14** — it does *not*
  mean the model is broken. What matters is the **relative ranking**, and
  whether a model beats the `seasonal_naive` row.
- Two files are written to `data/processed/`: `forecast_metrics.csv` (the table)
  and `forecast_predictions.csv` (actual vs each model's forecast).

If a model is missing from the table, the log says why (usually "not installed"
or a library-version mismatch) — the others still ran.

## 4. If a foundation model won't install locally — use Colab (free)

1. Go to https://colab.research.google.com → New notebook.
2. Upload `<field>_aligned_daily.csv` (from `data/processed/`).
3. In a cell:
   ```python
   !pip install neuralforecast chronos-forecasting timesfm[torch] -q
   ```
4. Paste the model functions from `src/forecast.py` and run the comparison.
   Colab gives you a free CPU/GPU, so nothing runs on your laptop.

## 5. Running the *actual* Chronos-2 (mandated)

`src/forecast.py` defaults to `amazon/chronos-bolt-base` (Chronos family, light,
CPU-friendly). To use the newest **Chronos-2** specifically:

```python
from autogluon.timeseries import TimeSeriesPredictor
# fit with the chronos-2 preset (zero-shot, supports covariates)
```
Swap it in once you confirm the exact model id available in your installed
`chronos-forecasting` / AutoGluon version.

## Next steps (Week 4)

- Add the **covariates** (weather, ET0, NDVI) to Chronos-2 and N-HiTS.
- Add **CRPS / quantile loss** for the probabilistic forecast (Chronos already
  returns quantiles; Lag-Llama is the probabilistic specialist if you want it).
- Longer history + synthetic gap-filling (brief §5.2) to strengthen training.
