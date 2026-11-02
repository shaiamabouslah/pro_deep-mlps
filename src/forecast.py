"""Short-horizon soil-moisture forecaster.

Compares foundation models (Chronos-2, TimesFM) with a supervised baseline (N-HiTS)
and a seasonal-naive reference. Metrics: MASE, sMAPE, RMSE, CRPS (lower is better).

    python -m src.forecast --config config/field_config.yaml --horizon 14
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "soil_moisture_sensor_mean"
# same levels for every probabilistic model -> comparable CRPS
QUANTILE_LEVELS = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1 ... 0.9


def load_target_series(csv_path, target: str = TARGET) -> pd.Series:
    """Load the aligned table and return a clean, gap-free daily target series."""
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not in {csv_path}. Available: {list(df.columns)}")
    s = df.set_index("date")[target].astype(float)
    s = s.asfreq("D").interpolate(method="time").ffill().bfill()
    s.name = target
    return s


def split(series: pd.Series, horizon: int):
    """Hold out the last `horizon` days as the test set."""
    if len(series) <= horizon + 10:
        raise ValueError(f"Series too short ({len(series)}) for horizon {horizon}.")
    return series.iloc[:-horizon], series.iloc[-horizon:]


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def smape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y) + np.abs(yhat)
    denom[denom == 0] = 1e-8
    return float(np.mean(2.0 * np.abs(y - yhat) / denom) * 100.0)


def mase(y, yhat, y_train, m: int = 1) -> float:
    y, yhat, y_train = np.asarray(y, float), np.asarray(yhat, float), np.asarray(y_train, float)
    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    scale = np.mean(naive_errors) if naive_errors.size else 1e-8
    return float(np.mean(np.abs(y - yhat)) / (scale if scale > 0 else 1e-8))


def crps_from_quantiles(y, q_preds, levels) -> float:
    """
    Discrete (pinball) approximation of the CRPS.
    q_preds: array of shape (horizon, n_levels). Lower = better.
    """
    y = np.asarray(y, float)[:, None]
    q = np.asarray(q_preds, float)
    tau = np.asarray(levels, float)[None, :]
    pinball = np.where(y >= q, tau * (y - q), (1.0 - tau) * (q - y))
    return float(2.0 * pinball.mean())


def crps_deterministic(y, yhat) -> float:
    """For a point forecast the predictive distribution is a Dirac -> CRPS = MAE."""
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def score(name, y_test, point, quantiles, y_train) -> dict:
    if quantiles is not None:
        crps = crps_from_quantiles(y_test, quantiles, QUANTILE_LEVELS)
        prob = True
    else:
        crps = crps_deterministic(y_test, point)
        prob = False
    return {
        "model": name,
        "MASE": round(mase(y_test, point, y_train), 4),
        "sMAPE": round(smape(y_test, point), 4),
        "RMSE": round(rmse(y_test, point), 5),
        "CRPS": round(crps, 5),
        "probabilistic": prob,
    }


# models -- each returns (point_forecast[H], quantiles[H, K] or None)
def run_seasonal_naive(train: pd.Series, horizon: int, m: int = 7):
    """Repeat the last m-day block. Deterministic -> no quantiles."""
    last = train.values[-m:]
    reps = int(np.ceil(horizon / m))
    return np.tile(last, reps)[:horizon], None


def run_nhits(train: pd.Series, horizon: int):
    """Supervised N-HiTS (Nixtla neuralforecast). Point forecast only here."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    dft = pd.DataFrame({"unique_id": "soil", "ds": train.index, "y": train.values})
    input_size = int(min(max(2 * horizon, 21), max(len(train) // 2, horizon + 1)))
    # deliberately small: ~200 training points can't support the default ~2.4M params
    model = NHITS(
        h=horizon, input_size=input_size,
        n_blocks=[1, 1, 1], mlp_units=[[64, 64], [64, 64], [64, 64]],
        max_steps=800, learning_rate=1e-3, scaler_type="standard",
        random_seed=1, enable_progress_bar=False,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(dft)
    return nf.predict()["NHITS"].to_numpy()[:horizon], None


def _extract_quantiles(pred_df: pd.DataFrame, horizon: int):
    """Pull the QUANTILE_LEVELS columns out of a Chronos predict_df result."""
    cols = []
    for lv in QUANTILE_LEVELS:
        found = next((c for c in (str(lv), lv, f"{lv:.1f}") if c in pred_df.columns), None)
        if found is None:
            return None
        cols.append(pred_df[found].to_numpy(dtype=float)[:horizon])
    return np.stack(cols, axis=1)


def run_chronos(train: pd.Series, horizon: int, model_id: str = "amazon/chronos-2"):
    """
    Chronos-2 (Amazon) -- a time-series foundation model. 120M params,
    zero-shot, CPU-friendly, natively supports covariates.
    API: chronos-forecasting >= 2.0 -> Chronos2Pipeline.predict_df().
    """
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained(model_id, device_map="cpu")
    context_df = pd.DataFrame({
        "id": "soil", "timestamp": train.index, "target": train.values,
    })
    pred = pipe.predict_df(
        context_df,
        prediction_length=horizon,
        quantile_levels=QUANTILE_LEVELS,
        id_column="id", timestamp_column="timestamp", target="target",
    )
    quantiles = _extract_quantiles(pred, horizon)
    # point forecast = median if available, else the middle quantile column
    for col in ("0.5", 0.5, "predictions", "mean"):
        if col in pred.columns:
            point = pred[col].to_numpy(dtype=float)[:horizon]
            break
    else:
        point = quantiles[:, len(QUANTILE_LEVELS) // 2] if quantiles is not None \
            else pred.select_dtypes("number").iloc[:, -1].to_numpy(float)[:horizon]
    return point, quantiles


def run_timesfm(train: pd.Series, horizon: int):
    """
    Google TimesFM 2.5, zero-shot. `compile()` needs a ForecastConfig.
    `forecast()` returns (point_forecast, quantile_forecast); the quantile array
    carries [mean, q0.1 ... q0.9] on its last axis.
    """
    import timesfm

    if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        model.compile(
            timesfm.ForecastConfig(
                max_context=1024, max_horizon=max(256, horizon),
                normalize_inputs=True, use_continuous_quantile_head=True,
                force_flip_invariance=True, infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        point, qf = model.forecast(horizon=horizon, inputs=[train.values.astype("float32")])
        point = np.asarray(point)[0][:horizon]

        quantiles = None
        qf = np.asarray(qf)
        if qf.ndim == 3 and qf.shape[-1] >= 10:
            # channel 0 is the mean, channels 1..9 are deciles 0.1..0.9
            quantiles = qf[0][:horizon, 1:10]
        return point, quantiles

    # older 2.x checkpoint API
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu", per_core_batch_size=1,
            horizon_len=horizon, context_len=min(512, len(train)),
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )
    point, _ = tfm.forecast([train.values.astype("float32")], freq=[0])
    return np.asarray(point)[0][:horizon], None


MODELS = {
    "seasonal_naive": run_seasonal_naive,
    "chronos": run_chronos,
    "timesfm": run_timesfm,
    "nhits": run_nhits,
}


def plot_forecasts(train, test, points, quantiles, out_path, context_days: int = 60):
    """Actual vs each model's forecast, with an uncertainty band for the best
    probabilistic model. Saved as a PNG you can drop straight into the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ctx = train.iloc[-context_days:]
    ax.plot(ctx.index, ctx.values, color="0.55", lw=1.2, label="history")
    ax.plot(test.index, test.values, color="black", lw=2.4, label="actual", zorder=5)

    # 10-90% band for the first probabilistic model available
    for name, q in quantiles.items():
        if q is not None:
            ax.fill_between(test.index, q[:, 0], q[:, -1], alpha=0.18, color="tab:blue",
                            label=f"{name} 10-90%")
            break

    for name, yhat in points.items():
        ax.plot(test.index, yhat, lw=1.7, ls="--", label=name)

    ax.axvline(test.index[0], color="0.7", ls=":", lw=1)
    ax.set_title(f"Soil-moisture forecast vs actual — last {len(test)} days held out")
    ax.set_ylabel("soil moisture (m³/m³)")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_comparison(csv_path, horizon: int, out_dir) -> pd.DataFrame:
    series = load_target_series(csv_path)
    train, test = split(series, horizon)
    logger.info("Target '%s': %d days total -> train %d / test %d (horizon %d)",
                TARGET, len(series), len(train), len(test), horizon)

    rows, points, quantiles = [], {}, {}
    for name, fn in MODELS.items():
        try:
            logger.info("Running %s ...", name)
            point, q = fn(train, horizon)
            point = np.asarray(point, dtype=float)[:horizon]
            if point.shape[0] != horizon or not np.all(np.isfinite(point)):
                raise ValueError(f"bad output shape/values ({point.shape})")
            if q is not None:
                q = np.asarray(q, dtype=float)[:horizon]
                if q.shape != (horizon, len(QUANTILE_LEVELS)):
                    logger.warning("  %s: unexpected quantile shape %s -- CRPS falls back to MAE",
                                   name, q.shape)
                    q = None
            points[name], quantiles[name] = point, q
            rows.append(score(name, test.values, point, q, train.values))
            r = rows[-1]
            logger.info("  %s -> MASE=%.3f sMAPE=%.2f RMSE=%.4f CRPS=%.4f%s",
                        name, r["MASE"], r["sMAPE"], r["RMSE"], r["CRPS"],
                        "" if r["probabilistic"] else "  (deterministic: CRPS=MAE)")
        except Exception as e:
            logger.warning("  SKIPPED %s: %s: %s", name, type(e).__name__, e)

    if not rows:
        raise RuntimeError("No model produced a forecast -- check installs (see WEEK3_FORECAST.md).")

    results = pd.DataFrame(rows).sort_values("MASE").reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "forecast_metrics.csv", index=False)

    pred_df = pd.DataFrame({"date": test.index, "actual": test.values})
    for name, yhat in points.items():
        pred_df[name] = yhat
        if quantiles.get(name) is not None:
            pred_df[f"{name}_q0.1"] = quantiles[name][:, 0]
            pred_df[f"{name}_q0.9"] = quantiles[name][:, -1]
    pred_df.to_csv(out_dir / "forecast_predictions.csv", index=False)

    png = out_dir / "forecast_comparison.png"
    try:
        plot_forecasts(train, test, points, quantiles, png)
        plotted = True
    except Exception as e:
        logger.warning("Plot skipped (%s: %s) -- try `pip install matplotlib`", type(e).__name__, e)
        plotted = False

    print("\n=== Forecast comparison (lower is better) ===")
    print(results.to_string(index=False))
    print(f"\nBest model by MASE: {results.iloc[0]['model']}")
    print("Note: CRPS for deterministic models (probabilistic=False) is the MAE.")
    print(f"Wrote: {out_dir / 'forecast_metrics.csv'}")
    print(f"Wrote: {out_dir / 'forecast_predictions.csv'}")
    if plotted:
        print(f"Wrote: {png}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Soil-moisture forecaster comparison")
    parser.add_argument("--config", default="config/field_config.yaml")
    parser.add_argument("--horizon", type=int, default=14, help="forecast horizon in days")
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_path = cfg.processed_dir / f"{cfg.name}_aligned_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `python -m src.pipeline` first.")
    run_comparison(csv_path, args.horizon, out_dir=cfg.processed_dir)


if __name__ == "__main__":
    main()
