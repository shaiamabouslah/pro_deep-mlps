"""Covariate study: measures whether adding weather and NDVI covariates improves the
soil-moisture forecast versus the univariate models.

    python -m src.forecast_cov --config config/field_config.yaml --horizon 14
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config_loader import load_config
from .forecast import (QUANTILE_LEVELS, _extract_quantiles, run_chronos, run_nhits,
                       run_seasonal_naive, run_timesfm, score)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET = "soil_moisture_sensor_mean"
FUTURE_COVS = ["precip_sum_mm", "et0_sum_mm", "temp_mean_c"]   # known from a weather forecast
PAST_COVS = ["ndvi"]                                           # only known up to "today"


def load_frame(csv_path) -> pd.DataFrame:
    """Aligned table -> clean daily frame with target + available covariates."""
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").set_index("date")
    cols = [c for c in ([TARGET] + FUTURE_COVS + PAST_COVS) if c in df.columns]
    df = df[cols].astype(float)
    df = df.asfreq("D").interpolate(method="time").ffill().bfill()
    return df


def _present(df, names):
    return [c for c in names if c in df.columns]


def run_chronos_cov(train_df, test_df, horizon):
    """Chronos-2 with covariates via predict_df(future_df=...)."""
    from chronos import Chronos2Pipeline

    fut_cols = _present(train_df, FUTURE_COVS)
    past_cols = _present(train_df, PAST_COVS)

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

    ctx = train_df.reset_index().rename(columns={"date": "timestamp"})
    ctx.insert(0, "id", "soil")
    ctx = ctx[["id", "timestamp", TARGET] + fut_cols + past_cols]

    fut = test_df.reset_index().rename(columns={"date": "timestamp"})
    fut.insert(0, "id", "soil")
    fut = fut[["id", "timestamp"] + fut_cols]  # only KNOWN-future covariates

    pred = pipe.predict_df(
        ctx, future_df=fut, prediction_length=horizon,
        quantile_levels=QUANTILE_LEVELS,
        id_column="id", timestamp_column="timestamp", target=TARGET,
    )
    quantiles = _extract_quantiles(pred, horizon)
    for col in ("0.5", 0.5, "predictions", "mean"):
        if col in pred.columns:
            return pred[col].to_numpy(float)[:horizon], quantiles
    point = quantiles[:, len(QUANTILE_LEVELS) // 2] if quantiles is not None \
        else pred.select_dtypes("number").iloc[:, -1].to_numpy(float)[:horizon]
    return point, quantiles


def run_nhits_cov(train_df, test_df, horizon):
    """N-HiTS with future + historical exogenous variables (neuralforecast)."""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    fut_cols = _present(train_df, FUTURE_COVS)
    past_cols = _present(train_df, PAST_COVS)

    dft = train_df.reset_index().rename(columns={"date": "ds", TARGET: "y"})
    dft.insert(0, "unique_id", "soil")
    dft = dft[["unique_id", "ds", "y"] + fut_cols + past_cols]

    input_size = int(min(max(2 * horizon, 21), max(len(train_df) // 2, horizon + 1)))
    model = NHITS(
        h=horizon, input_size=input_size,
        futr_exog_list=fut_cols or None, hist_exog_list=past_cols or None,
        n_blocks=[1, 1, 1], mlp_units=[[64, 64], [64, 64], [64, 64]],
        max_steps=800, learning_rate=1e-3, scaler_type="standard",
        random_seed=1, enable_progress_bar=False,
    )
    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(dft)

    futr = test_df.reset_index().rename(columns={"date": "ds"})
    futr.insert(0, "unique_id", "soil")
    futr = futr[["unique_id", "ds"] + fut_cols]
    return nf.predict(futr_df=futr)["NHITS"].to_numpy(float)[:horizon], None


def run_comparison(csv_path, horizon, out_dir) -> pd.DataFrame:
    df = load_frame(csv_path)
    train_df, test_df = df.iloc[:-horizon], df.iloc[-horizon:]
    train_s, test_s = df[TARGET].iloc[:-horizon], df[TARGET].iloc[-horizon:]
    logger.info("Covariate study: %d train / %d test days. future=%s past=%s",
                len(train_df), len(test_df), _present(df, FUTURE_COVS), _present(df, PAST_COVS))

    y_train, y_test = train_s.values, test_s.values
    rows = []

    def add(label, fn):
        try:
            logger.info("Running %s ...", label)
            point, q = fn()
            point = np.asarray(point, float)[:horizon]
            if not (point.shape[0] == horizon and np.all(np.isfinite(point))):
                raise ValueError(f"bad output {point.shape}")
            if q is not None:
                q = np.asarray(q, float)[:horizon]
                if q.shape != (horizon, len(QUANTILE_LEVELS)):
                    q = None
            rows.append(score(label, y_test, point, q, y_train))
            r = rows[-1]
            logger.info("   %-16s MASE=%.3f RMSE=%.4f CRPS=%.4f", label, r["MASE"], r["RMSE"], r["CRPS"])
        except Exception as e:
            logger.warning("   SKIPPED %s: %s: %s", label, type(e).__name__, e)

    # reference + univariate
    add("seasonal_naive", lambda: run_seasonal_naive(train_s, horizon))
    add("chronos",        lambda: run_chronos(train_s, horizon))
    add("chronos+cov",    lambda: run_chronos_cov(train_df, test_df, horizon))
    add("timesfm",        lambda: run_timesfm(train_s, horizon))
    add("nhits",          lambda: run_nhits(train_s, horizon))
    add("nhits+cov",      lambda: run_nhits_cov(train_df, test_df, horizon))

    if not rows:
        raise RuntimeError("No model produced a forecast.")
    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "forecast_cov_metrics.csv", index=False)

    print("\n=== Univariate vs +covariate (lower is better) ===")
    print(results.to_string(index=False))

    print("\n--- covariate effect (RMSE) ---")
    m = results.set_index("model")["RMSE"]
    for base in ("chronos", "nhits"):
        if base in m.index and f"{base}+cov" in m.index:
            delta = (m[base] - m[f"{base}+cov"]) / m[base] * 100
            print(f"  {base}: {m[base]:.4f} -> {m[f'{base}+cov']:.4f}  ({delta:+.1f}% RMSE)")
    print(f"\nWrote: {out_dir / 'forecast_cov_metrics.csv'}")
    return results


def main():
    p = argparse.ArgumentParser(description="Covariate study")
    p.add_argument("--config", default="config/field_config.yaml")
    p.add_argument("--horizon", type=int, default=14)
    args = p.parse_args()

    cfg = load_config(args.config)
    csv_path = cfg.processed_dir / f"{cfg.name}_aligned_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `python -m src.pipeline` first.")
    run_comparison(csv_path, args.horizon, out_dir=cfg.processed_dir)


if __name__ == "__main__":
    main()
