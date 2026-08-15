"""
Week-4 deliverable (part 3): synthetic gap-filling of the soil-moisture series.

The brief (§5.2) asks to "use time-series foundation models in zero-shot mode to
impute sparse/missing IoT sensor gaps". The ISMN quality filter left real holes
in the sensor series; so far we filled them with plain linear interpolation. Here
we test whether **Chronos-2 zero-shot** imputes them better, and we produce a
properly gap-filled series.

Method:
  1. Take the sensor series and its currently-known days.
  2. Punch several SYNTHETIC gaps into known stretches (we know the truth there).
  3. Fill each gap two ways:
        - linear interpolation (the baseline)
        - Chronos-2 forecasting forward from the values before the gap
  4. Compare the imputation error (MAE / RMSE) on the masked points.
  5. Fill the REAL missing days with the better method and save the result.

Usage:
    python -m src.gapfill --config config/field_config.yaml \
        --n-gaps 8 --gap-len 5
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
MIN_CONTEXT = 30  # days of history required before a gap to trust the model


def load_raw_target(csv_path) -> pd.Series:
    """Sensor target on a daily grid, keeping the real NaN holes (no interpolation)."""
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    s = df.set_index("date")[TARGET].astype(float).asfreq("D")
    s.name = TARGET
    return s


def _make_pipe():
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


def _chronos_forecast(pipe, context_values: np.ndarray, horizon: int) -> np.ndarray:
    """Median Chronos-2 forecast `horizon` steps ahead from a 1D context."""
    idx = pd.date_range("2000-01-01", periods=len(context_values), freq="D")
    ctx = pd.DataFrame({"id": "s", "timestamp": idx, "target": np.asarray(context_values, float)})
    pred = pipe.predict_df(ctx, prediction_length=horizon, quantile_levels=[0.5],
                           id_column="id", timestamp_column="timestamp", target="target")
    for col in ("0.5", 0.5, "predictions", "mean"):
        if col in pred.columns:
            return pred[col].to_numpy(float)[:horizon]
    return pred.select_dtypes("number").iloc[:, -1].to_numpy(float)[:horizon]


def choose_gaps(known_mask: np.ndarray, n_gaps: int, gap_len: int, seed: int = 0):
    """Pick non-overlapping windows that are fully known and have enough history."""
    rng = np.random.default_rng(seed)
    n = len(known_mask)
    candidates = []
    for start in range(MIN_CONTEXT, n - gap_len):
        block = known_mask[start:start + gap_len]
        ctx = known_mask[:start]
        if block.all() and ctx.sum() >= MIN_CONTEXT:
            candidates.append(start)
    rng.shuffle(candidates)
    chosen, used = [], np.zeros(n, dtype=bool)
    for start in candidates:
        if used[start:start + gap_len].any():
            continue
        chosen.append(start)
        used[start:start + gap_len] = True
        if len(chosen) >= n_gaps:
            break
    return sorted(chosen)


def evaluate(csv_path, n_gaps: int, gap_len: int, out_dir) -> pd.DataFrame:
    raw = load_raw_target(csv_path)
    values = raw.values.copy()
    known = ~np.isnan(values)
    clean = raw.interpolate(method="time").ffill().bfill().values  # context source & interp truth
    logger.info("Series %d days, %d known (%.0f%% coverage)",
                len(raw), known.sum(), 100 * known.mean())

    gaps = choose_gaps(known, n_gaps, gap_len)
    if not gaps:
        raise RuntimeError("No suitable stretch to punch a synthetic gap into "
                           "(try a smaller --gap-len).")
    logger.info("Evaluating on %d synthetic gaps of %d days", len(gaps), gap_len)

    logger.info(">>> Loading Chronos-2 into memory. The FIRST call can take "
                "30-90s with NO output -- this is normal, do NOT press any key. <<<")
    try:
        pipe = _make_pipe()
    except Exception as e:
        logger.error("Could not load Chronos-2 (%s). Aborting gap-fill.", e)
        raise
    logger.info(">>> Chronos-2 loaded. Filling gaps... <<<")

    rows, all_interp_err, all_chronos_err = [], [], []
    for gi, start in enumerate(gaps, 1):
        logger.info("   gap %d/%d (starts %s)", gi, len(gaps), raw.index[start].date())
        sl = slice(start, start + gap_len)
        truth = values[sl].copy()

        # linear interpolation over the punched gap
        masked = clean.copy()
        masked[sl] = np.nan
        interp = pd.Series(masked).interpolate(method="linear").ffill().bfill().values[sl]

        # Chronos-2: forecast forward from the (clean) history before the gap
        chronos = _chronos_forecast(pipe, clean[:start], gap_len)

        ie = np.abs(interp - truth)
        ce = np.abs(chronos - truth)
        all_interp_err += list(ie)
        all_chronos_err += list(ce)
        rows.append({"gap_start": str(raw.index[start].date()), "gap_len": gap_len,
                     "mae_interp": round(float(ie.mean()), 5),
                     "mae_chronos": round(float(ce.mean()), 5)})

    per_gap = pd.DataFrame(rows)
    mae_i, mae_c = np.mean(all_interp_err), np.mean(all_chronos_err)
    rmse_i = np.sqrt(np.mean(np.square(all_interp_err)))
    rmse_c = np.sqrt(np.mean(np.square(all_chronos_err)))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_gap.to_csv(out_dir / "gapfill_eval.csv", index=False)

    print("\n=== Synthetic gap-filling: Chronos-2 vs linear interpolation ===")
    print(per_gap.to_string(index=False))
    print(f"\nOverall MAE   -> interpolation {mae_i:.4f} | chronos {mae_c:.4f}")
    print(f"Overall RMSE  -> interpolation {rmse_i:.4f} | chronos {rmse_c:.4f}")
    better = "Chronos-2" if mae_c < mae_i else "linear interpolation"
    print(f"Better imputer on these gaps: {better}")

    # produce a real gap-filled series with the winning method
    filled = clean.copy()
    method = np.where(known, "observed", "interpolation")
    if mae_c < mae_i:
        for start in np.where(~known)[0]:
            if known[:start].sum() >= MIN_CONTEXT:
                try:
                    filled[start] = _chronos_forecast(pipe, clean[:start], 1)[0]
                    method[start] = "chronos"
                except Exception:
                    pass
    out = pd.DataFrame({"date": raw.index, "raw": raw.values, "filled": filled, "method": method})
    out.to_csv(out_dir / f"{Path(csv_path).stem.replace('_aligned_daily','')}_soil_gapfilled.csv", index=False)
    print(f"\nWrote: {out_dir / 'gapfill_eval.csv'}")
    print(f"Wrote gap-filled series (method column flags observed/interp/chronos).")
    return per_gap


def main():
    p = argparse.ArgumentParser(description="Week-4 synthetic gap-filling")
    p.add_argument("--config", default="config/field_config.yaml")
    p.add_argument("--n-gaps", type=int, default=8)
    p.add_argument("--gap-len", type=int, default=5)
    args = p.parse_args()

    cfg = load_config(args.config)
    csv_path = cfg.processed_dir / f"{cfg.name}_aligned_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `python -m src.pipeline` first.")
    evaluate(csv_path, args.n_gaps, args.gap_len, out_dir=cfg.processed_dir)


if __name__ == "__main__":
    main()
