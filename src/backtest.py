"""
Week-4 deliverable (part 1): sliding-origin backtest -> a defensible ranking.

The Week-3 comparison used ONE held-out window, so its ranking could be luck.
This module replays the same four models over several successive windows and
aggregates the metrics (mean AND standard deviation). A model that wins on
average *and* is stable (low std) is a defensible "best model" -- which is what
the brief asks for in §9 ("judged on rigour of evaluation, not a demo").

How the windows are built (rolling origin, no leakage):
    window 0 : test = last `horizon` days,            train = everything before
    window 1 : test = `horizon` days ending `step` earlier, train = before that
    ...
Each window only ever trains on data strictly before its test period.

Usage:
    python -m src.backtest --config config/field_config.yaml \
        --horizon 14 --windows 6 --step 7
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config_loader import load_config
from .forecast import MODELS, QUANTILE_LEVELS, load_target_series, score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def sliding_backtest(csv_path, horizon: int, n_windows: int, step: int, out_dir) -> pd.DataFrame:
    series = load_target_series(csv_path)
    n = len(series)

    # build the (train, test) folds, oldest first
    folds = []
    for i in range(n_windows):
        end = n - i * step
        start = end - horizon
        if start - (horizon + 20) < 0:
            logger.warning("Stopping at %d windows -- not enough history for more.", i)
            break
        folds.append((series.iloc[:start], series.iloc[start:end], i))
    folds.reverse()
    logger.info("Backtest: %d windows x %d models, horizon %d, step %d",
                len(folds), len(MODELS), horizon, step)

    records = []
    for train, test, wi in folds:
        logger.info("--- window %d: test %s -> %s ---", wi,
                    test.index[0].date(), test.index[-1].date())
        for name, fn in MODELS.items():
            try:
                point, q = fn(train, horizon)
                point = np.asarray(point, float)[:horizon]
                if not (point.shape[0] == horizon and np.all(np.isfinite(point))):
                    raise ValueError(f"bad output {point.shape}")
                if q is not None:
                    q = np.asarray(q, float)[:horizon]
                    if q.shape != (horizon, len(QUANTILE_LEVELS)):
                        q = None
                rec = score(name, test.values, point, q, train.values)
                rec["window"] = wi
                records.append(rec)
                logger.info("   %-15s MASE=%.3f RMSE=%.4f CRPS=%.4f",
                            name, rec["MASE"], rec["RMSE"], rec["CRPS"])
            except Exception as e:
                logger.warning("   SKIPPED %s (window %d): %s: %s", name, wi, type(e).__name__, e)

    if not records:
        raise RuntimeError("Backtest produced no results -- check installs.")

    per_window = pd.DataFrame(records)
    agg = (
        per_window.groupby("model")
        .agg(
            n_windows=("MASE", "count"),
            MASE_mean=("MASE", "mean"), MASE_std=("MASE", "std"),
            RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
            CRPS_mean=("CRPS", "mean"), CRPS_std=("CRPS", "std"),
        )
        .round(4)
        .sort_values("RMSE_mean")
        .reset_index()
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_window.to_csv(out_dir / "backtest_per_window.csv", index=False)
    agg.to_csv(out_dir / "backtest_summary.csv", index=False)

    print("\n=== Sliding-origin backtest (mean over windows, lower is better) ===")
    print(agg.to_string(index=False))
    print(f"\nMost accurate on average (RMSE): {agg.iloc[0]['model']}")
    print(f"Windows used: {int(agg['n_windows'].max())}")
    print("Read std alongside mean: a low std means the model is *consistently* good.")
    print(f"Wrote: {out_dir / 'backtest_summary.csv'}")
    print(f"Wrote: {out_dir / 'backtest_per_window.csv'}")
    return agg


def main():
    p = argparse.ArgumentParser(description="Week-4 sliding-origin backtest")
    p.add_argument("--config", default="config/field_config.yaml")
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--step", type=int, default=7, help="days between successive test windows")
    args = p.parse_args()

    cfg = load_config(args.config)
    csv_path = cfg.processed_dir / f"{cfg.name}_aligned_daily.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `python -m src.pipeline` first.")
    sliding_backtest(csv_path, args.horizon, args.windows, args.step, out_dir=cfg.processed_dir)


if __name__ == "__main__":
    main()
