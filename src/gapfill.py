"""Gap-filling of the soil-moisture series: fills short sensor dropouts by interpolation,
with an optional foundation-model imputer for comparison.

    python -m src.gapfill --config config/field_config.yaml --gap-len 5
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

    # Chronos is optional -- try to load it, but never abort the run if it
    # isn't available (offline / weights can't download / package missing).
    logger.info(">>> Trying to load Chronos-2. The FIRST call can take 30-90s with "
                "NO output -- this is normal. <<<")
    pipe, chronos_ok = None, False
    try:
        pipe = _make_pipe()
        chronos_ok = True
        logger.info(">>> Chronos-2 loaded. Filling gaps with interpolation AND Chronos. <<<")
    except Exception as e:
        logger.warning("Chronos-2 unavailable (%s: %s). Falling back to linear "
                       "interpolation only -- re-run where Chronos is installed to add "
                       "the foundation-model comparison.", type(e).__name__, e)

    rows, all_interp_err, all_chronos_err = [], [], []
    for gi, start in enumerate(gaps, 1):
        logger.info("   gap %d/%d (starts %s)", gi, len(gaps), raw.index[start].date())
        sl = slice(start, start + gap_len)
        truth = values[sl].copy()

        # linear interpolation over the punched gap (always available)
        masked = clean.copy()
        masked[sl] = np.nan
        interp = pd.Series(masked).interpolate(method="linear").ffill().bfill().values[sl]
        ie = np.abs(interp - truth)
        all_interp_err += list(ie)
        row = {"gap_start": str(raw.index[start].date()), "gap_len": gap_len,
               "mae_interp": round(float(ie.mean()), 5)}

        # Chronos-2: forecast forward from the (clean) history before the gap
        if chronos_ok:
            try:
                chronos = _chronos_forecast(pipe, clean[:start], gap_len)
                ce = np.abs(chronos - truth)
                all_chronos_err += list(ce)
                row["mae_chronos"] = round(float(ce.mean()), 5)
            except Exception as e:
                logger.warning("   Chronos failed on gap %d (%s) -- interpolation only here.", gi, e)
        rows.append(row)

    per_gap = pd.DataFrame(rows)
    mae_i = float(np.mean(all_interp_err))
    rmse_i = float(np.sqrt(np.mean(np.square(all_interp_err))))
    chronos_better = False
    mae_c = rmse_c = None
    if all_chronos_err:
        mae_c = float(np.mean(all_chronos_err))
        rmse_c = float(np.sqrt(np.mean(np.square(all_chronos_err))))
        chronos_better = mae_c < mae_i

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_gap.to_csv(out_dir / "gapfill_eval.csv", index=False)

    print("\n=== Synthetic gap-filling ===")
    print(per_gap.to_string(index=False))
    if all_chronos_err:
        print(f"\nOverall MAE   -> interpolation {mae_i:.4f} | chronos {mae_c:.4f}")
        print(f"Overall RMSE  -> interpolation {rmse_i:.4f} | chronos {rmse_c:.4f}")
        print(f"Better imputer on these gaps: {'Chronos-2' if chronos_better else 'linear interpolation'}")
    else:
        print(f"\nOverall MAE (linear interpolation): {mae_i:.4f} | RMSE: {rmse_i:.4f}")
        print("Chronos-2 not run in this environment -- interpolation is the imputer used.")

    # produce a real gap-filled series with the winning method
    filled = clean.copy()
    method = np.where(known, "observed", "interpolation")
    if chronos_better:
        for start in np.where(~known)[0]:
            if known[:start].sum() >= MIN_CONTEXT:
                try:
                    filled[start] = _chronos_forecast(pipe, clean[:start], 1)[0]
                    method[start] = "chronos"
                except Exception:
                    pass
    out = pd.DataFrame({"date": raw.index, "raw": raw.values, "filled": filled, "method": method})
    stem = Path(csv_path).stem.replace("_aligned_daily", "")
    out.to_csv(out_dir / f"{stem}_soil_gapfilled.csv", index=False)
    n_filled = int((~known).sum())
    print(f"\nGap-filled series: {n_filled} missing day(s) imputed "
          f"({'Chronos where it helped, ' if chronos_better else ''}interpolation otherwise).")
    print(f"Wrote: {out_dir / 'gapfill_eval.csv'}")
    print(f"Wrote: {out_dir / (stem + '_soil_gapfilled.csv')} (method column flags observed/interp/chronos).")
    return per_gap


def main():
    p = argparse.ArgumentParser(description="Gap-filling")
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
