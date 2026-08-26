"""Run the full pipeline, forecasting and backtest across several field configurations
for a cross-climate comparison.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config_loader import load_config
from src.pipeline import run_week1_pipeline
from src import backtest, forecast_cov

HORIZON = 14
N_WINDOWS = 6
STEP = 7


def run_one(cfg_path: str):
    cfg = load_config(cfg_path)
    name = cfg.name
    print(f"\n{'=' * 72}\nFIELD: {name}   ({cfg_path})\n{'=' * 72}")

    # 1. ingest + align (real Sentinel-2 + Open-Meteo + ISMN)
    try:
        res = run_week1_pipeline(cfg_path)
    except Exception as e:
        print(f"  [SKIP {name}] pipeline failed: {type(e).__name__}: {e}")
        return [], None

    csv = cfg.processed_dir / f"{name}_aligned_daily.csv"
    if not csv.exists():
        print(f"  [SKIP {name}] no aligned table produced")
        return [], res.get("iot_source")

    rows = []

    # 2. covariate study (single held-out window)
    try:
        cov = forecast_cov.run_comparison(csv, HORIZON, out_dir=cfg.processed_dir)
        for _, r in cov.iterrows():
            rows.append({"field": name, "study": "covariate", "model": r["model"],
                         "RMSE": r["RMSE"], "MASE": r["MASE"], "CRPS": r["CRPS"]})
    except Exception as e:
        print(f"  covariate study failed for {name}: {type(e).__name__}: {e}")

    # 3. sliding backtest (robustness across windows)
    try:
        bt = backtest.sliding_backtest(csv, HORIZON, N_WINDOWS, STEP, out_dir=cfg.processed_dir)
        for _, r in bt.iterrows():
            rows.append({"field": name, "study": "backtest", "model": r["model"],
                         "RMSE": r["RMSE_mean"], "RMSE_std": r["RMSE_std"], "MASE": r["MASE_mean"]})
    except Exception as e:
        print(f"  backtest failed for {name}: {type(e).__name__}: {e}")

    return rows, res.get("iot_source")


def main():
    configs = sys.argv[1:] or sorted(glob.glob("config/field_*.yaml"))
    if not configs:
        print("No field configs found. Pass them explicitly or create config/field_*.yaml")
        return
    print(f"Running multi-field validation over {len(configs)} field(s):")
    for c in configs:
        print(f"  - {c}")

    all_rows, sources = [], {}
    for c in configs:
        rows, src = run_one(c)
        all_rows += rows
        if rows:
            sources[rows[0]["field"]] = src

    if not all_rows:
        print("\nNo results produced -- check that each field's ISMN data is in data/raw/.")
        return

    df = pd.DataFrame(all_rows)
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)

    cov = df[df.study == "covariate"]
    bt = df[df.study == "backtest"]
    cov.to_csv(out / "multifield_covariate.csv", index=False)
    bt.to_csv(out / "multifield_backtest.csv", index=False)

    # human-readable pivots (RMSE, lower = better)
    lines = ["MULTI-FIELD VALIDATION — RMSE (lower is better)\n"]
    lines.append("Soil-moisture source per field: " +
                 ", ".join(f"{k}={v}" for k, v in sources.items()) + "\n")
    if not cov.empty:
        piv = cov.pivot_table(index="model", columns="field", values="RMSE")
        lines.append("\n=== Covariate study (single 14-day window) ===\n" + piv.round(4).to_string())
    if not bt.empty:
        piv2 = bt.pivot_table(index="model", columns="field", values="RMSE")
        lines.append("\n\n=== Sliding backtest (6 windows, mean RMSE) ===\n" + piv2.round(4).to_string())
    report = "\n".join(lines)
    (out / "multifield_summary.txt").write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nWrote: {out / 'multifield_covariate.csv'}")
    print(f"Wrote: {out / 'multifield_backtest.csv'}")
    print(f"Wrote: {out / 'multifield_summary.txt'}")
    print("\nPaste multifield_summary.txt back and I'll tabulate + interpret it for the report.")


if __name__ == "__main__":
    main()
