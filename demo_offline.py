"""
Offline smoke-test / demo for the Phase-1 deliverable (Weeks 1-2).

This script does NOT call Copernicus or Open-Meteo -- it needs no accounts and
no internet. It proves the whole Week-1 -> Week-2 chain end to end using
physically-plausible synthetic stand-ins:

  Week 1
    - the *real* NDVI/EVI/NDWI math (src/indices.py) on a synthetic S2-like tile
    - the *real* IoT simulator (src/iot_simulator.py) for the soil-moisture stream
  Week 2
    - the *real* alignment module (src/align.py) fusing a synthetic multi-date
      satellite index series + synthetic hourly weather + the simulated IoT
      soil stream into ONE clean daily table.

Run this first, from the repo root:

    python demo_offline.py

Then, from a machine with internet, run `python -m src.pipeline` to produce the
same outputs from REAL data for the field in config/field_config.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.align import build_aligned_timeseries
from src.config_loader import load_config
from src.indices import compute_evi, compute_ndvi, compute_ndwi, summarize_field, to_reflectance
from src.iot_simulator import run_iot_simulation


# ---------------------------------------------------------------------------
# synthetic stand-ins (only for the offline demo -- real data replaces these)
# ---------------------------------------------------------------------------
def make_synthetic_sentinel2_tile(size: int = 100, seed: int = 3) -> dict[str, np.ndarray]:
    """A plausible 100x100 px synthetic S2 L2A tile (digital numbers, 0-10000)."""
    rng = np.random.default_rng(seed)
    red = rng.normal(800, 60, (size, size)).clip(200, 2000)
    nir = rng.normal(4200, 200, (size, size)).clip(1000, 6000)
    blue = rng.normal(500, 40, (size, size)).clip(100, 1500)
    swir = rng.normal(2200, 150, (size, size)).clip(500, 4000)
    # inject a water-stressed patch (lower NIR, higher red -> lower NDVI)
    stress = np.s_[20:40, 60:85]
    nir[stress] -= 1800
    red[stress] += 500
    return {"red": red, "nir": nir, "blue": blue, "swir": swir}


def make_synthetic_satellite_series(cfg, revisit_days: int = 5, seed: int = 5) -> pd.DataFrame:
    """
    A sparse multi-date field-level NDVI/EVI/NDWI series -- the shape
    src/sentinel2_ingest.py::build_index_timeseries produces from real scenes.
    NDVI rises through the season then dips (a stress event) so the aligned
    table shows realistic structure.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(cfg.start_date, cfg.end_date, freq=f"{revisit_days}D")
    t = np.linspace(0, 1, len(dates))
    ndvi = np.clip(0.35 + 0.45 * np.sin(np.pi * t) + rng.normal(0, 0.02, len(dates)), 0.05, 0.95)
    ndvi[-2:] -= 0.15  # late-season stress dip
    evi = np.clip(ndvi * 0.7 + rng.normal(0, 0.01, len(dates)), 0.0, 1.0)
    ndwi = np.clip(0.10 + 0.20 * np.sin(np.pi * t) + rng.normal(0, 0.02, len(dates)), -0.2, 0.6)
    return pd.DataFrame({
        "date": dates,
        "scene_id": [f"SYNTH_S2_{i:02d}" for i in range(len(dates))],
        "cloud_cover_pct": rng.integers(0, 20, len(dates)),
        "ndvi": ndvi, "evi": evi, "ndwi": ndwi,
    })


def make_synthetic_weather(cfg, seed: int = 11) -> pd.DataFrame:
    """Hourly weather + multi-depth soil moisture, same schema as Open-Meteo output."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(cfg.start_date, cfg.end_date, freq="h")
    n = len(idx)
    hour = idx.hour.values
    season = np.linspace(0, 1, n)
    diurnal = np.sin(2 * np.pi * (hour - 6) / 24)

    temp = 18 + 8 * diurnal + 6 * season + rng.normal(0, 1.0, n)
    precip = (rng.random(n) < 0.03) * rng.gamma(2.0, 1.2, n)
    et0 = np.clip(0.15 * np.maximum(0, diurnal) + rng.normal(0, 0.01, n), 0, None)
    recharge = np.convolve(precip, np.exp(-np.arange(48) / 12.0) / 12.0, mode="full")[:n]
    base_sm = np.clip(0.30 - 0.05 * season + 0.02 * recharge, 0.05, 0.45)

    return pd.DataFrame({
        "time": idx,
        "temperature_2m": temp,
        "precipitation": precip,
        "et0_fao_evapotranspiration": et0,
        "soil_moisture_0_to_1cm": np.clip(base_sm + rng.normal(0, 0.005, n), 0, 1),
        "soil_moisture_1_to_3cm": np.clip(base_sm * 0.98, 0, 1),
        "soil_moisture_3_to_9cm": np.clip(base_sm * 0.95, 0, 1),
        "soil_moisture_9_to_27cm": np.clip(base_sm * 0.90, 0, 1),
        "soil_temperature_0cm": temp - 2.0,
        "field_name": cfg.name,
        "lat": cfg.centroid[1],
        "lon": cfg.centroid[0],
    })


def main():
    cfg = load_config(Path(__file__).resolve().parent / "config" / "field_config.yaml")
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    print(f"=== Offline Phase-1 demo for field '{cfg.name}' (bbox={cfg.bbox}) ===\n")

    # --- WEEK 1a: vegetation/stress indices from a synthetic tile --------------
    bands = make_synthetic_sentinel2_tile()
    ndvi = compute_ndvi(to_reflectance(bands["nir"]), to_reflectance(bands["red"]))
    evi = compute_evi(to_reflectance(bands["nir"]), to_reflectance(bands["red"]), to_reflectance(bands["blue"]))
    ndwi = compute_ndwi(to_reflectance(bands["nir"]), to_reflectance(bands["swir"]))

    print("Week-1 vegetation/stress index summary (synthetic scene):")
    rows = []
    for name, arr in (("ndvi", ndvi), ("evi", evi), ("ndwi", ndwi)):
        s = summarize_field(arr)
        rows.append({"field_name": cfg.name, "index": name, **s})
        print(f"  {name.upper():5s} mean={s['mean']:.3f}  std={s['std']:.3f}  "
              f"min={s['min']:.3f}  max={s['max']:.3f}  (p10={s['p10']:.3f}, p90={s['p90']:.3f})")
    pd.DataFrame(rows).to_csv(out_dir / "demo_index_summary.csv", index=False)

    # --- WEEK 1b: simulated IoT soil-moisture stream --------------------------
    iot_path = run_iot_simulation(cfg)  # synthetic baseline, no network
    iot_df = pd.read_csv(iot_path)
    print(f"\nWeek-1 simulated IoT stream: {len(iot_df)} rows across "
          f"{iot_df.sensor_id.nunique()} virtual sensors -> {iot_path}")

    # --- WEEK 2: fuse everything into ONE clean aligned daily table ------------
    weather_df = make_synthetic_weather(cfg)
    satellite_df = make_synthetic_satellite_series(cfg)
    aligned = build_aligned_timeseries(
        cfg, weather_df=weather_df, soil_df=iot_df, satellite_df=satellite_df
    )
    aligned_path = out_dir / "demo_aligned_daily.csv"
    aligned.to_csv(aligned_path, index=False)

    n_obs = int(aligned["ndvi_observed"].sum())
    print(f"\nWeek-2 ALIGNED multi-source daily table: {aligned.shape[0]} days x "
          f"{aligned.shape[1]} columns ({n_obs} days with a real satellite observation)")
    print(f"  columns: {list(aligned.columns)}")
    print("\n  first 5 aligned days:")
    preview_cols = [c for c in
                    ["date", "ndvi", "ndvi_observed", "temp_mean_c", "precip_sum_mm",
                     "et0_sum_mm", "soil_moisture_sensor_mean"] if c in aligned.columns]
    with pd.option_context("display.width", 120):
        print(aligned[preview_cols].head().to_string(index=False))

    print(f"\nWrote: {out_dir / 'demo_index_summary.csv'}")
    print(f"Wrote: {iot_path}")
    print(f"Wrote: {aligned_path}")
    print("\nNOTE: this demo stands in for the Sentinel-2/Open-Meteo network calls. "
          "Run `python -m src.pipeline` from a machine with internet to produce the "
          "same aligned table from REAL data for the configured field.")


if __name__ == "__main__":
    main()
