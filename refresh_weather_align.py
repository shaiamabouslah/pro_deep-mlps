"""
Re-fetch ONLY the Open-Meteo weather/soil series (fast) and rebuild the aligned
table, reusing the ISMN and Sentinel-2 CSVs already on disk. Use this after
changing the Open-Meteo variable list -- it avoids re-downloading the slow
Sentinel-2 bands.

Usage:
    python refresh_weather_align.py
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.align import run_alignment
from src.config_loader import load_config
from src.weather_ingest import fetch_weather_and_soil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

cfg = load_config("config/field_config.yaml")
p = cfg.processed_dir
name = cfg.name

# 1. re-fetch weather (uses the archive endpoint for past dates)
weather_df = fetch_weather_and_soil(cfg, use_archive=True)
weather_path = p / f"{name}_weather_soil.csv"
weather_df.to_csv(weather_path, index=False)
print(f"Re-fetched weather -> {weather_path}  ({len(weather_df)} rows, "
      f"{weather_df.shape[1]} cols)")

# 2. reuse existing soil + satellite CSVs
ismn_csv = p / f"{name}_ismn_soil_moisture.csv"
iot_csv = p / f"{name}_iot_soil_moisture.csv"
soil_path = ismn_csv if ismn_csv.exists() else (iot_csv if iot_csv.exists() else None)
satellite_path = p / f"{name}_sentinel2_indices.csv"

# 3. rebuild the aligned daily table
out = run_alignment(
    cfg,
    weather_path=weather_path,
    soil_path=soil_path,
    satellite_path=satellite_path if satellite_path.exists() else None,
)
print(f"\nAligned table rebuilt -> {out}")
