"""
Re-run ONLY the Week-2 alignment step, reusing the CSVs already written by a
previous `python -m src.pipeline` run. Handy when ingestion succeeded but the
alignment failed -- it avoids re-downloading the (slow) Sentinel-2 bands.

Usage:
    python align_only.py
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.align import run_alignment
from src.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

cfg = load_config("config/field_config.yaml")
p = cfg.processed_dir
name = cfg.name

# soil source: prefer real ISMN if present, else the simulated stream
ismn_csv = p / f"{name}_ismn_soil_moisture.csv"
iot_csv = p / f"{name}_iot_soil_moisture.csv"
soil_path = ismn_csv if ismn_csv.exists() else (iot_csv if iot_csv.exists() else None)

weather_path = p / f"{name}_weather_soil.csv"
satellite_path = p / f"{name}_sentinel2_indices.csv"

out = run_alignment(
    cfg,
    weather_path=weather_path if weather_path.exists() else None,
    soil_path=soil_path,
    satellite_path=satellite_path if satellite_path.exists() else None,
)
print(f"\nAligned table written to: {out}")
