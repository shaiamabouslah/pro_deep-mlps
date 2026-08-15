"""
Weather + multi-depth soil-moisture ingestion via the Open-Meteo API.

Open-Meteo has global model coverage (blends ECMWF/GFS/ICON/etc. depending
on region) and needs no API key, so -- like the Sentinel-2 module -- this
code makes no assumption about where on Earth the field is. Point it at any
lat/lon via config/field_config.yaml.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from .config_loader import FieldConfig

logger = logging.getLogger(__name__)


def fetch_weather_and_soil(cfg: FieldConfig, use_archive: bool = False) -> pd.DataFrame:
    """
    Pull hourly weather + multi-depth soil-moisture variables for the field
    centroid. use_archive=True hits the historical reanalysis endpoint
    (needed once start_date is in the past); otherwise the forecast endpoint
    is used, which also serves several days of recent history.
    """
    om_cfg = cfg.raw["open_meteo"]
    lon, lat = cfg.centroid
    base_url = om_cfg["archive_base_url"] if use_archive else om_cfg["forecast_base_url"]

    # The forecast and archive endpoints expose DIFFERENT soil-variable names,
    # so pick the matching set. Falls back gracefully to whatever is in
    # hourly_vars if the split keys aren't present (older config shape).
    hourly_vars = list(om_cfg.get("hourly_vars", []))
    soil_key = "soil_vars_archive" if use_archive else "soil_vars_forecast"
    hourly_vars += list(om_cfg.get(soil_key, []))
    if not hourly_vars:
        raise ValueError("No Open-Meteo hourly variables configured (open_meteo.hourly_vars).")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "timezone": om_cfg["timezone"],
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
    }

    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["field_name"] = cfg.name
    df["lat"] = lat
    df["lon"] = lon
    logger.info("Fetched %d hourly weather/soil records for field '%s'", len(df), cfg.name)
    return df


def run_weather_ingestion(cfg: FieldConfig) -> Path:
    """Week-1 entry point: fetch + persist weather/soil series for the field."""
    df = fetch_weather_and_soil(cfg, use_archive=True)
    out_path = cfg.processed_dir / f"{cfg.name}_weather_soil.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote weather/soil series to %s", out_path)
    return out_path
