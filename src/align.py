"""Align the multi-source streams (satellite indices, weather, soil moisture) onto one
clean daily table per field, carrying satellite indices forward between revisits with
an observed flag.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config_loader import FieldConfig

logger = logging.getLogger(__name__)


def _weather_to_daily(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Hourly Open-Meteo weather/soil -> one row per day."""
    df = weather_df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    daily = pd.DataFrame(index=df.resample("1D").mean(numeric_only=True).index)

    if "temperature_2m" in df:
        daily["temp_mean_c"] = df["temperature_2m"].resample("1D").mean()
        daily["temp_min_c"] = df["temperature_2m"].resample("1D").min()
        daily["temp_max_c"] = df["temperature_2m"].resample("1D").max()
    if "precipitation" in df:
        daily["precip_sum_mm"] = df["precipitation"].resample("1D").sum(min_count=1)
    if "et0_fao_evapotranspiration" in df:
        daily["et0_sum_mm"] = df["et0_fao_evapotranspiration"].resample("1D").sum(min_count=1)
    # every modelled soil-temperature / soil-moisture depth -> daily mean
    # (names differ between the forecast and archive APIs, so match by prefix)
    for col in df.columns:
        if col.startswith("soil_temperature_") or col.startswith("soil_moisture_"):
            daily[f"{col}_mean"] = df[col].resample("1D").mean()

    daily.index = daily.index.normalize()
    daily.index.name = "date"
    return daily.reset_index()


def _depth_label(depth_cm) -> str:
    try:
        return f"soil_moisture_sensor_{int(round(float(depth_cm)))}cm"
    except (TypeError, ValueError):
        return f"soil_moisture_sensor_depth_{depth_cm}"


def _sensor_soil_to_daily(soil_df: pd.DataFrame) -> pd.DataFrame:
    """
    IoT / ISMN soil-moisture (long format: timestamp, sensor_id, depth_cm,
    soil_moisture_vwc) -> one row per day, with an overall daily mean across
    all sensors plus a per-depth daily mean.
    """
    df = soil_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.normalize()

    # overall daily mean across all sensors (kept as a 1-col frame so the
    # join below never mixes Series/DataFrame dimensions)
    overall = (
        df.groupby("date")["soil_moisture_vwc"].mean()
        .rename("soil_moisture_sensor_mean")
        .to_frame()
    )

    # per-depth daily mean -- real ISMN data can carry NaN depths, which would
    # otherwise desync the two indexes; drop them and collapse duplicate labels.
    depth_df = df.dropna(subset=["depth_cm"])
    if not depth_df.empty:
        per_depth = (
            depth_df.groupby(["date", "depth_cm"])["soil_moisture_vwc"].mean()
            .unstack("depth_cm")
        )
        per_depth.columns = [_depth_label(c) for c in per_depth.columns]
        per_depth = per_depth.T.groupby(level=0).mean().T  # merge any duplicate depth labels
        out = overall.join(per_depth, how="outer")
    else:
        out = overall

    out.index = pd.DatetimeIndex(out.index).normalize()
    out.index.name = "date"
    return out.reset_index()


def _satellite_to_daily(sat_df: pd.DataFrame, master_dates: pd.DatetimeIndex,
                        ffill_limit_days: int) -> pd.DataFrame:
    """
    Sparse per-scene indices -> daily, forward-filled up to `ffill_limit_days`
    days, with an `ndvi_observed` flag marking days that carry a real scene.
    """
    s = sat_df.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    # collapse multiple scenes on the same day to their mean
    idx_cols = [c for c in ("ndvi", "evi", "ndwi") if c in s.columns]
    s = s.groupby("date")[idx_cols].mean()

    s = s.reindex(master_dates)
    observed = s["ndvi"].notna() if "ndvi" in s else pd.Series(False, index=master_dates)
    s = s.ffill(limit=ffill_limit_days)
    s["ndvi_observed"] = observed.values
    s.index.name = "date"
    return s.reset_index()


def build_aligned_timeseries(
    cfg: FieldConfig,
    weather_df: pd.DataFrame | None = None,
    soil_df: pd.DataFrame | None = None,
    satellite_df: pd.DataFrame | None = None,
    freq: str = "1D",
    ffill_satellite_days: int = 10,
) -> pd.DataFrame:
    """
    Fuse the (optional) weather, sensor-soil and satellite streams onto one
    daily grid spanning the config's time_range. Any stream that is None/empty
    is simply skipped -- the pipeline never hard-fails just because one source
    was unavailable. Returns a tidy DataFrame, one row per day.
    """
    master = pd.date_range(cfg.start_date, cfg.end_date, freq=freq).normalize()
    out = pd.DataFrame({"date": master})

    if weather_df is not None and len(weather_df):
        out = out.merge(_weather_to_daily(weather_df), on="date", how="left")

    if soil_df is not None and len(soil_df):
        out = out.merge(_sensor_soil_to_daily(soil_df), on="date", how="left")

    if satellite_df is not None and len(satellite_df):
        out = out.merge(_satellite_to_daily(satellite_df, master, ffill_satellite_days),
                        on="date", how="left")
    else:
        out["ndvi"] = np.nan
        out["evi"] = np.nan
        out["ndwi"] = np.nan
        out["ndvi_observed"] = False

    out.insert(1, "field_name", cfg.name)
    out = out.sort_values("date").reset_index(drop=True)

    n_days = len(out)
    n_obs = int(out["ndvi_observed"].sum()) if "ndvi_observed" in out else 0
    logger.info(
        "Aligned table: %d days x %d columns for field '%s' "
        "(%d days with a real Sentinel-2 observation)",
        n_days, out.shape[1], cfg.name, n_obs,
    )
    return out


def run_alignment(
    cfg: FieldConfig,
    weather_path: str | Path | None = None,
    soil_path: str | Path | None = None,
    satellite_path: str | Path | None = None,
) -> Path:
    """
    read whichever source CSVs exist,
    build the aligned daily table, and persist it to
    data/processed/<field>_aligned_daily.csv. Returns that path.
    """
    def _load(p):
        return pd.read_csv(p) if p and Path(p).exists() else None

    weather_df = _load(weather_path)
    soil_df = _load(soil_path)
    satellite_df = _load(satellite_path)

    al = cfg.raw.get("alignment", {})
    aligned = build_aligned_timeseries(
        cfg, weather_df, soil_df, satellite_df,
        freq=al.get("freq", "1D"),
        ffill_satellite_days=al.get("ffill_satellite_days", 10),
    )

    out_path = cfg.processed_dir / f"{cfg.name}_aligned_daily.csv"
    aligned.to_csv(out_path, index=False)
    logger.info("Wrote aligned multi-source daily series to %s", out_path)
    return out_path
