"""Soil-moisture sources for a field without a physical sensor: a modelled source built
directly from Open-Meteo, and a simulated multi-depth sensor network (seeded from
Open-Meteo) used for stress-tests and offline runs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config_loader import FieldConfig

logger = logging.getLogger(__name__)


def synthetic_baseline_soil_series(start_date: str, end_date: str,
                                    freq: str = "h", base_vwc: float = 0.28,
                                    seed: int = 42) -> pd.Series:
    """
    Fallback soil-moisture baseline (volumetric water content, m3/m3) used
    only when no real Open-Meteo series is available (e.g. offline unit
    tests). Models a slow seasonal decay with periodic rain-driven recharge
    spikes -- not a substitute for real data, just a physically-plausible
    placeholder so downstream code is testable without network access.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_date, end_date, freq=freq)
    n = len(idx)

    t = np.linspace(0, 1, n)
    drydown = base_vwc - 0.06 * t  # slow drying trend over the window
    rain_events = rng.poisson(lam=0.02, size=n).astype(float)
    recharge = np.convolve(rain_events, np.exp(-np.arange(48) / 12.0), mode="full")[:n] * 0.05
    series = np.clip(drydown + recharge, 0.05, 0.45)
    return pd.Series(series, index=idx, name="soil_moisture_vwc")


def simulate_sensor_network(baseline: pd.Series, cfg: FieldConfig) -> pd.DataFrame:
    """
    Expand a single baseline soil-moisture series into N virtual sensors at
    different depths, each with its own noise realization and random
    dropouts -- mimicking a real in-field sensor deployment.
    """
    iot_cfg = cfg.raw["iot_simulation"]
    rng = np.random.default_rng(7)

    records = []
    for sensor_idx, depth_cm in enumerate(iot_cfg["sensor_depth_cm"][: iot_cfg["n_virtual_sensors"]]):
        # deeper sensors respond to a damped/lagged version of the surface signal
        lag_steps = max(1, depth_cm // 10)
        depth_signal = baseline.shift(lag_steps).bfill() * (1.0 - 0.05 * (depth_cm / 30))

        noise = rng.normal(0, iot_cfg["noise_std_pct"] / 100.0, size=len(depth_signal))
        reading = np.clip(depth_signal.values + noise, 0.0, 1.0)

        dropout_mask = rng.random(len(reading)) < iot_cfg["dropout_prob"]
        reading = reading.astype(float)
        reading[dropout_mask] = np.nan

        df = pd.DataFrame(
            {
                "timestamp": baseline.index,
                "sensor_id": f"{cfg.name}_sensor_{sensor_idx}",
                "depth_cm": depth_cm,
                "soil_moisture_vwc": reading,
                "field_name": cfg.name,
            }
        )
        records.append(df)

    out = pd.concat(records, ignore_index=True)
    logger.info(
        "Simulated %d virtual sensors x %d timesteps (%.1f%% dropout) for field '%s'",
        iot_cfg["n_virtual_sensors"], len(baseline), 100 * iot_cfg["dropout_prob"], cfg.name,
    )
    return out


def run_modeled_soil_source(cfg: FieldConfig, weather_df: pd.DataFrame) -> Path | None:
    """
    TIER-2 soil source (global coverage, no station required).

    When no real ISMN station is near the field, use Open-Meteo's PHYSICALLY
    MODELLED multi-depth soil moisture directly as the soil series -- a real
    reanalysis value (ERA5-Land / model), not a synthetic simulation. This is
    the Open-Meteo soil source and, unlike ISMN, it has
    data for EVERY field on Earth, so the pipeline never gets stuck without soil
    data again.

    Writes a CSV in the same schema the alignment step expects for a sensor
    stream (timestamp, soil_moisture_vwc, depth_cm), so it drops straight into
    `run_alignment(soil_path=...)` with no other change.
    """
    soil_cols = [c for c in weather_df.columns if c.startswith("soil_moisture_")]
    if not soil_cols:
        logger.warning("No Open-Meteo soil-moisture column found -- cannot build modelled soil source.")
        return None
    # prefer a root-zone depth (what irrigation decisions care about)
    preferred = ["soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
                 "soil_moisture_9_to_27cm", "soil_moisture_0_to_7cm",
                 "soil_moisture_3_to_9cm", "soil_moisture_1_to_3cm", "soil_moisture_0_to_1cm"]
    col = next((c for c in preferred if c in soil_cols), soil_cols[0])

    tcol = "time" if "time" in weather_df.columns else weather_df.columns[0]
    out_df = pd.DataFrame({
        "timestamp": pd.to_datetime(weather_df[tcol]),
        "sensor_id": f"{cfg.name}_open_meteo_modeled",
        "depth_cm": 20,  # nominal root-zone depth label
        "soil_moisture_vwc": pd.to_numeric(weather_df[col], errors="coerce"),
        "field_name": cfg.name,
        "source": "open_meteo_modeled",
    }).dropna(subset=["soil_moisture_vwc"])

    out_path = cfg.processed_dir / f"{cfg.name}_soil_modeled.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Modelled soil source (Open-Meteo %s, %d rows) -> %s", col, len(out_df), out_path)
    return out_path


def run_iot_simulation(cfg: FieldConfig, real_soil_series: pd.Series | None = None) -> Path:
    """
    Pass in the real Open-Meteo soil-moisture series
    (preferred -- keeps the simulation grounded in the field's actual
    climate) or omit it to fall back to the synthetic baseline for a dry run.
    """
    baseline = real_soil_series if real_soil_series is not None else synthetic_baseline_soil_series(
        cfg.start_date, cfg.end_date
    )
    df = simulate_sensor_network(baseline, cfg)
    out_path = cfg.processed_dir / f"{cfg.name}_iot_soil_moisture.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote simulated IoT stream to %s", out_path)
    return out_path
