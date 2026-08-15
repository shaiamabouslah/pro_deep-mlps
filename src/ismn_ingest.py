"""
ISMN (International Soil Moisture Network) ingestion -- REAL sensor data.

https://ismn.earth -- a free, global network of ~60 in-situ soil-moisture
networks / several thousand stations (USA, Europe, Asia, Australia, South
America, some African sites), multi-depth, harmonized and quality-controlled.

IMPORTANT: the `ismn` Python package reads data that has already been
downloaded from the ISMN Data Portal -- it does NOT make live network calls.
You must, once:
    1. Create a free account at https://ismn.earth
    2. Use the Data Portal (map view) to pick a network/station near your
       field and a date range, then download the ZIP
    3. Extract it somewhere on disk and point `ismn.data_path` in
       config/field_config.yaml at that folder

This module then finds the nearest downloaded station to the field's
centroid and returns its data in the exact same tidy schema
src/iot_simulator.py produces (timestamp, field_name, sensor_id, depth_cm,
soil_moisture_vwc) -- so nothing downstream (forecasting, agent) needs to
know or care whether a reading came from a real station or the simulator.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config_loader import FieldConfig

logger = logging.getLogger(__name__)

# A station 50km away is still a reasonable regional proxy for most
# agricultural climate purposes; tune per field via config if you have
# denser or sparser coverage nearby.
DEFAULT_MAX_DIST_KM = 50

# ISMN's own per-reading quality flag -- 'G' (good) and 'G_high' are the
# values considered acceptable; everything else gets dropped rather than
# silently trusted.
ACCEPTABLE_QUALITY_FLAGS = ("G", "G_high")


def _get_interface(data_path: str):
    """Lazy import -- the `ismn` package is only needed when ISMN mode is used."""
    from ismn.interface import ISMN_Interface
    return ISMN_Interface(data_path, parallel=False)


def find_nearest_ismn_station(data_path: str, lon: float, lat: float,
                               max_dist_km: float = DEFAULT_MAX_DIST_KM):
    """
    Find the nearest ISMN station (among what's been downloaded to
    data_path) to the given coordinates.

    Returns (interface, station, distance_km) -- station is None if nothing
    is within max_dist_km.
    """
    interface = _get_interface(data_path)
    station, distance_m = interface.find_nearest_station(
        lon, lat, return_distance=True, max_dist=max_dist_km * 1000
    )
    if station is None:
        logger.warning(
            "No ISMN station within %.0f km of (lon=%.4f, lat=%.4f) in downloaded data at '%s'",
            max_dist_km, lon, lat, data_path,
        )
        return interface, None, None

    distance_km = distance_m / 1000
    logger.info("Nearest ISMN station: '%s' (%.1f km from field centroid)", station.name, distance_km)
    return interface, station, distance_km


def read_station_soil_moisture(interface, station, start_date: str, end_date: str,
                                field_name: str) -> pd.DataFrame:
    """
    Read every soil-moisture sensor at a station (each sensor = one depth),
    apply ISMN's own quality flag, and return one tidy long-format
    DataFrame across all of them -- the real-data equivalent of
    iot_simulator.simulate_sensor_network's output.
    """
    network_name = interface.network_for_station(station.name)
    records = []

    for i, sensor in enumerate(station.iter_sensors(variable="soil_moisture")):
        try:
            df = sensor.read_data()
        except Exception as e:
            logger.warning("Could not read sensor %d at station '%s': %s", i, station.name, e)
            continue

        if df is None or "soil_moisture" not in df.columns:
            continue

        meta = sensor.metadata.to_pd()
        depth_from_m = float(meta.loc["depth_from", "val"]) if "depth_from" in meta.index else np.nan
        depth_cm = depth_from_m * 100 if pd.notna(depth_from_m) else np.nan
        instrument = meta.loc["instrument", "val"] if "instrument" in meta.index else "unknown"

        series = df["soil_moisture"].copy()

        # apply ISMN's quality control flag if present -- don't trust
        # unflagged/failed readings just because a row exists
        flag_col = "soil_moisture_flag"
        if flag_col in df.columns:
            good = df[flag_col].isin(ACCEPTABLE_QUALITY_FLAGS)
            n_dropped = (~good).sum()
            if n_dropped:
                logger.info("Sensor %d at '%s': dropping %d/%d readings that failed ISMN QC",
                            i, station.name, n_dropped, len(series))
            series = series.where(good)

        series = series.loc[start_date:end_date]
        if series.dropna().empty:
            logger.info("Sensor %d at '%s' (depth %.0fcm) has no valid readings in [%s, %s] -- skipping",
                        i, station.name, depth_cm, start_date, end_date)
            continue

        records.append(pd.DataFrame({
            "timestamp": series.index,
            "field_name": field_name,
            "sensor_id": f"ismn_{network_name}_{station.name}_{instrument}_{i}".replace(" ", "_"),
            "depth_cm": depth_cm,
            "soil_moisture_vwc": series.values,
        }))

    if not records:
        raise ValueError(
            f"Station '{station.name}' has no usable soil_moisture sensors for "
            f"[{start_date}, {end_date}] after quality filtering."
        )

    out = pd.concat(records, ignore_index=True)
    logger.info("Read %d real ISMN rows across %d sensors at station '%s' (network '%s')",
                len(out), out["sensor_id"].nunique(), station.name, network_name)
    return out


def run_ismn_ingestion(cfg: FieldConfig) -> Path | None:
    """
    Week 1/2 entry point. Returns the output CSV path on success, or None
    if ISMN isn't configured, the data hasn't been downloaded, or no
    station is close enough to the field -- pipeline.py falls back to the
    simulator in any of those cases, so this never hard-fails the pipeline.
    """
    ismn_cfg = cfg.raw.get("ismn")
    if not ismn_cfg or not ismn_cfg.get("data_path"):
        logger.info("No 'ismn.data_path' configured in field_config.yaml -- skipping real-sensor ingestion")
        return None

    data_path = ismn_cfg["data_path"]
    if not Path(data_path).exists():
        logger.warning(
            "Configured ismn.data_path '%s' does not exist. Register a free account at "
            "https://ismn.earth, download data for a station near your field from the Data "
            "Portal, extract it, and point ismn.data_path at that folder. Skipping.",
            data_path,
        )
        return None

    max_dist_km = ismn_cfg.get("max_distance_km", DEFAULT_MAX_DIST_KM)
    lon, lat = cfg.centroid

    try:
        interface, station, distance_km = find_nearest_ismn_station(data_path, lon, lat, max_dist_km)
    except Exception as e:
        logger.error("Failed to open ISMN data at '%s': %s", data_path, e)
        return None

    if station is None:
        return None

    df = read_station_soil_moisture(interface, station, cfg.start_date, cfg.end_date, field_name=cfg.name)
    df["ismn_station"] = station.name
    df["distance_from_field_km"] = round(distance_km, 2)

    out_path = cfg.processed_dir / f"{cfg.name}_ismn_soil_moisture.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote real ISMN soil-moisture data to %s", out_path)
    return out_path
