"""
Phase-1 pipeline orchestrator (Weeks 1-2).

Runs the full ingestion + index-computation + multi-source ALIGNMENT
deliverable end to end for whichever field is defined in
config/field_config.yaml:

    python -m src.pipeline --config config/field_config.yaml

Week 1 output : Sentinel-2 index series + weather/soil + IoT/ISMN soil moisture.
Week 2 output : one clean, aligned daily multi-source table for the field
                (data/processed/<field>_aligned_daily.csv).

Because every geographic input comes from that config file, running this
against a field in Iowa vs. Andalusia vs. Punjab vs. anywhere else is a
one-line change to the YAML -- no code edits.

Network note: Sentinel-2 (STAC + COGs) and Open-Meteo calls require internet
egress. Run this from an environment with normal internet access (your laptop,
a cloud VM) -- not from a network-restricted sandbox. For a fully offline
end-to-end smoke test, run `python demo_offline.py` instead.
"""
from __future__ import annotations

import argparse
import logging

from .align import run_alignment
from .config_loader import load_config
from .ismn_ingest import run_ismn_ingestion
from .iot_simulator import run_iot_simulation
from .sentinel2_ingest import run_sentinel2_ingestion
from .weather_ingest import fetch_weather_and_soil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_week1_pipeline(config_path: str) -> dict:
    cfg = load_config(config_path)
    logger.info("=== Phase-1 pipeline (Weeks 1-2): field '%s' ===", cfg.name)

    results: dict = {}
    weather_path = soil_path = satellite_path = None

    # 1. Weather + soil (Open-Meteo) -- no key required, global coverage
    try:
        weather_df = fetch_weather_and_soil(cfg, use_archive=True)
        weather_path = cfg.processed_dir / f"{cfg.name}_weather_soil.csv"
        weather_df.to_csv(weather_path, index=False)
        results["weather_soil"] = str(weather_path)
        logger.info("[OK] Weather/soil ingestion -> %s", weather_path)
    except Exception as e:
        logger.error("[FAILED] Weather/soil ingestion: %s", e)
        weather_df = None
        results["weather_soil"] = None

    # 2. Real IoT soil-moisture data (ISMN) if a nearby station is configured;
    #    falls back to the simulator otherwise. Never hard-fails the pipeline.
    try:
        ismn_path = run_ismn_ingestion(cfg)
    except Exception as e:
        logger.error("[FAILED] ISMN ingestion: %s", e)
        ismn_path = None

    if ismn_path is not None:
        soil_path = ismn_path
        results["iot_soil_moisture"] = str(ismn_path)
        results["iot_source"] = "ismn_real"
        logger.info("[OK] Real ISMN station used -> %s", ismn_path)
    else:
        try:
            seed_series = None
            if weather_df is not None:
                # pick whatever soil-moisture column exists (names differ between
                # the forecast and archive Open-Meteo endpoints)
                soil_cols = [c for c in weather_df.columns if c.startswith("soil_moisture_")]
                if soil_cols:
                    seed_series = weather_df.set_index("time")[soil_cols[0]]
            iot_path = run_iot_simulation(cfg, real_soil_series=seed_series)
            soil_path = iot_path
            results["iot_soil_moisture"] = str(iot_path)
            results["iot_source"] = "simulated"
            logger.info("[OK] No ISMN station configured/nearby -- simulator used -> %s", iot_path)
        except Exception as e:
            logger.error("[FAILED] IoT simulation: %s", e)
            results["iot_soil_moisture"] = None
            results["iot_source"] = None

    # 3. Sentinel-2 scenes + vegetation/stress index TIME SERIES
    try:
        satellite_path = run_sentinel2_ingestion(cfg)
        results["sentinel2_indices"] = str(satellite_path)
        logger.info("[OK] Sentinel-2 ingestion -> %s", satellite_path)
    except Exception as e:
        logger.error("[FAILED] Sentinel-2 ingestion: %s", e)
        results["sentinel2_indices"] = None

    # 4. WEEK-2 DELIVERABLE: fuse everything into one clean aligned daily table
    try:
        aligned_path = run_alignment(
            cfg,
            weather_path=weather_path,
            soil_path=soil_path,
            satellite_path=satellite_path,
        )
        results["aligned_daily"] = str(aligned_path)
        logger.info("[OK] Aligned multi-source series -> %s", aligned_path)
    except Exception as e:
        logger.error("[FAILED] Alignment step: %s", e)
        results["aligned_daily"] = None

    logger.info("=== Phase-1 pipeline complete for '%s' ===", cfg.name)
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase-1 (Weeks 1-2) multi-source pipeline")
    parser.add_argument(
        "--config", default="config/field_config.yaml",
        help="Path to a field config YAML (change this to point at any field worldwide)",
    )
    args = parser.parse_args()
    results = run_week1_pipeline(args.config)

    print("\n--- Pipeline outputs ---")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
