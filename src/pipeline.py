"""Ingestion and alignment pipeline: runs the multi-source ingestion (Sentinel-2, Open-Meteo,
soil) and produces one aligned daily table for the configured field.

    python -m src.pipeline --config config/field_config.yaml
"""
from __future__ import annotations

import argparse
import logging

from .align import run_alignment
from .config_loader import load_config
from .ismn_ingest import run_ismn_ingestion
from .iot_simulator import run_iot_simulation, run_modeled_soil_source
from .sentinel2_ingest import run_sentinel2_ingestion
from .weather_ingest import fetch_weather_and_soil

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str) -> dict:
    cfg = load_config(config_path)
    logger.info("=== Pipeline: field '%s' ===", cfg.name)

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

    # Tiered soil source (best available first):
    #   1. real ISMN station (ground truth) -> for validation fields
    #   2. Open-Meteo MODELLED soil moisture (global, no station needed) -> DEFAULT fallback
    #   3. simulator seeded from Open-Meteo -> only if explicitly asked (stress-tests / offline)
    fallback = str(cfg.raw.get("soil", {}).get("fallback", "modeled")).lower()

    if ismn_path is not None:
        soil_path = ismn_path
        results["iot_soil_moisture"] = str(ismn_path)
        results["iot_source"] = "ismn_real"
        logger.info("[OK] Tier 1: real ISMN station used -> %s", ismn_path)
    else:
        modeled_path = None
        if weather_df is not None and fallback != "simulated":
            try:
                modeled_path = run_modeled_soil_source(cfg, weather_df)
            except Exception as e:
                logger.error("[FAILED] Modelled soil source: %s", e)

        if modeled_path is not None:
            soil_path = modeled_path
            results["iot_soil_moisture"] = str(modeled_path)
            results["iot_source"] = "open_meteo_modeled"
            logger.info("[OK] Tier 2: no ISMN station -- using Open-Meteo MODELLED soil moisture "
                        "(global coverage) -> %s", modeled_path)
        else:
            try:
                seed_series = None
                if weather_df is not None:
                    soil_cols = [c for c in weather_df.columns if c.startswith("soil_moisture_")]
                    if soil_cols:
                        seed_series = weather_df.set_index("time")[soil_cols[0]]
                iot_path = run_iot_simulation(cfg, real_soil_series=seed_series)
                soil_path = iot_path
                results["iot_soil_moisture"] = str(iot_path)
                results["iot_source"] = "simulated"
                logger.info("[OK] Tier 3: simulator (seeded from Open-Meteo) used -> %s", iot_path)
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

    # 4. Fuse everything into one clean aligned daily table
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

    logger.info("=== Pipeline complete for '%s' ===", cfg.name)
    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-source ingestion and alignment pipeline")
    parser.add_argument(
        "--config", default="config/field_config.yaml",
        help="Path to a field config YAML (change this to point at any field worldwide)",
    )
    args = parser.parse_args()
    results = run_pipeline(args.config)

    print("\n--- Pipeline outputs ---")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
