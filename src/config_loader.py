"""
Config loader for the precision-agriculture ingestion pipeline.

Loading the field definition from YAML (rather than hardcoding coordinates
anywhere in the ingestion code) is what makes this pipeline reusable for
ANY field, anywhere in the world — swap config/field_config.yaml and every
downstream module (Sentinel-2, Open-Meteo, IoT simulator, indices) follows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "field_config.yaml"


@dataclass
class FieldConfig:
    raw: dict[str, Any]

    # Convenience accessors
    @property
    def name(self) -> str:
        return self.raw["field"]["name"]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        b = self.raw["field"]["bbox"]
        return (b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"])

    @property
    def centroid(self) -> tuple[float, float]:
        c = self.raw["field"]["centroid"]
        return (c["lon"], c["lat"])

    @property
    def start_date(self) -> str:
        return self.raw["time_range"]["start_date"]

    @property
    def end_date(self) -> str:
        return self.raw["time_range"]["end_date"]

    @property
    def raw_dir(self) -> Path:
        p = Path(self.raw["output"]["raw_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def processed_dir(self) -> Path:
        p = Path(self.raw["output"]["processed_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> FieldConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Field config not found at {path}. "
            "Copy config/field_config.yaml and point --config at your own field."
        )
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return FieldConfig(raw=raw)


def resolve_env_credentials() -> dict[str, str | None]:
    """
    Copernicus Data Space and Google Earth Engine both require free
    registration (Week 1 action item). Credentials are read from environment
    variables / a local .env file — never hardcoded — so the same code runs for
    any user, anywhere, against their own account.

    A .env file at the project root (same folder as requirements.txt) is loaded
    automatically if python-dotenv is installed, e.g.:

        CDSE_CLIENT_ID=your-client-id
        CDSE_CLIENT_SECRET=your-client-secret
    """
    try:
        from dotenv import load_dotenv
        # look for a .env at the project root regardless of the current cwd
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        load_dotenv()  # also pick up a .env in the current working directory
    except Exception:
        pass  # python-dotenv optional; fall back to real environment variables

    return {
        "CDSE_CLIENT_ID": os.getenv("CDSE_CLIENT_ID"),
        "CDSE_CLIENT_SECRET": os.getenv("CDSE_CLIENT_SECRET"),
        "GEE_SERVICE_ACCOUNT": os.getenv("GEE_SERVICE_ACCOUNT"),
        "GEE_PRIVATE_KEY_PATH": os.getenv("GEE_PRIVATE_KEY_PATH"),
    }
