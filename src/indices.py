"""
Vegetation and water-stress index computation from Sentinel-2 L2A bands.

These formulas are universal — they work on reflectance values from any
Sentinel-2 tile on Earth, so this module has zero location-specific logic.
That's deliberate: the whole pipeline is designed to be pointed at any
field worldwide by changing config/field_config.yaml only.

References:
- NDVI (Rouse et al., 1974): (NIR - Red) / (NIR + Red)
- EVI  (Huete et al., 2002): 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
- NDWI (Gao, 1996, vegetation water content form): (NIR - SWIR) / (NIR + SWIR)
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8  # avoids divide-by-zero on masked / no-data pixels


def to_reflectance(band: np.ndarray, scale: float = 10000.0) -> np.ndarray:
    """Sentinel-2 L2A bands are delivered as scaled integers (0-10000 ~ 0-1 reflectance)."""
    return band.astype(np.float32) / scale


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    nir, red = nir.astype(np.float32), red.astype(np.float32)
    return (nir - red) / (nir + red + _EPS)


def compute_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    nir, red, blue = nir.astype(np.float32), red.astype(np.float32), blue.astype(np.float32)
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = 2.5 * (nir - red) / (denom + _EPS)
    return np.clip(evi, -1.0, 1.0)


def compute_ndwi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Gao NDWI — tracks vegetation/canopy water content, complements soil-moisture sensors."""
    nir, swir = nir.astype(np.float32), swir.astype(np.float32)
    return (nir - swir) / (nir + swir + _EPS)


def summarize_field(index_array: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
    """
    Reduce a per-pixel index raster to the field-level scalar summary that
    feeds the forecasting model (Week 3) and the fusion agent (Week 5).
    """
    arr = index_array if valid_mask is None else index_array[valid_mask]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "p10": None, "p90": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def flag_ndvi_anomaly(current_mean: float, historical_mean: float, historical_std: float,
                       z_threshold: float = -1.5) -> bool:
    """
    Simple z-score anomaly flag — the seed for the dashboard's 'NDVI anomaly'
    alert in Week 7. current well below the historical baseline => stress flag.
    """
    if historical_std <= 0:
        return False
    z = (current_mean - historical_mean) / historical_std
    return z <= z_threshold
