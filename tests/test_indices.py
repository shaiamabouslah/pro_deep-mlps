import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indices import (
    compute_ndvi, compute_evi, compute_ndwi, to_reflectance,
    summarize_field, flag_ndvi_anomaly,
)


def test_ndvi_healthy_vegetation():
    # Healthy crop: high NIR reflectance, low red reflectance -> NDVI near +1
    nir = np.array([4500.0])
    red = np.array([800.0])
    ndvi = compute_ndvi(to_reflectance(nir), to_reflectance(red))
    assert 0.6 < ndvi[0] < 1.0, f"expected healthy-vegetation NDVI, got {ndvi[0]}"


def test_ndvi_bare_soil_or_water():
    # Bare soil / water: NIR ~= red -> NDVI near 0, or NIR < red -> negative
    nir = np.array([1000.0])
    red = np.array([1100.0])
    ndvi = compute_ndvi(to_reflectance(nir), to_reflectance(red))
    assert ndvi[0] < 0, f"expected near-zero/negative NDVI for non-vegetation, got {ndvi[0]}"


def test_evi_bounded():
    nir = np.array([4500.0, 1000.0])
    red = np.array([800.0, 1100.0])
    blue = np.array([500.0, 900.0])
    evi = compute_evi(to_reflectance(nir), to_reflectance(red), to_reflectance(blue))
    assert np.all(evi >= -1.0) and np.all(evi <= 1.0)


def test_ndwi_range():
    nir = np.array([4500.0])
    swir = np.array([2000.0])
    ndwi = compute_ndwi(to_reflectance(nir), to_reflectance(swir))
    assert -1.0 <= ndwi[0] <= 1.0


def test_summarize_field_basic_stats():
    arr = np.array([0.2, 0.4, 0.6, 0.8, np.nan])
    stats = summarize_field(arr)
    assert stats["mean"] is not None
    assert round(stats["min"], 4) == 0.2
    assert round(stats["max"], 4) == 0.8


def test_summarize_field_empty_returns_none():
    arr = np.array([np.nan, np.nan])
    stats = summarize_field(arr)
    assert stats["mean"] is None


def test_flag_ndvi_anomaly_detects_drop():
    # Current mean far below historical baseline -> should flag
    assert flag_ndvi_anomaly(current_mean=0.2, historical_mean=0.6, historical_std=0.1) is True


def test_flag_ndvi_anomaly_no_false_alarm():
    # Current mean close to baseline -> should not flag
    assert flag_ndvi_anomaly(current_mean=0.58, historical_mean=0.6, historical_std=0.1) is False


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
