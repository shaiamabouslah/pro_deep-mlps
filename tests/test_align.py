"""
Offline, deterministic tests for the Week-2 alignment module (src/align.py).

These prove the fusion logic without any network: synthetic weather, sensor and
satellite frames go in; one clean daily table comes out, with sparse satellite
indices forward-filled and correctly flagged.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config
from src.align import build_aligned_timeseries

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "field_config.yaml"

START, END = "2026-05-01", "2026-05-10"


def _cfg():
    cfg = load_config(CONFIG_PATH)
    cfg.raw["time_range"] = {"start_date": START, "end_date": END}
    cfg.raw["field"]["name"] = "x"  # aligned table's field_name comes from cfg.name
    return cfg


def _weather():
    idx = pd.date_range(START, END, freq="h")
    n = len(idx)
    return pd.DataFrame({
        "time": idx,
        "temperature_2m": np.full(n, 20.0),
        "precipitation": np.full(n, 0.5),  # 0.5 mm every hour -> 12 mm/day
        "et0_fao_evapotranspiration": np.full(n, 0.1),
        "soil_moisture_3_to_9cm": np.full(n, 0.30),
        "field_name": "x",
    })


def _sensors():
    idx = pd.date_range(START, END, freq="3h")
    frames = []
    for depth, val in [(10, 0.25), (30, 0.30)]:
        frames.append(pd.DataFrame({
            "timestamp": idx,
            "field_name": "x",
            "sensor_id": f"s_{depth}",
            "depth_cm": depth,
            "soil_moisture_vwc": np.full(len(idx), val),
        }))
    return pd.concat(frames, ignore_index=True)


def _satellite():
    return pd.DataFrame({
        "date": ["2026-05-02", "2026-05-07"],
        "scene_id": ["A", "B"],
        "cloud_cover_pct": [5, 8],
        "ndvi": [0.50, 0.70],
        "evi": [0.35, 0.50],
        "ndwi": [0.10, 0.20],
    })


def test_one_row_per_day():
    out = build_aligned_timeseries(_cfg(), _weather(), _sensors(), _satellite())
    # 2026-05-01 .. 2026-05-10 inclusive = 10 days
    assert len(out) == 10
    assert out["date"].is_monotonic_increasing
    assert (out["field_name"] == "x").all()


def test_weather_daily_aggregation():
    out = build_aligned_timeseries(_cfg(), _weather(), None, None)
    assert "temp_mean_c" in out.columns
    assert np.isclose(out["temp_mean_c"].iloc[1], 20.0)
    # 0.5 mm * 24 h = 12 mm/day (interior day fully covered)
    assert np.isclose(out["precip_sum_mm"].iloc[1], 12.0, atol=1e-6)


def test_sensor_soil_mean_and_per_depth():
    out = build_aligned_timeseries(_cfg(), None, _sensors(), None)
    assert "soil_moisture_sensor_mean" in out.columns
    assert "soil_moisture_sensor_10cm" in out.columns
    assert "soil_moisture_sensor_30cm" in out.columns
    # mean of 0.25 and 0.30 across the two depths
    assert np.isclose(out["soil_moisture_sensor_mean"].iloc[1], 0.275, atol=1e-9)


def test_satellite_observed_flag_and_forward_fill():
    out = build_aligned_timeseries(_cfg(), None, None, _satellite()).set_index("date")
    d = pd.Timestamp

    # observed exactly on the two scene days
    assert bool(out.loc[d("2026-05-02"), "ndvi_observed"]) is True
    assert bool(out.loc[d("2026-05-07"), "ndvi_observed"]) is True
    assert bool(out.loc[d("2026-05-03"), "ndvi_observed"]) is False

    # value carried forward between scenes, not invented before the first one
    assert np.isclose(out.loc[d("2026-05-02"), "ndvi"], 0.50)
    assert np.isclose(out.loc[d("2026-05-05"), "ndvi"], 0.50)  # ffilled from 05-02
    assert np.isclose(out.loc[d("2026-05-07"), "ndvi"], 0.70)
    assert pd.isna(out.loc[d("2026-05-01"), "ndvi"])           # no prior observation


def test_missing_satellite_is_graceful():
    out = build_aligned_timeseries(_cfg(), _weather(), _sensors(), None)
    assert "ndvi" in out.columns
    assert out["ndvi"].isna().all()
    assert (~out["ndvi_observed"]).all()


def test_all_sources_none_still_returns_grid():
    out = build_aligned_timeseries(_cfg(), None, None, None)
    assert len(out) == 10
    assert "field_name" in out.columns


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
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
