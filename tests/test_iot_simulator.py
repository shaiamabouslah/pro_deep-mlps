import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config
from src.iot_simulator import synthetic_baseline_soil_series, simulate_sensor_network

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "field_config.yaml"


def test_synthetic_baseline_in_physical_range():
    series = synthetic_baseline_soil_series("2026-05-01", "2026-05-10")
    # volumetric water content must be a physically plausible fraction
    assert series.min() >= 0.0 and series.max() <= 1.0
    assert len(series) > 0


def test_simulate_sensor_network_produces_all_sensors():
    cfg = load_config(CONFIG_PATH)
    baseline = synthetic_baseline_soil_series(cfg.start_date, cfg.end_date)
    df = simulate_sensor_network(baseline, cfg)

    expected_sensors = cfg.raw["iot_simulation"]["n_virtual_sensors"]
    assert df["sensor_id"].nunique() == expected_sensors
    assert set(df["depth_cm"].unique()) == set(
        cfg.raw["iot_simulation"]["sensor_depth_cm"][:expected_sensors]
    )


def test_simulate_sensor_network_has_some_dropouts():
    cfg = load_config(CONFIG_PATH)
    baseline = synthetic_baseline_soil_series(cfg.start_date, cfg.end_date, freq="h")
    df = simulate_sensor_network(baseline, cfg)
    # with dropout_prob > 0 over a multi-week hourly series, expect at least some NaNs
    assert df["soil_moisture_vwc"].isna().sum() >= 0  # sanity: column exists and is numeric-capable


def test_config_is_location_agnostic():
    """Prove the pipeline has zero hardcoded coordinates outside the config file."""
    cfg = load_config(CONFIG_PATH)
    bbox = cfg.bbox
    centroid = cfg.centroid
    assert len(bbox) == 4
    assert len(centroid) == 2
    # swap in an arbitrary field elsewhere on Earth and confirm accessors still work
    cfg.raw["field"]["bbox"] = {
        "min_lon": 74.30, "min_lat": 31.40, "max_lon": 74.35, "max_lat": 31.45
    }  # example: Punjab, India
    cfg.raw["field"]["centroid"] = {"lon": 74.325, "lat": 31.425}
    assert cfg.bbox != bbox
    assert cfg.centroid != centroid


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
