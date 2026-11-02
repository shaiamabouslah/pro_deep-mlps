"""
Tests read_station_soil_moisture() against mock Station/Sensor objects
shaped exactly like the real `ismn` package's documented return types
(verified against the installed package's docstrings/signatures -- see
comments). This validates our parsing/QC/reshaping logic without requiring
a real ISMN account + download, which needs manual registration.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ismn_ingest import read_station_soil_moisture


class MockMetadata:
    """Mimics sensor.metadata.to_pd() -> DataFrame indexed by field name, column 'val'."""
    def __init__(self, depth_from_m, instrument):
        self._df = pd.DataFrame(
            {"val": [depth_from_m, depth_from_m + 0.1, instrument]},
            index=["depth_from", "depth_to", "instrument"],
        )

    def to_pd(self):
        return self._df


class MockSensor:
    """Mimics ismn.components.Sensor: .read_data() -> DataFrame, .metadata.to_pd(), .depth (meters)."""
    def __init__(self, depth_from_m, instrument, data_df):
        self.metadata = MockMetadata(depth_from_m, instrument)
        self.depth = SimpleNamespace(start=depth_from_m, end=depth_from_m + 0.1)
        self._data = data_df

    def read_data(self):
        return self._data


class MockStation:
    """Mimics ismn.components.Station: .name, .iter_sensors(variable=...)."""
    def __init__(self, name, sensors):
        self.name = name
        self._sensors = sensors

    def iter_sensors(self, variable=None):
        return iter(self._sensors)


class MockInterface:
    """Mimics ISMN_Interface.network_for_station(name) -> str."""
    def network_for_station(self, station_name):
        return "MOCK_NETWORK"


def _make_sensor_df(start, end, base_value, with_bad_flag=False):
    idx = pd.date_range(start, end, freq="1h")
    values = np.full(len(idx), base_value)
    df = pd.DataFrame({"soil_moisture": values}, index=idx)
    if with_bad_flag:
        flags = np.array(["G"] * len(idx), dtype=object)
        flags[: len(idx) // 4] = "D"  # first quarter flagged bad ('D' = doubtful, not in ACCEPTABLE list)
        df["soil_moisture_flag"] = flags
    return df


def test_reads_multiple_depths_correctly():
    sensors = [
        MockSensor(0.0, "Stevens-Hydra-Probe", _make_sensor_df("2026-05-01", "2026-05-03", 0.25)),
        MockSensor(0.3, "Stevens-Hydra-Probe", _make_sensor_df("2026-05-01", "2026-05-03", 0.30)),
    ]
    station = MockStation("TestStation", sensors)
    interface = MockInterface()

    df = read_station_soil_moisture(interface, station, "2026-05-01", "2026-05-03", field_name="my_field")

    assert df["sensor_id"].nunique() == 2
    assert set(df["depth_cm"].unique()) == {0.0, 30.0}  # meters -> cm conversion
    assert (df["field_name"] == "my_field").all()


def test_quality_flag_filtering_drops_bad_readings():
    sensors = [MockSensor(0.0, "Stevens-Hydra-Probe",
                           _make_sensor_df("2026-05-01", "2026-05-03", 0.25, with_bad_flag=True))]
    station = MockStation("TestStation", sensors)
    interface = MockInterface()

    df = read_station_soil_moisture(interface, station, "2026-05-01", "2026-05-03", field_name="my_field")
    # the first quarter of readings were flagged 'D' (doubtful) and should be NaN/dropped, not silently kept
    assert df["soil_moisture_vwc"].isna().sum() > 0


def test_sensor_id_includes_network_and_station():
    sensors = [MockSensor(0.0, "CS655", _make_sensor_df("2026-05-01", "2026-05-02", 0.28))]
    station = MockStation("Canizal", sensors)
    interface = MockInterface()

    df = read_station_soil_moisture(interface, station, "2026-05-01", "2026-05-02", field_name="my_field")
    assert "MOCK_NETWORK" in df["sensor_id"].iloc[0]
    assert "Canizal" in df["sensor_id"].iloc[0]


def test_raises_when_no_usable_sensors():
    station = MockStation("EmptyStation", [])
    interface = MockInterface()
    try:
        read_station_soil_moisture(interface, station, "2026-05-01", "2026-05-02", field_name="my_field")
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError when a station has no usable sensors"


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
