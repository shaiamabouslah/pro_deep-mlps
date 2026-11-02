import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.global_dataset import quality_flags, validate

DAYS = pd.date_range("2025-01-01", periods=10, freq="D")


def weather(precip=None, soil_temp=None):
    return pd.DataFrame({
        "temp_mean_c": 10.0,
        "precip_sum_mm": precip if precip is not None else 0.0,
        "soil_temperature_0_to_7cm_mean": soil_temp if soil_temp is not None else 10.0,
    }, index=DAYS)


def test_longest_gap_and_coverage():
    y = pd.Series([0.2, np.nan, np.nan, np.nan, 0.2, 0.2, np.nan, 0.2, 0.2, 0.2], index=DAYS)
    checks = validate(y, quality_flags(y, weather()), min_valid_days=1, min_coverage=0.0)
    assert checks["valid_days"] == 6
    assert checks["coverage"] == 0.6
    assert checks["longest_gap_days"] == 3


def test_jump_without_rain_is_flagged_but_rain_driven_jump_is_not():
    y = pd.Series([0.2, 0.2, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5, 0.5, 0.5], index=DAYS)
    precip = [0, 0, 0, 0, 20, 0, 0, 0, 0, 0]
    flags = quality_flags(y, weather(precip=precip))
    assert flags["jump_without_rain"].tolist() == [False, False, True] + [False] * 7


def test_frozen_days_and_missing_while_frozen():
    y = pd.Series([0.2, np.nan, np.nan, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], index=DAYS)
    soil_temp = [5, -1, -2, -1, 5, 5, 5, 5, 5, 5]
    checks = validate(y, quality_flags(y, weather(soil_temp=soil_temp)), min_valid_days=1, min_coverage=0.0)
    assert checks["frozen_days"] == 3
    assert checks["missing_while_frozen"] == 2


def test_rejection_reasons():
    y = pd.Series([0.2] * 5 + [np.nan] * 5, index=DAYS)
    flags = quality_flags(y, weather())
    assert validate(y, flags, min_valid_days=6, min_coverage=0.0)["status"] == "rejected"
    assert validate(y, flags, min_valid_days=1, min_coverage=0.6)["reason"] == "coverage 50%"
    assert validate(y, flags, min_valid_days=5, min_coverage=0.5)["status"] == "kept"
