"""Multi-station training table: every ISMN station in an archive -> daily surface soil
moisture + Open-Meteo weather, plus a per-station data-validation report.

    python -m src.global_dataset --archive data/raw/spain.zip --start 2025-08-24 --end 2026-08-24
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from . import tracking
from .align import _weather_to_daily
from .config_loader import load_config
from .ismn_ingest import _get_interface, read_station_soil_moisture

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
WEATHER_CACHE = REPO_ROOT / "data" / "raw" / "global_weather"
OUT_ROOT = REPO_ROOT / "data" / "processed" / "global"

SURFACE_MAX_DEPTH_CM = 10
SUSPECT_JUMP = 0.05
DRY_2DAY_MM = 1.0


class WeatherQuotaExceeded(RuntimeError):
    pass


def surface_daily(interface, station, start: str, end: str) -> pd.Series:
    sm = read_station_soil_moisture(interface, station, start, end, field_name=station.name)
    sm = sm[sm["depth_cm"] <= SURFACE_MAX_DEPTH_CM]
    if sm.empty:
        raise ValueError(f"no soil-moisture sensor at or above {SURFACE_MAX_DEPTH_CM} cm")
    sm = sm.assign(date=pd.to_datetime(sm["timestamp"]).dt.normalize())
    daily = sm.groupby("date")["soil_moisture_vwc"].mean()
    return daily.reindex(pd.date_range(start, end, freq="D"))


def _get_open_meteo(url: str, params: dict) -> dict:
    for _ in range(3):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        reason = resp.json().get("reason", "")
        if "Minutely" not in reason:
            raise WeatherQuotaExceeded(reason)
        time.sleep(60)
    raise WeatherQuotaExceeded("minutely limit still exceeded after retries")


def daily_weather(uid: str, lat: float, lon: float, start: str, end: str, om_cfg: dict) -> pd.DataFrame:
    cache = WEATHER_CACHE / f"{uid}_{start}_{end}.csv"
    if cache.exists():
        hourly = pd.read_csv(cache)
    else:
        params = {
            "latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "timezone": "UTC",
            "hourly": ",".join(om_cfg["hourly_vars"] + om_cfg["soil_vars_archive"]),
        }
        hourly = pd.DataFrame(_get_open_meteo(om_cfg["archive_base_url"], params)["hourly"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(cache, index=False)
    daily = _weather_to_daily(hourly).set_index("date")
    return daily.reindex(pd.date_range(start, end, freq="D"))


def quality_flags(y: pd.Series, weather: pd.DataFrame) -> pd.DataFrame:
    soil_temp = weather.get("soil_temperature_0_to_7cm_mean", weather["temp_mean_c"])
    rain_2day = weather["precip_sum_mm"].fillna(0).rolling(2, min_periods=1).sum()
    return pd.DataFrame({
        "frozen": soil_temp <= 0,
        "jump_without_rain": (y.diff().abs() > SUSPECT_JUMP) & (rain_2day < DRY_2DAY_MM),
    })


def validate(y: pd.Series, flags: pd.DataFrame, min_valid_days: int, min_coverage: float) -> dict:
    valid = y.notna()
    gap_lengths = (~valid).astype(int).groupby(valid.cumsum()).sum()
    checks = {
        "days": len(y),
        "valid_days": int(valid.sum()),
        "coverage": round(float(valid.mean()), 4),
        "longest_gap_days": int(gap_lengths.max()) if len(gap_lengths) else 0,
        "frozen_days": int(flags["frozen"].sum()),
        "missing_while_frozen": int((~valid & flags["frozen"]).sum()),
        "jumps_without_rain": int(flags["jump_without_rain"].sum()),
        "mean_abs_daily_change": round(float(y.diff().abs().mean()), 5),
    }
    if checks["valid_days"] < min_valid_days:
        return checks | {"status": "rejected", "reason": f"only {checks['valid_days']} valid days"}
    if checks["coverage"] < min_coverage:
        return checks | {"status": "rejected", "reason": f"coverage {checks['coverage']:.0%}"}
    return checks | {"status": "kept", "reason": ""}


def build(archive: Path, start: str, end: str, min_valid_days: int, min_coverage: float,
          om_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    interface = _get_interface(str(archive))
    frames, report, quota_hit = [], [], False

    for network in interface.networks.values():
        for station in network.stations.values():
            uid = f"{network.name}__{station.name}".replace(" ", "_")
            row = {"unique_id": uid, "network": network.name, "station": station.name,
                   "lat": station.lat, "lon": station.lon}
            if quota_hit:
                report.append(row | {"status": "pending", "reason": "weather quota exceeded"})
                continue
            try:
                y = surface_daily(interface, station, start, end)
                weather = daily_weather(uid, station.lat, station.lon, start, end, om_cfg)
            except ValueError as e:
                report.append(row | {"status": "rejected", "reason": str(e)})
                continue
            except WeatherQuotaExceeded as e:
                logger.error("Open-Meteo quota exceeded (%s) -- rerun later, cached stations are reused", e)
                quota_hit = True
                report.append(row | {"status": "pending", "reason": "weather quota exceeded"})
                continue

            flags = quality_flags(y, weather)
            checks = validate(y, flags, min_valid_days, min_coverage)
            report.append(row | checks)
            logger.info("%-45s %s %s", uid, checks["status"], checks["reason"])
            if checks["status"] == "kept":
                frames.append(pd.concat([y.rename("y"), flags, weather], axis=1)
                              .rename_axis("ds").reset_index().assign(unique_id=uid))

    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not daily.empty:
        daily = daily[["unique_id", "ds", "y"] + [c for c in daily.columns if c not in ("unique_id", "ds", "y")]]
    return daily, pd.DataFrame(report), quota_hit


def main():
    p = argparse.ArgumentParser(description="Build the multi-station global training table")
    p.add_argument("--archive", required=True, help="ISMN archive (.zip or extracted folder)")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--min-valid-days", type=int, default=180)
    p.add_argument("--min-coverage", type=float, default=0.6)
    p.add_argument("--config", default="config/field_config.yaml", help="source of the Open-Meteo settings")
    args = p.parse_args()

    archive = Path(args.archive)
    om_cfg = load_config(args.config).raw["open_meteo"]
    daily, report, quota_hit = build(archive, args.start, args.end, args.min_valid_days,
                                     args.min_coverage, om_cfg)

    out_dir = OUT_ROOT / archive.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "station_report.csv", index=False)
    if not daily.empty:
        daily.to_parquet(out_dir / "global_daily.parquet", index=False)

    kept = report[report["status"] == "kept"]
    params = {"archive": archive.name, "start": args.start, "end": args.end,
              "surface_max_depth_cm": SURFACE_MAX_DEPTH_CM, "min_valid_days": args.min_valid_days,
              "min_coverage": args.min_coverage}
    with tracking.study_run("dataset", archive.stem, params, tags={"quota_hit": str(quota_hit)}):
        tracking.log_metrics({
            "stations_total": len(report), "stations_kept": len(kept),
            "rows": len(daily), "valid_obs": int(daily["y"].notna().sum()) if not daily.empty else 0,
            "mean_coverage_kept": float(kept["coverage"].mean()) if not kept.empty else 0.0,
        })
        tracking.log_artifacts(out_dir / "station_report.csv")

    print(report["status"].value_counts().to_string())
    print(f"\nWrote {out_dir / 'station_report.csv'}")
    if not daily.empty:
        print(f"Wrote {out_dir / 'global_daily.parquet'}  ({len(daily)} rows, {daily['unique_id'].nunique()} stations)")
    if quota_hit:
        print("Open-Meteo quota hit: rerun the same command later to finish the pending stations.")


if __name__ == "__main__":
    main()
