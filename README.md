# Precision-Agriculture Pipeline — Phase 1 (Weeks 1–2)

**Week-1 deliverable:** multi-source ingestion pipeline (Sentinel-2 + weather + IoT soil
moisture) + computed NDVI/EVI/NDWI indices for a sample field.

**Week-2 deliverable:** a **clean, aligned multi-source time series** for the field —
all three streams fused onto one common daily grid
(`data/processed/<field>_aligned_daily.csv`), ready for the Week-3 forecaster.

> Reminder: Phase 1 also requires a mandatory **3–4 page state-of-the-art note**
> (brief §3) written *before* the code. That's a separate document deliverable —
> this repo is the code half of Phase 1.

This pipeline is **location-agnostic by design**: every geographic input (bounding
box, centroid, date range) lives in `config/field_config.yaml`. Nothing in `src/`
hardcodes a country or region, so the exact same code runs for a field in Iowa,
Andalusia, Punjab, the Nile Delta, or anywhere else with Sentinel-2 coverage
(which is everywhere) and Open-Meteo coverage (also everywhere). To point it at
your field, edit the `bbox` and `centroid` in the config — that's the only change
needed.

## Project layout

```
precision-ag-pipeline/
├── config/
│   └── field_config.yaml       # ALL geographic/time inputs live here
├── src/
│   ├── config_loader.py        # loads + validates the field config
│   ├── indices.py              # NDVI/EVI/NDWI math (pure, no network)
│   ├── sentinel2_ingest.py     # Copernicus Data Space STAC search + indices
│   ├── weather_ingest.py       # Open-Meteo weather + multi-depth soil moisture
│   ├── ismn_ingest.py          # REAL in-situ soil sensors (preferred IoT source)
│   ├── iot_simulator.py        # simulated IoT sensor network (fallback when no ISMN station is nearby)
│   ├── align.py                # WEEK 2: fuse all sources into one aligned daily table
│   └── pipeline.py             # orchestrates ingestion (Wk1) + alignment (Wk2)
├── tests/                      # offline unit tests (no network required)
├── demo_offline.py             # end-to-end Week1->Week2 smoke test with synthetic data
├── outputs/                    # demo output CSVs (incl. demo_aligned_daily.csv)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

### Where to get each API — all free

| Source | Link | Registration needed? |
|---|---|---|
| Copernicus Data Space (Sentinel-2) | https://dataspace.copernicus.eu | Yes — free account + OAuth client credentials |
| Open-Meteo (weather + soil) | https://open-meteo.com | No — works immediately, no key |
| ISMN (real in-situ soil sensors) | https://ismn.earth | Yes — free account, then download data from the Data Portal |
| SoilGrids (soil texture, optional) | https://rest.isric.org/soilgrids/v2.0/docs | No — public REST endpoint, 5 calls/min fair-use limit |
| Google Earth Engine (optional alt. to STAC) | https://earthengine.google.com | Yes — free account + service account key |

### Sentinel-2 provider: run today vs. mandated stack

`config/field_config.yaml` ships with `sentinel2.provider: earth_search`, which
reads Sentinel-2 L2A COGs from the **public AWS `sentinel-cogs` bucket** — no
account, no key — so you can run the real pipeline immediately while your
Copernicus account is being set up. It's the **same Sentinel-2 L2A data**.

The project brief mandates the **Copernicus** stack for your final submission.
When your account is ready, set `sentinel2.provider: copernicus` (and either
supply CDSE band-read credentials or use the openEO path in
`src/sentinel2_ingest.fetch_indices_timeseries_openeo`).

### 1. Copernicus Data Space (Sentinel-2) — register now, access has lead-time

1. Create a free account at https://dataspace.copernicus.eu
2. Under your account settings, create OAuth2 client credentials
3. Set them as environment variables:
   ```bash
   export CDSE_CLIENT_ID="your-client-id"
   export CDSE_CLIENT_SECRET="your-client-secret"
   ```

### 2. Open-Meteo (weather + soil) — no key needed

Nothing to configure. Global coverage, free tier (10k calls/day, non-commercial).

### 3. ISMN (real in-situ soil-moisture sensors) — preferred over the simulator when available

Unlike the other sources, the `ismn` Python package reads data **already downloaded to disk** — it does not call a live API at runtime. One-time setup:

1. Create a free account at https://ismn.earth
2. Go to the **Data Portal** (map-based interface) — https://ismn.earth/en/dataviewer/
3. Click near your field's coordinates to find a nearby network/station, select your date range, and request a download (you'll get an email with a download link)
4. Extract the downloaded ZIP to a folder, e.g. `data/raw/ismn/`
5. Point `ismn.data_path` in `config/field_config.yaml` at that folder:
   ```yaml
   ismn:
     data_path: "data/raw/ismn"
     max_distance_km: 50
   ```

Run `python -m src.pipeline` and it will automatically find the nearest downloaded station to your field's centroid and use its real sensor data. **If no station is within `max_distance_km`, or `data_path` doesn't exist yet, the pipeline logs a clear message and falls back to the simulator (`src/iot_simulator.py`) automatically** — nothing breaks either way.

### 4. (Optional) SoilGrids — soil texture, no account needed

Public REST endpoint at `https://rest.isric.org/soilgrids/v2.0/properties/query?lon=...&lat=...&property=clay&property=sand&depth=0-5cm&value=mean`. Not yet wired into the pipeline — ask if you want it added as a static per-field covariate.

### 5. (Optional) Google Earth Engine

Alternative/complement to raw STAC access for Sentinel-2. Register at https://earthengine.google.com and set `GEE_SERVICE_ACCOUNT` / `GEE_PRIVATE_KEY_PATH`.

## Point it at your field (anywhere in the world)

Edit `config/field_config.yaml`:

```yaml
field:
  name: "your_field_name"
  bbox:
    min_lon: <your min longitude>
    min_lat: <your min latitude>
    max_lon: <your max longitude>
    max_lat: <your max latitude>
  centroid:
    lon: <field center longitude>
    lat: <field center latitude>
time_range:
  start_date: "2026-05-01"
  end_date: "2026-07-01"
```

That's the only file you need to touch. The demo config ships pointed at a
field in the US Corn Belt (Iowa) purely as a worked example — swap in any
coordinates to run the identical pipeline for a field on a different continent.

## Run it

**Full pipeline (needs real internet access to Copernicus + Open-Meteo):**

```bash
python -m src.pipeline --config config/field_config.yaml
```

This writes to `data/processed/`:
- `<field>_weather_soil.csv` — hourly weather + multi-depth soil moisture
- `<field>_ismn_soil_moisture.csv` — **real** sensor data, if a station was found within `max_distance_km`
- `<field>_iot_soil_moisture.csv` — simulated multi-sensor soil-moisture stream, used only if ISMN wasn't available
- `<field>_sentinel2_indices.csv` — **per-date NDVI/EVI/NDWI series** over the field (Week-1 index output)
- `<field>_sentinel2_scenes.csv` — matched Sentinel-2 scene manifest (provenance)
- `<field>_aligned_daily.csv` — **Week-2 deliverable**: all sources fused onto one daily grid

Check `results["iot_source"]` (or the log line) to see which one was actually used — `"ismn_real"` or `"simulated"`.

**What's in the aligned table (`<field>_aligned_daily.csv`):** one row per day with
`ndvi/evi/ndwi` (sparse satellite indices forward-filled, with an `ndvi_observed`
flag), daily weather aggregates (`temp_mean_c`, `precip_sum_mm`, `et0_sum_mm`, …),
Open-Meteo modelled soil moisture per depth, and sensor soil moisture
(`soil_moisture_sensor_mean` + per-depth). This is the single tidy table the
Week-3 forecaster consumes.

**Offline smoke test (no internet/credentials needed, run this first):**

```bash
python demo_offline.py
```

Exercises the real NDVI/EVI/NDWI math, the IoT simulator, **and the Week-2
alignment** against synthetic stand-in data, writing sample outputs to
`outputs/` (including `demo_aligned_daily.csv`). Use this to sanity-check the
whole Week1→Week2 chain before spending API quota.

**Unit tests (offline):**

```bash
python tests/test_indices.py
python tests/test_iot_simulator.py
python tests/test_ismn_ingest.py
python tests/test_align.py
```

## Design notes / where each project requirement is met

| Requirement (from brief) | Where it's implemented |
|---|---|
| Sentinel-2 access (STAC) — Copernicus or public COGs | `src/sentinel2_ingest.py::search_sentinel2_scenes` |
| Sentinel-2 band reading → per-date index time series | `src/sentinel2_ingest.py::build_index_timeseries` |
| Weather + multi-depth soil via Open-Meteo | `src/weather_ingest.py::fetch_weather_and_soil` |
| Real in-situ IoT sensor data (ISMN), preferred | `src/ismn_ingest.py::run_ismn_ingestion` |
| Simulated IoT stream, seeded from Open-Meteo soil vars (fallback) | `src/iot_simulator.py::run_iot_simulation` |
| NDVI/EVI/NDWI computation | `src/indices.py` |
| Field-level index summary (for forecasting/agent input) | `src/indices.py::summarize_field` |
| NDVI anomaly flag (dashboard alert seed, Week 7) | `src/indices.py::flag_ndvi_anomaly` |
| **Week-1** ingestion pipeline deliverable | `src/pipeline.py::run_week1_pipeline` |
| **Week-2** clean, aligned multi-source time series | `src/align.py::build_aligned_timeseries` |

## Extending band reading (Sentinel-2)

`run_sentinel2_ingestion` currently handles auth + STAC search + scene
manifest export. Actually reading the four bands (`red`, `nir`, `blue`,
`swir`) per scene into arrays for `compute_indices_for_scene` is left as an
explicit extension point — the right access pattern (signed asset URLs +
`rasterio` `/vsicurl/`, vs. downloading COGs locally, vs. openEO server-side
processing) is a state-of-the-art-note design decision, not a fixed
requirement. `compute_indices_for_scene` already expects a `dict[str, np.ndarray]`
of the four bands, so wiring in your chosen access pattern is a small addition.

## Why this satisfies "worldwide, not region-locked"

- **Sentinel-2**: global revisit coverage over all land surfaces (~5 day cycle)
- **Open-Meteo**: global weather + soil-moisture model coverage, no regional restriction
- **Zero hardcoded coordinates** outside `config/field_config.yaml` — verified by
  `tests/test_iot_simulator.py::test_config_is_location_agnostic`
- Index math (`NDVI`/`EVI`/`NDWI`) is universal reflectance arithmetic — works
  identically on any Sentinel-2 tile regardless of location
