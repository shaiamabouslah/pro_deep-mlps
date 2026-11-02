# Precision-Agriculture Decision System

An agentic decision system that fuses **Sentinel-2** vegetation indices, **Open-Meteo**
weather forecasts and **soil-moisture** (in-situ or modelled) into short-horizon
soil-moisture forecasts and explained **irrigation and fertilization** recommendations.
The pipeline is location-agnostic: a single configuration file points it at any field.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows  (source .venv/bin/activate on macOS/Linux)
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional accounts (all free): Copernicus / AWS Sentinel-2 (no key on the default
`earth_search` provider), Open-Meteo (no key), ISMN (https://ismn.earth, one archive
download per station). For the fusion agent's hosted backend, put a Groq key in a `.env`
file at the project root:

```
GROQ_API_KEY=your_key_here
```

## Run

```bash
# Ingestion + alignment (Sentinel-2 + Open-Meteo + soil)
python -m src.pipeline --config config/field_config.yaml

# Forecasting and validation
python -m src.forecast      --config config/field_config.yaml --horizon 14
python -m src.backtest      --config config/field_config.yaml --windows 6 --step 7
python -m src.forecast_cov  --config config/field_config.yaml --horizon 14
python -m src.gapfill       --config config/field_config.yaml --gap-len 5

# Cross-climate comparison (add config/field_*.yaml per field)
python run_multifield.py config/field_config.yaml config/field_spain.yaml

# Multi-station training table from an ISMN archive (surface 0-10 cm soil moisture
# + Open-Meteo weather + per-station validation report), written to
# data/processed/global/<archive>/ and logged to the MLflow "dataset" experiment
python -m src.global_dataset --archive data/raw/spain.zip --start 2025-08-24 --end 2026-08-24

# Experiment tracking UI (forecast, backtest, covariate and dataset runs)
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1

# Tests
python -m pytest -q

# Fusion agent (local or hosted backend)
python -m src.agent --backend groq --model openai/gpt-oss-120b

# Agent decision-quality evaluation
python agent_eval.py --backend groq --model openai/gpt-oss-120b

# Dashboard
python -m streamlit run dashboard.py --server.fileWatcherType none
```

Point the pipeline at another field by editing only `config/field_config.yaml`
(bounding box, centroid, dates) — or copy `config/field_TEMPLATE.yaml`.

## Project layout

```
src/            ingestion, alignment, forecasting, backtest, gap-filling, agent
config/         per-field YAML configuration
data/           raw inputs and processed outputs (aligned tables, figures)
dashboard.py    interactive map-based decision surface
agent_eval.py   agent decision-quality benchmark
run_multifield.py   cross-climate comparison driver
```

## Notes and limitations

- Soil moisture uses a tiered source: a real ISMN station where one is near the field,
  otherwise Open-Meteo modelled soil moisture (global), with a simulator reserved for
  stress-tests.
- NDVI is computed only from clear (cloud-masked, SCL) Sentinel-2 pixels; heavily clouded
  windows yield fewer usable dates.
- The covariate study feeds recorded weather as a perfect-forecast stand-in, so its gain
  is an optimistic upper bound.
- The agent's ground truth is codified agronomic logic, not a human agronomist.
