"""MLflow experiment tracking: one run per (study, field), one nested run per model."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
METRICS = ("MASE", "sMAPE", "RMSE", "CRPS")


def field_name(csv_path) -> str:
    return Path(csv_path).name.removesuffix("_aligned_daily.csv")


@contextmanager
def study_run(experiment: str, run_name: str, params: dict, tags: dict | None = None):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    if mlflow.get_experiment_by_name(experiment) is None:
        artifacts = (REPO_ROOT / "mlartifacts" / experiment).as_uri()
        mlflow.create_experiment(experiment, artifact_location=artifacts)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags or {})
        mlflow.log_params(params)
        yield


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def log_model(model: str, metrics: dict, steps=(), tags: dict | None = None):
    with mlflow.start_run(run_name=model, nested=True):
        mlflow.set_tags({"model": model, **(tags or {})})
        mlflow.log_metrics(metrics)
        for step, values in steps:
            mlflow.log_metrics(values, step=step)


def log_skipped(model: str, error: str):
    log_model(model, {}, tags={"status": "skipped", "error": error[:500]})


def set_tags(tags: dict):
    mlflow.set_tags(tags)


def log_artifacts(*paths):
    for path in paths:
        if Path(path).exists():
            mlflow.log_artifact(str(path))
