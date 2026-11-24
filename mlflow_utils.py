"""
mlflow_utils.py — MLflow experiment tracking for video captioning project
Generative AI, FAST NUCES, Spring 2026 | Dr. Akhtar Jamil

Usage (in any member notebook):
    from mlflow_utils import start_run, log_eval_results, log_ablation

Logs:
  - Aggregate metrics (BLEU-1/4, METEOR, CIDEr, ROUGE-L)
  - Inference timing statistics
  - Model hyperparameters (frame count, beam size, quantisation)
  - Ablation experiment results as child runs
  - Saved artifact files (scores CSV, figures)
"""

import os
import json
import mlflow
import mlflow.pytorch
from contextlib import contextmanager

# ── MLflow server URI (overridden by MLFLOW_TRACKING_URI env var in Docker) ──
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "VideoCaption_Comparative_Study_2026"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


@contextmanager
def start_run(model_name: str, dataset_name: str, tags: dict = None):
    """
    Context manager for a single model-dataset evaluation run.

    Usage:
        with start_run("BLIP-2", "MSVD") as run:
            result = run_evaluation_full(...)
            log_eval_results(result, run)
    """
    run_tags = {
        "model"  : model_name,
        "dataset": dataset_name,
        "project": "VideoCaption_Spring2026",
    }
    if tags:
        run_tags.update(tags)

    with mlflow.start_run(run_name=f"{model_name}__{dataset_name}",
                          tags=run_tags) as run:
        yield run


def log_eval_results(result: dict, model_name: str, dataset_name: str,
                     params: dict = None):
    """
    Log all artifacts from run_evaluation_full() to the active MLflow run.

    Args:
        result     : dict returned by run_evaluation_full()
        model_name : string model identifier
        dataset_name: "MSVD" or "MSR-VTT"
        params     : optional dict of hyperparameters to log
    """
    # ── Metrics ──────────────────────────────────────────────────────────────
    scores = result.get("scores", {})
    for metric, value in scores.items():
        mlflow.log_metric(metric, value)

    # ── Timing ───────────────────────────────────────────────────────────────
    timing = result.get("timing", {})
    mlflow.log_metric("inference_mean_s",  timing.get("mean_s",  0))
    mlflow.log_metric("inference_std_s",   timing.get("std_s",   0))
    mlflow.log_metric("inference_total_s", timing.get("total_s", 0))
    mlflow.log_metric("n_videos",          len(result.get("hypotheses", {})))

    # ── Parameters ───────────────────────────────────────────────────────────
    mlflow.log_param("model_name",   model_name)
    mlflow.log_param("dataset_name", dataset_name)
    if params:
        for k, v in params.items():
            mlflow.log_param(k, v)

    # ── Tags ─────────────────────────────────────────────────────────────────
    mlflow.set_tag("model",   model_name)
    mlflow.set_tag("dataset", dataset_name)


def log_ablation(ablation_results: dict, model_name: str,
                 dataset_name: str, ablation_type: str = "frame_count"):
    """
    Log a complete ablation study as a series of child runs.

    Args:
        ablation_results : { "N frames": { "scores": {...}, "timing": {...} } }
        model_name       : model identifier
        dataset_name     : dataset identifier
        ablation_type    : "frame_count" | "vtm_reranking" | "quantisation"
    """
    parent_run_name = f"ABLATION__{ablation_type}__{model_name}__{dataset_name}"
    with mlflow.start_run(run_name=parent_run_name,
                          tags={"ablation_type": ablation_type,
                                "model": model_name,
                                "dataset": dataset_name}):
        for variant, data in ablation_results.items():
            with mlflow.start_run(run_name=f"{parent_run_name}__{variant}",
                                  nested=True,
                                  tags={"variant": variant}):
                scores = data.get("scores", data) if isinstance(data, dict) else {}
                timing = data.get("timing", {})   if isinstance(data, dict) else {}
                for metric, value in scores.items():
                    mlflow.log_metric(metric, float(value))
                if timing:
                    mlflow.log_metric("mean_latency_s", timing.get("mean_s", 0))
                mlflow.log_param("variant",      variant)
                mlflow.log_param("ablation_type", ablation_type)


def log_figure(fig_path: str):
    """Log a saved matplotlib figure as an MLflow artifact."""
    if os.path.exists(fig_path):
        mlflow.log_artifact(fig_path, artifact_path="figures")


def print_run_url(run=None):
    """Print the MLflow UI URL for the current or given run."""
    try:
        rid = run.info.run_id if run else mlflow.active_run().info.run_id
        print(f"\n  MLflow run: {MLFLOW_URI}/#/experiments/1/runs/{rid}")
    except Exception:
        print(f"\n  MLflow UI: {MLFLOW_URI}")
