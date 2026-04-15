from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return metrics


def build_prediction_frame(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_column: str = "Solubility",
) -> pd.DataFrame:
    report = frame.copy().reset_index(drop=True)
    report[f"actual_{target_column}"] = np.asarray(y_true, dtype=float)
    report[f"predicted_{target_column}"] = np.asarray(y_pred, dtype=float)
    report["residual"] = report[f"actual_{target_column}"] - report[f"predicted_{target_column}"]
    report["absolute_error"] = report["residual"].abs()
    return report


def save_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
