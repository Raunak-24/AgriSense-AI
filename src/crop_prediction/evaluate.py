from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
    }


def save_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return
    plt.figure(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_prediction_vs_actual(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    line_min = min(float(np.min(y_true)), float(np.min(y_pred)))
    line_max = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([line_min, line_max], [line_min, line_max], "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metrics_report(metrics: Dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
