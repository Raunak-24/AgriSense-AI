from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .evaluate import (
    regression_metrics,
    save_correlation_heatmap,
    save_metrics_report,
    save_prediction_vs_actual,
)
from .preprocess import TARGET_COLUMN, build_preprocessor, clean_crop_data, load_crop_data

LOGGER = logging.getLogger(__name__)
TEST_SIZE = 0.2
RANDOM_STATE = 42
RF_N_ESTIMATORS = 300
TOP_FEATURES_COUNT = 15


def _feature_importance_plot(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regressor = model.named_steps["regressor"]
    preprocessor = model.named_steps["preprocessor"]

    if not hasattr(regressor, "feature_importances_"):
        LOGGER.info("Skipping feature importance plot for model without feature_importances_.")
        return

    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    top_idx = importances.argsort()[-TOP_FEATURES_COUNT:]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_idx)), importances[top_idx])
    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_models(dataset_path: Path, model_out: Path, outputs_dir: Path) -> Dict[str, Any]:
    df = clean_crop_data(load_crop_data(dataset_path))
    preprocessor, feature_columns, _ = build_preprocessor(df)

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    best = {"name": None, "metrics": {"r2": float("-inf")}, "model": None}
    all_metrics: Dict[str, Dict[str, float]] = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])
        try:
            pipeline.fit(X_train, y_train)
        except (ValueError, MemoryError) as exc:
            LOGGER.exception("Model training failed for %s: %s", name, exc)
            continue
        preds = pipeline.predict(X_test)
        metrics = regression_metrics(y_test, preds)
        all_metrics[name] = metrics

        LOGGER.info("%s => R²=%.4f mae=%.4f rmse=%.4f", name, metrics["r2"], metrics["mae"], metrics["rmse"])

        score = (metrics["r2"], -metrics["rmse"], -metrics["mae"])
        best_score = (
            best["metrics"].get("r2", float("-inf")),
            -best["metrics"].get("rmse", float("inf")),
            -best["metrics"].get("mae", float("inf")),
        )
        if score > best_score:
            best = {"name": name, "metrics": metrics, "model": pipeline}

    if best["model"] is None:
        raise RuntimeError("No model could be trained.")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best["model"],
            "best_model": best["name"],
            "metrics": best["metrics"],
            "all_metrics": all_metrics,
            "feature_columns": feature_columns,
        },
        model_out,
    )

    save_correlation_heatmap(df, outputs_dir / "plots" / "crop_correlation_heatmap.png")
    final_preds = best["model"].predict(X_test)
    save_prediction_vs_actual(y_test, final_preds, outputs_dir / "plots" / "prediction_vs_actual.png")
    _feature_importance_plot(best["model"], outputs_dir / "plots" / "feature_importance.png")
    save_metrics_report(best["metrics"], outputs_dir / "reports" / "crop_model_metrics.json")

    return {"best_model": best["name"], "best_metrics": best["metrics"], "all_metrics": all_metrics}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train crop production prediction models")
    parser.add_argument(
        "--dataset",
        default="data/crop_production/PLACE_DATASET_HERE.csv",
        help="Path to crop dataset CSV (default is the placeholder file location)",
    )
    parser.add_argument("--model-output", default="models/crop_prediction_model.pkl")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    result = train_models(Path(args.dataset), Path(args.model_output), Path(args.outputs_dir))
    LOGGER.info("Training complete: %s", result)


if __name__ == "__main__":
    main()
