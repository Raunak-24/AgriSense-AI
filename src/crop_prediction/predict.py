from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from .preprocess import MISSING_CATEGORY_TOKEN


class CropPredictor:
    def __init__(self, model_path: str | Path = "models/crop_prediction_model.pkl") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        self.bundle: Dict[str, Any] = joblib.load(self.model_path)
        self.model = self.bundle["model"]
        self.feature_columns = self.bundle["feature_columns"]

    def predict(self, state: str, crop: str, season: str, cost: float) -> float:
        row = {
            "Crop": crop,
            "Variety": MISSING_CATEGORY_TOKEN,
            "State": state,
            "Quantity": 0,
            "Season": season,
            "Unit": MISSING_CATEGORY_TOKEN,
            "Cost": cost,
            "Recommended Zone": MISSING_CATEGORY_TOKEN,
        }
        frame = pd.DataFrame([row], columns=self.feature_columns)
        return float(self.model.predict(frame)[0])


def load_metadata(model_path: str | Path = "models/crop_prediction_model.pkl") -> Dict[str, Any]:
    if not Path(model_path).exists():
        return {}
    bundle = joblib.load(model_path)
    return {
        "best_model": bundle.get("best_model"),
        "metrics": bundle.get("metrics", {}),
        "all_metrics": bundle.get("all_metrics", {}),
    }
