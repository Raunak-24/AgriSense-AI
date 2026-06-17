from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Crop",
    "Variety",
    "State",
    "Quantity",
    "Production",
    "Season",
    "Unit",
    "Cost",
    "Recommended Zone",
]
TARGET_COLUMN = "Production"


def load_crop_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate crop production dataset."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    return df[REQUIRED_COLUMNS].copy()


def clean_crop_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and remove duplicates."""
    cleaned = df.drop_duplicates().copy()

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].fillna("Unknown")
        else:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    LOGGER.info("Dataset cleaned. Rows before=%s, after=%s", len(df), len(cleaned))
    return cleaned


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    """Create preprocessing pipeline for categorical encoding and numeric scaling."""
    feature_columns = [col for col in REQUIRED_COLUMNS if col != TARGET_COLUMN]
    categorical = [c for c in feature_columns if df[c].dtype == object]
    numeric = [c for c in feature_columns if c not in categorical]

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipe, categorical),
            ("numeric", numeric_pipe, numeric),
        ]
    )

    return preprocessor, feature_columns, numeric
