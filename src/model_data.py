"""Variant-aware loaders that materialize features for model scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.compose import ColumnTransformer

from src.featurization import (
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TEXT_LIKE_CATEGORICAL_FEATURES,
    build_feature_transformer,
)


VARIANT = os.environ.get("VARIANT", "full")
DEFAULT_PROCESSED_DIR = Path(f"data/processed_{VARIANT}")


def load_processed_splits(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read train/val/test CSVs from the processed-data directory."""
    train = pd.read_csv(processed_dir / "train.csv", parse_dates=["date"])
    val = pd.read_csv(processed_dir / "val.csv", parse_dates=["date"])
    test = pd.read_csv(processed_dir / "test.csv", parse_dates=["date"])
    return train, val, test


def get_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a processed dataframe into the feature frame and target series."""
    feature_columns = [*TEXT_LIKE_CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
    return df[feature_columns].copy(), df[TARGET_COLUMN].copy()


def _to_array(X) -> np.ndarray:
    """Densify a sparse matrix or pass through an array-like."""
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def prepare_features(
    train_df: pd.DataFrame,
    other_dfs: Iterable[pd.DataFrame],
    *,
    dense: bool = True,
):
    """Fit the shared transformer on ``train_df`` and transform every split."""
    x_train_raw, y_train = get_xy(train_df)
    feature_transformer = build_feature_transformer()
    x_train = feature_transformer.fit_transform(x_train_raw)
    if dense:
        x_train = _to_array(x_train)

    transformed = []
    for df in other_dfs:
        x_raw, y = get_xy(df)
        x = feature_transformer.transform(x_raw)
        if dense:
            x = _to_array(x)
        transformed.append((x, y))

    return x_train, y_train, transformed, feature_transformer


def get_data_for_model(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    *,
    dense: bool = True,
) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, np.ndarray, pd.Series, ColumnTransformer]:
    """Load the splits and return transformed train, val, and test sets."""
    train, val, test = load_processed_splits(processed_dir)
    x_train, y_train, transformed, feature_transformer = prepare_features(
        train,
        [val, test],
        dense=dense,
    )
    (x_val, y_val), (x_test, y_test) = transformed
    return x_train, y_train, x_val, y_val, x_test, y_test, feature_transformer
