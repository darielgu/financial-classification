from __future__ import annotations

from pathlib import Path
from typing import Tuple

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


def load_processed_splits(
    processed_dir: Path = Path("data/processed"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(processed_dir / "train.csv", parse_dates=["date"])
    val = pd.read_csv(processed_dir / "val.csv", parse_dates=["date"])
    test = pd.read_csv(processed_dir / "test.csv", parse_dates=["date"])
    return train, val, test


def get_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_columns = [*TEXT_LIKE_CATEGORICAL_FEATURES, *NUMERIC_FEATURES]
    x = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()
    return x, y


def _to_array(X) -> np.ndarray:
    """Convert a sparse matrix or array-like to a dense numpy array."""
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def get_data_for_model(
    processed_dir: Path = Path("data/processed"),
) -> Tuple[np.ndarray, pd.Series, np.ndarray, pd.Series, np.ndarray, pd.Series, ColumnTransformer]:
    """Load splits, encode features, and return ready-to-use arrays.

    The feature transformer is fitted on x_train only (no data leakage).
    Val and test sets are transformed using the fitted transformer.

    Returns
    -------
    x_train, y_train, x_val, y_val, x_test, y_test : numpy arrays / Series
        x_* are dense numpy arrays ready to pass directly to any sklearn estimator.
    feature_transformer : fitted ColumnTransformer
        Use for transforming new data at inference time.
    """
    train, val, test = load_processed_splits(processed_dir)

    x_train_raw, y_train = get_xy(train)
    x_val_raw, y_val = get_xy(val)
    x_test_raw, y_test = get_xy(test)

    # Fit on training data only — never on val or test
    feature_transformer = build_feature_transformer()
    x_train = _to_array(feature_transformer.fit_transform(x_train_raw))
    x_val   = _to_array(feature_transformer.transform(x_val_raw))
    x_test  = _to_array(feature_transformer.transform(x_test_raw))

    return x_train, y_train, x_val, y_val, x_test, y_test, feature_transformer
