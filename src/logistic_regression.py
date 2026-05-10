"""Train and evaluate the logistic regression model.

Tunes the regularization strength ``C`` via ``GridSearchCV`` (5-fold,
macro-F1). Best config is refit on train+val before final test prediction.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.evaluate import (
    compute_metrics,
    print_classification_report,
    print_report,
    save_confusion_matrix,
)
from src.model_data import load_processed_splits, prepare_features

VARIANT = os.environ.get("VARIANT", "full")
MODEL_NAME = "Logistic Regression"
MODEL_PATH = Path(f"models/{VARIANT}/logistic_regression.joblib")
RANDOM_STATE = 42

PARAM_GRID = {"C": [0.01, 0.1, 1.0, 10.0]}


def build_classifier(C: float = 1.0) -> LogisticRegression:
    """Return an unfitted LogisticRegression with class-balanced weighting."""
    return LogisticRegression(
        C=C,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def tune_classifier(x_train, y_train) -> dict:
    """Grid search over ``C`` with 5-fold stratified CV scored by macro-F1."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_classifier(),
        param_grid=PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=False,
    )
    print("Running hyperparameter search (4 configs x 5-fold CV)...")
    search.fit(x_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV F1  : {search.best_score_:.4f}")
    return search.best_params_


def main() -> None:
    print(f"Training logistic regression model (variant={VARIANT})...")

    train_df, val_df, test_df = load_processed_splits()
    x_train, y_train, transformed, _ = prepare_features(
        train_df, [val_df, test_df], dense=True,
    )
    (x_val, y_val), (x_test, y_test) = transformed

    best_params = tune_classifier(x_train, y_train)

    if sp.issparse(x_train):
        x_train_val = sp.vstack([x_train, x_val])
    else:
        x_train_val = np.vstack([x_train, x_val])
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    final = build_classifier(**best_params)
    final.fit(x_train_val, y_train_val)
    y_pred = final.predict(x_test)

    metrics = compute_metrics(y_test, y_pred, model_name=MODEL_NAME)
    print_report(metrics)
    print_classification_report(y_test, y_pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    save_confusion_matrix(
        y_test, y_pred, model_name=MODEL_NAME, output_dir=MODEL_PATH.parent
    )


if __name__ == "__main__":
    main()
