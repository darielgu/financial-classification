"""Train and evaluate the logistic regression model."""

from __future__ import annotations

import argparse
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

MODEL_NAME = "Logistic Regression"
MODEL_FILENAME = "logistic_regression.joblib"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train and evaluate {MODEL_NAME}.")
    parser.add_argument(
        "--variant",
        default=os.environ.get("VARIANT", "full"),
        help="Preprocessing variant (default: $VARIANT or 'full').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(f"data/processed_{args.variant}")
    model_path = Path(f"models/{args.variant}/{MODEL_FILENAME}")

    print(f"Training logistic regression model (variant={args.variant})...")

    train_df, val_df, test_df = load_processed_splits(processed_dir)
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

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_path)
    print(f"Model saved to {model_path}")

    save_confusion_matrix(
        y_test, y_pred, model_name=MODEL_NAME, output_dir=model_path.parent
    )


if __name__ == "__main__":
    main()
