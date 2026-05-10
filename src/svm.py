"""Train and evaluate the SVM model.

An RBF kernel grid is included but disabled by ``USE_RBF_KERNEL`` because
RBF is too slow on the ~10k-feature TF-IDF input and ``LinearSVC`` matches
it in practice.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC

from src.baseline import run_baseline_diagnostic
from src.evaluate import (
    compute_metrics,
    print_classification_report,
    print_report,
    save_confusion_matrix,
)
from src.model_data import load_processed_splits, prepare_features

MODEL_NAME = "SVM"
MODEL_FILENAME = "svm.joblib"
RANDOM_STATE = 42

# LinearSVC: tune regularization strength C only.
PARAM_GRID_LINEAR = [
    {"C": 0.01, "class_weight": "balanced"},
    {"C": 0.05, "class_weight": "balanced"},
    {"C": 0.1, "class_weight": "balanced"},
    {"C": 0.5, "class_weight": "balanced"},
    {"C": 1.0, "class_weight": "balanced"},
    {"C": 2.0, "class_weight": "balanced"},
    {"C": 5.0, "class_weight": "balanced"},
    {"C": 10.0, "class_weight": "balanced"},
]

# RBF SVC: kept for reference, disabled by USE_RBF_KERNEL below.
PARAM_GRID_RBF = [
    {"C": 0.1, "gamma": "scale"},
    {"C": 0.5, "gamma": "scale"},
    {"C": 1.0, "gamma": "scale"},
    {"C": 5.0, "gamma": "scale"},
    {"C": 10.0, "gamma": "scale"},
    {"C": 0.1, "gamma": "auto"},
    {"C": 0.5, "gamma": "auto"},
    {"C": 1.0, "gamma": "auto"},
]

USE_RBF_KERNEL = False


def build_classifier(
    *, C: float, class_weight: str | None = None, **kwargs
) -> LinearSVC | SVC:
    """Build LinearSVC or RBF SVC depending on the USE_RBF_KERNEL flag."""
    if USE_RBF_KERNEL:
        return SVC(
            kernel="rbf",
            C=C,
            class_weight=class_weight,
            gamma=kwargs.get("gamma", "scale"),
            random_state=RANDOM_STATE,
            probability=True,
        )
    return LinearSVC(
        C=C,
        class_weight=class_weight,
        dual="auto",
        max_iter=10000,
        random_state=RANDOM_STATE,
    )


def tune_with_validation(x_train, y_train, x_val, y_val) -> dict:
    """Grid search over the active param grid; pick best macro-F1 on val."""
    param_grid = PARAM_GRID_RBF if USE_RBF_KERNEL else PARAM_GRID_LINEAR
    best_result: dict | None = None

    for params in param_grid:
        kwargs = {k: v for k, v in params.items() if k not in ("C", "class_weight")}
        clf = build_classifier(C=params["C"], class_weight="balanced", **kwargs)
        clf.fit(x_train, y_train)
        y_val_pred = clf.predict(x_val)
        score = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

        result = {"params": params, "score": score}
        if best_result is None or score > best_result["score"]:
            best_result = result

    assert best_result is not None
    return best_result


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

    train_df, val_df, test_df = load_processed_splits(processed_dir)

    x_train, y_train, transformed, feature_transformer = prepare_features(
        train_df, [val_df, test_df], dense=True,
    )
    (x_val, y_val), (x_test, y_test) = transformed

    run_baseline_diagnostic(x_train, y_train, x_test, y_test)

    best_result = tune_with_validation(x_train, y_train, x_val, y_val)
    best_params = best_result["params"]

    print("Validation tuning results")
    print(f"Best params : {best_params}")
    print(f"Best val F1 : {best_result['score']:.4f}")
    print(f"Kernel      : {'RBF' if USE_RBF_KERNEL else 'Linear'}")

    if sp.issparse(x_train):
        x_train_val = sp.vstack([x_train, x_val])
    else:
        x_train_val = np.vstack([x_train, x_val])
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)

    kwargs = {k: v for k, v in best_params.items() if k not in ("C", "class_weight")}
    clf = build_classifier(C=best_params["C"], class_weight="balanced", **kwargs)
    clf.fit(x_train_val, y_train_val)
    y_pred = clf.predict(x_test)

    metrics = compute_metrics(y_test, y_pred, model_name=MODEL_NAME)
    print_report(metrics)
    print_classification_report(y_test, y_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_transformer": feature_transformer}, model_path)
    print(f"Model saved to {model_path}")

    save_confusion_matrix(
        y_test, y_pred, model_name=MODEL_NAME, output_dir=model_path.parent
    )


if __name__ == "__main__":
    main()
