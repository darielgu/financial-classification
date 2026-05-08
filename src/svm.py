# DARIELS PART
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
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
MODEL_PATH = Path("models/svm.joblib")

# Expanded hyperparameter grid with RBF kernel option
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

USE_RBF_KERNEL = False  # RBF is too slow for 10k features; LinearSVC is much faster


def build_classifier(
    *, C: float, class_weight: str | None = None, **kwargs
) -> LinearSVC | SVC:
    """Build LinearSVC or RBF SVC based on USE_RBF_KERNEL flag."""
    if USE_RBF_KERNEL:
        return SVC(
            kernel="rbf",
            C=C,
            class_weight=class_weight,
            gamma=kwargs.get("gamma", "scale"),
            random_state=42,
            probability=True,
        )
    else:
        return LinearSVC(
            C=C,
            class_weight=class_weight,
            dual="auto",
            max_iter=10000,
            random_state=42,
        )


def tune_with_validation(x_train, y_train, x_val, y_val) -> dict:
    """Tune SVM hyperparameters on validation set."""
    param_grid = PARAM_GRID_RBF if USE_RBF_KERNEL else PARAM_GRID_LINEAR
    best_result: dict | None = None

    for params in param_grid:
        clf = build_classifier(C=params["C"], class_weight="balanced", **{k: v for k, v in params.items() if k != "C"})
        clf.fit(x_train, y_train)
        y_val_pred = clf.predict(x_val)
        score = f1_score(y_val, y_val_pred, average="macro", zero_division=0)

        result = {"params": params, "score": score}
        if best_result is None or score > best_result["score"]:
            best_result = result

    assert best_result is not None
    return best_result




def main() -> None:
    train_df, val_df, test_df = load_processed_splits()

    x_train, y_train, transformed, feature_transformer = prepare_features(
        train_df,
        [val_df, test_df],
        dense=True,
    )
    (x_val, y_val), (x_test, y_test) = transformed

    # Run baseline diagnostic
    run_baseline_diagnostic(x_train, y_train, x_test, y_test)

    # Tune hyperparameters
    best_result = tune_with_validation(x_train, y_train, x_val, y_val)
    best_params = best_result["params"]

    print("Validation tuning results")
    print(f"Best params : {best_params}")
    print(f"Best val F1 : {best_result['score']:.4f}")
    if USE_RBF_KERNEL:
        print(f"Kernel      : RBF")
    else:
        print(f"Kernel      : Linear")

    # Retrain on combined train+val using SAME feature_transformer and SAME labels
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    x_train_val_raw, _ = get_xy(train_val_df)
    x_train_val = feature_transformer.transform(x_train_val_raw)  # Use existing transformer, don't refit
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)  # Use same label arrays (consistent encoding)

    clf = build_classifier(C=best_params["C"], class_weight="balanced", **{k: v for k, v in best_params.items() if k != "C"})
    clf.fit(x_train_val, y_train_val)
    y_pred = clf.predict(x_test)

    metrics = compute_metrics(y_test, y_pred, model_name=MODEL_NAME)
    print_report(metrics)
    print_classification_report(y_test, y_pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_transformer": feature_transformer}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    save_confusion_matrix(
        y_test, y_pred, model_name=MODEL_NAME, output_dir=MODEL_PATH.parent
    )



if __name__ == "__main__":
    main()
