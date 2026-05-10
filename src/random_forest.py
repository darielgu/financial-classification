"""Train and evaluate the random forest model.

Tunes the standard tree knobs via ``RandomizedSearchCV`` (5-fold,
macro-F1, 20 iterations) and saves the best estimator.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.evaluate import (
    compute_metrics,
    print_classification_report,
    print_report,
    save_confusion_matrix,
)
from src.model_data import get_data_for_model

VARIANT = os.environ.get("VARIANT", "full")
MODEL_NAME = "Random Forest"
MODEL_PATH = Path(f"models/{VARIANT}/random_forest.joblib")
RANDOM_STATE = 42

PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "max_features": ["sqrt", "log2", 0.3],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", "balanced_subsample"],
}


def build_classifier() -> RandomForestClassifier:
    """Return an unfitted RandomForestClassifier with class-balanced weighting."""
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def tune_classifier(
    clf: RandomForestClassifier, x_train, y_train
) -> RandomForestClassifier:
    """Randomized search over ``PARAM_DIST`` with 5-fold stratified CV scored by macro-F1."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=PARAM_DIST,
        n_iter=20,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        refit=True,
    )
    print("Running hyperparameter search (this may take a few minutes)...")
    search.fit(x_train, y_train)
    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV F1  : {search.best_score_:.4f}")
    return search.best_estimator_


def main() -> None:
    x_train, y_train, _x_val, _y_val, x_test, y_test, _ = get_data_for_model()

    clf = build_classifier()
    best_clf = tune_classifier(clf, x_train, y_train)

    y_pred = best_clf.predict(x_test)

    metrics = compute_metrics(y_test, y_pred, model_name=MODEL_NAME)
    print_report(metrics)
    print_classification_report(y_test, y_pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    save_confusion_matrix(y_test, y_pred, model_name=MODEL_NAME, output_dir=MODEL_PATH.parent)


if __name__ == "__main__":
    main()
