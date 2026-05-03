"""Random Forest classifier for financial transaction categorization.

Usage
-----
    python -m src.random_forest

This script:
1. Loads the shared preprocessed train / val / test splits.
2. Builds an sklearn Pipeline (TF-IDF + OHE + scaler → RandomForest).
3. Tunes hyperparameters with RandomizedSearchCV on the training set only.
4. Evaluates the best model on the held-out test set.
5. Saves the fitted pipeline to models/random_forest.joblib.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Random Forest"
MODEL_PATH = Path("models/random_forest.joblib")
RANDOM_STATE = 42

# Hyperparameter search space
PARAM_DIST = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "max_features": ["sqrt", "log2", 0.3],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": ["balanced", "balanced_subsample"],
}


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_classifier() -> RandomForestClassifier:
    """Return an unfitted RandomForestClassifier with sensible defaults.

    Feature encoding and SMOTE oversampling are handled upstream by
    get_data_for_model(), so only the classifier is needed here.
    ``class_weight="balanced"`` is kept as a secondary guard against any
    residual imbalance after SMOTE.
    """
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def tune_classifier(
    clf: RandomForestClassifier, x_train, y_train
) -> RandomForestClassifier:
    """Run RandomizedSearchCV over PARAM_DIST and return the best estimator.

    Uses stratified 5-fold CV scored by macro-F1 (appropriate for
    multi-class problems with potential class imbalance).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=PARAM_DIST,
        n_iter=20,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Running hyperparameter search (this may take a few minutes)...")
    search.fit(x_train, y_train)

    print(f"\nBest params : {search.best_params_}")
    print(f"Best CV F1  : {search.best_score_:.4f}")

    return search.best_estimator_


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load data — transformer is fitted inside get_data_for_model
    x_train, y_train, x_val, y_val, x_test, y_test, _ = get_data_for_model()

    # 2. Build and tune classifier on the SMOTE-oversampled training set
    clf = build_classifier()
    best_clf = tune_classifier(clf, x_train, y_train)

    # 3. Evaluate on the held-out test set (never seen during training/tuning)
    y_pred = best_clf.predict(x_test)

    metrics = compute_metrics(y_test, y_pred, model_name=MODEL_NAME)
    print_report(metrics)
    print_classification_report(y_test, y_pred)

    # 4. Save the fitted classifier
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 5. Save confusion matrix plot
    save_confusion_matrix(y_test, y_pred, model_name=MODEL_NAME, output_dir=MODEL_PATH.parent)


if __name__ == "__main__":
    main()
