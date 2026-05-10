"""Train and evaluate the neural network (MLPClassifier) model.

Pipeline::

    sparse features (TF-IDF + OHE + scaled amount)
        -> StandardScaler(with_mean=False)   # sparse-safe; equalize feature scales
        -> RandomOverSampler (capped)        # mitigate class imbalance, train-time only
        -> MLPClassifier                     # adam + early stopping

The full pipeline is dumped to disk so the comparison notebook can keep
calling ``model.predict(x_test)`` against sparse features unchanged.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.baseline import run_baseline_diagnostic
from src.evaluate import (
    compute_metrics,
    print_classification_report,
    print_report,
    print_runtime_summary,
    save_confusion_matrix,
    save_learning_curve,
)
from src.model_data import get_data_for_model

MODEL_NAME = "Neural Network"
MODEL_FILENAME = "neural_network.joblib"
RANDOM_STATE = 42
MINORITY_FLOOR = 200  # cap for the capped RandomOverSampler

# Tuning surface: layers and learning rate (named in the proposal) plus alpha
# (L2) for regularization. Activation and batch_size are held constant.
# The ``mlp__`` prefix routes parameters into the MLP step of the pipeline.
PARAM_DIST = {
    "mlp__hidden_layer_sizes": [(128,), (256,), (256, 128), (512, 256)],
    "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
    "mlp__alpha":              [1e-4, 1e-3, 1e-2],
}


class LabelDecodingClassifier:
    """Wrap a fitted classifier and decode integer predictions back to strings.

    sklearn >= 1.5 ``MLPClassifier``'s early-stopping path calls
    ``np.isnan(y_pred)`` on raw class labels, which raises ``TypeError`` on
    string-typed ``y``. Encoding ``y`` to ints sidesteps the bug while
    preserving the string-label ``predict()`` interface the comparison
    notebook expects.
    """

    def __init__(self, estimator, label_encoder: LabelEncoder):
        self.estimator = estimator
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X):
        y_pred_int = self.estimator.predict(X)
        return self.label_encoder.inverse_transform(y_pred_int)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def capped_sampling_strategy(y, cap: int = MINORITY_FLOOR) -> dict:
    """Return ``{class -> cap}`` for every class with fewer than ``cap`` samples."""
    counts = pd.Series(y).value_counts()
    return {cls: cap for cls, count in counts.items() if int(count) < cap}


def build_pipeline(sampling_strategy) -> ImbPipeline:
    """Build the scaler + capped oversampler + MLP pipeline.

    ``RandomOverSampler`` is wrapped in ``imblearn.pipeline.Pipeline`` so it
    only runs during ``fit`` (no leakage at predict time).
    """
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("ros", RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=RANDOM_STATE,
            )),
            (
                "mlp",
                MLPClassifier(
                    solver="adam",
                    max_iter=120,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_pipeline(pipeline: ImbPipeline, x_train, y_train) -> dict:
    """Randomized search over ``PARAM_DIST`` with 5-fold stratified CV scored by macro-F1."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DIST,
        n_iter=15,
        scoring="f1_macro",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=2,
        refit=False,
    )
    print("Running hyperparameter search (15 configs x 5-fold CV)...")
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

    x_train, y_train, x_val, y_val, x_test, y_test, _ = get_data_for_model(
        processed_dir=processed_dir, dense=False
    )

    # Encode string categories to ints; works around the sklearn >=1.5 MLP
    # early-stopping isnan-on-strings bug. See LabelDecodingClassifier.
    label_encoder = LabelEncoder()
    label_encoder.fit(pd.concat([y_train, y_val, y_test], ignore_index=True))
    y_train_enc = pd.Series(label_encoder.transform(y_train))
    y_val_enc = pd.Series(label_encoder.transform(y_val))

    # DummyClassifier baseline needs dense inputs.
    x_train_dense = x_train.toarray() if sp.issparse(x_train) else np.asarray(x_train)
    x_test_dense = x_test.toarray() if sp.issparse(x_test) else np.asarray(x_test)
    run_baseline_diagnostic(x_train_dense, y_train, x_test_dense, y_test)

    cv_strategy = capped_sampling_strategy(y_train_enc)
    print(
        f"\nCapped oversampling strategy (cap={MINORITY_FLOOR}): "
        f"{len(cv_strategy)} of {y_train_enc.nunique()} classes lifted"
    )

    pipeline = build_pipeline(cv_strategy)

    t_search_start = time.perf_counter()
    best_params = tune_pipeline(pipeline, x_train, y_train_enc)
    search_elapsed = time.perf_counter() - t_search_start

    if sp.issparse(x_train):
        x_train_val = sp.vstack([x_train, x_val])
    else:
        x_train_val = np.vstack([x_train, x_val])
    y_train_val_enc = pd.concat([y_train_enc, y_val_enc], ignore_index=True)

    refit_strategy = capped_sampling_strategy(y_train_val_enc)
    final_pipeline = build_pipeline(refit_strategy)
    final_pipeline.set_params(**best_params)

    print("\nRefitting best config on train+val...")
    t_refit_start = time.perf_counter()
    final_pipeline.fit(x_train_val, y_train_val_enc)
    refit_elapsed = time.perf_counter() - t_refit_start

    mlp = final_pipeline.named_steps["mlp"]
    print(f"Refit completed in {refit_elapsed:.1f}s over {mlp.n_iter_} epochs.")

    final = LabelDecodingClassifier(final_pipeline, label_encoder)
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
    save_learning_curve(mlp, model_name=MODEL_NAME, output_dir=model_path.parent)
    print_runtime_summary(search_elapsed, refit_elapsed)


if __name__ == "__main__":
    main()
