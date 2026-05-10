"""Shared ``ColumnTransformer`` used by every model script."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DESCRIPTION_FEATURE = "description_clean"
CATEGORICAL_FEATURES = [
    "date_month",
    "date_day_of_week",
    "transaction_type",
    "account_name",
]
TEXT_LIKE_CATEGORICAL_FEATURES = [DESCRIPTION_FEATURE, *CATEGORICAL_FEATURES]
NUMERIC_FEATURES = ["amount"]
TARGET_COLUMN = "category"


def build_feature_transformer() -> ColumnTransformer:
    """Build the shared ColumnTransformer.

    - ``description_clean`` — TF-IDF (1-4 grams, sublinear, max 10k features)
    - ``date_month``, ``date_day_of_week``, ``transaction_type``,
      ``account_name`` — one-hot encoding (ignores unseen values)
    - ``amount`` — standard scaling (zero mean, unit variance)
    """
    return ColumnTransformer(
        transformers=[
            (
                "text_tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 4),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
                DESCRIPTION_FEATURE,
            ),
            (
                "categorical_one_hot",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric_scaled", StandardScaler(), NUMERIC_FEATURES),
        ],
        sparse_threshold=0.3,
    )
