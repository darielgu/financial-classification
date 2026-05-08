from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DESCRIPTION_FEATURE = "description_clean"
CATEGORICAL_FEATURES = ["date_month", "date_day_of_week", "transaction_type", "account_name"]
TEXT_LIKE_CATEGORICAL_FEATURES = [DESCRIPTION_FEATURE, *CATEGORICAL_FEATURES]
NUMERIC_FEATURES = ["amount"]
TARGET_COLUMN = "category"


def build_feature_transformer() -> ColumnTransformer:
    """Shared featurization for all models.

    Transformations applied:
    - description_clean : TF-IDF with a larger n-gram vocabulary for transaction text
    - date_month, date_day_of_week : one-hot encoding (ignores unseen values)
    - amount : standard scaling (zero mean, unit variance)

    NOTE: Fit this transformer on x_train only. Apply transform() to val and
    test sets separately to avoid data leakage::

        x_train_t = feature_transformer.fit_transform(x_train)
        x_val_t   = feature_transformer.transform(x_val)
        x_test_t  = feature_transformer.transform(x_test)
    """
    return ColumnTransformer(
        transformers=[
            (
                "text_tfidf",
                TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 3),
                    min_df=2,
                    sublinear_tf=True,
                ),
                DESCRIPTION_FEATURE,  # scalar str → 1-D Series, required by TfidfVectorizer
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
