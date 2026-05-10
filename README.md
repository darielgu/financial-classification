# Financial Classification

CS 549 (Machine Learning, Spring 2026) project on classifying personal financial transactions into categories (Dining, Groceries, Mortgage & Rent, Entertainment, Income, ...).

## Project Team

- **Joshua Sherrod** — Logistic Regression
- **Bradley Mustoe** — Random Forest
- **Dariel Gutierrez** — Support Vector Machine
- **Zander Barajas** — Neural Network
- **Group** — Data preprocessing, evaluation pipeline, comparative analysis, final report

## Project Structure

```text
financial-classification/
├── README.md
├── CLAUDE.md                                         # internal contributor notes
├── requirements.txt
├── data/
│   ├── Personal_Finance_Dataset.csv                  # raw D1 (Kaggle: ramyapintchy/personal-finance-data)
│   ├── aug_personal_transactions_with_UserId.csv    # raw D2 (Kaggle: shyakanobledavid/personal-transactions-userid-new-transactions)
│   └── processed_<variant>/                          # generated per variant: train.csv / val.csv / test.csv
├── models/
│   └── <variant>/                                    # generated: <model>.joblib + <model>_confusion_matrix.png
├── notebooks/
│   └── MODEL_COMPARISON.ipynb                        # cross-variant comparison + figures
└── src/
    ├── data_preprocessing.py     # load + clean + split (--variant flag)
    ├── featurization.py          # shared TF-IDF + OHE + StandardScaler ColumnTransformer
    ├── model_data.py             # load processed splits + transform features
    ├── evaluate.py               # metrics + confusion matrix + learning curve + runtime helpers
    ├── baseline.py               # stratified DummyClassifier diagnostic
    ├── logistic_regression.py
    ├── random_forest.py
    ├── svm.py
    └── neural_network.py
```

## Setup

The repo uses [`uv`](https://docs.astral.sh/uv/) as the runner. Plain `pip + venv` also works.

```bash
# with uv (recommended)
uv venv
uv pip install -r requirements.txt

# or with pip
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Environment used for the report:
- Python 3.14
- scikit-learn 1.5+, imbalanced-learn 0.12+, pandas 2.2+, numpy 1.26+, scipy 1.12+, matplotlib 3.8+, seaborn 0.13+, joblib 1.3+

## Preprocessing Variants

The pipeline supports four variants designed to isolate the impact of data quality on classification performance:

| Variant | Data | Why |
|---------|------|-----|
| `d1` | D1 only | D1 has Faker-generated descriptions; useful as a "no real signal" baseline |
| `full` | D1 + D2 | proposal baseline (both raw Kaggle CSVs) |
| `d2` | D2 only | drops D1's synthetic descriptions |
| `d2_clean` | D2 + modal-category denoising | "clean ceiling" — keeps only rows whose category equals the modal category for that description |

Generate the processed splits:

```bash
uv run python -m src.data_preprocessing --variant full
uv run python -m src.data_preprocessing --variant d1
uv run python -m src.data_preprocessing --variant d2
uv run python -m src.data_preprocessing --variant d2_clean
```

Each variant writes to `data/processed_<variant>/`.

## Run Models

Each model script reads the `VARIANT` environment variable (default `full`) and saves to `models/<variant>/`.

```bash
# Linux / macOS
VARIANT=full uv run python -m src.logistic_regression
VARIANT=full uv run python -m src.random_forest
VARIANT=full uv run python -m src.svm
VARIANT=full uv run python -m src.neural_network

# Windows PowerShell
$env:VARIANT="full"
uv run python -m src.logistic_regression
uv run python -m src.random_forest
uv run python -m src.svm
uv run python -m src.neural_network
```

Repeat with `VARIANT=d1`, `VARIANT=d2`, `VARIANT=d2_clean` to populate all 16 (variant × model) artifacts.

## Cross-Variant Comparison

`notebooks/MODEL_COMPARISON.ipynb` runs preprocessing + training for all variants, then loads every `models/<variant>/<model>.joblib` and produces:

- `models/cross_variant_results.csv` — long-form results
- `models/cross_variant_f1_pivot.csv` — macro-F1 by (model × variant)
- `models/cross_variant_accuracy_pivot.csv` — accuracy by (model × variant)
- `models/cross_variant_comparison.png` — grouped bar chart, 4 metrics × 4 models × 4 variants
- `models/cross_variant_heatmap.png` — macro-F1 heatmap

## Shared Features

All models reuse the same `ColumnTransformer` from `src/featurization.py`:

- `description_clean` → `TfidfVectorizer(max_features=10000, ngram_range=(1,4), min_df=2, max_df=0.95, sublinear_tf=True)`
- `date_month`, `date_day_of_week`, `transaction_type`, `account_name` → `OneHotEncoder(handle_unknown="ignore")`
- `amount` → `StandardScaler`

Output is sparse by default; `get_data_for_model(dense=True)` densifies for models that need it (RF, LR-default).

## Model Tuning

| Model | Tuning method | Knobs |
|-------|--------------|-------|
| Logistic Regression | `GridSearchCV` (5-fold, macro-F1) | `C` |
| Random Forest | `RandomizedSearchCV` (5-fold, macro-F1, 20 iter) | `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`, `class_weight` |
| SVM | manual val-set grid | `C`, `class_weight` (LinearSVC) |
| Neural Network | `RandomizedSearchCV` (5-fold, macro-F1, 15 iter) | `hidden_layer_sizes`, `learning_rate_init`, `alpha` |

All models refit on `train + val` before the final test prediction.

## Reproducibility

- Random seed `42` everywhere.
- Test set held out only for final evaluation; tuning happens on train (CV) or val.
- Stratified train/val/test split (64/16/20) by `category`.
