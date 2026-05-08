# Financial Classification

This repository contains our CS 549 project on classifying personal financial transactions into categories such as groceries, rent, transportation, and entertainment.

## Project Structure

```text
financial-classification/
├── README.md
├── RESULTS.md
├── requirements.txt
├── data/
│   ├── Personal_Finance_Dataset.csv
│   ├── aug_personal_transactions_with_UserId.csv
│   └── processed/
├── models/
├── notebooks/
│   └── MODEL_COMPARISON.ipynb
└── src/
    ├── data_preprocessing.py
    ├── evaluate.py
    ├── featurization.py
    ├── logistic_regression.py
    ├── model_data.py
    ├── random_forest.py
    └── svm.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Preprocessing

```bash
python3 -m src.data_preprocessing
```

This creates the shared dataset splits used by every model:

- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`
- `data/processed/transactions_preprocessed.csv`

## Run Models

### Local Execution

```bash
python3 -m src.random_forest
python3 -m src.svm
python3 -m src.logistic_regression
```

## Shared Features

All models use the same feature set:

- `description_clean` with TF-IDF (5000 features, 1-3 grams)
- `date_month` and `date_day_of_week` (one-hot encoded)
- `transaction_type` (one-hot encoded)
- `account_name` (one-hot encoded)
- `amount` (standardized)

The preprocessing script also normalizes category names, removes duplicates, and creates a stratified train/validation/test split.

## Project Team

- **Data Preprocessing & Evaluation**: Team
- **SVM Implementation**: Dariel Gutierrez
- **Random Forest Implementation**: Team
- **Logistic Regression**: Team
