from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CANONICAL_COLUMNS = [
    "date",
    "description",
    "amount",
    "transaction_type",
    "account_name",
    "category",
    "source_dataset",
]


CATEGORY_MAP = {
    # --- Dining consolidation ---
    # Merge Restaurants, Fast Food, Food & Dining, and dataset-1 "Food & Drink"
    # into a single "Dining" label. All represent money spent eating out.
    "food & drink":  "Dining",
    "restaurants":   "Dining",
    "fast food":     "Dining",
    "food & dining": "Dining",

    # --- Entertainment consolidation ---
    # Merge Movies & DVDs, Television, and Music into "Entertainment".
    # Television (52 samples) and Music (124) are too small to learn alone,
    # and all three represent discretionary media spending.
    "entertainment": "Entertainment",
    "movies & dvds": "Entertainment",
    "television":    "Entertainment",
    "music":         "Entertainment",

    # --- Income consolidation ---
    # Salary and Paycheck are semantically identical income transactions.
    "salary":        "Income",
    "paycheck":      "Income",

    # --- Other cross-dataset normalisations ---
    "rent":          "Mortgage & Rent",
    "utilities":     "Bills & Utilities",
    "shopping":      "Shopping",
    "investment":    "Investment",
    "transportation":"Transportation",
    "health":        "Health",
    "insurance":     "Insurance",
    "grocery":       "Groceries",
    "education":     "Education",
}


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _to_title_or_empty(value: object) -> str:
    text = _normalize_text(value)
    return text.title() if text else ""


def load_dataset_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {
        "Date": "date",
        "Transaction Description": "description",
        "Category": "category",
        "Amount": "amount",
        "Type": "transaction_type",
    }
    df = df.rename(columns=rename_map)
    df["account_name"] = np.nan
    df["source_dataset"] = path.name
    return df


def load_dataset_two(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {
        "Date": "date",
        "Description": "description",
        "Amount": "amount",
        "Category": "category",
        "Transaction Type": "transaction_type",
        "Account Name": "account_name",
    }
    df = df.rename(columns=rename_map)
    df["source_dataset"] = path.name
    return df


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    out = df[CANONICAL_COLUMNS].copy()
    out["description"] = out["description"].map(_normalize_text)

    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out.dropna(subset=["amount"]).copy()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()

    out["category"] = out["category"].map(_normalize_text)
    out["category"] = out["category"].replace("", np.nan)

    # Build normalized categories to reduce mismatches between datasets.
    labeled_mask = out["category"].notna()
    cat_lower = out.loc[labeled_mask, "category"].str.lower()
    out.loc[labeled_mask, "category"] = (
        cat_lower.map(CATEGORY_MAP).fillna(out.loc[labeled_mask, "category"].map(_to_title_or_empty))
    )

    out["description_clean"] = (
        out["description"].str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    out["date_month"] = out["date"].dt.month.astype(str)
    out["date_day_of_week"] = out["date"].dt.day_name()

    # Normalize transaction_type to lowercase; fill missing as "unknown"
    out["transaction_type"] = (
        out["transaction_type"].map(_normalize_text).str.lower().replace("", "unknown")
    )
    # Normalize account_name to title case; fill missing as "unknown"
    out["account_name"] = (
        out["account_name"].map(_to_title_or_empty).replace("", "unknown")
    )

    return out


def combine_and_preprocess(path_one: Path, path_two: Path) -> pd.DataFrame:
    df1 = standardize_schema(load_dataset_one(path_one))
    df2 = standardize_schema(load_dataset_two(path_two))
    combined = pd.concat([df1, df2], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "description_clean", "amount", "category"],
        keep="first",
    )
    combined = combined.sort_values(["date", "description_clean"]).reset_index(drop=True)
    return combined


def make_splits(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labeled = df[df["category"].notna()].copy()

    # Keep classes with >=2 labeled records so stratified split is valid.
    counts = labeled["category"].value_counts()
    valid_categories = counts[counts >= 2].index
    filtered = labeled[labeled["category"].isin(valid_categories)].copy()

    train_val, test = train_test_split(
        filtered,
        test_size=0.2,
        random_state=seed,
        stratify=filtered["category"],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.2,
        random_state=seed,
        stratify=train_val["category"],
    )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_outputs(df: pd.DataFrame, output_dir: Path, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / "transactions_preprocessed.csv"
    df.to_csv(combined_path, index=False)

    train, val, test = make_splits(df, seed=seed)
    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    feature_columns = [
        "description_clean",
        "amount",
        "date_month",
        "date_day_of_week",
        "transaction_type",
        "account_name",
    ]
    metadata_path = output_dir / "feature_columns.txt"
    metadata_path.write_text("\n".join(feature_columns) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine two transaction CSVs and create one shared preprocessed dataset for all models."
    )
    parser.add_argument(
        "--dataset-one",
        type=Path,
        default=Path("data/Personal_Finance_Dataset.csv"),
        help="Path to Personal_Finance_Dataset.csv",
    )
    parser.add_argument(
        "--dataset-two",
        type=Path,
        default=Path("data/aug_personal_transactions_with_UserId.csv"),
        help="Path to aug_personal_transactions_with_UserId.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where preprocessed outputs are written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = combine_and_preprocess(args.dataset_one, args.dataset_two)
    save_outputs(combined, args.output_dir, seed=args.seed)

    print(f"Combined rows: {len(combined):,}")
    print(f"Unique categories: {combined['category'].nunique()}")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
