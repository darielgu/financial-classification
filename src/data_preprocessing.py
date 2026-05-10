"""Combine the two raw transaction CSVs into a shared, variant-aware dataset.

Run as a module: ``python -m src.data_preprocessing --variant <V>``.

Steps:
1. Load the raw CSV(s) selected by ``--variant`` (D1 only / D2 only / both).
2. Standardize the schema, normalize categories via ``CATEGORY_MAP``,
   drop rows with unparseable date or amount, and dedupe.
3. For ``--variant d2_clean`` only, apply modal-category denoising
   (keep rows whose label matches the modal label for their description).
4. Stratified 64/16/20 train/val/test split by category.
5. Write splits to ``data/processed_<variant>/`` plus a feature-columns
   metadata file.
"""

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


# Lowercased raw labels mapped to canonical category names. Groups
# semantically equivalent labels across the two source datasets.
CATEGORY_MAP = {
    # Dining: restaurants, fast food, coffee, and D1's "Food & Drink".
    "food & drink":  "Dining",
    "restaurants":   "Dining",
    "fast food":     "Dining",
    "food & dining": "Dining",
    "coffee shops":  "Dining",

    # Entertainment: TV (52 rows) and Music (124) are too small to learn
    # alone; all three are discretionary media spending.
    "entertainment": "Entertainment",
    "movies & dvds": "Entertainment",
    "television":    "Entertainment",
    "music":         "Entertainment",

    # Income: Salary and Paycheck are semantically identical.
    "salary":        "Income",
    "paycheck":      "Income",

    # Cross-dataset normalizations.
    "rent":          "Mortgage & Rent",
    "utilities":     "Bills & Utilities",
    "internet":      "Bills & Utilities",
    "mobile phone":  "Bills & Utilities",
    "shopping":      "Shopping",
    "investment":    "Investment",
    "transportation":"Transportation",
    "health":        "Health",
    "insurance":     "Insurance",
    "grocery":       "Groceries",
    "education":     "Education",
}


VARIANTS = ("full", "d1", "d2", "d2_clean")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _to_title_or_empty(value: object) -> str:
    text = _normalize_text(value)
    return text.title() if text else ""


def load_dataset_one(path: Path) -> pd.DataFrame:
    """Load D1 (Personal_Finance_Dataset.csv) and rename to canonical columns."""
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
    """Load D2 (aug_personal_transactions_with_UserId.csv) into canonical columns."""
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
    """Coerce types, drop bad rows, normalize labels, and derive feature columns."""
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

    # Normalize categories via CATEGORY_MAP; unmapped labels fall back to title-case.
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

    out["transaction_type"] = (
        out["transaction_type"].map(_normalize_text).str.lower().replace("", "unknown")
    )
    out["account_name"] = (
        out["account_name"].map(_to_title_or_empty).replace("", "unknown")
    )

    return out


def combine_and_preprocess(
    path_one: Path,
    path_two: Path,
    variant: str = "full",
) -> pd.DataFrame:
    """Load + standardize + dedupe; optionally drop D1 and/or denoise D2.

    Variants:
      - ``full``     — load both D1 and D2.
      - ``d1``       — D1 only (Faker-generated descriptions, clean labels).
      - ``d2``       — D2 only (real descriptions, noisy labels).
      - ``d2_clean`` — D2 only, then keep rows whose category equals the modal
        category for that ``description_clean`` (mitigates D2 label noise).
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {VARIANTS}.")

    if variant == "full":
        df1 = standardize_schema(load_dataset_one(path_one))
        df2 = standardize_schema(load_dataset_two(path_two))
        combined = pd.concat([df1, df2], ignore_index=True)
    elif variant == "d1":
        combined = standardize_schema(load_dataset_one(path_one))
    else:  # d2 or d2_clean
        combined = standardize_schema(load_dataset_two(path_two))

    combined = combined.drop_duplicates(
        subset=["date", "description_clean", "amount", "category"],
        keep="first",
    )

    if variant == "d2_clean":
        combined = _denoise_modal_category(combined)

    combined = combined.sort_values(["date", "description_clean"]).reset_index(drop=True)
    return combined


def _denoise_modal_category(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose category matches the modal category for their description.

    For each unique ``description_clean``, drop rows whose category disagrees
    with the most-frequent label among labeled rows. Unlabeled rows pass
    through unchanged — ``make_splits`` filters them anyway.
    """
    labeled = df[df["category"].notna()].copy()
    modal = labeled.groupby("description_clean")["category"].agg(
        lambda s: s.mode().iloc[0]
    )
    labeled["_modal"] = labeled["description_clean"].map(modal)
    labeled = labeled[labeled["category"] == labeled["_modal"]].drop(columns="_modal")

    unlabeled = df[df["category"].isna()]
    return pd.concat([labeled, unlabeled], ignore_index=True)


def make_splits(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 64/16/20 train/val/test split on labeled rows."""
    labeled = df[df["category"].notna()].copy()

    # Stratified split needs at least 2 samples per class.
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
    """Write the combined preprocessed CSV, train/val/test splits, and metadata."""
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
        description="Combine two transaction CSVs into one shared preprocessed dataset for all models."
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="full",
        help=(
            "Preprocessing variant: 'full' (D1+D2 baseline), "
            "'d1' (Faker-only D1, clean labels), "
            "'d2' (D2 only, drops Faker-text D1), "
            "'d2_clean' (D2 only + modal-category denoising)."
        ),
    )
    parser.add_argument(
        "--dataset-one",
        type=Path,
        default=Path("data/Personal_Finance_Dataset.csv"),
        help="Path to Personal_Finance_Dataset.csv (used by --variant full and d1).",
    )
    parser.add_argument(
        "--dataset-two",
        type=Path,
        default=Path("data/aug_personal_transactions_with_UserId.csv"),
        help="Path to aug_personal_transactions_with_UserId.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to data/processed_<variant>/.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path(f"data/processed_{args.variant}")

    combined = combine_and_preprocess(args.dataset_one, args.dataset_two, variant=args.variant)
    save_outputs(combined, output_dir, seed=args.seed)

    print(f"Variant          : {args.variant}")
    print(f"Combined rows    : {len(combined):,}")
    print(f"Unique categories: {combined['category'].nunique()}")
    print(f"Saved outputs to : {output_dir}")


if __name__ == "__main__":
    main()
