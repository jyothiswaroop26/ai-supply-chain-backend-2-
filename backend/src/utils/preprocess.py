"""
Data preprocessing utilities for sales data cleaning.
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default path relative to this file's location
DEFAULT_DATA_PATH = str(Path(__file__).resolve().parents[2] / "data" / "sales_data.csv")

NUMERIC_COLUMNS = ["quantity", "unit_price", "discount"]
DATE_COLUMNS = ["order_date", "delivery_date"]
REQUIRED_COLUMNS = [
    "order_id", "product_id", "product_name", "category",
    "supplier_id", "supplier_name", "region", "quantity",
    "unit_price", "discount", "order_date", "delivery_date",
    "status", "customer_id", "warehouse_id",
]


def load_data(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load sales CSV data into a DataFrame."""
    filepath = str(Path(filepath).resolve())
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows from %s", len(df), filepath)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric columns to proper types and remove invalid values.

    - Coerces non-numeric strings (e.g. 'N/A') to NaN.
    - Fills missing discount with 0.
    - Drops rows where quantity or unit_price cannot be parsed.
    - Removes rows with negative quantity or unit_price.
    """
    df = df.copy()
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    # Discount is optional; default to 0 when absent
    df["discount"] = df["discount"].fillna(0.0)

    before = len(df)
    df = df.dropna(subset=["quantity", "unit_price"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with unparseable quantity/unit_price", dropped)

    # Remove physically invalid negative values
    invalid_mask = (df["quantity"] < 0) | (df["unit_price"] < 0)
    if invalid_mask.any():
        logger.warning("Dropped %d rows with negative quantity/unit_price", invalid_mask.sum())
        df = df.loc[~invalid_mask]

    df["quantity"] = df["quantity"].astype(int)
    return df


def clean_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date columns; keep NaT for missing delivery dates (in-transit orders)."""
    df = df.copy()
    df[DATE_COLUMNS] = df[DATE_COLUMNS].apply(pd.to_datetime, errors="coerce")
    return df


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all object (string) columns."""
    df = df.copy()
    str_cols = df.select_dtypes(include="object").columns
    if len(str_cols) > 0:
        df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows, keeping the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=["order_id"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Removed %d duplicate order_id rows", dropped)
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed columns useful for analysis:
      - total_price: quantity * unit_price * (1 - discount)
      - lead_time_days: delivery_date - order_date in calendar days
    """
    df = df.copy()
    df["total_price"] = (
        df["quantity"] * df["unit_price"] * (1 - df["discount"])
    ).round(2)

    df["lead_time_days"] = (
        df["delivery_date"] - df["order_date"]
    ).dt.days

    return df


def preprocess(filepath: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Load CSV
      2. Validate required columns
      3. Remove duplicates
      4. Clean string columns
      5. Clean numeric columns
      6. Parse date columns
      7. Add derived columns

    Returns a clean DataFrame ready for analysis or model training.
    """
    df = load_data(filepath)
    validate_columns(df)
    # Keep one working copy to avoid repeated full DataFrame copies between steps.
    df = df.copy()
    df = remove_duplicates(df)

    str_cols = df.select_dtypes(include="object").columns
    if len(str_cols) > 0:
        df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df["discount"] = df["discount"].fillna(0.0)

    before = len(df)
    df = df.dropna(subset=["quantity", "unit_price"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with unparseable quantity/unit_price", dropped)

    invalid_mask = (df["quantity"] < 0) | (df["unit_price"] < 0)
    if invalid_mask.any():
        logger.warning("Dropped %d rows with negative quantity/unit_price", invalid_mask.sum())
        df = df.loc[~invalid_mask]

    df["quantity"] = df["quantity"].astype(int)

    df[DATE_COLUMNS] = df[DATE_COLUMNS].apply(pd.to_datetime, errors="coerce")
    df = add_derived_columns(df)
    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    clean_df = preprocess()
    print(clean_df.head())
    print(f"\nShape: {clean_df.shape}")
    print(f"Columns: {list(clean_df.columns)}")
    print(f"\nNull counts:\n{clean_df.isnull().sum()}")
