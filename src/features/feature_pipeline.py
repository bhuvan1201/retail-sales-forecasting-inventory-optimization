from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.features.build_lag_features import add_lag_features
from src.features.build_rolling_features import add_rolling_features
from src.features.build_calendar_features import add_calendar_features
from src.features.build_price_features import add_price_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "interim" / "joined_base.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "feature_store.parquet"


def optimize_final_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "weekday",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "d",
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading joined base data from: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)

    print(f"Initial shape: {df.shape}")

    print("Converting date column to datetime...")
    df["date"] = pd.to_datetime(df["date"])

    print("Sorting data...")
    df = df.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    print("Adding calendar features...")
    df = add_calendar_features(df)
    print(f"Shape after calendar features: {df.shape}")

    print("Adding lag features...")
    df = add_lag_features(df, target_col="sales", lags=[7, 14, 28, 56])
    print(f"Shape after lag features: {df.shape}")

    print("Adding rolling features...")
    df = add_rolling_features(df, target_col="sales", windows=[7, 28, 56])
    print(f"Shape after rolling features: {df.shape}")

    print("Adding price features...")
    df = add_price_features(df)
    print(f"Shape after price features: {df.shape}")

    print("Optimizing dtypes...")
    df = optimize_final_dtypes(df)

    print("Dropping rows with missing critical lag features...")
    required_cols = ["lag_7", "lag_28", "rolling_mean_7", "rolling_mean_28"]
    before_drop = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    after_drop = len(df)

    print(f"Rows before drop: {before_drop}")
    print(f"Rows after drop:  {after_drop}")
    print(f"Rows removed:     {before_drop - after_drop}")

    print("\nFinal columns:")
    print(df.columns.tolist())

    print("\nSample rows:")
    print(df.head())

    print(f"\nSaving feature store to: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)

    print("Feature pipeline completed successfully.")


if __name__ == "__main__":
    main()