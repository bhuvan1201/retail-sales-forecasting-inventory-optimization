from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MELTED_SALES_FILE = PROJECT_ROOT / "data" / "interim" / "melted_sales.parquet"
CALENDAR_FILE = PROJECT_ROOT / "data" / "raw" / "calendar.csv"
SELL_PRICES_FILE = PROJECT_ROOT / "data" / "raw" / "sell_prices.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "joined_base.parquet"


def optimize_calendar_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    category_cols = [
        "weekday",
        "d",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]

    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    int_cols = ["wm_yr_wk", "wday", "month", "year", "snap_CA", "snap_TX", "snap_WI"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def optimize_prices_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["store_id", "item_id"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if "wm_yr_wk" in df.columns:
        df["wm_yr_wk"] = pd.to_numeric(df["wm_yr_wk"], downcast="integer")

    if "sell_price" in df.columns:
        df["sell_price"] = pd.to_numeric(df["sell_price"], downcast="float")

    return df


def main() -> None:
    for file_path in [MELTED_SALES_FILE, CALENDAR_FILE, SELL_PRICES_FILE]:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing input file: {file_path}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Loading melted sales data...")
    sales_df = pd.read_parquet(MELTED_SALES_FILE)
    print(f"sales_df shape: {sales_df.shape}")

    print("\nLoading calendar data...")
    calendar_df = pd.read_csv(CALENDAR_FILE)
    calendar_df = optimize_calendar_dtypes(calendar_df)
    print(f"calendar_df shape: {calendar_df.shape}")

    print("\nLoading sell prices data...")
    prices_df = pd.read_csv(SELL_PRICES_FILE)
    prices_df = optimize_prices_dtypes(prices_df)
    print(f"prices_df shape: {prices_df.shape}")

    print("\nMerging sales with calendar on 'd'...")
    merged_df = sales_df.merge(
        calendar_df,
        on="d",
        how="left",
        validate="many_to_one"
    )
    print(f"Shape after calendar merge: {merged_df.shape}")

    print("\nMerging with sell prices on ['store_id', 'item_id', 'wm_yr_wk']...")
    merged_df = merged_df.merge(
        prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one"
    )
    print(f"Final joined shape: {merged_df.shape}")

    print("\nMissing sell_price values:")
    print(merged_df["sell_price"].isna().sum())

    print("\nSample rows:")
    print(merged_df.head())

    print("\nColumn list:")
    print(merged_df.columns.tolist())

    print(f"\nSaving joined base table to {OUTPUT_FILE} ...")
    merged_df.to_parquet(OUTPUT_FILE, index=False)

    print("Done.")


if __name__ == "__main__":
    main()