from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "sales_train_validation.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "melted_sales.parquet"

ID_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        df[col] = df[col].astype("category")

    day_columns = [col for col in df.columns if col.startswith("d_")]
    for col in day_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def reshape_sales_to_long(df: pd.DataFrame) -> pd.DataFrame:
    day_columns = [col for col in df.columns if col.startswith("d_")]

    long_df = df.melt(
        id_vars=ID_COLUMNS,
        value_vars=day_columns,
        var_name="d",
        value_name="sales"
    )

    long_df["d"] = long_df["d"].astype("category")
    long_df["sales"] = pd.to_numeric(long_df["sales"], downcast="integer")

    return long_df


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {RAW_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {RAW_FILE} ...")
    sales_df = pd.read_csv(RAW_FILE)

    print(f"Original shape: {sales_df.shape}")

    print("Optimizing dtypes...")
    sales_df = optimize_dtypes(sales_df)

    print("Reshaping wide sales data to long format...")
    melted_df = reshape_sales_to_long(sales_df)

    print(f"Reshaped dataframe shape: {melted_df.shape}")
    print("\nSample rows:")
    print(melted_df.head())

    print("\nData types:")
    print(melted_df.dtypes)

    print(f"\nSaving to {OUTPUT_FILE} ...")
    melted_df.to_parquet(OUTPUT_FILE, index=False)

    print("Done.")


if __name__ == "__main__":
    main()