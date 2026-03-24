from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_csv(file_name: str) -> pd.DataFrame:
    file_path = RAW_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    print(f"Loading {file_path} ...")
    df = pd.read_csv(file_path)
    print(f"Loaded {file_name} with shape: {df.shape}")
    return df


def summarize_dataframe(name: str, df: pd.DataFrame) -> None:
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    print(df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))
    print("\nSample rows:")
    print(df.head(3))


def main() -> None:
    files = {
        "calendar": "calendar.csv",
        "sales_train_validation": "sales_train_validation.csv",
        "sales_train_evaluation": "sales_train_evaluation.csv",
        "sell_prices": "sell_prices.csv",
    }

    loaded_data = {}

    for name, file_name in files.items():
        df = load_csv(file_name)
        loaded_data[name] = df

    for name, df in loaded_data.items():
        summarize_dataframe(name, df)

    print("\nAll raw files loaded successfully.")


if __name__ == "__main__":
    main()