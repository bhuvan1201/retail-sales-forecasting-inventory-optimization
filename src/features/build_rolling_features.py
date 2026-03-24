from __future__ import annotations

import pandas as pd


GROUP_COLS = ["item_id", "store_id"]


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [7, 28, 56]

    df = df.copy()
    df = df.sort_values(GROUP_COLS + ["date"])

    grouped = df.groupby(GROUP_COLS, observed=True)[target_col]

    for window in windows:
        shifted = grouped.shift(1)

        df[f"rolling_mean_{window}"] = (
            shifted.groupby([df["item_id"], df["store_id"]], observed=True)
            .rolling(window=window, min_periods=window)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

        df[f"rolling_std_{window}"] = (
            shifted.groupby([df["item_id"], df["store_id"]], observed=True)
            .rolling(window=window, min_periods=window)
            .std()
            .reset_index(level=[0, 1], drop=True)
        )

    return df