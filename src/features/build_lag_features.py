from __future__ import annotations

import pandas as pd


GROUP_COLS = ["item_id", "store_id"]


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    if lags is None:
        lags = [7, 14, 28, 56]

    df = df.copy()
    df = df.sort_values(GROUP_COLS + ["date"])

    grouped = df.groupby(GROUP_COLS, observed=True)[target_col]

    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag)

    return df