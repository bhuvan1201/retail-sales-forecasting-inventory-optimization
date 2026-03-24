from __future__ import annotations

import pandas as pd


GROUP_COLS = ["item_id", "store_id"]


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(GROUP_COLS + ["date"])

    grouped_price = df.groupby(GROUP_COLS, observed=True)["sell_price"]

    df["price_lag_1"] = grouped_price.shift(1)
    df["price_lag_7"] = grouped_price.shift(7)

    df["price_change_1"] = df["sell_price"] - df["price_lag_1"]
    df["price_change_7"] = df["sell_price"] - df["price_lag_7"]

    df["price_change_pct_1"] = (df["sell_price"] - df["price_lag_1"]) / df["price_lag_1"]
    df["price_change_pct_7"] = (df["sell_price"] - df["price_lag_7"]) / df["price_lag_7"]

    df["rolling_price_mean_7"] = (
        grouped_price.shift(1)
        .groupby([df["item_id"], df["store_id"]], observed=True)
        .rolling(window=7, min_periods=7)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["rolling_price_mean_28"] = (
        grouped_price.shift(1)
        .groupby([df["item_id"], df["store_id"]], observed=True)
        .rolling(window=28, min_periods=28)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["price_to_avg_ratio_7"] = df["sell_price"] / df["rolling_price_mean_7"]
    df["price_to_avg_ratio_28"] = df["sell_price"] / df["rolling_price_mean_28"]

    return df