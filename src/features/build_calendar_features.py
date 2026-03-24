from __future__ import annotations

import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype("int16")
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    df["is_weekend"] = np.where(df["day_of_week"].isin([5, 6]), 1, 0)

    df["has_event_1"] = df["event_name_1"].notna().astype("int8")
    df["has_event_2"] = df["event_name_2"].notna().astype("int8")
    df["has_any_event"] = ((df["has_event_1"] + df["has_event_2"]) > 0).astype("int8")

    if "snap_CA" in df.columns:
        df["snap_CA"] = df["snap_CA"].astype("int8")
    if "snap_TX" in df.columns:
        df["snap_TX"] = df["snap_TX"].astype("int8")
    if "snap_WI" in df.columns:
        df["snap_WI"] = df["snap_WI"].astype("int8")

    return df