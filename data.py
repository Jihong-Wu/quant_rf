from __future__ import annotations

from datetime import time

import pandas as pd

from . import indicators as ti


def load_tick_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Date", "TimeOfDay", "Spot"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["TimeOfDay"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "timestamp"]).reset_index(drop=True)
    return df


def build_bars(
    df: pd.DataFrame,
    slice_minutes: int = 10,
    session_start: str = "10:00",
    session_end: str = "15:30",
) -> pd.DataFrame:
    start_h, start_m = map(int, session_start.split(":"))
    end_h, end_m = map(int, session_end.split(":"))

    filtered = df[
        (df["timestamp"].dt.time >= time(start_h, start_m))
        & (df["timestamp"].dt.time <= time(end_h, end_m))
    ].copy()
    filtered = filtered.set_index("timestamp")

    rule = f"{slice_minutes}min"
    bars = filtered["Spot"].resample(rule).agg(["max", "min"])
    bars["Average"] = filtered["Spot"].resample(rule).mean()
    bars["Endprice"] = filtered["Spot"].resample(rule).last()
    bars["Range"] = bars["max"] - bars["min"]
    bars["Drift"] = (
        filtered["Spot"].resample(rule).last()
        - filtered["Spot"].resample(rule).first()
    )
    bars["Volume"] = filtered["Spot"].resample(rule).size().clip(lower=1)
    bars["Std"] = filtered["Spot"].resample(rule).std().fillna(0)
    bars["S"] = bars["Range"] / bars["Volume"].pow(0.5)

    bars["TradeCount"] = filtered["Spot"].resample(rule).count()
    bars["TradeCount_Fast"] = bars["TradeCount"].rolling(window=3, min_periods=1).mean()
    bars["TradeCount_Slow"] = bars["TradeCount"].rolling(window=10, min_periods=1).mean()
    bars["TradeCount_Diff"] = bars["TradeCount_Fast"] - bars["TradeCount_Slow"]

    grouped = filtered.resample(rule)
    bars["UpTrends"] = grouped.apply(ti.count_up_trends).fillna(0)
    bars["DownTrends"] = grouped.apply(ti.count_down_trends).fillna(0)
    bars["TrendDiff"] = bars["UpTrends"] - bars["DownTrends"]
    bars = bars.drop(columns=["UpTrends", "DownTrends"])
    return bars
