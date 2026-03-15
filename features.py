from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import indicators as ti


def prepare_labels(bars: pd.DataFrame, horizon: int = 1, threshold: float = 0.001) -> pd.Series:
    future_return = bars["Average"].shift(-horizon) - bars["Average"]
    labels = np.where(future_return > threshold, 2, np.where(future_return < -threshold, 0, 1))
    out = pd.Series(labels, index=bars.index, dtype="Int64")
    out.iloc[-horizon:] = pd.NA
    return out


def build_feature_matrix(bars: pd.DataFrame, windows: Iterable[int] = (2, 5, 10)) -> pd.DataFrame:
    data = bars.copy()
    for n in windows:
        data = ti.relative_strength_index(data, n)
        data = ti.average_true_range(data, n)
        data = ti.stochastic_oscillator(data, n)
        data = ti.accumulation_distribution(data, n)
        data = ti.momentum(data, n)
        data = ti.rate_of_change(data, n)
        data = ti.on_balance_volume(data, n)
        data = ti.commodity_channel_index(data, n)
        data = ti.trix(data, n)
        data[f"ema_{n}"] = data["Average"] / data["Average"].ewm(span=n, adjust=False).mean()

    data = ti.ease_of_movement(data)
    data = ti.macd(data, n_fast=5, n_slow=20)
    return data


def clip_outliers(df: pd.DataFrame, lower_q: float = 0.001, upper_q: float = 0.999) -> pd.DataFrame:
    data = df.copy()
    numeric_cols = data.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        lower = data[col].quantile(lower_q)
        upper = data[col].quantile(upper_q)
        data[col] = data[col].clip(lower=lower, upper=upper)
    return data


def make_xy(
    bars: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.001,
    windows: Iterable[int] = (2, 5, 10),
) -> tuple[pd.DataFrame, pd.Series]:
    data = build_feature_matrix(bars, windows=windows)
    data["Prediction"] = prepare_labels(data, horizon=horizon, threshold=threshold)

    clean = data.dropna().copy()
    y = clean["Prediction"].astype(int)
    x = clean.drop(columns=["Prediction"])
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[x.index]
    x = clip_outliers(x)
    return x, y
