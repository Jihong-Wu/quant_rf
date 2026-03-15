from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def relative_strength_index(df: pd.DataFrame, n: int) -> pd.DataFrame:
    data = df.copy()
    up_move = data["max"].diff()
    down_move = -data["min"].diff()

    pos = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=data.index,
    )
    neg = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=data.index,
    )

    pos_di = pos.ewm(span=n, min_periods=n, adjust=False).mean()
    neg_di = neg.ewm(span=n, min_periods=n, adjust=False).mean()
    data[f"RSI_{n}"] = _safe_divide(pos_di, pos_di + neg_di)
    return data


def average_true_range(df: pd.DataFrame, n: int) -> pd.DataFrame:
    data = df.copy()
    prev_max = data["max"].shift(1)
    prev_min = data["min"].shift(1)
    tr = np.maximum(data["max"], prev_min) - np.minimum(data["min"], prev_max)
    data[f"ATR_{n}"] = pd.Series(tr, index=data.index).ewm(
        span=n, min_periods=n, adjust=False
    ).mean()
    return data


def stochastic_oscillator(df: pd.DataFrame, n: int) -> pd.DataFrame:
    data = df.copy()
    k = _safe_divide(data["Average"] - data["min"], data["max"] - data["min"])
    data[f"S_D_{n}"] = k.rolling(window=n, min_periods=n).mean()
    return data


def accumulation_distribution(df: pd.DataFrame, n: int) -> pd.DataFrame:
    data = df.copy()
    ad = _safe_divide(
        (2 * data["Endprice"] - data["max"] - data["min"]) * data["Volume"],
        data["max"] - data["min"],
    )
    roc = _safe_divide(ad.diff(n - 1), ad.shift(n - 1))
    data[f"Acc{n}"] = roc
    return data


def momentum(df: pd.DataFrame, n: int) -> pd.DataFrame:
    data = df.copy()
    data[f"Mom{n}"] = data["Average"].diff(n)
    return data


def rate_of_change(df: pd.DataFrame, n: int) -> pd.DataFrame:
    data = df.copy()
    data[f"ROC{n}"] = _safe_divide(data["Average"].diff(n), data["Average"].shift(n))
    return data


def on_balance_volume(df: pd.DataFrame, n: int) -> pd.DataFrame:
    data = df.copy()
    direction = np.sign(data["Endprice"].diff()).fillna(0.0)
    obv = (direction * data["Volume"]).cumsum()
    data[f"OBV_MA{n}"] = obv.rolling(window=n, min_periods=n).mean()
    return data


def commodity_channel_index(df: pd.DataFrame, n: int) -> pd.DataFrame:
    data = df.copy()
    typical_price = (data["max"] + data["min"] + data["Average"]) / 3
    sma = typical_price.rolling(window=n, min_periods=n).mean()
    dev = typical_price.rolling(window=n, min_periods=n).std()
    data[f"CCI{n}"] = _safe_divide(typical_price - sma, dev)
    return data


def ease_of_movement(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    movement = data["max"].diff(1) + data["min"].diff(1)
    box_ratio = _safe_divide(data["max"] - data["min"], 2 * data["Volume"])
    data["EOM"] = movement * box_ratio
    return data


def trix(df: pd.DataFrame, n: int) -> pd.DataFrame:
    data = df.copy()
    ex1 = data["Average"].ewm(span=n, adjust=False).mean()
    ex2 = ex1.ewm(span=n, adjust=False).mean()
    ex3 = ex2.ewm(span=n, adjust=False).mean()
    data[f"TRIX{n}"] = ex3.pct_change() * 100
    return data


def macd(df: pd.DataFrame, n_fast: int, n_slow: int, signal_span: int = 7) -> pd.DataFrame:
    data = df.copy()
    ema_fast = data["Average"].ewm(span=n_fast, adjust=False).mean()
    ema_slow = data["Average"].ewm(span=n_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_span, adjust=False).mean()
    data[f"MACD_{n_fast}_{n_slow}"] = macd
    data[f"MACDsign_{n_fast}_{n_slow}"] = macd_signal
    data[f"MACDdiff_{n_fast}_{n_slow}"] = macd - macd_signal
    return data


def count_up_trends(group: pd.DataFrame) -> int:
    return int(group["Spot"].diff().gt(0).sum())


def count_down_trends(group: pd.DataFrame) -> int:
    return int(group["Spot"].diff().lt(0).sum())
