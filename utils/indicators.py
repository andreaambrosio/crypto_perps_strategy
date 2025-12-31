"""
Technical indicators for the trading strategy
"""
import numpy as np
import pandas as pd
from typing import Tuple


def ema(data: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()


def sma(data: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return data.rolling(window=period).mean()


def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD indicator"""
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    middle = sma(data, period)
    std = data.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


def momentum(data: pd.Series, period: int = 10) -> pd.Series:
    """Price momentum (rate of change)"""
    return (data - data.shift(period)) / data.shift(period)


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume Simple Moving Average"""
    return sma(volume, period)


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Supertrend indicator
    Returns: (supertrend_line, direction) where direction is 1 for uptrend, -1 for downtrend
    """
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = hl2 + (multiplier * atr_val)
    lower_band = hl2 - (multiplier * atr_val)

    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(index=close.index, dtype=float)

    for i in range(period, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1] if i > period else 1

            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            elif direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]

    return supertrend, direction


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index - measures trend strength"""
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = atr(high, low, close, 1) * period  # True range sum approximation

    plus_di = 100 * ema(plus_dm, period) / atr(high, low, close, period)
    minus_di = 100 * ema(minus_dm, period) / atr(high, low, close, period)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = ema(dx, period)

    return adx_val


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()
