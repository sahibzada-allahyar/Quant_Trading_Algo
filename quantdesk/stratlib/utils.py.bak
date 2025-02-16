"""Reusable helpers such as ATR stops & sizing."""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr_stop(price: pd.Series, atr: pd.Series, mult: float = 2.0) -> pd.Series:
    return price - mult * atr


def position_size(
    capital: float, vol_target: float, vol_est: float, price: float
) -> int:
    dollar_qty = capital * (vol_target / max(vol_est, 1e-8))
    return int(dollar_qty // price)


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range calculated purely with pandas/NumPy.

    :param high: High price series.
    :param low: Low price series.
    :param close: Close price series.
    :param period: Rolling window length.
    :return: ATR series.
    """
    prev_close = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
    return tr.rolling(period, min_periods=1).mean()


def rolling_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Annualised rolling volatility (σ) based on daily returns.

    :param returns: Daily return series.
    :param window: Look-back window length.
    :return: Annualised rolling volatility.
    """
    return returns.rolling(window).std(ddof=0) * np.sqrt(252) 