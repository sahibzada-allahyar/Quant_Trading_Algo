"""Simple GBM Monte-Carlo simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd


def gbm_paths(
    s0: float, mu: float, sigma: float, days: int, n_paths: int = 1000
) -> pd.DataFrame:
    """Generate Geometric Brownian Motion price paths.
    
    :param s0: Initial price.
    :param mu: Drift parameter (annualized).
    :param sigma: Volatility parameter (annualized).
    :param days: Number of trading days to simulate.
    :param n_paths: Number of simulation paths.
    :return: DataFrame with price paths indexed by business days.
    """
    dt = 1 / 252
    shocks = np.random.normal(mu * dt, sigma * np.sqrt(dt), size=(days, n_paths))
    price = s0 * np.exp(shocks.cumsum(axis=0))
    idx = pd.date_range("today", periods=days, freq="B")
    return pd.DataFrame(price, index=idx) 