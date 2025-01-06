"""1-dimensional Kalman filter for time-varying hedge ratio."""
from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_filter(y: pd.Series, x: pd.Series) -> pd.Series:
    """Apply 1-dimensional Kalman filter to estimate time-varying hedge ratio.
    
    :param y: Dependent variable series.
    :param x: Independent variable series.
    :return: Time-varying beta estimates.
    """
    n = len(y)
    delta = 1e-5
    vt, rt, qt, at, bt = (np.zeros(n) for _ in range(5))
    Pt = np.zeros(n)
    beta = 0.0
    P = 1.0

    for t in range(n):
        at[t] = beta
        rt[t] = P + delta
        vt[t] = y.iloc[t] - at[t] * x.iloc[t]
        qt[t] = rt[t] * x.iloc[t] ** 2 + 1.0  # obs noise σ² = 1
        bt[t] = at[t] + rt[t] * x.iloc[t] * vt[t] / qt[t]
        Pt = rt[t] - rt[t] ** 2 * x.iloc[t] ** 2 / qt[t]
        beta = bt[t]
        P = Pt
    return pd.Series(bt, index=y.index, name="kalman_beta") 