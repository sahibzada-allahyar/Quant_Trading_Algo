"""GARCH(1,1) volatility forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model  # optional dependency, pinned in conda/poetry


def garch_vol(returns: pd.Series, horizon: int = 1) -> float:
    """Fit GARCH(1,1) model and forecast volatility.
    
    :param returns: Return series.
    :param horizon: Forecast horizon in periods.
    :return: Volatility forecast.
    """
    model = arch_model(returns * 100, p=1, q=1)  # %
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=horizon)
    return float(np.sqrt(forecast.variance.iloc[-1, -1]) / 100) 