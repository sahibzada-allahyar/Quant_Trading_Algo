"""Statistical hypothesis tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


def ols(y: pd.Series, x: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    return model


def hurst(ts: pd.Series) -> float:
    lags = range(2, 100)
    log_rs = np.log([np.sqrt(((ts.diff(lag).dropna()) ** 2).mean()) for lag in lags])
    log_lags = np.log(lags)
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    return slope * 2.0


def adf(ts: pd.Series) -> float:
    return adfuller(ts, maxlag=1, regression="c")[1]  # p‑value 