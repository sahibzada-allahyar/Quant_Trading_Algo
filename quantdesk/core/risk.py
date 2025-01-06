"""Position sizing and risk calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


def kelly_fraction(edge: float, variance: float) -> float:
    """Kelly % for given edge and variance."""
    k = edge / variance
    return max(0.0, min(k, 1.0))  # cap at 100 %


def cvar(returns: pd.Series, level: float = 0.95) -> float:
    """Conditional Value‑at‑Risk (expected shortfall)."""
    var = returns.quantile(1 - level)
    return returns[returns <= var].mean()


def volatility_target(position_value: float, annual_vol_target: float, vol_est: float) -> float:
    """Target dollar allocation given vol forecast."""
    if vol_est == 0:
        return 0
    target = position_value * (annual_vol_target / vol_est)
    log.debug("risk.vol_target", target=target, vol_est=vol_est)
    return target 