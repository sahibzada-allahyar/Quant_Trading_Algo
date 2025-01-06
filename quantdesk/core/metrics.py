"""Performance metrics with deflated Sharpe ratio."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False

from quantdesk.utils.logging import get_logger

log = get_logger(__name__)

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    return np.sqrt(TRADING_DAYS) * excess.mean() / excess.std(ddof=1)


def deflated_sharpe(
    sharpes: np.ndarray, n_trials: int, n_obs: int, skew: float, kurt: float
) -> np.ndarray:
    """Implements Bailey & Lopez‑de‑Prado (2020)."""
    sr = sharpes
    sr_hat = sr * np.sqrt((n_obs - 1) / (n_obs - 2))
    z = sr_hat * np.sqrt(n_obs - 1)
    delta = 0.5 * (skew * sr_hat + ((kurt - 3) / 4) * sr_hat**2)
    p = 1.0 - norm.cdf(z + delta)
    deflated = sr_hat - norm.ppf(1.0 - p / n_trials) / np.sqrt(n_obs - 1)
    return deflated


def tearsheet(returns: pd.Series) -> dict[str, float]:
    sr = sharpe_ratio(returns)
    df = pd.DataFrame({"r": returns})
    skew = df["r"].skew()
    kurt = df["r"].kurtosis()
    dsr = float(deflated_sharpe(np.array([sr]), 1, len(df), skew, kurt))
    cagr = (1 + returns).prod() ** (TRADING_DAYS / len(returns)) - 1
    mdd = (df["r"].cumsum().expanding().max() - df["r"].cumsum()).max()
    return {"sharpe": sr, "deflated_sharpe": dsr, "cagr": cagr, "max_drawdown": mdd}


def quantstats_report(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    output_file: str | Path | None = None,
    title: str = "Strategy Performance",
) -> dict[str, Any] | None:
    """Generate comprehensive QuantStats performance report.
    
    :param returns: Strategy returns series.
    :param benchmark: Benchmark returns series (optional).
    :param output_file: Path to save HTML report (optional).
    :param title: Report title.
    :return: Dictionary of metrics or None if QuantStats not available.
    """
    if not HAS_QUANTSTATS:
        log.warning("QuantStats not installed - skipping report generation")
        return None
    
    # Configure QuantStats
    qs.extend_pandas()
    
    # Generate metrics
    metrics = {
        "total_return": qs.stats.comp(returns),
        "cagr": qs.stats.cagr(returns),
        "volatility": qs.stats.volatility(returns),
        "sharpe": qs.stats.sharpe(returns),
        "sortino": qs.stats.sortino(returns),
        "max_drawdown": qs.stats.max_drawdown(returns),
        "calmar": qs.stats.calmar(returns),
        "skew": qs.stats.skew(returns),
        "kurtosis": qs.stats.kurtosis(returns),
        "tail_ratio": qs.stats.tail_ratio(returns),
        "common_sense_ratio": qs.stats.common_sense_ratio(returns),
        "value_at_risk": qs.stats.value_at_risk(returns),
        "conditional_value_at_risk": qs.stats.conditional_value_at_risk(returns),
    }
    
    # Add benchmark-relative metrics if provided
    if benchmark is not None:
        metrics.update({
            "alpha": qs.stats.alpha(returns, benchmark),
            "beta": qs.stats.beta(returns, benchmark),
            "information_ratio": qs.stats.information_ratio(returns, benchmark),
            "treynor_ratio": qs.stats.treynor_ratio(returns, benchmark),
        })
    
    # Generate HTML report if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if benchmark is not None:
            qs.reports.html(returns, benchmark, output=str(output_path), title=title)
        else:
            qs.reports.html(returns, output=str(output_path), title=title)
        
        log.info("quantstats.report_generated", path=str(output_path))
    
    return metrics


def quantstats_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    live_start_date: str | None = None,
) -> None:
    """Display QuantStats tearsheet in console/notebook.
    
    :param returns: Strategy returns series.
    :param benchmark: Benchmark returns series (optional).
    :param live_start_date: Date when live trading started (optional).
    """
    if not HAS_QUANTSTATS:
        log.warning("QuantStats not installed - skipping tearsheet")
        return
    
    qs.extend_pandas()
    
    if benchmark is not None:
        qs.reports.full(returns, benchmark, live_start_date=live_start_date)
    else:
        qs.reports.basic(returns, live_start_date=live_start_date)


def combined_metrics(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    include_quantstats: bool = True,
) -> dict[str, Any]:
    """Combine custom metrics with QuantStats metrics.
    
    :param returns: Strategy returns series.
    :param benchmark: Benchmark returns series (optional).
    :param include_quantstats: Whether to include QuantStats metrics.
    :return: Combined metrics dictionary.
    """
    # Start with existing custom metrics
    metrics = tearsheet(returns)
    
    # Add QuantStats metrics if available and requested
    if include_quantstats and HAS_QUANTSTATS:
        qs_metrics = quantstats_report(returns, benchmark)
        if qs_metrics:
            metrics.update({"quantstats": qs_metrics})
    
    return metrics 