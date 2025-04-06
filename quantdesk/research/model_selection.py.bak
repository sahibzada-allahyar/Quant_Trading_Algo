"""Walk‑forward cross‑validation & Bayesian optimisation."""
from __future__ import annotations

import itertools
from datetime import timedelta
from typing import Callable

import mlflow
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer

from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


def walk_forward_splits(
    df: pd.DataFrame, train_size: int, test_size: int
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Return list of index splits (train_idx, test_idx)."""
    splits = []
    start = 0
    while start + train_size + test_size <= len(df):
        train_idx = df.index[start : start + train_size]
        test_idx = df.index[start + train_size : start + train_size + test_size]
        splits.append((train_idx, test_idx))
        start += test_size
    return splits


def bayes_opt(
    obj: Callable[[dict[str, float]], float],
    space: list[Real | Integer],
    n_calls: int = 30,
) -> dict[str, float]:
    res = gp_minimize(obj, space, n_calls=n_calls, random_state=42)
    return {f"x{i}": v for i, v in enumerate(res.x)} 