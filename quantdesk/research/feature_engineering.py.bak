"""TA features and PCA factor reduction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta  # ta‑lib wrapper

from sklearn.decomposition import PCA


def ta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    out["macd"] = ta.trend.macd_diff(df["close"])
    out["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
    return out.fillna(method="bfill")


def pca_factors(features: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
    pca = PCA(n_components=n_components, whiten=True, svd_solver="full")
    comps = pca.fit_transform(features)
    cols = [f"pca_{i}" for i in range(comps.shape[1])]
    return pd.DataFrame(comps, index=features.index, columns=cols) 