"""Unified data access across Yahoo, Alpaca, Polygon, Binance‑ccxt."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Literal

import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import ccxt

from quantdesk.utils.logging import get_logger
from quantdesk.utils.env import SETTINGS

log = get_logger(__name__)

Provider = Literal["yahoo", "alpaca", "binance"]
Freq = Literal["1m", "5m", "1h", "1d"]

_alpaca_client = StockHistoricalDataClient(
    api_key=SETTINGS.alpaca_key, secret_key=SETTINGS.alpaca_secret
)

_binance = ccxt.binance()
if SETTINGS.binance_key and SETTINGS.binance_secret:
    _binance = ccxt.binance({
        "apiKey": SETTINGS.binance_key,
        "secret": SETTINGS.binance_secret,
        "enableRateLimit": True,
    })


def _load_yahoo(symbol: str, start: datetime, end: datetime, freq: Freq) -> pd.DataFrame:
    interval_map: dict[Freq, str] = {"1m": "1m", "5m": "5m", "1h": "60m", "1d": "1d"}
    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval=interval_map[freq],
        progress=False,
        auto_adjust=False,
        threads=True,
    )
    return df.rename(columns=str.lower)  # std: open, high, low, close, adj close, volume


def _load_alpaca(symbol: str, start: datetime, end: datetime, freq: Freq) -> pd.DataFrame:
    tf_map: dict[Freq, TimeFrame] = {
        "1m": TimeFrame.Minute,
        "5m": TimeFrame(5, TimeFrame.Unit.Minute),
        "1h": TimeFrame.Hour,
        "1d": TimeFrame.Day,
    }
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf_map[freq], start=start, end=end)
    bars = _alpaca_client.get_stock_bars(req).df
    return bars.xs(symbol, level=0).rename(columns=str.lower)


async def _load_binance(symbol: str, start: datetime, end: datetime, freq: Freq) -> pd.DataFrame:
    limit = 1000
    timeframe = freq
    since = int(start.timestamp() * 1000)
    out: list[list[Any]] = []

    while since < end.timestamp() * 1000:
        batch = await asyncio.to_thread(
            _binance.fetch_ohlcv, symbol, timeframe=timeframe, since=since, limit=limit
        )
        if not batch:
            break
        since = batch[-1][0] + 1
        out.extend(batch)

    df = pd.DataFrame(
        out, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def load(
    symbol: str,
    start: datetime,
    end: datetime,
    freq: Freq = "1d",
    provider: Provider = "yahoo",
) -> pd.DataFrame:
    """Public entry point."""
    log.info("load.start", symbol=symbol, provider=provider, freq=freq)
    if provider == "yahoo":
        return _load_yahoo(symbol, start, end, freq)
    if provider == "alpaca":
        return _load_alpaca(symbol, start, end, freq)
    if provider == "binance":
        return asyncio.run(_load_binance(symbol, start, end, freq))
    raise ValueError(f"Unsupported provider {provider}") 