#!/usr/bin/env python
"""Stream Alpaca market data and save to parquet files."""
from __future__ import annotations

import asyncio
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import pandas as pd
from alpaca.data.live import StockDataStream
from alpaca.data.models import Bar, Quote, Trade

from quantdesk.utils.env import SETTINGS
from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


class AlpacaDataIngester:
    """Real-time Alpaca data ingestion with parquet storage."""

    def __init__(self, output_dir: str = "data/cache/alpaca") -> None:
        """Initialize the data ingester.
        
        :param output_dir: Directory to save parquet files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stream = StockDataStream(
            api_key=SETTINGS.alpaca_key,
            secret_key=SETTINGS.alpaca_secret,
        )
        
        self.data_buffer: dict[str, list[dict[str, Any]]] = {
            "bars": [],
            "quotes": [],
            "trades": [],
        }
        self.buffer_size = 1000
        self.running = False

    async def handle_bar(self, bar: Bar) -> None:
        """Handle incoming bar data."""
        data = {
            "timestamp": bar.timestamp,
            "symbol": bar.symbol,
            "open": float(bar.open),
            "high": float(bar.high),
            "low": float(bar.low),
            "close": float(bar.close),
            "volume": int(bar.volume),
            "trade_count": int(bar.trade_count) if bar.trade_count else 0,
            "vwap": float(bar.vwap) if bar.vwap else None,
        }
        
        self.data_buffer["bars"].append(data)
        log.debug("bar.received", symbol=bar.symbol, timestamp=bar.timestamp)
        
        if len(self.data_buffer["bars"]) >= self.buffer_size:
            await self._flush_buffer("bars")

    async def handle_quote(self, quote: Quote) -> None:
        """Handle incoming quote data."""
        data = {
            "timestamp": quote.timestamp,
            "symbol": quote.symbol,
            "bid_price": float(quote.bid_price) if quote.bid_price else None,
            "bid_size": int(quote.bid_size) if quote.bid_size else None,
            "ask_price": float(quote.ask_price) if quote.ask_price else None,
            "ask_size": int(quote.ask_size) if quote.ask_size else None,
        }
        
        self.data_buffer["quotes"].append(data)
        
        if len(self.data_buffer["quotes"]) >= self.buffer_size:
            await self._flush_buffer("quotes")

    async def handle_trade(self, trade: Trade) -> None:
        """Handle incoming trade data."""
        data = {
            "timestamp": trade.timestamp,
            "symbol": trade.symbol,
            "price": float(trade.price),
            "size": int(trade.size),
        }
        
        self.data_buffer["trades"].append(data)
        
        if len(self.data_buffer["trades"]) >= self.buffer_size:
            await self._flush_buffer("trades")

    async def _flush_buffer(self, data_type: str) -> None:
        """Flush buffer to parquet file."""
        if not self.data_buffer[data_type]:
            return
            
        df = pd.DataFrame(self.data_buffer[data_type])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        
        # Create filename with current date
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        filename = self.output_dir / f"{data_type}_{date_str}.parquet"
        
        # Append to existing file or create new one
        if filename.exists():
            existing_df = pd.read_parquet(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_parquet(filename, index=False)
        
        log.info(
            "data.flushed",
            data_type=data_type,
            records=len(self.data_buffer[data_type]),
            filename=str(filename),
        )
        
        # Clear buffer
        self.data_buffer[data_type] = []

    async def start_streaming(self, symbols: list[str]) -> None:
        """Start streaming data for given symbols."""
        log.info("ingester.starting", symbols=symbols)
        
        # Register handlers
        self.stream.subscribe_bars(self.handle_bar, *symbols)
        self.stream.subscribe_quotes(self.handle_quote, *symbols)
        self.stream.subscribe_trades(self.handle_trade, *symbols)
        
        self.running = True
        
        # Start the stream
        await self.stream._run_forever()

    async def stop_streaming(self) -> None:
        """Stop streaming and flush remaining data."""
        log.info("ingester.stopping")
        self.running = False
        
        # Flush remaining data
        for data_type in self.data_buffer:
            await self._flush_buffer(data_type)
        
        await self.stream.stop_ws()
        log.info("ingester.stopped")


# Global ingester instance for signal handling
ingester: AlpacaDataIngester | None = None


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    log.info("signal.received", signal=signum)
    if ingester:
        asyncio.create_task(ingester.stop_streaming())


@click.command()
@click.option(
    "--symbols",
    default="SPY,QQQ,AAPL,MSFT,GOOGL",
    help="Comma-separated list of symbols to stream",
)
@click.option(
    "--output-dir",
    default="data/cache/alpaca",
    help="Output directory for parquet files",
)
def main(symbols: str, output_dir: str) -> None:
    """Stream Alpaca market data to parquet files."""
    global ingester
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse symbols
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    # Create ingester
    ingester = AlpacaDataIngester(output_dir)
    
    # Start streaming
    try:
        asyncio.run(ingester.start_streaming(symbol_list))
    except KeyboardInterrupt:
        log.info("ingester.interrupted")
    except Exception:
        log.exception("ingester.error")
        sys.exit(1)


if __name__ == "__main__":
    main() 