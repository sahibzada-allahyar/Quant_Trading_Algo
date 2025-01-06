#!/usr/bin/env python
"""Stream Binance market data and save to parquet files."""
from __future__ import annotations

import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed

from quantdesk.utils.env import SETTINGS
from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


class BinanceDataIngester:
    """Real-time Binance data ingestion with parquet storage."""

    def __init__(self, output_dir: str = "data/cache/binance") -> None:
        """Initialize the data ingester.
        
        :param output_dir: Directory to save parquet files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_url = "wss://stream.binance.com:9443/ws"
        self.testnet_url = "wss://testnet.binance.vision/ws"
        
        # Use testnet if configured
        self.ws_url = self.testnet_url if SETTINGS.binance_key else self.base_url
        
        self.data_buffer: dict[str, list[dict[str, Any]]] = {
            "klines": [],
            "trades": [],
            "depth": [],
            "ticker": [],
        }
        self.buffer_size = 1000
        self.running = False
        self.websocket = None

    async def handle_kline(self, data: dict[str, Any]) -> None:
        """Handle kline (candlestick) data."""
        kline = data["k"]
        record = {
            "timestamp": pd.to_datetime(kline["t"], unit="ms", utc=True),
            "symbol": kline["s"],
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"]),
            "quote_volume": float(kline["q"]),
            "trade_count": int(kline["n"]),
            "is_closed": kline["x"],  # Whether this kline is closed
        }
        
        self.data_buffer["klines"].append(record)
        log.debug("kline.received", symbol=kline["s"], timestamp=record["timestamp"])
        
        if len(self.data_buffer["klines"]) >= self.buffer_size:
            await self._flush_buffer("klines")

    async def handle_trade(self, data: dict[str, Any]) -> None:
        """Handle trade data."""
        record = {
            "timestamp": pd.to_datetime(data["T"], unit="ms", utc=True),
            "symbol": data["s"],
            "price": float(data["p"]),
            "quantity": float(data["q"]),
            "trade_id": int(data["t"]),
            "buyer_maker": data["m"],  # True if buyer is market maker
        }
        
        self.data_buffer["trades"].append(record)
        
        if len(self.data_buffer["trades"]) >= self.buffer_size:
            await self._flush_buffer("trades")

    async def handle_depth(self, data: dict[str, Any]) -> None:
        """Handle order book depth data."""
        record = {
            "timestamp": pd.to_datetime(data["E"], unit="ms", utc=True),
            "symbol": data["s"],
            "first_update_id": int(data["U"]),
            "final_update_id": int(data["u"]),
            "bids": json.dumps(data["b"][:10]),  # Top 10 bids
            "asks": json.dumps(data["a"][:10]),  # Top 10 asks
        }
        
        self.data_buffer["depth"].append(record)
        
        if len(self.data_buffer["depth"]) >= self.buffer_size:
            await self._flush_buffer("depth")

    async def handle_ticker(self, data: dict[str, Any]) -> None:
        """Handle 24hr ticker statistics."""
        record = {
            "timestamp": pd.to_datetime(data["E"], unit="ms", utc=True),
            "symbol": data["s"],
            "price_change": float(data["p"]),
            "price_change_percent": float(data["P"]),
            "weighted_avg_price": float(data["w"]),
            "last_price": float(data["c"]),
            "last_qty": float(data["Q"]),
            "bid_price": float(data["b"]),
            "bid_qty": float(data["B"]),
            "ask_price": float(data["a"]),
            "ask_qty": float(data["A"]),
            "open_price": float(data["o"]),
            "high_price": float(data["h"]),
            "low_price": float(data["l"]),
            "volume": float(data["v"]),
            "quote_volume": float(data["q"]),
            "open_time": pd.to_datetime(data["O"], unit="ms", utc=True),
            "close_time": pd.to_datetime(data["C"], unit="ms", utc=True),
            "count": int(data["n"]),
        }
        
        self.data_buffer["ticker"].append(record)
        
        if len(self.data_buffer["ticker"]) >= self.buffer_size:
            await self._flush_buffer("ticker")

    async def _flush_buffer(self, data_type: str) -> None:
        """Flush buffer to parquet file."""
        if not self.data_buffer[data_type]:
            return
            
        df = pd.DataFrame(self.data_buffer[data_type])
        
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

    def create_stream_params(self, symbols: list[str]) -> list[str]:
        """Create WebSocket stream parameters."""
        streams = []
        
        for symbol in symbols:
            symbol_lower = symbol.lower()
            # Add different stream types
            streams.extend([
                f"{symbol_lower}@kline_1m",  # 1-minute klines
                f"{symbol_lower}@trade",     # Individual trades
                f"{symbol_lower}@depth20@100ms",  # Order book depth
                f"{symbol_lower}@ticker",    # 24hr ticker
            ])
        
        return streams

    async def start_streaming(self, symbols: list[str]) -> None:
        """Start streaming data for given symbols."""
        streams = self.create_stream_params(symbols)
        stream_names = "/".join(streams)
        ws_url = f"{self.ws_url}/{stream_names}"
        
        log.info("ingester.starting", symbols=symbols, url=ws_url)
        
        self.running = True
        
        while self.running:
            try:
                async with websockets.connect(ws_url) as websocket:
                    self.websocket = websocket
                    log.info("websocket.connected")
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        try:
                            data = json.loads(message)
                            await self._process_message(data)
                        except json.JSONDecodeError:
                            log.warning("invalid.json", message=message[:100])
                        except Exception:
                            log.exception("message.processing.error")
                            
            except ConnectionClosed:
                if self.running:
                    log.warning("websocket.disconnected.reconnecting")
                    await asyncio.sleep(5)  # Wait before reconnecting
                else:
                    log.info("websocket.disconnected.shutdown")
                    break
            except Exception:
                log.exception("websocket.error")
                if self.running:
                    await asyncio.sleep(5)  # Wait before reconnecting

    async def _process_message(self, data: dict[str, Any]) -> None:
        """Process incoming WebSocket message."""
        if "stream" not in data:
            return
            
        stream = data["stream"]
        message_data = data["data"]
        
        if "@kline" in stream:
            await self.handle_kline(message_data)
        elif "@trade" in stream:
            await self.handle_trade(message_data)
        elif "@depth" in stream:
            await self.handle_depth(message_data)
        elif "@ticker" in stream:
            await self.handle_ticker(message_data)

    async def stop_streaming(self) -> None:
        """Stop streaming and flush remaining data."""
        log.info("ingester.stopping")
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
        
        # Flush remaining data
        for data_type in self.data_buffer:
            await self._flush_buffer(data_type)
        
        log.info("ingester.stopped")


# Global ingester instance for signal handling
ingester: BinanceDataIngester | None = None


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    log.info("signal.received", signal=signum)
    if ingester:
        asyncio.create_task(ingester.stop_streaming())


@click.command()
@click.option(
    "--symbols",
    default="BTCUSDT,ETHUSDT,ADAUSDT,SOLUSDT,DOTUSDT",
    help="Comma-separated list of symbols to stream",
)
@click.option(
    "--output-dir",
    default="data/cache/binance",
    help="Output directory for parquet files",
)
def main(symbols: str, output_dir: str) -> None:
    """Stream Binance market data to parquet files."""
    global ingester
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse symbols
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    
    # Create ingester
    ingester = BinanceDataIngester(output_dir)
    
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