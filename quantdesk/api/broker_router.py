"""Route orders to Alpaca or Binance."""
from __future__ import annotations

import asyncio
from typing import Protocol

import alpaca_trade_api as alpaca
import ccxt.async_support as accxt
from fastapi import HTTPException

from quantdesk.utils.env import SETTINGS
from quantdesk.utils.logging import get_logger
from quantdesk.api.schemas import Order

log = get_logger(__name__)


class Broker(Protocol):
    async def submit(self, order: Order) -> str: ...
    async def cancel(self, order_id: str) -> None: ...


class AlpacaBroker:
    def __init__(self) -> None:
        self._client = alpaca.REST(
            key_id=SETTINGS.alpaca_key,
            secret_key=SETTINGS.alpaca_secret,
            base_url="https://paper-api.alpaca.markets",
        )

    async def submit(self, order: Order) -> str:
        o = await asyncio.to_thread(
            self._client.submit_order,
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            type=order.order_type,
            time_in_force="day",
            limit_price=order.limit_price,
        )
        return o.id

    async def cancel(self, order_id: str) -> None:
        await asyncio.to_thread(self._client.cancel_order, order_id)


class BinanceBroker:
    def __init__(self) -> None:
        self._client = accxt.binance({
            "apiKey": SETTINGS.binance_key,
            "secret": SETTINGS.binance_secret,
            "enableRateLimit": True,
        })

    async def submit(self, order: Order) -> str:
        side = "buy" if order.side == "buy" else "sell"
        response = await self._client.create_order(
            symbol=order.symbol,
            type="MARKET" if order.order_type == "market" else "LIMIT",
            side=side.upper(),
            amount=order.qty,
            price=order.limit_price,
        )
        return str(response["id"])

    async def cancel(self, order_id: str) -> None:
        await self._client.cancel_order(order_id)


class BrokerRouter:
    def __init__(self) -> None:
        self.alpaca = AlpacaBroker()
        self.binance = BinanceBroker()

    async def route(self, order: Order) -> str:
        if order.symbol.endswith("USD") or order.symbol.endswith("USDT"):
            return await self.binance.submit(order)
        if len(order.symbol) <= 5:  # crude equity heuristic
            return await self.alpaca.submit(order)
        raise HTTPException(400, f"Un‑routable symbol {order.symbol}") 