"""Cash & position ledger with vectorised analytics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from quantdesk.core.event_engine import FillEvent
from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Position:
    qty: int = 0
    avg_price: float = 0.0

    def update(self, fill_price: float, quantity: int) -> None:
        total_cost = self.avg_price * self.qty + fill_price * quantity
        self.qty += quantity
        self.avg_price = 0.0 if self.qty == 0 else total_cost / self.qty


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)

    def update_fill(self, fill: FillEvent) -> None:
        symbol = fill.symbol
        pos = self.positions.setdefault(symbol, Position())
        pos.update(fill.fill_price, fill.direction * fill.quantity)
        cost = fill.fill_price * fill.quantity + fill.commission
        self.cash -= cost
        log.info(
            "portfolio.fill",
            symbol=symbol,
            qty=pos.qty,
            cash=self.cash,
            cost=cost,
        )

    def value(self, mark: dict[str, float]) -> float:
        equity = sum(pos.qty * mark.get(sym, 0.0) for sym, pos in self.positions.items())
        return self.cash + equity

    def to_dataframe(self, mark: dict[str, float]) -> pd.DataFrame:
        rows = []
        for sym, pos in self.positions.items():
            market = mark.get(sym, 0.0)
            rows.append(
                {
                    "symbol": sym,
                    "qty": pos.qty,
                    "avg_price": pos.avg_price,
                    "market_price": market,
                    "unrealised_pnl": (market - pos.avg_price) * pos.qty,
                }
            )
        return pd.DataFrame(rows) 