"""Abstract Backtrader strategy wrapper."""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

import backtrader as bt
from quantdesk.utils.logging import get_logger


class BaseStrategy(bt.Strategy, ABC):
    params: dict[str, Any] = {
        "size": 1,  # position size in contracts/shares
        "atr_period": 14,
        "printlog": False,
    }

    def __init__(self) -> None:  # noqa: D401; Backtrader API
        """Initialise common state & indicators."""
        self.order: bt.Order | None = None  # last submitted order
        self.log = get_logger(self.__class__.__name__)

        # Common indicators (sub-classes free to ignore/use)
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params["atr_period"])
        self.sma_fast = bt.indicators.SMA(self.datas[0], period=20)
        self.sma_slow = bt.indicators.SMA(self.datas[0], period=50)

    # -------------------------------------------------------
    @abstractmethod
    def next(self) -> None:  # noqa: D401; Backtrader API
        """Implement trading logic here (called every bar)."""

    # -------------------------------------------------------
    # helper / callback utilities used by most concrete strats
    def _print(self, txt: str) -> None:  # pragma: no cover
        """Console print if *printlog* param is truthy (non-critical)."""
        if self.params.get("printlog", False):
            dt = self.datas[0].datetime.datetime(0)
            print(f"{dt.isoformat()} {txt}")

    # Alias retained for backward compatibility
    _log = _print

    # -------------------------------------------------------
    # Backtrader notifications
    def notify_order(self, order: bt.Order) -> None:  # pragma: no cover
        if order.status in (order.Submitted, order.Accepted):
            return  # not yet executed

        dt = self.datas[0].datetime.datetime(0)
        if order.status == order.Completed:
            side = "BUY" if order.isbuy() else "SELL"
            self.log.info(
                "order.completed",
                dt=dt.isoformat(),
                symbol=order.data._name,
                side=side,
                price=order.executed.price,
                qty=order.executed.size,
                value=order.executed.value,
                commission=order.executed.comm,
            )
            self.order = None
        elif order.status in (order.Canceled, order.Margin, order.Rejected):
            self.log.warning(
                "order.failed", status=order.Status[order.status], dt=dt.isoformat()
            )
            self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:  # pragma: no cover
        if trade.isclosed:
            self.log.info(
                "trade.closed",
                symbol=trade.data._name,
                pnl=trade.pnl,
                pnl_comm=trade.pnlcomm,
                barlen=trade.barlen,
            ) 