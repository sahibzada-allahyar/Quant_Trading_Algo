"""Simple event queue driving Backtrader + live trading."""
from __future__ import annotations

import queue
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Protocol

from quantdesk.utils.logging import get_logger

log = get_logger(__name__)


class EventType(Enum):
    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


@dataclass(slots=True)
class Event:
    type: EventType
    timestamp: datetime


@dataclass(slots=True)
class MarketEvent(Event):
    symbol: str
    price: float
    volume: float


@dataclass(slots=True)
class SignalEvent(Event):
    symbol: str
    direction: int  # +1 long, -1 short
    strength: float


@dataclass(slots=True)
class OrderEvent(Event):
    symbol: str
    direction: int
    quantity: int
    order_type: str = "MKT"


@dataclass(slots=True)
class FillEvent(Event):
    symbol: str
    direction: int
    quantity: int
    fill_price: float
    commission: float


class EventHandler(Protocol):
    def __call__(self, event: Event) -> None: ...


class EventEngine:
    """Thread‑safe queue with pub/sub callbacks."""

    def __init__(self) -> None:
        self._q: queue.Queue[Event] = queue.Queue(maxsize=10000)
        self._subs: dict[EventType, list[EventHandler]] = {t: [] for t in EventType}

    def put(self, event: Event) -> None:
        self._q.put(event, block=False)

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        self._subs[event_type].append(handler)

    def run_once(self) -> None:
        """Process a single event if available."""
        try:
            event = self._q.get(block=False)
        except queue.Empty:
            return
        for handler in self._subs[event.type]:
            handler(event)
        self._q.task_done() 