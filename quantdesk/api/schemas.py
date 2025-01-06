"""Pydantic request/response models."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    side: str = Field(pattern="^(buy|sell)$")
    qty: int
    order_type: str = Field(default="market", pattern="^(market|limit)$")
    limit_price: float | None = None
    timestamp: datetime | None = None


class Position(BaseModel):
    symbol: str
    qty: int
    avg_price: float 