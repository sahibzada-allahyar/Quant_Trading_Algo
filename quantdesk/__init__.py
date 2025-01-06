"""QuantDesk: Open-source quantitative trading research platform."""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "QuantDesk Team"
__email__ = "team@quantdesk.io"
__license__ = "AGPL-3.0"

# Core imports for easy access
from quantdesk.core.data_loader import load
from quantdesk.core.portfolio import Portfolio
from quantdesk.core.event_engine import EventEngine
from quantdesk.utils.logging import get_logger

__all__ = [
    "__version__",
    "load",
    "Portfolio", 
    "EventEngine",
    "get_logger",
] 