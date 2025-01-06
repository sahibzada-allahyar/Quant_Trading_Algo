"""Structured JSON logging."""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

LOG_LEVEL = logging.INFO

_console = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    handlers=[_console], level=LOG_LEVEL, format="%(message)s", force=True
)

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL),
    logger_factory=structlog.stdlib.LoggerFactory(),
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger[Any]:
    return structlog.get_logger(name or "quantdesk") 