"""Environment variable loader & validator."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH, override=False)


class Settings(BaseModel):
    """Runtime configuration pulled from the environment."""

    alpaca_key: str = Field(..., env="ALPACA_KEY_ID")
    alpaca_secret: str = Field(..., env="ALPACA_SECRET_KEY")
    binance_key: str = Field(..., env="BINANCE_API_KEY")
    binance_secret: str = Field(..., env="BINANCE_API_SECRET")
    discord_webhook: str = Field(..., env="DISCORD_WEBHOOK_URL")
    mlflow_uri: str = Field("http://mlflow:5000", env="MLFLOW_TRACKING_URI")

    class Config:
        extra = "forbid"
        frozen = True


try:
    SETTINGS = Settings()  # validated at import‑time
except ValidationError as e:  # pragma: no cover
    raise RuntimeError(f"Invalid .env ‑ {e}") from e 