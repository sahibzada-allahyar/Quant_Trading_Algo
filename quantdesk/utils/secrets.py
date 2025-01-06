"""Helpers for secret discovery inside containers or local OS keychains."""
from __future__ import annotations

import os
from base64 import b64decode
from pathlib import Path

_SECRET_DIR = Path("/run/secrets")  # Docker Swarm / Compose secrets mount


def get_secret(key: str, default: str | None = None) -> str:
    """Fetch secret value, prioritising Docker secrets over env."""
    file_path = _SECRET_DIR / key
    if file_path.exists():
        return file_path.read_text().strip()
    try:
        return os.environ[key]
    except KeyError as exc:
        if default is not None:
            return default
        raise RuntimeError(f"Missing secret: {key}") from exc 