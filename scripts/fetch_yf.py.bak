#!/usr/bin/env python
"""CLI: Bulk‑download Yahoo Finance data."""
from __future__ import annotations

import click
from datetime import datetime
import pandas as pd
from pathlib import Path

from quantdesk.core.data_loader import load


@click.command()
@click.argument("symbol")
@click.option("--start", default="2010‑01‑01")
@click.option("--end", default=datetime.utcnow().strftime("%Y‑%m‑%d"))
@click.option("--freq", default="1d")
@click.option("--outdir", default="data/cache")
def main(symbol: str, start: str, end: str, freq: str, outdir: str) -> None:
    df = load(symbol, datetime.fromisoformat(start), datetime.fromisoformat(end), freq=freq)
    out_path = Path(outdir) / "yahoo" / symbol / f"{freq}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    click.echo(f"Saved {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main() 