"""FastAPI service."""
from __future__ import annotations

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from quantdesk.api.schemas import Order, Position
from quantdesk.api.broker_router import BrokerRouter
from quantdesk.core.portfolio import Portfolio

app = FastAPI(title="QuantDesk API")
router = BrokerRouter()
PORTFOLIO = Portfolio(cash=1_000_000.0)


@app.post("/orders")
async def submit_order(order: Order) -> dict[str, str]:
    order_id = await router.route(order)
    return {"order_id": order_id}


@app.get("/positions")
async def positions() -> list[Position]:
    mark = {}  # could hit real‑time data
    df = PORTFOLIO.to_dataframe(mark)
    return df.to_dict(orient="records")


@app.websocket("/ws/fills")
async def stream_fills(ws: WebSocket) -> None:  # simplistic demo
    await ws.accept()
    try:
        while True:
            await ws.send_json({"msg": "heartbeat"})
            await ws.receive_text()
    except WebSocketDisconnect:
        return 


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    """Basic liveness probe used by orchestrators."""
    return {"status": "ok"}


@app.get("/portfolio/value")
async def portfolio_value() -> dict[str, float]:
    """Return current portfolio gross value (cash + equity)."""
    mark = {}
    return {"value": PORTFOLIO.value(mark)} 