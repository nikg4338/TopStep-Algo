from __future__ import annotations

import asyncio
import json

import pytest

from execution.api_client import ProjectXClient


class _FakeWebSocket:
    def __init__(self, messages: list[dict]) -> None:
        self._messages = [json.dumps(msg) for msg in messages]
        self.sent: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        self._iter = iter(self._messages)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise asyncio.CancelledError() from exc

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


@pytest.mark.asyncio
async def test_subscribe_market_data_routes_tick_bar_and_order_updates(monkeypatch):
    messages = [
        {"type": "tick", "price": 100.0},
        {"type": "bar", "close": 101.0},
        {"type": "fill", "orderId": "abc", "filledPrice": 101.25},
    ]
    fake_ws = _FakeWebSocket(messages)

    class _FakeWebSocketsModule:
        @staticmethod
        def connect(*args, **kwargs):
            return fake_ws

    monkeypatch.setitem(__import__("sys").modules, "websockets", _FakeWebSocketsModule)

    client = ProjectXClient(api_key="k", base_url="https://example.test", account_id="acct")
    client._token = "token"
    client._token_expiry = None

    ticks: list[dict] = []
    bars: list[dict] = []
    updates: list[dict] = []

    with pytest.raises(asyncio.CancelledError):
        await client.subscribe_market_data(
            symbol="MES",
            on_tick=ticks.append,
            on_bar=bars.append,
            on_order_update=updates.append,
        )

    assert fake_ws.sent == [{"action": "subscribe", "symbol": "MES"}]
    assert ticks == [{"type": "tick", "price": 100.0}]
    assert bars == [{"type": "bar", "close": 101.0}]
    assert updates == [{"type": "fill", "orderId": "abc", "filledPrice": 101.25}]