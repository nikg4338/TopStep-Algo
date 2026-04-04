from __future__ import annotations

import pytest

from session_manager import SessionManager


def test_on_order_update_forwards_to_order_manager() -> None:
    manager = SessionManager()
    seen: list[dict] = []
    manager.order_manager.handle_order_update = seen.append  # type: ignore[method-assign]

    msg = {"type": "fill", "orderId": "stop-1", "filledPrice": 99.0}
    manager._on_order_update(msg)

    assert seen == [msg]


@pytest.mark.asyncio
async def test_start_market_data_flattens_with_last_price_on_disconnect() -> None:
    manager = SessionManager()
    manager._last_market_price = 4321.25

    async def _boom(**kwargs):
        raise RuntimeError("ws down")

    calls: list[tuple[str, float | None]] = []
    manager.api.subscribe_market_data = _boom  # type: ignore[method-assign]
    manager.order_manager.flatten_all_with_price = lambda reason, exit_price=None: calls.append((reason, exit_price))  # type: ignore[method-assign]

    await manager._start_market_data()

    assert calls == [("WS_DISCONNECT", 4321.25)]


def test_reconcile_broker_state_falls_back_to_broker_flatten() -> None:
    manager = SessionManager()
    flattened: list[bool] = []

    def _raise() -> dict:
        raise RuntimeError("reconcile failed")

    manager.order_manager.reconcile_with_broker = _raise  # type: ignore[method-assign]
    manager.api.close_all_positions = lambda: flattened.append(True) or True  # type: ignore[method-assign]

    manager._reconcile_broker_state()

    assert flattened == [True]