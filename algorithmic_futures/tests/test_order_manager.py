"""Focused tests for live order-manager reconciliation and order updates."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.order_manager import OrderManager, SessionState, TradeRecord
from regime.regime_state import OrderSide


class _StubPosition:
    def __init__(self, qty: int, side: str = "BUY", avg_price: float = 100.0) -> None:
        self.qty = qty
        self.side = side
        self.avg_price = avg_price


class _StubAPI:
    def __init__(self) -> None:
        self.positions = []
        self.open_orders = []
        self.cancelled: list[str] = []
        self.closed_all = False

    def get_positions(self):
        return self.positions

    def get_open_orders(self):
        return list(self.open_orders)

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return True

    def close_all_positions(self) -> bool:
        self.closed_all = True
        return True


class _StubBreakers:
    events = []


class _StubSizer:
    pass


def _make_manager() -> tuple[OrderManager, _StubAPI, SessionState]:
    api = _StubAPI()
    state = SessionState(account_balance=50_000.0, account_high_water_mark=50_000.0)
    manager = OrderManager(api=api, breakers=_StubBreakers(), sizer=_StubSizer(), state=state)
    return manager, api, state


def test_handle_order_update_records_exit_fill_and_clears_position():
    manager, _, state = _make_manager()
    state.open_position = {
        "trade_id": "t1",
        "side": OrderSide.BUY.value,
        "qty": 1,
        "entry_price": 100.0,
        "stop_price": 99.0,
        "target_price": 102.0,
        "strategy": "TEST",
    }
    state.pending_exit_orders = ["stop-1", "target-1"]
    manager._trade_log.append(
        TradeRecord(
            trade_id="t1",
            timestamp_entry="2026-03-29T14:35:00+00:00",
            strategy="TEST",
            side=OrderSide.BUY.value,
            qty=1,
            entry_price=100.0,
            fill_price=100.0,
            stop_price=99.0,
            target_price=102.0,
            order_id_stop="stop-1",
            order_id_target="target-1",
        )
    )
    manager.handle_order_update({"type": "fill", "orderId": "stop-1", "filledPrice": 99.0})

    assert state.open_position is None
    assert state.pending_exit_orders == []
    assert state.daily_pnl == -5.0
    assert manager._trade_log[0].exit_reason == "STOP"
    assert manager._trade_log[0].exit_price == 99.0


def test_reconcile_with_broker_clears_local_orphans_without_position():
    manager, api, state = _make_manager()
    state.open_position = {
        "trade_id": "t1",
        "side": OrderSide.BUY.value,
        "qty": 1,
        "entry_price": 100.0,
        "stop_price": 99.0,
        "target_price": 102.0,
        "strategy": "TEST",
    }
    state.pending_exit_orders = ["stop-1"]
    result = manager.reconcile_with_broker()

    assert result["cleared_local_position"] is True
    assert state.open_position is None
    assert state.pending_exit_orders == []


def test_reconcile_with_broker_flattens_untracked_broker_position():
    manager, api, state = _make_manager()
    api.positions = [_StubPosition(qty=1)]

    result = manager.reconcile_with_broker()

    assert result["flattened_broker_position"] is True
    assert api.closed_all is True


def test_reconcile_with_broker_handles_dict_positions():
    manager, api, _ = _make_manager()
    api.positions = [{"qty": 1, "side": "BUY", "avgPrice": 100.0}]

    result = manager.reconcile_with_broker()

    assert result["broker_positions"] == 1
    assert result["flattened_broker_position"] is True


def test_handle_order_update_infers_exit_fill_price_from_pending_leg():
    manager, _, state = _make_manager()
    state.open_position = {
        "trade_id": "t1",
        "side": OrderSide.BUY.value,
        "qty": 1,
        "entry_price": 100.0,
        "stop_price": 99.0,
        "target_price": 102.0,
        "strategy": "TEST",
    }
    state.pending_exit_orders = ["stop-1", "target-1"]
    manager._trade_log.append(
        TradeRecord(
            trade_id="t1",
            timestamp_entry="2026-03-29T14:35:00+00:00",
            strategy="TEST",
            side=OrderSide.BUY.value,
            qty=1,
            entry_price=100.0,
            fill_price=100.0,
            stop_price=99.0,
            target_price=102.0,
            order_id_stop="stop-1",
            order_id_target="target-1",
        )
    )

    manager.handle_order_update({"type": "order", "orderId": "target-1", "status": "FILLED"})

    assert state.open_position is None
    assert manager._trade_log[0].exit_reason == "TARGET"
    assert manager._trade_log[0].exit_price == 102.0
