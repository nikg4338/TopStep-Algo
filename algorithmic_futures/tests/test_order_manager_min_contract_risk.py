"""
tests/test_order_manager_min_contract_risk.py — pre-trade min-contract risk gate.
"""

from __future__ import annotations

from typing import Any

from execution.api_client import OrderResponse
from execution.circuit_breakers import BreakerCheckResult, CircuitBreakers
from execution.order_manager import OrderManager, SessionState
from regime.regime_state import OrderSide, RegimeState
from risk.position_sizer import PositionSizer


class FakeBreakers:
    @property
    def events(self) -> list:
        return []

    def check_all(self, **kwargs: Any) -> BreakerCheckResult:
        return BreakerCheckResult(allowed=True)


class FakeAPI:
    def __init__(self) -> None:
        self.positions_checked = False
        self.orders_placed: list[dict[str, Any]] = []
        self._order_id = 0

    def get_positions(self) -> list:
        self.positions_checked = True
        return []

    def place_order(self, **kwargs: Any) -> OrderResponse:
        self.orders_placed.append(kwargs)
        self._order_id += 1
        return OrderResponse(order_id=f"order-{self._order_id}", status="ACCEPTED")

    def get_open_orders(self) -> list:
        return []

    def cancel_order(self, order_id: str) -> None:
        pass


def _manager(state: SessionState) -> tuple[OrderManager, FakeAPI]:
    api = FakeAPI()
    manager = OrderManager(
        api=api,  # type: ignore[arg-type]
        breakers=FakeBreakers(),  # type: ignore[arg-type]
        sizer=PositionSizer(risk_per_trade=20),
        state=state,
    )
    return manager, api


def test_order_manager_records_min_contract_risk_rejection() -> None:
    manager, api = _manager(
        SessionState(
            daily_pnl=0.0,
            account_balance=50_000.0,
            account_high_water_mark=50_000.0,
            current_regime=RegimeState.BALANCED.value,
        )
    )

    trade = manager.submit_entry(
        side=OrderSide.BUY,
        stop_price=5700.0,
        target_price=5810.0,
        entry_price=5800.0,
        strategy="VWAP_MR",
        regime=RegimeState.BALANCED,
    )

    assert trade is None
    assert manager.last_rejection_reason == CircuitBreakers.MIN_CONTRACT_RISK_TOO_HIGH
    assert manager.rejection_log[-1]["stage"] == "position_sizer"
    assert api.positions_checked is False
    assert api.orders_placed == []


def test_order_manager_allows_trade_with_healthy_mll_headroom() -> None:
    manager, api = _manager(
        SessionState(
            daily_pnl=0.0,
            account_balance=50_000.0,
            account_high_water_mark=50_000.0,
            current_regime=RegimeState.BALANCED.value,
        )
    )

    trade = manager.submit_entry(
        side=OrderSide.BUY,
        stop_price=5796.0,
        target_price=5810.0,
        entry_price=5800.0,
        strategy="VWAP_MR",
        regime=RegimeState.BALANCED,
    )

    assert trade is not None
    assert trade.qty == 1
    assert manager.rejection_log == []
    assert api.positions_checked is True
    assert len(api.orders_placed) == 3


def test_order_manager_rejects_trade_with_tight_mll_headroom() -> None:
    manager, api = _manager(
        SessionState(
            daily_pnl=0.0,
            account_balance=48_100.0,
            account_high_water_mark=50_000.0,
            current_regime=RegimeState.BALANCED.value,
        )
    )

    trade = manager.submit_entry(
        side=OrderSide.BUY,
        stop_price=5796.0,
        target_price=5810.0,
        entry_price=5800.0,
        strategy="VWAP_MR",
        regime=RegimeState.BALANCED,
    )

    assert trade is None
    assert manager.last_rejection_reason == CircuitBreakers.MIN_CONTRACT_RISK_TOO_HIGH
    assert manager.rejection_log[-1]["metadata"]["remaining_mll_headroom"] == 100.0
    assert manager.rejection_log[-1]["metadata"]["current_mll_floor"] == 48_000.0
    assert api.positions_checked is False
    assert api.orders_placed == []


def test_order_manager_rejection_reason_is_clear_for_projected_mll_risk() -> None:
    manager, api = _manager(
        SessionState(
            daily_pnl=0.0,
            account_balance=48_100.0,
            account_high_water_mark=50_000.0,
            current_regime=RegimeState.BALANCED.value,
        )
    )

    trade = manager.submit_entry(
        side=OrderSide.BUY,
        stop_price=5799.0,
        target_price=5810.0,
        entry_price=5800.0,
        strategy="VWAP_MR",
        regime=RegimeState.BALANCED,
    )

    assert trade is None
    assert manager.last_rejection_reason == CircuitBreakers.MIN_CONTRACT_RISK_TOO_HIGH
    assert manager.rejection_log[-1]["metadata"]["projected_trade_risk"] == 20.0
    assert manager.rejection_log[-1]["metadata"]["mll_headroom_safety_fraction"] == 0.10
    assert api.positions_checked is False
    assert api.orders_placed == []
