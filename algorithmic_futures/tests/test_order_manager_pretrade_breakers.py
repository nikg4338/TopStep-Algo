"""OrderManager pre-trade circuit-breaker integration tests."""

from __future__ import annotations

from typing import Any

from execution.api_client import OrderResponse
from execution.circuit_breakers import CircuitBreakers
from execution.order_manager import OrderManager, SessionState
from regime.regime_state import OrderSide, RegimeState
from risk.position_sizer import PositionSizer


class FakeAPI:
    def __init__(self) -> None:
        self.orders_placed: list[dict[str, Any]] = []
        self._order_id = 0

    def get_positions(self) -> list:
        return []

    def place_order(self, **kwargs: Any) -> OrderResponse:
        self.orders_placed.append(kwargs)
        self._order_id += 1
        return OrderResponse(order_id=f"order-{self._order_id}", status="ACCEPTED")

    def get_open_orders(self) -> list:
        return []

    def cancel_order(self, order_id: str) -> None:
        pass

    def close_all_positions(self) -> None:
        pass


def _manager(state: SessionState) -> OrderManager:
    return OrderManager(
        api=FakeAPI(),  # type: ignore[arg-type]
        breakers=CircuitBreakers(account_mode="combine"),
        sizer=PositionSizer(risk_per_trade=20),
        state=state,
    )


def _submit(manager: OrderManager, *, stop_price: float = 5796.0) -> None:
    trade = manager.submit_entry(
        side=OrderSide.BUY,
        stop_price=stop_price,
        target_price=5810.0,
        entry_price=5800.0,
        strategy="VWAP_MR",
        regime=RegimeState.BALANCED,
    )
    assert trade is None


def test_order_manager_rejects_when_pass_state_reached() -> None:
    manager = _manager(
        SessionState(
            daily_pnl=100.0,
            cumulative_pnl=3_100.0,
            best_day_pnl=1_200.0,
            account_balance=53_100.0,
            account_high_water_mark=53_100.0,
            current_regime=RegimeState.BALANCED.value,
        )
    )
    _submit(manager)
    assert manager.last_rejection_reason == CircuitBreakers.PASS_STATE_REACHED


def test_order_manager_rejects_when_mll_headroom_too_low() -> None:
    manager = _manager(
        SessionState(
            daily_pnl=0.0,
            cumulative_pnl=-1_850.0,
            best_day_pnl=300.0,
            account_balance=48_150.0,
            account_high_water_mark=50_000.0,
            current_regime=RegimeState.BALANCED.value,
        )
    )
    _submit(manager)
    assert manager.last_rejection_reason == CircuitBreakers.MLL_HEADROOM_TOO_LOW


def test_order_manager_rejects_when_consistency_cap_risk() -> None:
    manager = _manager(
        SessionState(
            daily_pnl=100.0,
            cumulative_pnl=2_500.0,
            best_day_pnl=1_450.0,
            account_balance=52_500.0,
            account_high_water_mark=52_500.0,
            current_regime=RegimeState.BALANCED.value,
        )
    )
    _submit(manager, stop_price=5400.0)
    assert manager.last_rejection_reason == CircuitBreakers.CONSISTENCY_CAP_RISK


def test_order_manager_rejects_when_daily_loss_budget_low() -> None:
    manager = _manager(
        SessionState(
            daily_pnl=-230.0,
            cumulative_pnl=100.0,
            best_day_pnl=200.0,
            account_balance=50_100.0,
            account_high_water_mark=50_100.0,
            current_regime=RegimeState.BALANCED.value,
        )
    )
    _submit(manager)
    assert manager.last_rejection_reason == CircuitBreakers.DAILY_LOSS_BUDGET_LOW
