"""
tests/test_session_manager_research_mr.py — live MR adapter routing tests.

These tests cover only SessionManager routing. They do not exercise broker
execution, sizing, exits, or risk management.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from data.market_data import Bar
from regime.regime_state import OrderSide, RegimeState
from session_manager import SessionManager
from strategies.signal_adapter import SignalContext, SignalDecision


def _bar() -> Bar:
    return Bar(
        timestamp=datetime(2026, 2, 18, 10, 0),
        open=5800.0,
        high=5803.0,
        low=5790.0,
        close=5792.0,
        volume=100.0,
    )


def test_session_manager_initializes_research_mr_adapter(monkeypatch) -> None:
    import session_manager

    monkeypatch.setattr(session_manager, "USE_RESEARCH_MR_ENGINE", True)
    manager = SessionManager()

    manager._init_strategies(RegimeState.BALANCED)

    assert manager._mr_signal_adapter is not None
    assert manager._vwap_strategy is None
    assert manager._orb_strategy is not None


def test_session_manager_mr_signal_calls_flow_through_adapter(monkeypatch) -> None:
    import session_manager

    monkeypatch.setattr(session_manager, "USE_RESEARCH_MR_ENGINE", True)
    manager = SessionManager()
    calls: list[tuple[Bar, SignalContext]] = []
    submitted: list[dict[str, Any]] = []

    class FakeAdapter:
        def on_bar(self, bar: Bar, context: SignalContext) -> SignalDecision:
            calls.append((bar, context))
            return SignalDecision(
                engine="mr",
                signal_type="MR",
                timestamp=bar.timestamp,
                side="BUY",
                approved=True,
                entry_reference_price=bar.close,
                stop_reference=bar.close - 3.0,
                target_reference=5800.0,
                band_level_hit=1.4,
                vwap_at_signal=5800.0,
                sigma_at_signal=5.0,
                z_at_signal=-1.6,
                bar_index=7,
                metadata={"source": "test_adapter"},
            )

    @dataclass
    class FakeOrderManager:
        def submit_entry(self, **kwargs: Any) -> None:
            submitted.append(kwargs)

    manager._mr_signal_adapter = FakeAdapter()  # type: ignore[assignment]
    manager.order_manager = FakeOrderManager()  # type: ignore[assignment]

    manager._on_research_mr_bar(_bar())

    assert len(calls) == 1
    assert calls[0][1].regime == "range"
    assert submitted == [
        {
            "side": OrderSide.BUY,
            "stop_price": 5789.0,
            "target_price": 5800.0,
            "entry_price": 5792.0,
            "strategy": "VWAP_MR",
            "regime": RegimeState.BALANCED,
            "metadata": {
                "signal_engine": "mr",
                "signal_type": "MR",
                "rejection_reason": "",
                "band_level_hit": 1.4,
                "vwap_at_signal": 5800.0,
                "sigma_at_signal": 5.0,
                "z_at_signal": -1.6,
                "bar_index": 7,
                "bar_timestamp": "2026-02-18 10:00:00",
                "source": "test_adapter",
            },
        }
    ]


def test_session_manager_legacy_mr_mode_still_initializes_vwap(monkeypatch) -> None:
    import session_manager

    monkeypatch.setattr(session_manager, "USE_RESEARCH_MR_ENGINE", False)
    manager = SessionManager()

    manager._init_strategies(RegimeState.BALANCED)

    assert manager._mr_signal_adapter is None
    assert manager._vwap_strategy is not None
    assert manager._orb_strategy is not None
