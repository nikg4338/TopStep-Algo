"""
tests/test_signal_adapter.py — shared signal adapter tests.

The adapter is intentionally thin in v1: it normalizes MRSignalEngine output
without changing strategy logic or parameter values.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from data.indicators import VWAPState
from data.market_data import Bar
from strategies.mr_signal_engine import MRSignalEngine
from strategies.signal_adapter import MRSignalAdapter, SignalAdapter, SignalContext


def _bar(ts: datetime, close: float, high: float | None = None, low: float | None = None) -> Bar:
    high = close + 1.0 if high is None else high
    low = close - 1.0 if low is None else low
    return Bar(timestamp=ts, open=close, high=high, low=low, close=close, volume=100.0)


def _vwap_state() -> VWAPState:
    return VWAPState(bar_count=20, vwap=5800.0, std_dev=5.0)


def _context() -> SignalContext:
    return SignalContext(regime="range", vwap_state=_vwap_state(), atr=2.0, adx=0.0)


def test_mr_adapter_normalizes_engine_signal() -> None:
    adapter = MRSignalAdapter(MRSignalEngine(reclaim_mode="on"))

    assert adapter.on_bar(_bar(datetime(2026, 2, 18, 14, 30), close=5787.0), _context()) is None

    decision = adapter.on_bar(
        _bar(datetime(2026, 2, 18, 14, 35), close=5794.0, high=5795.5, low=5793.5),
        _context(),
    )

    assert decision is not None
    assert decision.engine == "mr"
    assert decision.signal_type == "MR"
    assert decision.side == "BUY"
    assert decision.approved is True
    assert decision.entry_reference_price == pytest.approx(5794.0)
    assert decision.target_reference == pytest.approx(5800.0)
    assert decision.stop_reference == pytest.approx(5791.0)


def test_signal_adapter_facade_supports_mr_engine() -> None:
    adapter = SignalAdapter(engine="mr", reclaim_mode="on")

    assert adapter.on_bar(_bar(datetime(2026, 2, 18, 14, 30), close=5813.0), _context()) is None
    decision = adapter.on_bar(
        _bar(datetime(2026, 2, 18, 14, 35), close=5806.0, high=5806.5, low=5804.0),
        _context(),
    )

    assert decision is not None
    assert decision.engine == "mr"
    assert decision.side == "SELL"
    assert decision.approved is True


def test_signal_adapter_rejects_unsupported_engine() -> None:
    with pytest.raises(ValueError, match="supports engine='mr' only"):
        SignalAdapter(engine="orb")  # type: ignore[arg-type]
