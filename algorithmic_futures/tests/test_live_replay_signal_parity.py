"""
tests/test_live_replay_signal_parity.py — scaffold for live/replay signal parity.

Future contract:
  - Given the same input bars,
  - the same preset/config values,
  - and the same strategy engine,
  - replay mode and live-style dispatch should emit the same signal decisions.

This is intentionally an xfail scaffold today. Replay signal generation is
available through MRSignalEngine, but live dispatch still routes through the
legacy VWAPMeanReversion / ORBBreakout classes rather than a shared signal
router that can be exercised without broker/API wiring.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from data.indicators import ATRCalculator, VWAPCalculator
from data.market_data import Bar
from strategies.signal_adapter import SignalAdapter, SignalContext


Decision = tuple[str, str, bool, str]


def _make_bar(
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
) -> Bar:
    return Bar(
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_parity_bars() -> list[Bar]:
    """Small deterministic bar set for future replay/live parity checks."""
    base = datetime(2026, 2, 18, 14, 30)
    prices = [
        (5800.0, 5801.0, 5799.0, 5800.0),
        (5800.0, 5801.0, 5799.0, 5800.2),
        (5800.2, 5801.0, 5799.2, 5800.1),
        (5800.1, 5800.6, 5798.0, 5798.5),
        (5798.5, 5799.2, 5796.8, 5797.2),
        (5797.2, 5799.8, 5797.0, 5799.4),
        (5799.4, 5801.0, 5799.0, 5800.5),
    ]
    return [
        _make_bar(base + timedelta(minutes=5 * idx), o, h, l, c)
        for idx, (o, h, l, c) in enumerate(prices)
    ]


def _decision_tuple(signal) -> Decision:
    return (
        signal.signal_type,
        signal.side,
        bool(signal.approved),
        signal.rejection_reason,
    )


def _collect_replay_mr_decisions(bars: list[Bar]) -> list[Decision]:
    """Replay-style MR signal collection through the shared adapter."""
    vwap = VWAPCalculator()
    atr = ATRCalculator(period=14)
    adapter = SignalAdapter(engine="mr", reclaim_mode="on", regime_enabled=False)

    decisions: list[Decision] = []
    for bar in bars:
        vwap_state = vwap.update(bar.high, bar.low, bar.close, bar.volume)
        atr_value = atr.update(bar.high, bar.low, bar.close)
        signal = adapter.on_bar(
            bar,
            SignalContext(regime="range", vwap_state=vwap_state, atr=atr_value, adx=0.0),
        )
        if signal is not None:
            decisions.append(_decision_tuple(signal))
    return decisions


def _collect_live_style_decisions(_bars: list[Bar]) -> list[Decision]:
    """Placeholder for the future shared live-style dispatch path."""
    pytest.xfail(
        "Live/replay parity is blocked until live dispatch uses a shared "
        "signal router instead of legacy VWAPMeanReversion/ORBBreakout classes."
    )


def test_live_style_dispatch_matches_replay_signal_decisions() -> None:
    """Future parity contract for feature/live-research-parity-risk-gates."""
    bars = _make_parity_bars()

    replay_decisions = _collect_replay_mr_decisions(bars)
    live_decisions = _collect_live_style_decisions(bars)

    assert live_decisions == replay_decisions
