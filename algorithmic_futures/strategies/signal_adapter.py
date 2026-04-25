"""
strategies/signal_adapter.py — shared signal adapter layer.

This module gives replay/research and live-style dispatch a small common
interface for strategy signal generation.  It intentionally does not change
strategy logic or parameters; v1 wraps MRSignalEngine only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from data.indicators import VWAPState
from data.market_data import Bar
from strategies.mr_signal_engine import MRSignal, MRSignalEngine


EngineName = Literal["mr"]


@dataclass(frozen=True)
class SignalContext:
    """Per-bar context required by shared signal adapters."""

    regime: str | None
    vwap_state: VWAPState
    atr: float
    adx: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalDecision:
    """Normalized signal decision returned by strategy adapters."""

    engine: EngineName
    signal_type: str
    timestamp: datetime
    side: str
    approved: bool
    rejection_reason: str = ""
    entry_reference_price: float = 0.0
    stop_reference: float = 0.0
    target_reference: float = 0.0
    band_level_hit: float = 0.0
    vwap_at_signal: float = 0.0
    sigma_at_signal: float = 0.0
    z_at_signal: float = 0.0
    bar_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mr_signal(
        cls,
        signal: MRSignal,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SignalDecision:
        """Normalize an MRSignalEngine signal."""
        return cls(
            engine="mr",
            signal_type=signal.signal_type,
            timestamp=signal.timestamp,
            side=signal.side,
            approved=signal.approved,
            rejection_reason=signal.rejection_reason,
            entry_reference_price=signal.entry_reference_price,
            stop_reference=signal.stop_reference,
            target_reference=signal.target_reference,
            band_level_hit=signal.band_level_hit,
            vwap_at_signal=signal.vwap_at_signal,
            sigma_at_signal=signal.sigma_at_signal,
            z_at_signal=signal.z_at_signal,
            bar_index=signal.bar_index,
            metadata=metadata or {},
        )


class MRSignalAdapter:
    """Adapter around MRSignalEngine with a stable shared interface."""

    engine_name: EngineName = "mr"

    def __init__(self, engine: MRSignalEngine | None = None, **engine_kwargs: Any) -> None:
        self.engine = engine or MRSignalEngine(**engine_kwargs)

    def reset(self) -> None:
        self.engine.reset()

    def on_bar(self, bar: Bar, context: SignalContext) -> SignalDecision | None:
        """Return a normalized decision for this bar, or None if no candidate forms."""
        signal = self.engine.on_bar(
            bar=bar,
            regime=context.regime,
            vwap_state=context.vwap_state,
            atr=context.atr,
            adx=context.adx,
        )
        if signal is None:
            return None
        return SignalDecision.from_mr_signal(signal, metadata=context.metadata)


class SignalAdapter:
    """Small strategy adapter facade.

    v1 supports MR only.  ORB/live routing can be added once the ORB research
    logic is extracted from replay_debug.py into a reusable engine.
    """

    def __init__(self, engine: EngineName = "mr", **engine_kwargs: Any) -> None:
        if engine != "mr":
            raise ValueError("SignalAdapter v1 supports engine='mr' only")
        self._adapter = MRSignalAdapter(**engine_kwargs)

    def reset(self) -> None:
        self._adapter.reset()

    def on_bar(self, bar: Bar, context: SignalContext) -> SignalDecision | None:
        return self._adapter.on_bar(bar, context)
