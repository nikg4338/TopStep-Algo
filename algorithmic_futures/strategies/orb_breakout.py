"""
strategies/orb_breakout.py — 15-Minute Opening Range Breakout (ORB).

Activated exclusively when the HMM regime classifier outputs State 1
(DIRECTIONAL).  Captures momentum following a decisive break of the
first 15 minutes' high or low of the RTH session.

Lifecycle (per session):
  1. reset() at session open.
  2. on_bar() called on every completed 5-min bar during RTH.
  3. 09:30–09:45 ET  → accumulate bars, build ORB range.
  4. 09:45–11:00 ET  → monitor for breakout (full candle close outside range).
  5. After breakout  → submit entry via OrderManager; stop = ORB midline,
                        target = 1:1.5 R:R in breakout direction.
  6. Max 2 entries per session (ORB_TRADES_PER_DAY).
  7. No entries after 11:00 ET (stale range) or within 15 min of EOD close.
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dt_time

import pytz

from config import (
    EOD_CLOSE,
    LAST_ENTRY_CUTOFF,
    ORB_END,
    ORB_RR_RATIO,
    ORB_STALE_CUTOFF,
    ORB_TRADES_PER_DAY,
    RTH_OPEN,
    TIMEZONE,
)
from data.market_data import Bar
from execution.order_manager import OrderManager, SessionState
from regime.regime_state import OrderSide, RegimeState

logger = logging.getLogger(__name__)
ET = pytz.timezone(TIMEZONE)

# Pre-parse session time boundaries once at import time.
_RTH_OPEN = dt_time.fromisoformat(RTH_OPEN)          # 09:30
_ORB_END = dt_time.fromisoformat(ORB_END)             # 09:45
_STALE_CUTOFF = dt_time.fromisoformat(ORB_STALE_CUTOFF)  # 11:00
_LAST_ENTRY = dt_time.fromisoformat(LAST_ENTRY_CUTOFF)    # 15:50
_EOD_CLOSE = dt_time.fromisoformat(EOD_CLOSE)             # 16:05


class ORBBreakout:
    """15-Minute Opening Range Breakout strategy for MES futures.

    Parameters
    ----------
    order_manager : OrderManager
        Handles the full trade lifecycle (pre-trade checks, sizing,
        placement, fill confirmation, exit management).
    state : SessionState
        Shared mutable intra-day state (daily P&L, trade count, etc.).
    """

    STRATEGY_NAME = "ORB_BREAKOUT"

    def __init__(self, order_manager: OrderManager, state: SessionState) -> None:
        self.order_manager = order_manager
        self.state = state

        # ORB range tracking
        self._orb_high: float = float("-inf")
        self._orb_low: float = float("inf")
        self._orb_defined: bool = False
        self._orb_bars: list[Bar] = []

        # Session-level trade control
        self._trade_count: int = 0
        self._breakout_triggered: bool = False

    # ── Public API ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all strategy state for a new trading session.

        Must be called at the start of every RTH session before the first
        ``on_bar`` invocation.
        """
        self._orb_high = float("-inf")
        self._orb_low = float("inf")
        self._orb_defined = False
        self._orb_bars.clear()
        self._trade_count = 0
        self._breakout_triggered = False
        logger.info("[%s] Strategy reset for new session.", self.STRATEGY_NAME)

    def on_bar(self, bar: Bar) -> None:
        """Process a completed 5-minute bar.

        Called by the bar aggregator for every intraday bar during RTH.
        Handles three phases:
          1. ORB accumulation (09:30–09:45)
          2. Breakout detection & entry (09:45–11:00, before LAST_ENTRY)
          3. EOD position close (at or after EOD_CLOSE)
        """
        bar_time = self._to_et(bar.timestamp).time()

        # ── Phase 0: EOD time-stop — close any open position ────────────
        if bar_time >= _EOD_CLOSE:
            self._handle_eod_close(bar)
            return

        # ── Phase 1: ORB range accumulation (09:30–09:44:59) ────────────
        if bar_time < _ORB_END:
            self._accumulate_orb(bar)
            return

        # If the ORB window just ended and we haven't finalised yet, do so.
        if not self._orb_defined:
            self._finalise_orb()

        # If the ORB could not be defined (no bars received), skip.
        if not self._orb_defined:
            return

        # ── Phase 2: Breakout detection & entry ─────────────────────────
        self._evaluate_breakout(bar, bar_time)

    # ── Internal Helpers ────────────────────────────────────────────────

    def _accumulate_orb(self, bar: Bar) -> None:
        """Record bar during the ORB accumulation window."""
        self._orb_bars.append(bar)
        self._orb_high = max(self._orb_high, bar.high)
        self._orb_low = min(self._orb_low, bar.low)
        logger.debug(
            "[%s] ORB accumulating | bar_time=%s high=%.2f low=%.2f | "
            "running ORB high=%.2f low=%.2f",
            self.STRATEGY_NAME,
            bar.timestamp,
            bar.high,
            bar.low,
            self._orb_high,
            self._orb_low,
        )

    def _finalise_orb(self) -> None:
        """Lock in the ORB range after the accumulation window closes."""
        if not self._orb_bars:
            logger.warning(
                "[%s] No bars received during ORB window — cannot define range.",
                self.STRATEGY_NAME,
            )
            return

        self._orb_defined = True
        orb_width = self._orb_high - self._orb_low
        midline = self._orb_midline
        logger.info(
            "[%s] ORB DEFINED | high=%.2f low=%.2f width=%.2f midline=%.2f | "
            "%d bars accumulated",
            self.STRATEGY_NAME,
            self._orb_high,
            self._orb_low,
            orb_width,
            midline,
            len(self._orb_bars),
        )

    def _evaluate_breakout(self, bar: Bar, bar_time: dt_time) -> None:
        """Check whether the current bar triggers a valid ORB breakout entry."""
        # ── Guard: max trade attempts reached ───────────────────────────
        if self._trade_count >= ORB_TRADES_PER_DAY:
            logger.debug(
                "[%s] Trade limit reached (%d/%d) — skipping.",
                self.STRATEGY_NAME,
                self._trade_count,
                ORB_TRADES_PER_DAY,
            )
            return

        # ── Guard: stale ORB (after 11:00 ET) ──────────────────────────
        if bar_time >= _STALE_CUTOFF:
            logger.debug(
                "[%s] Past ORB_STALE_CUTOFF (%s) — no new ORB entries.",
                self.STRATEGY_NAME,
                ORB_STALE_CUTOFF,
            )
            return

        # ── Guard: too close to EOD ─────────────────────────────────────
        if bar_time >= _LAST_ENTRY:
            logger.debug(
                "[%s] Past LAST_ENTRY_CUTOFF (%s) — no new entries.",
                self.STRATEGY_NAME,
                LAST_ENTRY_CUTOFF,
            )
            return

        # ── Guard: regime must be DIRECTIONAL ───────────────────────────
        if self.state.current_regime != RegimeState.DIRECTIONAL:
            logger.debug(
                "[%s] Regime is %s (need DIRECTIONAL) — skipping.",
                self.STRATEGY_NAME,
                RegimeState(self.state.current_regime).name,
            )
            return

        # ── Guard: already have an open position ────────────────────────
        if self.state.open_position is not None:
            logger.debug(
                "[%s] Position already open — waiting for exit.",
                self.STRATEGY_NAME,
            )
            return

        # ── Breakout detection: full candle close outside range ─────────
        if bar.close > self._orb_high:
            self._attempt_entry(OrderSide.BUY, bar)
        elif bar.close < self._orb_low:
            self._attempt_entry(OrderSide.SELL, bar)

    def _attempt_entry(self, side: OrderSide, bar: Bar) -> None:
        """Submit a breakout entry through the OrderManager."""
        entry_price = bar.close
        stop_price = self._orb_midline
        risk = abs(entry_price - stop_price)

        if risk <= 0:
            logger.warning(
                "[%s] Computed risk is zero — cannot calculate target. Skipping.",
                self.STRATEGY_NAME,
            )
            return

        reward = risk * ORB_RR_RATIO

        if side == OrderSide.BUY:
            target_price = entry_price + reward
        else:
            target_price = entry_price - reward

        metadata = {
            "orb_high": self._orb_high,
            "orb_low": self._orb_low,
            "orb_midline": self._orb_midline,
            "orb_width": self._orb_high - self._orb_low,
            "breakout_direction": side.value,
            "entry_bar_timestamp": str(bar.timestamp),
            "risk": round(risk, 4),
            "reward": round(reward, 4),
            "rr_ratio": ORB_RR_RATIO,
        }

        logger.info(
            "[%s] BREAKOUT %s | entry=%.2f stop=%.2f target=%.2f | "
            "ORB high=%.2f low=%.2f mid=%.2f | attempt %d/%d",
            self.STRATEGY_NAME,
            side.value,
            entry_price,
            stop_price,
            target_price,
            self._orb_high,
            self._orb_low,
            self._orb_midline,
            self._trade_count + 1,
            ORB_TRADES_PER_DAY,
        )

        trade = self.order_manager.submit_entry(
            side=side,
            stop_price=stop_price,
            target_price=target_price,
            entry_price=entry_price,
            strategy=self.STRATEGY_NAME,
            regime=RegimeState.DIRECTIONAL,
            metadata=metadata,
        )

        self._trade_count += 1

        if trade is not None:
            logger.info(
                "[%s] Entry FILLED | trade_id=%s fill=%.2f side=%s",
                self.STRATEGY_NAME,
                trade.trade_id,
                trade.fill_price,
                side.value,
            )
        else:
            logger.warning(
                "[%s] Entry REJECTED / TIMED OUT | side=%s attempt=%d/%d",
                self.STRATEGY_NAME,
                side.value,
                self._trade_count,
                ORB_TRADES_PER_DAY,
            )

    def _handle_eod_close(self, bar: Bar) -> None:
        """Close any open position at end of day.

        The OrderManager / circuit-breaker layer should also enforce this,
        but we log the intent here for the strategy audit trail.
        """
        if self.state.open_position is not None:
            logger.info(
                "[%s] EOD_CLOSE reached — requesting position flatten.",
                self.STRATEGY_NAME,
            )
            # The main event loop / OrderManager is responsible for the
            # actual flatten call.  We log the signal here for OBS.

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def _orb_midline(self) -> float:
        """Midpoint of the opening range (structural stop level)."""
        return (self._orb_high + self._orb_low) / 2.0

    @property
    def orb_high(self) -> float:
        return self._orb_high

    @property
    def orb_low(self) -> float:
        return self._orb_low

    @property
    def orb_defined(self) -> bool:
        return self._orb_defined

    @property
    def trade_count(self) -> int:
        return self._trade_count

    # ── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _to_et(ts: datetime) -> datetime:
        """Ensure a timestamp is in US/Eastern."""
        if ts.tzinfo is None:
            return ET.localize(ts)
        return ts.astimezone(ET)
