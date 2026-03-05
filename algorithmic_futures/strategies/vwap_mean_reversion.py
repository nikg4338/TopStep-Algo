"""
strategies/vwap_mean_reversion.py — VWAP Mean Reversion strategy for MES futures.

Activated when HMM regime is State 0 (BALANCED).  Enters at ±2.5σ–3.0σ
VWAP bands with CVD divergence confirmation, targets the VWAP baseline,
and uses a 1.5× ATR stop beyond the entry extreme.

Entry rules
-----------
1. HMM must output State 0 (BALANCED).
2. VWAP anchored to RTH open (09:30 ET), recalculated every 5-min bar.
3. Price must be in the ±2.5σ to ±3.0σ band zone — no entries inside ±2.5σ.
4. CVD divergence must confirm:
   - Lower band → BULLISH divergence (price lower-low + CVD higher-low).
   - Upper band → BEARISH divergence (price higher-high + CVD lower-high).
5. No new entries within 15 minutes of the 16:05 ET close (cutoff 15:50 ET).
6. Max 3 trade attempts per session.

Exit rules
----------
- Profit target = current VWAP value (centre band).
- Stop loss = 1.5× 5-min ATR beyond entry extreme.
- Time stop: flatten at EOD_CLOSE.
"""

from __future__ import annotations

import logging
from datetime import datetime, time as dt_time

import pytz

from config import (
    EOD_CLOSE,
    LAST_ENTRY_CUTOFF,
    TIMEZONE,
    VWAP_BAR_INTERVAL_MIN,
    VWAP_SD_ENTRY_MAX,
    VWAP_SD_ENTRY_MIN,
    VWAP_STOP_ATR_MULT,
    VWAP_TRADES_PER_DAY,
)
from data.indicators import ATRCalculator, CVDCalculator, VWAPCalculator
from data.market_data import Bar
from execution.order_manager import OrderManager, SessionState, TradeRecord
from regime.regime_state import OrderSide, RegimeState

logger = logging.getLogger(__name__)

ET = pytz.timezone(TIMEZONE)

# Pre-parse cutoff / close times once at import
_CUTOFF_TIME: dt_time = dt_time(
    *map(int, LAST_ENTRY_CUTOFF.split(":"))
)  # 15:50 ET
_CLOSE_TIME: dt_time = dt_time(
    *map(int, EOD_CLOSE.split(":"))
)  # 16:05 ET

STRATEGY_NAME = "VWAP_MR"


class VWAPMeanReversion:
    """VWAP Mean Reversion strategy — activated in BALANCED regime (State 0).

    Parameters
    ----------
    order_manager : OrderManager
        Pre-wired order execution manager.
    state : SessionState
        Shared mutable session state (daily P&L, trade count, open position).
    """

    def __init__(
        self,
        order_manager: OrderManager,
        state: SessionState,
    ) -> None:
        self.order_manager = order_manager
        self.state = state

        # Indicators
        self._vwap = VWAPCalculator()
        self._atr = ATRCalculator(period=14)
        self._cvd = CVDCalculator()

        # Bar history for CVD divergence detection
        self._bar_lows: list[float] = []
        self._bar_highs: list[float] = []

        # Session-level trade counter (independent of SessionState.daily_trade_count
        # which tracks all strategies; this tracks VWAP-specific attempts).
        self._vwap_trade_count: int = 0

    # ── Session lifecycle ───────────────────────────────────────────────

    def reset(self) -> None:
        """Call at session open (RTH 09:30 ET) to re-anchor all state."""
        self._vwap.reset()
        self._atr.reset()
        self._cvd.reset()
        self._bar_lows.clear()
        self._bar_highs.clear()
        self._vwap_trade_count = 0
        logger.info("%s: session reset — VWAP re-anchored", STRATEGY_NAME)

    # ── Main bar handler ────────────────────────────────────────────────

    def on_bar(self, bar: Bar, current_regime: RegimeState) -> TradeRecord | None:
        """Process a completed 5-min bar.

        This is the primary entry point called by the orchestrator on every
        bar close.  Returns a ``TradeRecord`` if a trade was placed and
        filled, otherwise ``None``.

        Parameters
        ----------
        bar : Bar
            Completed 5-minute OHLCV bar.
        current_regime : RegimeState
            Current HMM-classified regime.
        """
        # 1. Always update indicators regardless of regime (keeps state warm)
        vwap_state = self._vwap.update(bar.high, bar.low, bar.close, bar.volume)
        atr_value = self._atr.update(bar.high, bar.low, bar.close)
        self._cvd.update_bar(bar.open, bar.close, bar.volume)
        self._bar_lows.append(bar.low)
        self._bar_highs.append(bar.high)

        # 2. Gate: regime must be BALANCED
        if current_regime != RegimeState.BALANCED:
            return None

        # 3. Gate: no entry if already holding a position
        if self.state.open_position is not None:
            return None

        # 4. Gate: max VWAP trades per session
        if self._vwap_trade_count >= VWAP_TRADES_PER_DAY:
            return None

        # 5. Gate: time-based cutoff — no entries within 15 min of close
        bar_time_et = self._to_et(bar.timestamp)
        if bar_time_et.time() >= _CUTOFF_TIME:
            return None

        # 6. Gate: ensure indicators have enough data
        if vwap_state.bar_count < 2 or atr_value <= 0:
            return None

        # 7. Determine if price is at a VWAP SD extreme
        price = bar.close
        at_lower = self._vwap.is_at_lower_extreme(price)
        at_upper = self._vwap.is_at_upper_extreme(price)

        if not at_lower and not at_upper:
            return None

        # 8. CVD divergence confirmation
        divergence = self._cvd.detect_divergence(
            self._bar_lows, self._bar_highs
        )

        if at_lower and divergence != "BULLISH":
            logger.debug(
                "%s: price at lower extreme (%.2f) but no BULLISH divergence — skipping",
                STRATEGY_NAME, price,
            )
            return None

        if at_upper and divergence != "BEARISH":
            logger.debug(
                "%s: price at upper extreme (%.2f) but no BEARISH divergence — skipping",
                STRATEGY_NAME, price,
            )
            return None

        # ── All conditions met — compute order parameters ───────────────
        side = OrderSide.BUY if at_lower else OrderSide.SELL
        target_price = vwap_state.vwap  # profit target = VWAP baseline
        stop_distance = VWAP_STOP_ATR_MULT * atr_value

        if side == OrderSide.BUY:
            # Long entry at lower extreme — stop below the entry low
            stop_price = price - stop_distance
        else:
            # Short entry at upper extreme — stop above the entry high
            stop_price = price + stop_distance

        # Determine which SD band the price is in for logging
        sd_band = self._compute_sd_level(price, vwap_state)

        metadata = {
            "sd_band": round(sd_band, 3),
            "cvd_divergence": divergence,
            "atr_5min": round(atr_value, 4),
            "vwap": round(vwap_state.vwap, 2),
            "vwap_std": round(vwap_state.std_dev, 4),
            "bar_timestamp": str(bar.timestamp),
            "bar_count": vwap_state.bar_count,
        }

        logger.info(
            "%s SIGNAL: side=%s price=%.2f vwap=%.2f sd=%.3fσ "
            "cvd=%s atr=%.4f stop=%.2f target=%.2f",
            STRATEGY_NAME,
            side.value,
            price,
            vwap_state.vwap,
            sd_band,
            divergence,
            atr_value,
            stop_price,
            target_price,
        )

        # ── Submit to order manager ─────────────────────────────────────
        trade = self.order_manager.submit_entry(
            side=side,
            stop_price=stop_price,
            target_price=target_price,
            entry_price=price,
            strategy=STRATEGY_NAME,
            regime=RegimeState.BALANCED,
            metadata=metadata,
        )

        if trade is not None:
            self._vwap_trade_count += 1
            logger.info(
                "%s ENTRY FILLED: trade_id=%s side=%s qty=%d entry=%.2f "
                "stop=%.2f target=%.2f | session trade #%d/%d",
                STRATEGY_NAME,
                trade.trade_id,
                trade.side,
                trade.qty,
                trade.fill_price,
                trade.stop_price,
                trade.target_price,
                self._vwap_trade_count,
                VWAP_TRADES_PER_DAY,
            )
        else:
            logger.warning(
                "%s: order submission returned None (rejected or timeout)",
                STRATEGY_NAME,
            )

        return trade

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _to_et(ts: datetime) -> datetime:
        """Ensure a timestamp is localized to US/Eastern."""
        if ts.tzinfo is None:
            return ET.localize(ts)
        return ts.astimezone(ET)

    @staticmethod
    def _compute_sd_level(price: float, vwap_state) -> float:
        """Return the signed SD distance of *price* from VWAP.

        Positive = above VWAP, negative = below.  Returns 0.0 if std_dev
        is zero to avoid division-by-zero.
        """
        if vwap_state.std_dev == 0:
            return 0.0
        return (price - vwap_state.vwap) / vwap_state.std_dev

    @property
    def vwap_trade_count(self) -> int:
        """Number of VWAP trades placed this session."""
        return self._vwap_trade_count
