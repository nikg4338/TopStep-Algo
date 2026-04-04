"""
session_manager.py — Daily lifecycle orchestrator for Algorithmic Futures.

Ties all modules together and manages the full trading day lifecycle
using APScheduler for timed events. All times are US/Eastern.

Daily Schedule:
  18:30 prev eve  — Nightly HMM refit (fetch OHLCV+VIX, refit, save)
  09:30           — Pre-market init  (load regime, reset breakers, connect WS)
  09:30           — RTH open         (reset indicators, start strategy)
  09:30–09:45     — ORB window       (State 1: record bars for opening range)
  09:45           — ORB armed        (State 1: begin monitoring for breakout)
  09:30–16:00     — Active trading   (execute signals, enforce breakers)
  16:05           — EOD hard close   (flatten all, record EOD balance)
  16:10           — Daily log        (write session summary, trade log)
  18:30           — HMM refit        (repeat nightly cycle)
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import (
    ACCOUNT_MODE,
    EOD_CLOSE,
    INSTRUMENT,
    LOG_DIR,
    MAX_LOSS_LIMIT,
    NIGHTLY_REFIT_TIME,
    PROFIT_TARGET,
    RISK_PER_TRADE,
    RTH_OPEN,
    STATE_FILE,
    TIMEZONE,
    VWAP_BAR_INTERVAL_MIN,
)
from data.indicators import FeatureBuilder
from data.market_data import Bar, DailyDataProvider, IntradayBarAggregator
from execution.api_client import ProjectXClient
from execution.circuit_breakers import CircuitBreakers
from execution.order_manager import OrderManager, SessionState
from regime.hmm_classifier import RegimeClassifier
from regime.regime_state import ChallengeStatus, RegimeState
from risk.position_sizer import PositionSizer

logger = logging.getLogger(__name__)

ET = pytz.timezone(TIMEZONE)


def _parse_time(t: str) -> time:
    """Parse an 'HH:MM' string into a ``datetime.time``."""
    parts = t.split(":")
    return time(int(parts[0]), int(parts[1]))


class SessionManager:
    """Top-level orchestrator — owns every sub-module and drives the daily schedule."""

    def __init__(self) -> None:
        # ── Core infrastructure ─────────────────────────────────────────
        self.api = ProjectXClient()
        self.breakers = CircuitBreakers(account_mode=ACCOUNT_MODE)
        self.sizer = PositionSizer(risk_per_trade=RISK_PER_TRADE)
        self.state = SessionState()
        self.order_manager = OrderManager(
            api=self.api,
            breakers=self.breakers,
            sizer=self.sizer,
            state=self.state,
        )

        # ── Regime / HMM ───────────────────────────────────────────────
        self.classifier = RegimeClassifier()
        self.daily_data = DailyDataProvider()

        # ── Bar aggregator (feeds strategies) ───────────────────────────
        self.bar_aggregator = IntradayBarAggregator(
            interval_minutes=VWAP_BAR_INTERVAL_MIN,
            on_bar_callback=self._on_bar,
        )

        # ── Strategies (lazily initialised; import at runtime) ──────────
        self._vwap_strategy: Any = None
        self._orb_strategy: Any = None

        # ── Scheduler ──────────────────────────────────────────────────
        self._scheduler: AsyncIOScheduler | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._last_market_price: float | None = None

        logger.info("SessionManager initialised")

    # ═══════════════════════════════════════════════════════════════════
    #  Public interface
    # ═══════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """Load persisted state, set up the APScheduler, and enter the event loop."""
        logger.info("=" * 60)
        logger.info("  Algorithmic Futures — SessionManager START")
        logger.info("=" * 60)

        # Restore previous session state (survives restarts)
        self._load_persisted_state()

        # Check if the challenge is already over
        if self.state.challenge_status != ChallengeStatus.IN_PROGRESS.value:
            logger.warning(
                "Challenge already %s — not starting scheduler",
                self.state.challenge_status,
            )
            return

        # Build the scheduler
        self._scheduler = AsyncIOScheduler(timezone=ET)
        self._schedule_daily_events()
        self._scheduler.start()

        self._running = True

        # Enter the asyncio event loop
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Register OS signals for clean shutdown
            assert self._loop is not None
            _loop = self._loop
            for sig in (signal.SIGINT, signal.SIGTERM):
                _loop.add_signal_handler(
                    sig, lambda lp=_loop: lp.call_soon_threadsafe(self._signal_shutdown)
                )

            logger.info("Event loop running — waiting for scheduled events")
            self._loop.run_forever()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        """Graceful shutdown: flatten positions, save state, stop scheduler."""
        if not self._running:
            return
        self._running = False

        logger.info("SessionManager shutting down…")

        # Safety flatten
        try:
            if self.state.open_position is not None:
                logger.warning("Open position detected during shutdown — flattening")
                self.order_manager.flatten_all(reason="SHUTDOWN")
        except Exception:
            logger.exception("Error flattening during shutdown")

        # Persist state
        try:
            self.order_manager.save_state()
            logger.info("State saved")
        except Exception:
            logger.exception("Error saving state during shutdown")

        # Stop scheduler
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

        # Stop event loop
        if self._loop and self._loop.is_running():
            self._loop.stop()

        logger.info("SessionManager shutdown complete")

    # ═══════════════════════════════════════════════════════════════════
    #  Scheduled lifecycle events
    # ═══════════════════════════════════════════════════════════════════

    def _schedule_daily_events(self) -> None:
        """Register all daily cron jobs on the scheduler."""
        assert self._scheduler is not None, "Scheduler not initialised"
        rth = _parse_time(RTH_OPEN)
        eod = _parse_time(EOD_CLOSE)
        refit = _parse_time(NIGHTLY_REFIT_TIME)

        # Pre-market / RTH open  (09:30 ET, Mon–Fri)
        self._scheduler.add_job(
            self._pre_market_init,
            CronTrigger(hour=rth.hour, minute=rth.minute, day_of_week="mon-fri", timezone=ET),
            id="pre_market_init",
            name="Pre-Market Init + RTH Open",
            replace_existing=True,
        )

        # EOD hard close (16:05 ET, Mon–Fri)
        self._scheduler.add_job(
            self._eod_close,
            CronTrigger(hour=eod.hour, minute=eod.minute, day_of_week="mon-fri", timezone=ET),
            id="eod_close",
            name="EOD Hard Close",
            replace_existing=True,
        )

        # Daily log (16:10 ET, Mon–Fri)
        self._scheduler.add_job(
            self._daily_log,
            CronTrigger(hour=16, minute=10, day_of_week="mon-fri", timezone=ET),
            id="daily_log",
            name="Daily Log",
            replace_existing=True,
        )

        # Nightly HMM refit (18:30 ET, Mon–Fri)
        self._scheduler.add_job(
            self._nightly_refit,
            CronTrigger(hour=refit.hour, minute=refit.minute, day_of_week="mon-fri", timezone=ET),
            id="nightly_refit",
            name="Nightly HMM Refit",
            replace_existing=True,
        )

        logger.info(
            "Scheduled jobs: pre_market=%s, eod=%s, daily_log=16:10, refit=%s",
            RTH_OPEN, EOD_CLOSE, NIGHTLY_REFIT_TIME,
        )

    # ── Pre-Market Init (09:30 ET) ──────────────────────────────────────

    async def _pre_market_init(self) -> None:
        """Load regime, reset breakers, connect API, start market data, arm strategies."""
        logger.info("━" * 50)
        logger.info("PRE-MARKET INIT — %s", datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"))
        logger.info("━" * 50)

        try:
            # 1. Authenticate with broker
            self.api.authenticate()
            self._reconcile_broker_state()
            balance = self.api.get_account_balance()
            self.state.account_balance = balance.balance
            if self.state.account_high_water_mark == 0.0:
                self.state.account_high_water_mark = balance.balance
            logger.info("Account balance: $%.2f  HWM: $%.2f",
                        self.state.account_balance, self.state.account_high_water_mark)

            # 2. Load regime from last nightly refit
            regime = RegimeClassifier.load_regime()
            self.state.current_regime = regime.value
            logger.info("Loaded regime: %s", regime.name)

            # 3. Reset daily counters
            self.state.daily_pnl = 0.0
            self.state.daily_trade_count = 0
            self.state.date = datetime.now(ET).strftime("%Y-%m-%d")

            # 4. Reset circuit breakers
            self.breakers.reset()

            # 5. Reset bar aggregator and indicators
            self.bar_aggregator.reset()

            # 6. Initialise strategies based on regime
            self._init_strategies(regime)

            # 7. Check challenge status
            self._check_challenge_complete()
            if self.state.challenge_status != ChallengeStatus.IN_PROGRESS.value:
                logger.warning("Challenge %s — skipping trading today", self.state.challenge_status)
                return

            # 8. Start WebSocket market data stream
            asyncio.ensure_future(self._start_market_data())

            logger.info("PRE-MARKET INIT complete — regime=%s, ready for trading", regime.name)

        except Exception:
            logger.exception("PRE-MARKET INIT FAILED — no trading today")

    # ── Strategy Initialisation ─────────────────────────────────────────

    def _init_strategies(self, regime: RegimeState) -> None:
        """Create / reset strategy instances based on current regime."""
        try:
            from strategies import VWAPMeanReversion, ORBBreakout  # type: ignore[import-untyped]

            self._vwap_strategy = VWAPMeanReversion(
                order_manager=self.order_manager,
                state=self.state,
            )
            self._orb_strategy = ORBBreakout(
                order_manager=self.order_manager,
                state=self.state,
            )
            self._vwap_strategy.reset()
            self._orb_strategy.reset()

            logger.info(
                "Strategies initialised: VWAP (State 0), ORB (State 1), CRISIS → skip (State 2)"
            )
        except ImportError:
            logger.warning(
                "Strategy modules not yet available — bar dispatch will be no-op. "
                "Build strategies/ in parallel."
            )
            self._vwap_strategy = None
            self._orb_strategy = None

    # ── Market Data Stream ──────────────────────────────────────────────

    async def _start_market_data(self) -> None:
        """Subscribe to WebSocket ticks and route them through the bar aggregator."""
        logger.info("Subscribing to market data for %s…", INSTRUMENT)
        try:
            await self.api.subscribe_market_data(
                symbol=INSTRUMENT,
                on_tick=self._on_tick,
                on_bar=None,  # we build our own bars via IntradayBarAggregator
                on_order_update=self._on_order_update,
            )
        except Exception:
            logger.exception("Market data stream terminated — attempting emergency flatten")
            try:
                self.order_manager.flatten_all_with_price(
                    reason="WS_DISCONNECT",
                    exit_price=self._last_market_price,
                )
            except Exception:
                logger.exception("Emergency flatten failed after WS disconnect")

    def _on_tick(self, tick_msg: dict) -> None:
        """Process a raw tick message from the WebSocket."""
        try:
            ts_raw = tick_msg.get("timestamp") or tick_msg.get("t")
            price = float(tick_msg.get("price") or tick_msg.get("p", 0))
            size = float(tick_msg.get("size") or tick_msg.get("s", 1))

            if price <= 0:
                return

            self._last_market_price = price

            # Parse timestamp (ISO or epoch)
            if isinstance(ts_raw, (int, float)):
                timestamp = datetime.fromtimestamp(ts_raw, tz=ET)
            elif isinstance(ts_raw, str):
                timestamp = datetime.fromisoformat(ts_raw)
                if timestamp.tzinfo is None:
                    timestamp = ET.localize(timestamp)
            else:
                timestamp = datetime.now(ET)

            self.bar_aggregator.on_tick(timestamp, price, size)

        except Exception:
            logger.exception("Error processing tick: %s", tick_msg)

    def _on_order_update(self, msg: dict) -> None:
        """Process a broker order/fill update from the WebSocket."""
        try:
            self.order_manager.handle_order_update(msg)
        except Exception:
            logger.exception("Error processing order update: %s", msg)

    # ── Bar Dispatch (strategy routing) ─────────────────────────────────

    def _on_bar(self, bar: Bar) -> None:
        """Route a completed bar to the active strategy based on current regime.

        BALANCED (State 0)     → VWAP Mean Reversion
        DIRECTIONAL (State 1)  → ORB Breakout
        CRISIS (State 2)       → No trading
        """
        regime = RegimeState(self.state.current_regime)
        now = datetime.now(ET)

        # Enforce EOD cutoff — no processing after close
        eod_time = _parse_time(EOD_CLOSE)
        if now.time() >= eod_time:
            return

        logger.debug(
            "Bar: %s O=%.2f H=%.2f L=%.2f C=%.2f V=%.0f | regime=%s",
            bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume,
            regime.name,
        )

        try:
            if regime == RegimeState.BALANCED:
                if self._vwap_strategy is not None:
                    self._vwap_strategy.on_bar(bar, regime)
                else:
                    logger.debug("VWAP strategy not loaded — skipping bar")

            elif regime == RegimeState.DIRECTIONAL:
                if self._orb_strategy is not None:
                    self._orb_strategy.on_bar(bar)
                else:
                    logger.debug("ORB strategy not loaded — skipping bar")

            elif regime == RegimeState.CRISIS:
                logger.debug("CRISIS regime — no trading, bar ignored")

            else:
                logger.warning("Unknown regime %s — skipping bar", regime)

        except Exception:
            logger.exception("Error in strategy on_bar dispatch (regime=%s)", regime.name)

        # Persist state after every bar (trade may have occurred)
        try:
            self.order_manager.save_state()
        except Exception:
            logger.exception("Error saving state after bar")

    # ── EOD Hard Close (16:05 ET) ───────────────────────────────────────

    async def _eod_close(self) -> None:
        """Flatten all positions and flush the bar aggregator."""
        logger.info("━" * 50)
        logger.info("EOD HARD CLOSE — %s", datetime.now(ET).strftime("%H:%M ET"))
        logger.info("━" * 50)

        try:
            # Flush any incomplete bar
            self.bar_aggregator.flush()

            # Flatten everything
            self.order_manager.flatten_all_with_price(
                reason="EOD_CLOSE",
                exit_price=self._last_market_price,
            )

            # Record EOD balance
            try:
                balance = self.api.get_account_balance()
                self.state.account_balance = balance.balance
                if balance.balance > self.state.account_high_water_mark:
                    self.state.account_high_water_mark = balance.balance
                logger.info("EOD balance: $%.2f", balance.balance)
            except Exception:
                logger.exception("Failed to fetch EOD balance — using tracked value")

            # Check challenge completion
            self._check_challenge_complete()

            # Save state
            self.order_manager.save_state()

            logger.info("EOD CLOSE complete — daily P&L: $%.2f, cumulative: $%.2f",
                        self.state.daily_pnl, self.state.cumulative_pnl)

        except Exception:
            logger.exception("EOD CLOSE encountered errors")

    # ── Daily Log (16:10 ET) ────────────────────────────────────────────

    async def _daily_log(self) -> None:
        """Write session summary and trade log to disk."""
        logger.info("Writing daily log…")

        try:
            # Write trade log
            self.order_manager.write_trade_log()

            # Write daily summary
            summary = self.order_manager.daily_summary()
            summary["challenge_status"] = self.state.challenge_status

            log_dir = Path(LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)

            today = datetime.now(ET).strftime("%Y-%m-%d")
            summary_file = log_dir / f"summary_{today}.json"
            summary_file.write_text(json.dumps(summary, indent=2))

            logger.info("Daily summary written to %s", summary_file)
            logger.info("Summary: %s", json.dumps(summary, indent=2))

        except Exception:
            logger.exception("Error writing daily log")

    # ── Nightly HMM Refit (18:30 ET) ───────────────────────────────────

    async def _nightly_refit(self) -> None:
        """Fetch daily data, build features, refit HMM, predict next regime, persist."""
        logger.info("━" * 50)
        logger.info("NIGHTLY HMM REFIT — %s", datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"))
        logger.info("━" * 50)

        try:
            # 1. Fetch daily OHLCV + VIX data
            logger.info("Fetching daily data…")
            daily_ohlcv, vix_close, vix3m_close = self.daily_data.fetch()

            # 2. Build feature matrix
            logger.info("Building HMM features…")
            feature_df = FeatureBuilder.build(daily_ohlcv, vix_close, vix3m_close)
            logger.info("Feature matrix: %d rows, %d features", len(feature_df), len(feature_df.columns))

            # 3. Fit HMM
            logger.info("Fitting HMM classifier…")
            self.classifier.fit(feature_df)

            # 4. Predict next session regime
            next_regime = self.classifier.predict_next_regime(feature_df)
            logger.info("Predicted next regime: %s", next_regime.name)

            # 5. Save regime to state file
            tomorrow = (datetime.now(ET) + timedelta(days=1)).strftime("%Y-%m-%d")
            self.classifier.save_regime(next_regime, target_date=tomorrow)

            # 6. Log diagnostics
            diag = self.classifier.diagnostics()
            logger.info("HMM diagnostics: %s", json.dumps(diag, indent=2, default=str))

            logger.info("NIGHTLY REFIT complete — next regime: %s", next_regime.name)

        except Exception:
            logger.exception(
                "NIGHTLY REFIT FAILED — next session will default to CRISIS (no-trade)"
            )

    # ═══════════════════════════════════════════════════════════════════
    #  Challenge tracking
    # ═══════════════════════════════════════════════════════════════════

    def _check_challenge_complete(self) -> None:
        """Evaluate whether the challenge has been passed or failed."""
        prev_status = self.state.challenge_status

        # Win condition: cumulative P&L reaches profit target
        if self.state.cumulative_pnl >= PROFIT_TARGET:
            self.state.challenge_status = ChallengeStatus.PASSED.value
            if prev_status != ChallengeStatus.PASSED.value:
                logger.info(
                    "🏆 CHALLENGE PASSED — cumulative P&L $%.2f ≥ target $%d",
                    self.state.cumulative_pnl, PROFIT_TARGET,
                )

        # Lose condition: trailing drawdown from high-water mark reached MLL
        elif (
            self.state.account_high_water_mark > 0
            and (self.state.account_high_water_mark - self.state.account_balance) >= MAX_LOSS_LIMIT
        ):
            self.state.challenge_status = ChallengeStatus.FAILED.value
            if prev_status != ChallengeStatus.FAILED.value:
                logger.critical(
                    "CHALLENGE FAILED — account balance $%.2f, HWM $%.2f, drawdown $%.2f ≥ MLL $%d",
                    self.state.account_balance,
                    self.state.account_high_water_mark,
                    self.state.account_high_water_mark - self.state.account_balance,
                    MAX_LOSS_LIMIT,
                )

    # ═══════════════════════════════════════════════════════════════════
    #  State persistence
    # ═══════════════════════════════════════════════════════════════════

    def _load_persisted_state(self) -> None:
        """Reload session state from disk if available."""
        state_path = Path(STATE_FILE)
        if state_path.exists():
            try:
                self.order_manager.load_state()
                # Re-bind state reference (load_state replaces the object)
                self.state = self.order_manager.state
                logger.info(
                    "Restored persisted state: date=%s, cumulative_pnl=$%.2f, status=%s",
                    self.state.date, self.state.cumulative_pnl, self.state.challenge_status,
                )
            except Exception:
                logger.exception("Failed to load persisted state — starting fresh")
        else:
            logger.info("No persisted state found at %s — starting fresh", state_path)

    def _reconcile_broker_state(self) -> None:
        """Normalise persisted state against live broker state before trading."""
        try:
            result = self.order_manager.reconcile_with_broker()
            logger.info("Broker reconciliation complete: %s", json.dumps(result, default=str))
        except Exception:
            logger.exception("Broker reconciliation failed; flattening all positions as safety fallback")
            try:
                self.api.close_all_positions()
            except Exception:
                logger.exception("Safety flatten failed during reconciliation")

    # ═══════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ═══════════════════════════════════════════════════════════════════

    def _signal_shutdown(self) -> None:
        """Handle OS signal by scheduling clean shutdown."""
        logger.info("Shutdown signal received")
        self.stop()
