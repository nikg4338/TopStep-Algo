"""
simulation/mr_exit_simulator.py — Deterministic mean-reversion exit simulator.

Replays approved MR signals against 5-minute bars to produce a complete
trade log with PnL, MAE/MFE, and exit-reason attribution.  Designed for
offline research; no broker interaction.

Key assumptions (documented inline):
  - Entry at the open of the NEXT bar after the signal timestamp.
  - Stop is checked before target within the same bar (conservative).
  - No slippage by default (configurable via ExitSimConfig.slippage_ticks).
  - ATR is computed from the same bar series used for simulation.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from data.databento_provider import DatabentoReplayProvider
from data.indicators import ATRCalculator
from data.market_data import Bar, IntradayBarAggregator

import config as _cfg

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ExitSimConfig:
    """Tunable knobs for the exit simulator.

    ``entry_mode``
        "next_bar_open" — enter at the open of the first bar whose
        timestamp is strictly after the signal.  "next_tick" is reserved
        for a future version and currently falls back to next_bar_open.

    ``stop_mode``
        "atr" — stop placed at ``atr_stop_mult × ATR`` beyond entry.

    ``target_mode``
        "vwap_at_signal" — target is the VWAP value recorded in
        ``signals.csv`` at signal time (the MR trade's mean-reversion
        destination).
        "fixed_r_multiple" — target placed at ``target_r_multiple × risk``
        from entry.
    """

    entry_mode: str = "next_bar_open"
    stop_mode: str = "atr"
    atr_stop_mult: float = _cfg.MR_EXIT_SIM_STOP_ATR_MULT
    target_mode: str = "vwap_at_signal"
    target_r_multiple: float = _cfg.MR_EXIT_SIM_FIXED_R_MULT
    time_stop_bars: int = _cfg.MR_EXIT_SIM_TIME_STOP_BARS
    session_cutoff_time: str = _cfg.MR_EXIT_SIM_SESSION_CUTOFF
    tick_value: float = 1.25
    point_value: float = 5.00
    tick_size: float = 0.25
    slippage_ticks: int = _cfg.MR_EXIT_SIM_SLIPPAGE_TICKS
    runner_enabled: bool = _cfg.MR_EXIT_SIM_RUNNER_ENABLED
    runner_primary_pct: float = _cfg.MR_EXIT_SIM_RUNNER_PRIMARY_PCT
    runner_target_r: float = _cfg.MR_EXIT_SIM_RUNNER_TARGET_R
    runner_trail_r: float = _cfg.MR_EXIT_SIM_RUNNER_TRAIL_R
    runner_step_enabled: bool = _cfg.MR_EXIT_SIM_RUNNER_STEP_ENABLED
    runner_step_trigger_r: float = _cfg.MR_EXIT_SIM_RUNNER_STEP_TRIGGER_R
    runner_step_lock_r: float = _cfg.MR_EXIT_SIM_RUNNER_STEP_LOCK_R


# ═══════════════════════════════════════════════════════════════════════
#  Trade Record
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SimulatedTrade:
    """Full record for one simulated round-trip."""

    trade_id: str
    session_id: str
    signal_timestamp: str        # ISO-8601
    side: str                    # "BUY" or "SELL"
    entry_timestamp: str
    entry_price: float
    stop_price: float
    target_price: float
    exit_timestamp: str
    exit_price: float
    exit_reason: str             # target | stop | time_stop | session_cutoff | replay_end
    pnl_points: float
    pnl_dollars: float
    pnl_r: float
    mae_points: float            # max adverse excursion (points)
    mfe_points: float            # max favourable excursion (points)
    hold_minutes: float
    hold_bars: int
    regime_at_entry: str
    sigma_band_level: float


# ═══════════════════════════════════════════════════════════════════════
#  Simulator
# ═══════════════════════════════════════════════════════════════════════


class MRExitSimulator:
    """Walk-forward bar-by-bar exit simulator for mean-reversion signals.

    Workflow
    --------
    1. Load approved MR signals from ``signals.csv``.
    2. Compute a rolling ATR series from the bar list.
    3. For each signal, find the entry bar, set stop & target, then walk
       forward checking exit conditions on every subsequent bar.
    4. Record MAE / MFE, PnL, and exit reason.
    """

    def __init__(self, config: ExitSimConfig | None = None) -> None:
        self.config = config or ExitSimConfig()
        self.last_run_diagnostics: dict[str, int] = {
            "signals_received": 0,
            "entries_opened": 0,
            "trades_emitted": 0,
            "forced_replay_exits": 0,
            "skipped_invalid_levels": 0,
            "skipped_no_entry_bar": 0,
        }

    # ── public API ───────────────────────────────────────────────────────

    def simulate_session(
        self,
        signals_csv_path: str,
        bars: list[Bar],
    ) -> list[SimulatedTrade]:
        """Run the simulator for one session.

        Parameters
        ----------
        signals_csv_path : str
            Path to ``signals.csv`` produced by the replay system.
        bars : list[Bar]
            Chronologically sorted 5-minute bars covering the session.

        Returns
        -------
        list[SimulatedTrade]
        """
        if not bars:
            logger.warning("No bars supplied — nothing to simulate")
            return []

        signals = self._load_signals(signals_csv_path)
        if signals.empty:
            logger.info("No approved MR signals in %s", signals_csv_path)
            return []

        # Pre-compute ATR at every bar index
        atr_by_idx = self._compute_atr_series(bars)

        signals_received = len(signals)
        entries_opened = 0
        trades_emitted = 0
        forced_replay_exits = 0
        skipped_invalid_levels = 0
        skipped_no_entry_bar = 0

        trades: list[SimulatedTrade] = []
        for _, sig in signals.iterrows():
            trade, status = self._simulate_one(sig, bars, atr_by_idx)
            if status == "no_entry_bar":
                skipped_no_entry_bar += 1
                continue
            if status == "invalid_levels":
                skipped_invalid_levels += 1
                continue
            entries_opened += 1
            if trade is not None:
                trades.append(trade)
                trades_emitted += 1
                if trade.exit_reason == "replay_end":
                    forced_replay_exits += 1

        self.last_run_diagnostics = {
            "signals_received": signals_received,
            "entries_opened": entries_opened,
            "trades_emitted": trades_emitted,
            "forced_replay_exits": forced_replay_exits,
            "skipped_invalid_levels": skipped_invalid_levels,
            "skipped_no_entry_bar": skipped_no_entry_bar,
        }

        print("EXIT SIM DEBUG")
        print("signals_received =", signals_received)
        print("entries_opened =", entries_opened)
        print("trades_emitted =", trades_emitted)
        print("forced_replay_exits =", forced_replay_exits)
        print("skipped_invalid_levels =", skipped_invalid_levels)

        logger.info(
            "Simulated %d trades from %d approved signals",
            len(trades),
            len(signals),
        )
        return trades

    def simulate_from_replay_artifacts(
        self,
        session_dir: str,
        replay_start: str,
        replay_end: str,
        symbol: str = "MES.c.0",
    ) -> list[SimulatedTrade]:
        """End-to-end convenience: fetch bars, simulate, write outputs.

        Parameters
        ----------
        session_dir : str
            Directory containing ``signals.csv``; outputs are written here.
        replay_start, replay_end : str
            ISO-8601 timestamps bounding the replay window.
        symbol : str
            Databento continuous-contract symbol.

        Returns
        -------
        list[SimulatedTrade]
        """
        signals_csv = os.path.join(session_dir, "signals.csv")
        if not os.path.isfile(signals_csv):
            raise FileNotFoundError(f"signals.csv not found in {session_dir}")

        # Replay ticks → bars
        provider = DatabentoReplayProvider()
        aggregator = IntradayBarAggregator(interval_minutes=5)
        provider.replay_trades(
            start=replay_start,
            end=replay_end,
            on_tick=aggregator.on_tick,
            symbol=symbol,
        )
        aggregator.flush()
        bars = aggregator.bars
        logger.info("Replayed %d bars for simulation", len(bars))

        # Simulate
        trades = self.simulate_session(signals_csv, bars)

        # Derive session_id from the first trade or from directory name
        session_id = trades[0].session_id if trades else Path(session_dir).name

        # Write artefacts
        trades_csv_path = os.path.join(session_dir, "trades.csv")
        summary_path = os.path.join(session_dir, "trade_summary.json")
        self.write_trades_csv(trades, trades_csv_path)
        self.write_trade_summary(trades, session_id, summary_path)

        return trades

    # ── I/O helpers ──────────────────────────────────────────────────────

    def write_trades_csv(
        self,
        trades: list[SimulatedTrade],
        output_path: str,
    ) -> None:
        """Serialize trade list to CSV.  Always creates the file (even if empty)."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if not trades:
            # Always create the file so downstream knows the sim ran
            fieldnames = [
                "trade_id", "session_id", "signal_timestamp", "side",
                "entry_timestamp", "entry_price", "stop_price", "target_price",
                "exit_timestamp", "exit_price", "exit_reason",
                "pnl_points", "pnl_dollars", "pnl_r",
                "mae_points", "mfe_points", "hold_minutes", "hold_bars",
                "regime_at_entry", "sigma_band_level",
            ]
            with open(output_path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
            print(f"  [exit_sim] writing trades.csv -> {output_path} (0 trades, header-only)")
            logger.info("Wrote header-only trades.csv to %s (no trades)", output_path)
            return

        fieldnames = list(asdict(trades[0]).keys())
        with open(output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for t in trades:
                writer.writerow(asdict(t))
        print(f"  [exit_sim] writing trades.csv -> {output_path} ({len(trades)} trades)")
        logger.info("Wrote %d trades to %s", len(trades), output_path)

    def write_trade_summary(
        self,
        trades: list[SimulatedTrade],
        session_id: str,
        output_path: str,
    ) -> None:
        """Write a JSON summary with aggregate statistics."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        summary = self._build_summary(trades, session_id)
        with open(output_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        logger.info("Wrote trade summary to %s", output_path)

    # ── internal: signal loading ─────────────────────────────────────────

    @staticmethod
    def _load_signals(csv_path: str) -> pd.DataFrame:
        """Load and filter for approved MR/ORB signals."""
        df = pd.read_csv(csv_path)

        # Normalise column names (strip whitespace, lowercase)
        df.columns = df.columns.str.strip().str.lower()

        # Filter: approved == True AND signal_type in {MR, ORB}
        # The CSV stores booleans as strings; handle both forms.
        approved_mask = df["approved"].astype(str).str.strip().str.lower() == "true"
        type_mask = df["signal_type"].astype(str).str.strip().str.upper().isin({"MR", "ORB"})
        filtered = df[approved_mask & type_mask].copy()

        # Parse timestamps
        filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], utc=True)
        filtered = filtered.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "Loaded %d approved MR/ORB signals from %s (total rows: %d)",
            len(filtered),
            csv_path,
            len(df),
        )
        return filtered

    # ── internal: ATR series ─────────────────────────────────────────────

    @staticmethod
    def _compute_atr_series(bars: list[Bar], period: int = 14) -> dict[int, float]:
        """Return {bar_index: ATR} for every bar in the list.

        Uses the same Wilder-smoothed ATR as the live system
        (``data.indicators.ATRCalculator``).
        """
        calc = ATRCalculator(period=period)
        atr_map: dict[int, float] = {}
        for idx, bar in enumerate(bars):
            atr_map[idx] = calc.update(bar.high, bar.low, bar.close)
        return atr_map

    # ── internal: single-trade simulation ────────────────────────────────

    def _simulate_one(
        self,
        signal: pd.Series,
        bars: list[Bar],
        atr_by_idx: dict[int, float],
    ) -> tuple[SimulatedTrade | None, str]:
        """Simulate a single signal through the bar series.

                Returns ``(trade, status)`` where status is one of:
                    - "emitted"
                    - "no_entry_bar"
                    - "invalid_levels"
        """
        cfg = self.config
        sig_ts = signal["timestamp"]  # already tz-aware datetime

        # -- Resolve entry_mode ------------------------------------------------
        # "next_tick" is not yet implemented; fall back to "next_bar_open".
        if cfg.entry_mode == "next_tick":
            logger.debug(
                "next_tick entry mode not implemented in v1; "
                "falling back to next_bar_open"
            )

        # Assumption: entry at the open of the first bar whose timestamp
        # is strictly after the signal timestamp.
        entry_idx: int | None = None
        for idx, bar in enumerate(bars):
            bar_ts = _ensure_utc(bar.timestamp)
            if bar_ts > sig_ts:
                entry_idx = idx
                break

        if entry_idx is None:
            logger.debug("No bar after signal at %s — skipping", sig_ts)
            return None, "no_entry_bar"

        entry_bar = bars[entry_idx]
        entry_price = entry_bar.open

        # Apply slippage (adverse direction)
        slippage_points = cfg.slippage_ticks * cfg.tick_size
        side = signal["side"].strip().upper()
        if side == "BUY":
            entry_price += slippage_points
        else:
            entry_price -= slippage_points

        # -- ATR at entry bar ---------------------------------------------------
        atr_at_entry = atr_by_idx.get(entry_idx, 0.0)
        if atr_at_entry <= 0:
            logger.warning(
                "ATR is zero at bar %d; stop will equal entry price", entry_idx
            )

        signal_type = str(signal.get("signal_type", "MR")).strip().upper()

        # -- Stop price ---------------------------------------------------------
        if signal_type == "ORB":
            stop_price = float(signal.get("stop_reference", float("nan")))
            if not math.isfinite(stop_price):
                if side == "BUY":
                    stop_price = entry_price - cfg.atr_stop_mult * atr_at_entry
                else:
                    stop_price = entry_price + cfg.atr_stop_mult * atr_at_entry
        else:
            # Assumption: stop_mode == "atr" (default MR mode).
            if side == "BUY":
                stop_price = entry_price - cfg.atr_stop_mult * atr_at_entry
            else:
                stop_price = entry_price + cfg.atr_stop_mult * atr_at_entry

        # -- Target price -------------------------------------------------------
        risk_points = abs(entry_price - stop_price)

        if signal_type == "ORB":
            target_price = float(signal.get("target_reference", signal.get("vwap", entry_price)))
        elif cfg.target_mode == "vwap_at_signal":
            # MR BUY: price is below VWAP → target is VWAP (above entry).
            # MR SELL: price is above VWAP → target is VWAP (below entry).
            target_price = float(signal.get("target_reference", signal.get("vwap", entry_price)))
        elif cfg.target_mode == "fixed_r_multiple":
            if side == "BUY":
                target_price = entry_price + cfg.target_r_multiple * risk_points
            else:
                target_price = entry_price - cfg.target_r_multiple * risk_points
        else:
            logger.warning("Unknown target_mode %r; using VWAP", cfg.target_mode)
            target_price = float(signal.get("vwap", entry_price))

        # Validate numeric and directional integrity.
        if not (
            math.isfinite(entry_price)
            and math.isfinite(stop_price)
            and math.isfinite(target_price)
        ):
            logger.warning(
                "Skipping signal at %s due to non-finite levels: entry=%s stop=%s target=%s",
                sig_ts,
                entry_price,
                stop_price,
                target_price,
            )
            return None, "invalid_levels"

        if side == "BUY":
            valid_direction = stop_price < entry_price and target_price > entry_price
        else:
            valid_direction = stop_price > entry_price and target_price < entry_price

        if not valid_direction:
            logger.warning(
                "Skipping signal at %s due to invalid stop/target direction: "
                "side=%s entry=%s stop=%s target=%s",
                sig_ts,
                side,
                entry_price,
                stop_price,
                target_price,
            )
            return None, "invalid_levels"

        # -- Walk forward checking exit conditions each bar ---------------------
        exit_price: float = entry_price
        exit_reason: str = "replay_end"   # force-close at last bar, never discard
        exit_timestamp = entry_bar.timestamp
        bars_in_trade = 0
        mae_points = 0.0
        mfe_points = 0.0
        realized_pnl_points = 0.0

        runner_enabled = bool(cfg.runner_enabled) or signal_type == "ORB"
        if signal_type == "ORB":
            primary_pct = 0.5
        else:
            primary_pct = min(max(cfg.runner_primary_pct, 0.0), 1.0)
        runner_pct = 1.0 - primary_pct
        runner_active = False
        runner_target_price: float | None = None
        runner_trail_stop: float | None = None
        runner_best_price = entry_price
        runner_step_armed = False
        last_mark_points = 0.0

        # -- Thesis-break state ------------------------------------------------
        vwap_at_signal = float(signal.get("vwap", 0.0))
        sigma_at_signal = float(signal.get("sigma_points", signal.get("sigma_value", 0.0)))
        z_inside_count = 0  # bars where |z| < thesis-break threshold

        def _signed_points(fill_price: float) -> float:
            if side == "BUY":
                return fill_price - entry_price
            return entry_price - fill_price

        def _effective_exit_from_points(points: float) -> float:
            if side == "BUY":
                return entry_price + points
            return entry_price - points

        for bar_idx in range(entry_idx, len(bars)):
            bar = bars[bar_idx]
            bars_in_trade += 1

            # Update MAE / MFE
            if side == "BUY":
                adverse = max(entry_price - bar.low, 0.0)
                favourable = max(bar.high - entry_price, 0.0)
            else:
                adverse = max(bar.high - entry_price, 0.0)
                favourable = max(entry_price - bar.low, 0.0)
            mae_points = max(mae_points, adverse)
            mfe_points = max(mfe_points, favourable)

            # --- Exit checks (order matters) ---

            # 1. Session cutoff
            bar_time_str = _local_time_str(bar.timestamp)
            if bar_time_str >= cfg.session_cutoff_time:
                if runner_active:
                    realized_pnl_points += runner_pct * _signed_points(bar.close)
                    exit_reason = "target_runner_session_cutoff"
                else:
                    realized_pnl_points = _signed_points(bar.close)
                    exit_reason = "session_cutoff"
                exit_price = _effective_exit_from_points(realized_pnl_points)
                exit_timestamp = bar.timestamp
                break

            if runner_active:
                if side == "BUY":
                    if cfg.runner_step_enabled:
                        if runner_trail_stop is None:
                            runner_trail_stop = entry_price
                        if (
                            not runner_step_armed
                            and bar.high >= entry_price + cfg.runner_step_trigger_r * risk_points
                        ):
                            runner_step_armed = True
                            runner_trail_stop = max(
                                runner_trail_stop,
                                entry_price + cfg.runner_step_lock_r * risk_points,
                            )
                    else:
                        trail_dist = cfg.runner_trail_r * risk_points
                        runner_best_price = max(runner_best_price, bar.high)
                        runner_trail_stop = max(entry_price, runner_best_price - trail_dist)
                    if bar.low <= runner_trail_stop:
                        realized_pnl_points += runner_pct * _signed_points(runner_trail_stop)
                        exit_reason = "target_runner_trail"
                        exit_price = _effective_exit_from_points(realized_pnl_points)
                        exit_timestamp = bar.timestamp
                        break
                    if runner_target_price is not None and bar.high >= runner_target_price:
                        realized_pnl_points += runner_pct * _signed_points(runner_target_price)
                        exit_reason = "target_runner"
                        exit_price = _effective_exit_from_points(realized_pnl_points)
                        exit_timestamp = bar.timestamp
                        break
                else:
                    if cfg.runner_step_enabled:
                        if runner_trail_stop is None:
                            runner_trail_stop = entry_price
                        if (
                            not runner_step_armed
                            and bar.low <= entry_price - cfg.runner_step_trigger_r * risk_points
                        ):
                            runner_step_armed = True
                            runner_trail_stop = min(
                                runner_trail_stop,
                                entry_price - cfg.runner_step_lock_r * risk_points,
                            )
                    else:
                        trail_dist = cfg.runner_trail_r * risk_points
                        runner_best_price = min(runner_best_price, bar.low)
                        runner_trail_stop = min(entry_price, runner_best_price + trail_dist)
                    if bar.high >= runner_trail_stop:
                        realized_pnl_points += runner_pct * _signed_points(runner_trail_stop)
                        exit_reason = "target_runner_trail"
                        exit_price = _effective_exit_from_points(realized_pnl_points)
                        exit_timestamp = bar.timestamp
                        break
                    if runner_target_price is not None and bar.low <= runner_target_price:
                        realized_pnl_points += runner_pct * _signed_points(runner_target_price)
                        exit_reason = "target_runner"
                        exit_price = _effective_exit_from_points(realized_pnl_points)
                        exit_timestamp = bar.timestamp
                        break

            if not runner_active:
                # 2. Stop checked BEFORE target (conservative assumption)
                stop_hit = False
                if side == "BUY" and bar.low <= stop_price:
                    stop_hit = True
                elif side == "SELL" and bar.high >= stop_price:
                    stop_hit = True

                if stop_hit:
                    realized_pnl_points = _signed_points(stop_price)
                    exit_price = _effective_exit_from_points(realized_pnl_points)
                    exit_reason = "stop"
                    exit_timestamp = bar.timestamp
                    break

                # 3. Target hit
                target_hit = False
                if side == "BUY" and bar.high >= target_price:
                    target_hit = True
                elif side == "SELL" and bar.low <= target_price:
                    target_hit = True

                if target_hit:
                    if runner_enabled and runner_pct > 0:
                        realized_pnl_points += primary_pct * _signed_points(target_price)
                        runner_active = True
                        runner_best_price = target_price
                        if side == "BUY":
                            runner_target_price = max(
                                entry_price + cfg.runner_target_r * risk_points,
                                target_price + cfg.tick_size,
                            )
                            runner_trail_stop = entry_price
                            if bar.high >= runner_target_price:
                                realized_pnl_points += runner_pct * _signed_points(runner_target_price)
                                exit_reason = "target_runner"
                                exit_price = _effective_exit_from_points(realized_pnl_points)
                                exit_timestamp = bar.timestamp
                                break
                        else:
                            runner_target_price = min(
                                entry_price - cfg.runner_target_r * risk_points,
                                target_price - cfg.tick_size,
                            )
                            runner_trail_stop = entry_price
                            if bar.low <= runner_target_price:
                                realized_pnl_points += runner_pct * _signed_points(runner_target_price)
                                exit_reason = "target_runner"
                                exit_price = _effective_exit_from_points(realized_pnl_points)
                                exit_timestamp = bar.timestamp
                                break
                    else:
                        realized_pnl_points = _signed_points(target_price)
                        exit_price = _effective_exit_from_points(realized_pnl_points)
                        exit_reason = "target"
                        exit_timestamp = bar.timestamp
                        break

            # 4. Thesis-break exit (replaces blind time-stop)
            if _cfg.THESIS_BREAK_ENABLED and sigma_at_signal > 0:
                # a) Z-score reversion stall: z reverted inside threshold but
                #    price stalls (doesn't reach target) for N bars → cut
                current_z = abs((bar.close - vwap_at_signal) / sigma_at_signal)
                if current_z < _cfg.THESIS_BREAK_ZSCORE:
                    z_inside_count += 1
                else:
                    z_inside_count = 0  # reset if pushed back out

                if z_inside_count >= _cfg.THESIS_BREAK_STALL_BARS:
                    if runner_active:
                        realized_pnl_points += runner_pct * _signed_points(bar.close)
                        exit_reason = "target_runner_thesis_break_stall"
                    else:
                        realized_pnl_points = _signed_points(bar.close)
                        exit_reason = "thesis_break_stall"
                    exit_price = _effective_exit_from_points(realized_pnl_points)
                    exit_timestamp = bar.timestamp
                    break

                # b) Early MAE excess: if MAE > fraction of stop distance
                #    within the first N bars → thesis is failing early
                if (bars_in_trade <= _cfg.THESIS_BREAK_MAE_BAR_LIMIT
                        and risk_points > 0
                        and mae_points / risk_points > _cfg.THESIS_BREAK_MAE_FRAC):
                    if runner_active:
                        realized_pnl_points += runner_pct * _signed_points(bar.close)
                        exit_reason = "target_runner_thesis_break_mae"
                    else:
                        realized_pnl_points = _signed_points(bar.close)
                        exit_reason = "thesis_break_mae"
                    exit_price = _effective_exit_from_points(realized_pnl_points)
                    exit_timestamp = bar.timestamp
                    break

                # c) Backstop: if none of the thesis-break conditions fire
                #    within time_stop_bars, still exit (prevents holding forever)
                if bars_in_trade >= cfg.time_stop_bars:
                    if runner_active:
                        realized_pnl_points += runner_pct * _signed_points(bar.close)
                        exit_reason = "target_runner_time_stop"
                    else:
                        realized_pnl_points = _signed_points(bar.close)
                        exit_reason = "time_stop"
                    exit_price = _effective_exit_from_points(realized_pnl_points)
                    exit_timestamp = bar.timestamp
                    break

            # 4b. Fallback time-stop (only when thesis-break disabled or
            #     sigma is zero — keeps backward compat)
            elif bars_in_trade >= cfg.time_stop_bars:
                if runner_active:
                    realized_pnl_points += runner_pct * _signed_points(bar.close)
                    exit_reason = "target_runner_time_stop"
                else:
                    realized_pnl_points = _signed_points(bar.close)
                    exit_reason = "time_stop"
                exit_price = _effective_exit_from_points(realized_pnl_points)
                exit_timestamp = bar.timestamp
                break

            # 5. End of data — will fall through if this is the last bar
            if runner_active:
                provisional_points = realized_pnl_points + runner_pct * _signed_points(bar.close)
            else:
                provisional_points = _signed_points(bar.close)
            last_mark_points = provisional_points
            exit_price = _effective_exit_from_points(provisional_points)
            exit_timestamp = bar.timestamp

        # -- Compute PnL -------------------------------------------------------
        if exit_reason == "replay_end":
            realized_pnl_points = last_mark_points

        pnl_points = realized_pnl_points

        pnl_dollars = pnl_points * cfg.point_value
        pnl_r = pnl_points / risk_points if risk_points > 0 else 0.0

        # -- Holding duration ---------------------------------------------------
        entry_dt = _ensure_utc(entry_bar.timestamp)
        exit_dt = _ensure_utc(exit_timestamp)
        hold_minutes = (exit_dt - entry_dt).total_seconds() / 60.0

        # -- Build trade record ------------------------------------------------
        session_id = str(signal.get("session_id", ""))
        regime = str(signal.get("regime", ""))
        band_level = float(signal.get("band_level", 0.0))

        return SimulatedTrade(
            trade_id=uuid.uuid4().hex[:12],
            session_id=session_id,
            signal_timestamp=sig_ts.isoformat(),
            side=side,
            entry_timestamp=_ensure_utc(entry_bar.timestamp).isoformat(),
            entry_price=round(entry_price, 6),
            stop_price=round(stop_price, 6),
            target_price=round(target_price, 6),
            exit_timestamp=exit_dt.isoformat(),
            exit_price=round(exit_price, 6),
            exit_reason=exit_reason,
            pnl_points=round(pnl_points, 6),
            pnl_dollars=round(pnl_dollars, 2),
            pnl_r=round(pnl_r, 4),
            mae_points=round(mae_points, 6),
            mfe_points=round(mfe_points, 6),
            hold_minutes=round(hold_minutes, 2),
            hold_bars=bars_in_trade,
            regime_at_entry=regime,
            sigma_band_level=band_level,
        ), "emitted"

    # ── internal: summary builder ────────────────────────────────────────

    @staticmethod
    def _build_summary(
        trades: list[SimulatedTrade],
        session_id: str,
    ) -> dict[str, Any]:
        """Aggregate trade-level data into a session summary dict."""
        total = len(trades)
        if total == 0:
            return {
                "session_id": session_id,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_win_points": 0.0,
                "avg_loss_points": 0.0,
                "avg_win_dollars": 0.0,
                "avg_loss_dollars": 0.0,
                "payoff_ratio": 0.0,
                "expectancy_points": 0.0,
                "expectancy_dollars": 0.0,
                "avg_mae": 0.0,
                "avg_mfe": 0.0,
                "avg_hold_minutes": 0.0,
                "exit_reason_breakdown": {},
            }

        wins = [t for t in trades if t.pnl_points > 0]
        losses = [t for t in trades if t.pnl_points <= 0]

        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = n_wins / total if total > 0 else 0.0

        avg_win_pts = (
            sum(t.pnl_points for t in wins) / n_wins if n_wins else 0.0
        )
        avg_loss_pts = (
            sum(t.pnl_points for t in losses) / n_losses if n_losses else 0.0
        )
        avg_win_dollars = (
            sum(t.pnl_dollars for t in wins) / n_wins if n_wins else 0.0
        )
        avg_loss_dollars = (
            sum(t.pnl_dollars for t in losses) / n_losses if n_losses else 0.0
        )

        payoff_ratio = (
            abs(avg_win_pts / avg_loss_pts) if avg_loss_pts != 0 else 0.0
        )

        expectancy_pts = sum(t.pnl_points for t in trades) / total
        expectancy_dollars = sum(t.pnl_dollars for t in trades) / total

        avg_mae = sum(t.mae_points for t in trades) / total
        avg_mfe = sum(t.mfe_points for t in trades) / total
        avg_hold = sum(t.hold_minutes for t in trades) / total

        # Exit-reason breakdown
        reason_counts: dict[str, int] = {}
        for t in trades:
            reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1

        return {
            "session_id": session_id,
            "total_trades": total,
            "wins": n_wins,
            "losses": n_losses,
            "win_rate": round(win_rate, 4),
            "avg_win_points": round(avg_win_pts, 4),
            "avg_loss_points": round(avg_loss_pts, 4),
            "avg_win_dollars": round(avg_win_dollars, 2),
            "avg_loss_dollars": round(avg_loss_dollars, 2),
            "payoff_ratio": round(payoff_ratio, 4),
            "expectancy_points": round(expectancy_pts, 4),
            "expectancy_dollars": round(expectancy_dollars, 2),
            "avg_mae": round(avg_mae, 4),
            "avg_mfe": round(avg_mfe, 4),
            "avg_hold_minutes": round(avg_hold, 2),
            "exit_reason_breakdown": reason_counts,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _ensure_utc(dt: datetime) -> datetime:
    """Return a timezone-aware UTC datetime.

    If *dt* is naïve, assume it is already UTC and attach the tzinfo.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=pd.Timestamp.now("UTC").tzinfo)
    return dt


def _local_time_str(dt: datetime, tz: str = "US/Eastern") -> str:
    """Return 'HH:MM' in the given timezone (default ET)."""
    import pytz

    eastern = pytz.timezone(tz)
    local_dt = _ensure_utc(dt).astimezone(eastern)
    return local_dt.strftime("%H:%M")
