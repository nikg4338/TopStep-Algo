"""
validation/sizing_policy.py — Dynamic contract sizing for Topstep Combine.

Provides a centralized sizing decision engine that determines how many
contracts to trade per session/day.  Two policies:

  ``fixed``       — constant contract count every day.
  ``dynamic_v1``  — 1↔2 contracts based on risk headroom, loss streak,
                     regime alignment, and profit-lock protection.

The sizing decision is day-scoped (set at start of session, can be
downshifted intraday but never upsized intraday).

Usage:
    config = SizingConfig(policy="dynamic_v1")
    policy = SizingPolicy(config)

    # Before each session:
    day = policy.decide_day_start(regime="range", active_engine="mr")

    # After each trade within the session:
    day = policy.on_trade(trade_pnl_dollars=−42.50)

    # After session ends:
    policy.end_of_day()
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class SizingConfig:
    """All tunable parameters for the sizing policy.

    Mirrors CLI flags so every value can be overridden without code changes.
    """

    policy: str = "fixed"           # "fixed", "dynamic_v1", "dynamic_v2", or "dynamic_v3"
    fixed_contracts: int = 2        # contract count when policy == "fixed"

    # ── Dynamic v1 upsize thresholds ────────────────────────────────────
    up_trail_headroom: float = 1400.0      # trail_headroom >= X to upsize
    up_day_headroom: float = 700.0         # day_headroom >= X to upsize
    loss_streak_up_max: int = 1            # loss_streak <= X to upsize

    # ── Dynamic v1 downshift thresholds ─────────────────────────────────
    down_trail_headroom: float = 1200.0    # trail_headroom < X => force 1c
    down_day_headroom: float = 600.0       # day_headroom < X => force 1c
    loss_streak_down_min: int = 2          # loss_streak >= X => force 1c
    shock_loss_frac: float = 0.6           # single loss >= frac * daily_loss_limit => force 1c

    # ── Dynamic v2: volatility throttle ────────────────────────────────
    vol_atr_cap: float = 14.0              # median first-hour ATR >= X => cap 1c (≈p75 of historical)

    # ── Dynamic v2: earned upsize ──────────────────────────────────────
    earned_traction: float = 150.0         # day_pnl >= X to unlock 2c intraday
    earned_giveback: float = 50.0          # if day_pnl drops below (traction - giveback) => revert 1c

    # ── Dynamic v3: calibrated upsize triggers ─────────────────────────
    v3_earned_traction: float = 75.0       # day_pnl >= X to unlock 2c
    v3_giveback_floor: float = 25.0        # day_pnl < X => revert 1c
    v3_orb_upsize_allowed: bool = False    # if allocator selected ORB, auto-eligible for 2c
    v3_day_headroom_up: float = 800.0      # day_headroom >= X required for upsize
    v3_day_headroom_down: float = 600.0    # day_headroom < X => force 1c
    v3_trail_headroom_up: float = 1400.0   # trail_headroom >= X required for upsize
    v3_trail_headroom_down: float = 1200.0 # trail_headroom < X => force 1c

    # ── Profit protection ──────────────────────────────────────────────
    profit_lock: float = 2000.0            # equity >= X => lock 1c for remainder

    # ── Topstep boundaries (used for headroom computation) ─────────────
    daily_loss_limit: float = 1000.0       # Topstep daily loss limit
    trail_dd_limit: float = 2000.0         # Topstep max loss limit (trailing DD)

    def __post_init__(self) -> None:
        if self.policy not in {"fixed", "dynamic_v1", "dynamic_v2", "dynamic_v3"}:
            raise ValueError(f"sizing policy must be 'fixed', 'dynamic_v1', 'dynamic_v2', or 'dynamic_v3', got '{self.policy}'")
        if self.fixed_contracts < 1:
            raise ValueError(f"fixed_contracts must be >= 1, got {self.fixed_contracts}")


# ═══════════════════════════════════════════════════════════════════════
#  Per-day state
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DaySizingRecord:
    """Snapshot of sizing state for one session, written to artifacts."""

    day_index: int = 0
    session_id: str = ""
    regime: str = ""
    active_engine: str = ""
    contracts_start: int = 1
    contracts_final: int = 1
    trail_headroom: float = 0.0
    day_headroom: float = 0.0
    loss_streak: int = 0
    equity_before: float = 0.0
    equity_after: float = 0.0
    profit_lock_triggered: bool = False
    downshift_reason: str = ""
    upsize_eligible: bool = False
    # v2-specific
    session_atr_median: float = 0.0
    vol_throttled: bool = False
    earned_upsize_triggered: bool = False
    # v3-specific
    v3_upsize_trigger: str = ""   # "traction", "first_trade_win", "orb_day", or ""
    v3_orb_day: bool = False
    v3_day_pnl: float = 0.0
    allocator_engine: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  Sizing Policy
# ═══════════════════════════════════════════════════════════════════════


class SizingPolicy:
    """Stateful sizing engine that tracks equity / trailing DD across sessions.

    Lifecycle per pack run::

        policy = SizingPolicy(config)
        for session in pack:
            policy.decide_day_start(session_id, regime, engine, day_index)
            for trade in session_trades:
                policy.on_trade(trade.pnl_dollars)
            policy.end_of_day()
        report = policy.daily_log
    """

    def __init__(self, config: SizingConfig) -> None:
        self.config = config

        # ── Run-level state ─────────────────────────────────────────────
        self.equity: float = 0.0             # cumulative PnL from start of combine
        self.peak_equity: float = 0.0        # high-water mark
        self.loss_streak: int = 0            # consecutive losing trades
        self.profit_lock_triggered: bool = False

        # ── Day-level state (reset each session) ────────────────────────
        self.day_pnl: float = 0.0
        self.day_contracts: int = 1
        self._downshift_reason: str = ""
        self._day_index: int = 0
        self._session_id: str = ""
        self._regime: str = ""
        self._active_engine: str = ""
        self._session_atr_median: float = 0.0
        self._vol_throttled: bool = False
        self._earned_upsize_triggered: bool = False
        self._v2_upsize_eligible: bool = False  # headroom + streak OK for v2

        # ── v3 day-level state ──────────────────────────────────────────
        self._v3_headroom_ok: bool = False      # headroom + streak prereqs met
        self._v3_upsize_trigger: str = ""       # which rule triggered 2c
        self._v3_orb_day: bool = False           # allocator chose ORB
        self._v3_first_trade_seen: bool = False  # have we processed the 1st trade?

        # ── Logging ─────────────────────────────────────────────────────
        self.daily_log: list[DaySizingRecord] = []

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def trailing_dd_used(self) -> float:
        """How much of the trailing DD budget has been consumed."""
        return max(0.0, self.peak_equity - self.equity)

    @property
    def trail_headroom(self) -> float:
        """Remaining trailing DD headroom before ruin."""
        return self.config.trail_dd_limit - self.trailing_dd_used

    @property
    def day_headroom(self) -> float:
        """Remaining daily loss headroom."""
        return self.config.daily_loss_limit + self.day_pnl  # day_pnl is negative when losing

    @property
    def contracts(self) -> int:
        """Current contract count for the active day."""
        return self.day_contracts

    # ── Day-start decision ──────────────────────────────────────────────

    def decide_day_start(
        self,
        session_id: str = "",
        regime: str = "",
        active_engine: str = "",
        day_index: int = 0,
        session_atr_median: float = 0.0,
    ) -> int:
        """Determine contract count for the start of a new session.

        Must be called before processing any trades for the session.

        Parameters
        ----------
        session_atr_median:
            Median ATR of the first hour of the session (from
            features_snapshot.csv).  Used by dynamic_v2 for volatility
            throttling.  Ignored by other policies.

        Returns
        -------
        int
            The contract count for this day (1 or 2 for dynamic policies).
        """
        self.day_pnl = 0.0
        self._downshift_reason = ""
        self._day_index = day_index
        self._session_id = session_id
        self._regime = regime
        self._active_engine = active_engine
        self._session_atr_median = session_atr_median
        self._vol_throttled = False
        self._earned_upsize_triggered = False
        self._v2_upsize_eligible = False
        self._v3_headroom_ok = False
        self._v3_upsize_trigger = ""
        self._v3_orb_day = (active_engine == "orb")
        self._v3_first_trade_seen = False

        if self.config.policy == "fixed":
            self.day_contracts = self.config.fixed_contracts
            return self.day_contracts

        # ── dynamic_v1 ──────────────────────────────────────────────────
        if self.config.policy == "dynamic_v1":
            self.day_contracts = 1

            if self.profit_lock_triggered:
                self._downshift_reason = "profit_lock"
                return self.day_contracts

            regime_aligned = (
                (regime == "range" and active_engine == "mr")
                or (regime == "trend" and active_engine == "orb")
            )

            upsize_eligible = (
                self.trail_headroom >= self.config.up_trail_headroom
                and self.day_headroom >= self.config.up_day_headroom
                and self.loss_streak <= self.config.loss_streak_up_max
                and regime_aligned
            )

            if upsize_eligible:
                self.day_contracts = 2

            return self.day_contracts

        # ── dynamic_v2 ──────────────────────────────────────────────────
        if self.config.policy == "dynamic_v2":
            # Always start at 1c.  Upsize is *earned* intraday via on_trade().
            self.day_contracts = 1

            if self.profit_lock_triggered:
                self._downshift_reason = "profit_lock"
                return self.day_contracts

            # Volatility throttle: if session ATR is high, cap at 1c all day
            if session_atr_median >= self.config.vol_atr_cap:
                self._vol_throttled = True
                self._downshift_reason = f"vol_throttle_atr_{session_atr_median:.1f}>={self.config.vol_atr_cap:.1f}"
                return self.day_contracts

            # Pre-check eligibility for earned upsize (headroom + streak + regime)
            regime_aligned = (
                (regime == "range" and active_engine == "mr")
                or (regime == "trend" and active_engine == "orb")
            )
            self._v2_upsize_eligible = (
                self.trail_headroom >= self.config.up_trail_headroom
                and self.day_headroom >= self.config.up_day_headroom
                and self.loss_streak <= self.config.loss_streak_up_max
                and regime_aligned
            )

            return self.day_contracts

        # ── dynamic_v3 ──────────────────────────────────────────────────
        # Start every day at 1 contract.  Upsize to 2 is earned intraday.
        self.day_contracts = 1

        if self.profit_lock_triggered:
            self._downshift_reason = "profit_lock"
            return self.day_contracts

        # Pre-check headroom + streak prerequisites for any upsize path
        self._v3_headroom_ok = (
            self.trail_headroom >= self.config.v3_trail_headroom_up
            and self.day_headroom >= self.config.v3_day_headroom_up
            and self.loss_streak <= self.config.loss_streak_up_max
        )

        # ORB-day auto-upsize: if allocator selected ORB and config allows,
        # upsize immediately at day start (before any trades).
        if (
            self._v3_headroom_ok
            and self.config.v3_orb_upsize_allowed
            and self._v3_orb_day
        ):
            self.day_contracts = 2
            self._v3_upsize_trigger = "orb_day"
            self._earned_upsize_triggered = True

        return self.day_contracts

    # ── Per-trade update ────────────────────────────────────────────────

    def on_trade(self, trade_pnl_dollars: float) -> int:
        """Process a completed trade and check for downshift triggers.

        Parameters
        ----------
        trade_pnl_dollars:
            The dollar PnL of the trade **after** contract scaling
            (i.e., already multiplied by current contracts).

        Returns
        -------
        int
            The (possibly updated) contract count after this trade.
        """
        # Update PnL tracking
        self.day_pnl += trade_pnl_dollars
        self.equity += trade_pnl_dollars

        # Update peak equity (high-water mark)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Update loss streak
        if trade_pnl_dollars < 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0

        # ── Profit lock check ───────────────────────────────────────────
        if not self.profit_lock_triggered and self.equity >= self.config.profit_lock:
            self.profit_lock_triggered = True
            self.day_contracts = 1
            self._downshift_reason = "profit_lock"
            return self.day_contracts

        # ── Downshift checks (relevant for dynamic_v1 / v2 / v3 at 2c) ─────
        if self.config.policy in {"dynamic_v1", "dynamic_v2", "dynamic_v3"} and self.day_contracts > 1:
            reason = self._check_downshift(trade_pnl_dollars)
            if reason:
                self.day_contracts = 1
                self._downshift_reason = reason
                return self.day_contracts

        # ── dynamic_v2: earned upsize + giveback ────────────────────────
        if self.config.policy == "dynamic_v2" and not self._vol_throttled:
            if self.day_contracts == 1 and self._v2_upsize_eligible:
                # Check if traction threshold met to unlock 2c
                if self.day_pnl >= self.config.earned_traction:
                    self.day_contracts = 2
                    self._earned_upsize_triggered = True
            elif self.day_contracts == 2 and self._earned_upsize_triggered:
                # Giveback revert: if day_pnl drops below threshold
                giveback_floor = self.config.earned_traction - self.config.earned_giveback
                if self.day_pnl < giveback_floor:
                    self.day_contracts = 1
                    self._downshift_reason = (
                        f"earned_giveback_{self.day_pnl:.0f}<{giveback_floor:.0f}"
                    )

        # ── dynamic_v3: multi-trigger earned upsize + giveback ──────────
        if self.config.policy == "dynamic_v3":
            is_first_trade = not self._v3_first_trade_seen
            self._v3_first_trade_seen = True

            if self.day_contracts == 1 and self._v3_headroom_ok and not self._v3_upsize_trigger:
                # Trigger a) earned traction
                if self.day_pnl >= self.config.v3_earned_traction:
                    self.day_contracts = 2
                    self._v3_upsize_trigger = "traction"
                    self._earned_upsize_triggered = True
                # Trigger b) first trade of day is a winner
                elif is_first_trade and trade_pnl_dollars > 0:
                    self.day_contracts = 2
                    self._v3_upsize_trigger = "first_trade_win"
                    self._earned_upsize_triggered = True

            elif self.day_contracts == 2 and self._v3_upsize_trigger:
                # Giveback revert: if day_pnl drops below giveback_floor
                if self.day_pnl < self.config.v3_giveback_floor:
                    self.day_contracts = 1
                    self._downshift_reason = (
                        f"v3_giveback_{self.day_pnl:.0f}<{self.config.v3_giveback_floor:.0f}"
                    )

            # v3 downshift checks: day headroom and trail headroom
            if self.day_contracts > 1:
                if self.trail_headroom < self.config.v3_trail_headroom_down:
                    self.day_contracts = 1
                    self._downshift_reason = (
                        f"v3_trail_{self.trail_headroom:.0f}<{self.config.v3_trail_headroom_down:.0f}"
                    )
                elif self.day_headroom < self.config.v3_day_headroom_down:
                    self.day_contracts = 1
                    self._downshift_reason = (
                        f"v3_day_{self.day_headroom:.0f}<{self.config.v3_day_headroom_down:.0f}"
                    )

        return self.day_contracts

    def _check_downshift(self, trade_pnl_dollars: float) -> str:
        """Check all downshift conditions. Returns reason string or empty."""
        if self.trail_headroom < self.config.down_trail_headroom:
            return f"trail_headroom_{self.trail_headroom:.0f}<{self.config.down_trail_headroom:.0f}"

        if self.day_headroom < self.config.down_day_headroom:
            return f"day_headroom_{self.day_headroom:.0f}<{self.config.down_day_headroom:.0f}"

        if self.loss_streak >= self.config.loss_streak_down_min:
            return f"loss_streak_{self.loss_streak}>={self.config.loss_streak_down_min}"

        # Shock loss: single trade loss >= fraction of daily limit
        shock_threshold = self.config.shock_loss_frac * self.config.daily_loss_limit
        if trade_pnl_dollars < 0 and abs(trade_pnl_dollars) >= shock_threshold:
            return f"shock_loss_{abs(trade_pnl_dollars):.0f}>={shock_threshold:.0f}"

        return ""

    # ── End-of-day bookkeeping ──────────────────────────────────────────

    def end_of_day(self) -> DaySizingRecord:
        """Finalize the day and log the sizing record.

        Returns the DaySizingRecord for this session.
        """
        # contracts_start: what size did the day begin at?
        if self.config.policy == "dynamic_v2":
            cs = 1  # v2 always starts at 1
        elif self.config.policy == "dynamic_v3":
            # v3 starts at 1 unless ORB-day auto-upsize triggered at day start
            cs = 2 if (self._v3_upsize_trigger == "orb_day") else 1
        elif self.config.policy == "dynamic_v1":
            if self._downshift_reason:
                cs = 2
            else:
                cs = self.day_contracts
        else:
            cs = self.day_contracts

        record = DaySizingRecord(
            day_index=self._day_index,
            session_id=self._session_id,
            regime=self._regime,
            active_engine=self._active_engine,
            contracts_start=cs,
            contracts_final=self.day_contracts,
            trail_headroom=round(self.trail_headroom, 2),
            day_headroom=round(self.day_headroom, 2),
            loss_streak=self.loss_streak,
            equity_before=round(self.equity - self.day_pnl, 2),
            equity_after=round(self.equity, 2),
            profit_lock_triggered=self.profit_lock_triggered,
            downshift_reason=self._downshift_reason,
            upsize_eligible=False,
            session_atr_median=round(self._session_atr_median, 2),
            vol_throttled=self._vol_throttled,
            earned_upsize_triggered=self._earned_upsize_triggered,
            # v3-specific diagnostics
            v3_upsize_trigger=self._v3_upsize_trigger,
            v3_orb_day=self._v3_orb_day,
            v3_day_pnl=round(self.day_pnl, 2),
            allocator_engine=self._active_engine,
        )

        # Mark upsize eligibility
        if self.config.policy == "dynamic_v1":
            if self._downshift_reason:
                record.upsize_eligible = True
            elif self.day_contracts == 2:
                record.upsize_eligible = True
        elif self.config.policy == "dynamic_v2":
            record.upsize_eligible = self._v2_upsize_eligible
        elif self.config.policy == "dynamic_v3":
            record.upsize_eligible = self._v3_headroom_ok

        self.daily_log.append(record)
        return record

    # ── Serialization ───────────────────────────────────────────────────

    def write_daily_log(self, path: Path) -> None:
        """Write the daily sizing log to a JSON file."""
        records = [asdict(r) for r in self.daily_log]
        path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    def config_snapshot(self) -> dict[str, Any]:
        """Return config as a dict for artifact snapshots."""
        return asdict(self.config)


# ═══════════════════════════════════════════════════════════════════════
#  Trade-level sizing application
# ═══════════════════════════════════════════════════════════════════════


def apply_sizing_to_trades(
    trades_csv: Path,
    policy: SizingPolicy,
    regime: str = "",
    active_engine: str = "",
    session_id: str = "",
    day_index: int = 0,
    session_atr_median: float = 0.0,
) -> list[dict[str, Any]]:
    """Read trades.csv, apply contract sizing, update policy state, return scaled trades.

    Processes trades sequentially to support intra-day downshifts and
    earned upsize (dynamic_v2).

    Parameters
    ----------
    trades_csv:
        Path to the session's trades.csv (from MRExitSimulator).
    policy:
        Active SizingPolicy instance with run-level state.
    regime, active_engine:
        Allocator v2's day-level regime/engine assignments for this session.
    session_id:
        Identifier for logging.
    day_index:
        1-based day index within the combine run.
    session_atr_median:
        Median ATR of the first hour (for v2 vol throttle).

    Returns
    -------
    list[dict]
        Trades with ``pnl_dollars``, ``pnl_points``, ``mae_points``,
        ``mfe_points`` scaled by contracts, plus a ``contracts`` column.
    """
    import pandas as pd

    if not trades_csv.is_file():
        # No trades for this session — still register the day
        policy.decide_day_start(session_id, regime, active_engine, day_index,
                                session_atr_median=session_atr_median)
        policy.end_of_day()
        return []

    df = pd.read_csv(trades_csv)
    if df.empty:
        policy.decide_day_start(session_id, regime, active_engine, day_index,
                                session_atr_median=session_atr_median)
        policy.end_of_day()
        return []

    # Decide contracts at day start
    policy.decide_day_start(session_id, regime, active_engine, day_index,
                            session_atr_median=session_atr_median)

    scaled_trades: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        c = policy.contracts
        base_pnl_dollars = float(row.get("pnl_dollars", 0.0))
        base_pnl_points = float(row.get("pnl_points", 0.0))

        # Scale by current contracts
        scaled_pnl_dollars = base_pnl_dollars * c
        scaled_pnl_points = base_pnl_points * c
        scaled_mae = float(row.get("mae_points", 0.0)) * c
        scaled_mfe = float(row.get("mfe_points", 0.0)) * c

        trade_dict: dict[str, Any] = {str(k): v for k, v in row.to_dict().items()}
        trade_dict["pnl_dollars"] = round(scaled_pnl_dollars, 2)
        trade_dict["pnl_points"] = round(scaled_pnl_points, 6)
        trade_dict["mae_points"] = round(scaled_mae, 6)
        trade_dict["mfe_points"] = round(scaled_mfe, 6)
        trade_dict["contracts"] = c
        scaled_trades.append(trade_dict)

        # Update policy state with the *scaled* PnL
        policy.on_trade(scaled_pnl_dollars)

    policy.end_of_day()
    return scaled_trades
