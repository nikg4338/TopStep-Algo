"""
simulation/mc_survival.py — Monte Carlo "Combine Survival" simulation.

Models the Topstep combine challenge as a random walk with:
  - True path-based trailing drawdown ruin  (MAX_LOSS_LIMIT)
  - Timestamp-aware daily loss limit        (DAILY_LOSS_LIMIT_EXTERNAL)
  - Profit target                           (PROFIT_TARGET)
  - Consistency cap                         (no single day > X% of total profit at target)

Supports two sampling modes:
  - ``iid``   — classic IID resampling of individual trades
  - ``block`` — block bootstrap resampling (by session or day)

Supports optional adversarial stress transforms:
  - Loss multiplier, win multiplier, win-rate shift, slippage

Inputs are *R-values* (or PnL-dollar values) from the replay-derived
MC profile.  Each scenario samples trades until a boundary is hit or
the trade budget is exhausted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import config as _cfg

logger = logging.getLogger(__name__)


# ── Wilson binomial confidence interval ─────────────────────────────────

def wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score interval for a binomial proportion.

    Pure-math implementation — no SciPy dependency.

    Parameters
    ----------
    successes:
        Number of successes (e.g., simulations that hit target).
    trials:
        Total number of trials.
    confidence:
        Confidence level (default 0.95).

    Returns
    -------
    (lower, upper)
        Bounds of the Wilson interval.
    """
    if trials == 0:
        return (0.0, 1.0)
    import math
    # Z-score for common confidence levels
    z_map = {0.90: 1.6449, 0.95: 1.96, 0.99: 2.5758}
    z = z_map.get(confidence, 1.96)

    p_hat = successes / trials
    denom = 1 + z * z / trials
    centre = p_hat + z * z / (2 * trials)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials))
    lo = max(0.0, (centre - spread) / denom)
    hi = min(1.0, (centre + spread) / denom)
    return (lo, hi)


# ── Stress preset definitions ──────────────────────────────────────────

STRESS_PRESETS: dict[str, dict[str, float]] = {
    "base": {
        "loss_multiplier": 1.0,
        "win_multiplier": 1.0,
        "win_rate_shift": 0.0,
        "slippage_ticks": 0,
    },
    "mild": {
        "loss_multiplier": 1.2,
        "win_multiplier": 0.9,
        "win_rate_shift": 0.0,
        "slippage_ticks": 0,
    },
    "severe": {
        "loss_multiplier": 1.5,
        "win_multiplier": 0.8,
        "win_rate_shift": -0.1,
        "slippage_ticks": 0,
    },
    "tilt_bad_week": {
        "loss_multiplier": 1.0,
        "win_multiplier": 1.0,
        "win_rate_shift": 0.0,
        "slippage_ticks": 0,
        "tilt_frac": _cfg.MC_TILT_BAD_FRAC,
        "tilt_quantile": _cfg.MC_TILT_BAD_QUANTILE,
    },
}


# ── Result container ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SurvivalResult:
    """Immutable result from a combine-survival Monte Carlo run."""

    n_simulations: int
    p_target_before_ruin: float
    p_ruin: float
    p_fail_consistency_given_target: float
    dd_p95: float
    losing_streak_p95: float
    median_trades_to_target: float
    p_daily_loss_breach: float
    notes: str

    # ── diagnostics ─────────────────────────────────────────────────────
    avg_trade_pnl_dollars: float = 0.0
    expected_trades_to_target: float = 0.0
    termination_hit_target: int = 0
    termination_ruin: int = 0
    termination_max_trades: int = 0
    termination_daily_loss: int = 0
    nan_trades_skipped: int = 0

    # ── distribution diagnostics ────────────────────────────────────────
    trade_count_input: int = 0
    std_trade_pnl_dollars: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    worst_sampled_loss: float = 0.0
    worst_sampled_drawdown: float = 0.0
    equity_p1: float = 0.0
    equity_p10: float = 0.0
    equity_p50: float = 0.0
    equity_p99: float = 0.0
    worst_intraday_dd: float = 0.0

    # ── mode / stress metadata ──────────────────────────────────────────
    mode: str = "iid"
    stress_scenario: str = "base"

    # ── expanded diagnostics (Phase 2) ──────────────────────────────────
    trades_to_target_p5: float = 0.0
    trades_to_target_p25: float = 0.0
    trades_to_target_p75: float = 0.0
    trades_to_target_p95: float = 0.0
    equity_at_50: float = 0.0
    equity_at_100: float = 0.0
    equity_at_200: float = 0.0
    frac_terminated_max_trades: float = 0.0
    dd_p50: float = 0.0
    dd_p90: float = 0.0
    dd_p99: float = 0.0

    # ── confidence interval fields (Phase 3) ────────────────────────────
    p_target_ci_lo: float = 0.0
    p_target_ci_hi: float = 0.0
    p_target_batch_median: float = 0.0
    p_target_batch_min: float = 0.0
    p_target_batch_max: float = 0.0

    # ── day-horizon fields ──────────────────────────────────────────────
    days_to_target_median: float = 0.0
    days_to_target_p5: float = 0.0
    days_to_target_p95: float = 0.0
    max_days_used: int = 0


# ── Simulator ───────────────────────────────────────────────────────────


class MonteCarloSurvivalSimulator:
    """Run combine-survival Monte Carlo scenarios.

    Parameters
    ----------
    risk_per_trade:
        Dollar risk per trade (used to convert R-values → dollars).
    profit_target:
        Dollar equity level for "pass" (equity >= target = success).
    max_loss_limit:
        Trailing drawdown threshold.  If ``peak - equity >= max_loss_limit``
        the scenario is ruined.
    daily_loss_limit:
        Maximum dollar loss allowed in a single "day".
    consistency_cap:
        Max fraction of total profit that any single day may contribute
        at the point the target is reached.
    n_simulations:
        Number of Monte Carlo scenarios.
    max_trades:
        Budget per scenario.
    trades_per_day:
        Approximate number of trades per day for daily-boundary grouping
        (used only in IID mode when no timestamps are available).
    mode:
        ``"iid"`` for classic IID resampling, ``"block"`` for block bootstrap.
    block_type:
        ``"session"`` or ``"day"`` — how to group trades into blocks
        (only used when ``mode="block"``).
    """

    def __init__(
        self,
        risk_per_trade: float = _cfg.RISK_PER_TRADE,
        profit_target: float = _cfg.PROFIT_TARGET,
        max_loss_limit: float = _cfg.MAX_LOSS_LIMIT,
        daily_loss_limit: float = _cfg.DAILY_LOSS_LIMIT_EXTERNAL,
        consistency_cap: float = _cfg.CONSISTENCY_CAP.get(_cfg.ACCOUNT_MODE, 0.50),
        n_simulations: int = _cfg.MC_SIMULATIONS,
        max_trades: int = _cfg.MC_MAX_TRADES,
        trades_per_day: int = _cfg.RG_MAX_TRADES_PER_DAY,
        mode: str = _cfg.MC_MODE,
        block_type: str = _cfg.MC_BLOCK_TYPE,
    ) -> None:
        self.risk_per_trade = risk_per_trade
        self.profit_target = profit_target
        self.max_loss_limit = max_loss_limit
        self.daily_loss_limit = daily_loss_limit
        self.consistency_cap = consistency_cap
        self.n_simulations = n_simulations
        self.max_trades = max_trades
        self.trades_per_day = max(trades_per_day, 1)
        self.mode = mode
        self.block_type = block_type

    # ── Public API ──────────────────────────────────────────────────────

    def run(
        self,
        r_values: list[float] | np.ndarray,
        *,
        seed: int | None = None,
        use_dollar_values: bool = False,
        session_ids: list[str] | None = None,
        timestamps: list[str] | None = None,
        stress: dict[str, Any] | None = None,
    ) -> SurvivalResult:
        """Simulate *n_simulations* combine paths.

        Parameters
        ----------
        r_values:
            Historical trade outcomes.  Interpreted as R-multiples unless
            ``use_dollar_values=True``.
        seed:
            RNG seed for reproducibility.
        use_dollar_values:
            When *True*, skip risk_per_trade scaling.
        session_ids:
            Per-trade session labels (length must match r_values).
            Used for block-mode grouping when ``block_type="session"``.
        timestamps:
            Per-trade ISO-8601 timestamps.
            Used for block-mode grouping when ``block_type="day"``.
        stress:
            Optional stress overrides: ``loss_multiplier``, ``win_multiplier``,
            ``win_rate_shift``, ``slippage_ticks``.
        """
        r_arr = np.asarray(r_values, dtype=np.float64)
        valid_mask = np.isfinite(r_arr)
        nan_input_count = int((~valid_mask).sum())
        r_arr = r_arr[valid_mask]

        if r_arr.size == 0:
            return SurvivalResult(
                n_simulations=self.n_simulations,
                p_target_before_ruin=0.0,
                p_ruin=0.0,
                p_fail_consistency_given_target=0.0,
                dd_p95=0.0,
                losing_streak_p95=0.0,
                median_trades_to_target=0.0,
                p_daily_loss_breach=0.0,
                notes="No r_values supplied — degenerate result.",
                nan_trades_skipped=nan_input_count,
                mode=self.mode,
                stress_scenario=stress.get("_name", "base") if stress else "base",
            )

        # Convert to dollars
        if use_dollar_values:
            outcomes_pool = r_arr.copy()
        else:
            outcomes_pool = r_arr * self.risk_per_trade

        # Apply stress transforms
        stress_name = "base"
        if stress:
            stress_name = stress.get("_name", "custom")
            outcomes_pool = self._apply_stress(outcomes_pool, stress, seed)

        trade_count_input = len(outcomes_pool)

        # Determine grouping for block mode
        if self.mode == "block" and trade_count_input > 0:
            blocks = self._build_blocks(
                outcomes_pool, session_ids, timestamps, valid_mask
            )
            # Extract tilt parameters from stress dict
            tilt_frac = stress.get("tilt_frac", 0.0) if stress else 0.0
            bad_indices = stress.get("_bad_block_indices") if stress else None

            # Day-horizon mode: simulate by days (blocks) instead of trades
            if _cfg.MC_DAY_HORIZON_ENABLED:
                return self._simulate_block_day_horizon(
                    blocks, outcomes_pool, seed, nan_input_count, stress_name,
                    trade_count_input,
                    max_days=_cfg.MC_MAX_DAYS,
                    bad_block_indices=bad_indices,
                    tilt_frac=tilt_frac,
                )

            return self._simulate_block(
                blocks, outcomes_pool, seed, nan_input_count, stress_name,
                trade_count_input,
                bad_block_indices=bad_indices,
                tilt_frac=tilt_frac,
            )
        else:
            return self._simulate_iid(
                outcomes_pool, seed, nan_input_count, stress_name,
                trade_count_input,
            )

    # ── Stress transforms ───────────────────────────────────────────────

    def _apply_stress(
        self,
        outcomes: np.ndarray,
        stress: dict[str, Any],
        seed: int | None = None,
    ) -> np.ndarray:
        """Apply adversarial stress transforms to dollar outcomes."""
        out = outcomes.copy()
        loss_mult = stress.get("loss_multiplier", 1.0)
        win_mult = stress.get("win_multiplier", 1.0)
        wr_shift = stress.get("win_rate_shift", 0.0)
        slip_ticks = int(stress.get("slippage_ticks", 0))

        # Multiply losses and wins
        out[out < 0] *= loss_mult
        out[out > 0] *= win_mult

        # Win-rate shift: flip a fraction of winning trades to losses
        if wr_shift < 0:
            win_mask = np.where(out > 0)[0]
            n_flip = int(abs(wr_shift) * len(out))
            if n_flip > 0 and len(win_mask) > 0:
                rng = np.random.default_rng(seed)
                flip_count = min(n_flip, len(win_mask))
                flip_idx = rng.choice(win_mask, size=flip_count, replace=False)
                out[flip_idx] = -np.abs(out[flip_idx])

        # Slippage
        if slip_ticks > 0:
            slippage_dollars = slip_ticks * _cfg.TICK_VALUE
            out -= slippage_dollars

        return out

    # ── Block building ──────────────────────────────────────────────────

    def _build_blocks(
        self,
        outcomes: np.ndarray,
        session_ids: list[str] | None,
        timestamps: list[str] | None,
        valid_mask: np.ndarray,
    ) -> list[np.ndarray]:
        """Partition dollar outcomes into blocks for block bootstrap."""
        n = len(outcomes)

        if self.block_type == "session" and session_ids is not None:
            filtered_sids = [
                s for s, v in zip(session_ids, valid_mask) if v
            ]
            if len(filtered_sids) == n:
                return self._group_by_label(outcomes, filtered_sids)

        if self.block_type == "day" and timestamps is not None:
            filtered_ts = [
                t for t, v in zip(timestamps, valid_mask) if v
            ]
            if len(filtered_ts) == n:
                day_labels = [ts[:10] for ts in filtered_ts]
                return self._group_by_label(outcomes, day_labels)

        # Fallback: split into day-sized chunks based on trades_per_day
        blocks: list[np.ndarray] = []
        for i in range(0, n, self.trades_per_day):
            blocks.append(outcomes[i:i + self.trades_per_day])
        return blocks if blocks else [outcomes]

    @staticmethod
    def _group_by_label(
        outcomes: np.ndarray, labels: list[str]
    ) -> list[np.ndarray]:
        """Group outcomes by label, preserving within-group order."""
        from collections import OrderedDict
        groups: dict[str, list[int]] = OrderedDict()
        for i, lab in enumerate(labels):
            groups.setdefault(lab, []).append(i)
        return [outcomes[np.array(idx)] for idx in groups.values()]

    # ── IID simulation ──────────────────────────────────────────────────

    def _simulate_iid(
        self,
        outcomes_pool: np.ndarray,
        seed: int | None,
        nan_input_count: int,
        stress_name: str,
        trade_count_input: int,
    ) -> SurvivalResult:
        """Run IID-mode simulation (classic resampling)."""
        rng = np.random.default_rng(seed)
        n = self.n_simulations
        T = self.max_trades
        tpd = self.trades_per_day

        sample_idx = rng.integers(0, len(outcomes_pool), size=(n, T))
        outcomes_matrix = outcomes_pool[sample_idx]

        return self._walk_and_collect(
            outcomes_matrix, n, T, tpd, outcomes_pool,
            nan_input_count, stress_name, trade_count_input,
            mode="iid",
        )

    # ── Block bootstrap simulation ──────────────────────────────────────

    def _simulate_block(
        self,
        blocks: list[np.ndarray],
        outcomes_pool: np.ndarray,
        seed: int | None,
        nan_input_count: int,
        stress_name: str,
        trade_count_input: int,
        bad_block_indices: list[int] | None = None,
        tilt_frac: float = 0.0,
    ) -> SurvivalResult:
        """Run block-bootstrap simulation.

        Each scenario samples full blocks (with replacement) until
        max_trades is reached.  Within each block, trade order is
        preserved.

        When *tilt_frac > 0* and *bad_block_indices* are provided,
        ``tilt_frac`` proportion of blocks are drawn from the bad-bucket
        (bottom-quantile sessions by base PnL).  This implements the
        "bad week" tilt stress.
        """
        rng = np.random.default_rng(seed)
        n = self.n_simulations
        T = self.max_trades
        n_blocks = len(blocks)
        tpd = self.trades_per_day

        # Pre-build outcome matrix by concatenating sampled blocks
        outcomes_matrix = np.zeros((n, T), dtype=np.float64)

        # Tilt support: separate bad vs normal block indices
        if tilt_frac > 0 and bad_block_indices and n_blocks > 1:
            bad_set = set(bad_block_indices)
            normal_indices = [i for i in range(n_blocks) if i not in bad_set]
            if not normal_indices:
                normal_indices = list(range(n_blocks))
            bad_arr = np.array(bad_block_indices)
            normal_arr = np.array(normal_indices)
        else:
            bad_arr = None
            normal_arr = None

        for i in range(n):
            filled = 0
            while filled < T:
                # Choose block index — with tilt bias if configured
                if bad_arr is not None and rng.random() < tilt_frac:
                    block = blocks[rng.choice(bad_arr)]
                elif normal_arr is not None:
                    block = blocks[rng.choice(normal_arr)]
                else:
                    block = blocks[rng.integers(0, n_blocks)]
                room = T - filled
                chunk = block[:room]
                outcomes_matrix[i, filled:filled + len(chunk)] = chunk
                filled += len(chunk)

        return self._walk_and_collect(
            outcomes_matrix, n, T, tpd, outcomes_pool,
            nan_input_count, stress_name, trade_count_input,
            mode="block",
        )

    # ── Day-horizon block simulation ────────────────────────────────────

    def _simulate_block_day_horizon(
        self,
        blocks: list[np.ndarray],
        outcomes_pool: np.ndarray,
        seed: int | None,
        nan_input_count: int,
        stress_name: str,
        trade_count_input: int,
        max_days: int,
        bad_block_indices: list[int] | None = None,
        tilt_frac: float = 0.0,
    ) -> SurvivalResult:
        """Day-horizon block bootstrap: P(target within N days) before ruin.

        Each "day" = one sampled session block.  Simulation terminates
        after *max_days* blocks (days) are consumed, or on target/ruin.
        This directly answers: "can this strategy pass a combine in N days?"
        """
        rng = np.random.default_rng(seed)
        n = self.n_simulations
        n_blocks = len(blocks)

        # Tilt support
        if tilt_frac > 0 and bad_block_indices and n_blocks > 1:
            bad_set = set(bad_block_indices)
            normal_indices = [i for i in range(n_blocks) if i not in bad_set]
            if not normal_indices:
                normal_indices = list(range(n_blocks))
            bad_arr = np.array(bad_block_indices)
            normal_arr = np.array(normal_indices)
        else:
            bad_arr = None
            normal_arr = None

        # Per-sim state
        equity = np.zeros(n, dtype=np.float64)
        peak = np.zeros(n, dtype=np.float64)
        max_dd = np.zeros(n, dtype=np.float64)
        streak = np.zeros(n, dtype=np.int64)
        max_streak = np.zeros(n, dtype=np.int64)

        hit_target = np.zeros(n, dtype=bool)
        hit_ruin = np.zeros(n, dtype=bool)
        active = np.ones(n, dtype=bool)

        days_to_target = np.full(n, max_days, dtype=np.int64)
        trades_to_target = np.full(n, 0, dtype=np.int64)
        total_trades = np.zeros(n, dtype=np.int64)

        # Per-day PnL for consistency cap
        day_pnl_matrix = np.zeros((n, max_days), dtype=np.float64)

        # Daily loss tracking
        daily_breach_ever = np.zeros(n, dtype=bool)
        worst_intraday = np.zeros(n, dtype=np.float64)

        # Equity snapshots
        equity_snapshots: dict[int, np.ndarray] = {}
        _MILESTONES = (50, 100, 200)

        for day in range(max_days):
            if not active.any():
                break

            # Sample a block for each active sim
            for i in range(n):
                if not active[i]:
                    continue

                # Choose block
                if bad_arr is not None and rng.random() < tilt_frac:
                    block = blocks[rng.choice(bad_arr)]
                elif normal_arr is not None:
                    block = blocks[rng.choice(normal_arr)]
                else:
                    block = blocks[rng.integers(0, n_blocks)]

                daily_pnl = 0.0
                daily_peak = 0.0

                for trade_pnl in block:
                    equity[i] += trade_pnl
                    daily_pnl += trade_pnl
                    day_pnl_matrix[i, day] += trade_pnl
                    total_trades[i] += 1

                    # Trailing DD
                    peak[i] = max(peak[i], equity[i])
                    dd = peak[i] - equity[i]
                    max_dd[i] = max(max_dd[i], dd)

                    # Daily peak/drawdown
                    daily_peak = max(daily_peak, daily_pnl)
                    intra_dd = daily_peak - daily_pnl
                    worst_intraday[i] = max(worst_intraday[i], intra_dd)

                    # Streak
                    if trade_pnl < 0:
                        streak[i] += 1
                    else:
                        streak[i] = 0
                    max_streak[i] = max(max_streak[i], streak[i])

                    # Equity snapshots
                    t_num = int(total_trades[i])
                    if t_num in _MILESTONES:
                        if t_num not in equity_snapshots:
                            equity_snapshots[t_num] = np.full(n, np.nan)
                        equity_snapshots[t_num][i] = equity[i]

                    # Ruin check (trailing DD)
                    if dd >= self.max_loss_limit:
                        hit_ruin[i] = True
                        active[i] = False
                        break

                    # Target check
                    if equity[i] >= self.profit_target:
                        if not hit_target[i]:
                            hit_target[i] = True
                            days_to_target[i] = day + 1
                            trades_to_target[i] = t_num
                        active[i] = False
                        break

                # Daily loss breach
                if daily_pnl <= -self.daily_loss_limit:
                    daily_breach_ever[i] = True
                    active[i] = False

        # ── Aggregate metrics ───────────────────────────────────────────
        target_mask = hit_target & (~hit_ruin)
        n_target = int(target_mask.sum())
        n_ruin = int(hit_ruin.sum())
        n_max_days = n - n_target - n_ruin
        n_daily_breach = int(daily_breach_ever.sum())

        p_target = n_target / n
        p_ruin = n_ruin / n
        dd_p95 = float(np.percentile(max_dd, 95))
        streak_p95 = float(np.percentile(max_streak, 95))
        p_daily_breach = float(daily_breach_ever.sum() / n)

        # Consistency check
        fail_consistency = np.zeros(n, dtype=bool)
        if target_mask.any():
            total_profit = equity[target_mask]
            max_day = np.max(day_pnl_matrix[target_mask], axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(total_profit > 0, max_day / total_profit, 0.0)
            fail_consistency[target_mask] = ratio > self.consistency_cap

        p_fail_cons = int(fail_consistency.sum()) / n_target if n_target > 0 else 0.0

        median_ttt = float(
            np.median(trades_to_target[target_mask]) if target_mask.any() else 0.0
        )

        # Day-horizon percentiles
        if target_mask.any():
            dtt = days_to_target[target_mask]
            dtt_median = float(np.median(dtt))
            dtt_p5 = float(np.percentile(dtt, 5))
            dtt_p95 = float(np.percentile(dtt, 95))
            ttt_arr = trades_to_target[target_mask]
            ttt_p5 = float(np.percentile(ttt_arr, 5))
            ttt_p25 = float(np.percentile(ttt_arr, 25))
            ttt_p75 = float(np.percentile(ttt_arr, 75))
            ttt_p95 = float(np.percentile(ttt_arr, 95))
        else:
            dtt_median = dtt_p5 = dtt_p95 = 0.0
            ttt_p5 = ttt_p25 = ttt_p75 = ttt_p95 = 0.0

        # Drawdown percentiles
        dd_p50 = float(np.percentile(max_dd, 50))
        dd_p90 = float(np.percentile(max_dd, 90))
        dd_p99 = float(np.percentile(max_dd, 99))

        # Equity snapshots
        equity_at_50 = float(np.nanmean(equity_snapshots.get(50, np.array([0.0])))) if 50 in equity_snapshots else 0.0
        equity_at_100 = float(np.nanmean(equity_snapshots.get(100, np.array([0.0])))) if 100 in equity_snapshots else 0.0
        equity_at_200 = float(np.nanmean(equity_snapshots.get(200, np.array([0.0])))) if 200 in equity_snapshots else 0.0

        frac_max_days = max(n_max_days, 0) / n

        # Distribution diagnostics
        avg_trade_pnl = float(np.mean(outcomes_pool))
        std_trade_pnl = float(np.std(outcomes_pool, ddof=1)) if len(outcomes_pool) > 1 else 0.0

        skewness_val = 0.0
        kurtosis_val = 0.0
        if len(outcomes_pool) >= 3:
            try:
                from scipy.stats import skew as _skew, kurtosis as _kurtosis
                sample_std = float(np.std(outcomes_pool, ddof=0))
                if sample_std < 1e-8:
                    skewness_val = 0.0
                    kurtosis_val = 0.0
                else:
                    skewness_val = float(_skew(outcomes_pool))
                    kurtosis_val = float(_kurtosis(outcomes_pool, fisher=True))
            except ImportError:
                m = np.mean(outcomes_pool)
                s = np.std(outcomes_pool, ddof=0)
                if s > 0:
                    z = (outcomes_pool - m) / s
                    skewness_val = float(np.mean(z**3))
                    kurtosis_val = float(np.mean(z**4) - 3.0)

        worst_loss = float(np.min(outcomes_pool))
        worst_dd = float(np.max(max_dd))
        eq_p1 = float(np.percentile(equity, 1))
        eq_p10 = float(np.percentile(equity, 10))
        eq_p50 = float(np.percentile(equity, 50))
        eq_p99 = float(np.percentile(equity, 99))
        worst_intra = float(np.max(worst_intraday))

        eps = 1e-9
        expected_ttt = self.profit_target / max(avg_trade_pnl, eps) if avg_trade_pnl > eps else 0.0

        notes_parts = [
            "Day-horizon block bootstrap.",
            f"max_days={max_days}.",
            f"Trailing DD ruin.",
            f"Consistency cap = {self.consistency_cap:.0%}.",
            f"avg_trade_pnl_dollars={avg_trade_pnl:.2f}",
            f"termination: target={n_target} ruin={n_ruin} max_days={max(n_max_days, 0)}",
            f"mode=block_day_horizon",
        ]
        if stress_name != "base":
            notes_parts.append(f"stress={stress_name}")

        return SurvivalResult(
            n_simulations=n,
            p_target_before_ruin=round(p_target, 4),
            p_ruin=round(p_ruin, 4),
            p_fail_consistency_given_target=round(p_fail_cons, 4),
            dd_p95=round(dd_p95, 2),
            losing_streak_p95=round(streak_p95, 1),
            median_trades_to_target=round(median_ttt, 1),
            p_daily_loss_breach=round(p_daily_breach, 4),
            notes=" | ".join(notes_parts),
            avg_trade_pnl_dollars=round(avg_trade_pnl, 2),
            expected_trades_to_target=round(expected_ttt, 1),
            termination_hit_target=n_target,
            termination_ruin=n_ruin,
            termination_max_trades=max(n_max_days, 0),
            termination_daily_loss=n_daily_breach,
            nan_trades_skipped=nan_input_count,
            trade_count_input=trade_count_input,
            std_trade_pnl_dollars=round(std_trade_pnl, 2),
            skewness=round(skewness_val, 4),
            kurtosis=round(kurtosis_val, 4),
            worst_sampled_loss=round(worst_loss, 2),
            worst_sampled_drawdown=round(worst_dd, 2),
            equity_p1=round(eq_p1, 2),
            equity_p10=round(eq_p10, 2),
            equity_p50=round(eq_p50, 2),
            equity_p99=round(eq_p99, 2),
            worst_intraday_dd=round(worst_intra, 2),
            mode="block_day_horizon",
            stress_scenario=stress_name,
            trades_to_target_p5=round(ttt_p5, 1),
            trades_to_target_p25=round(ttt_p25, 1),
            trades_to_target_p75=round(ttt_p75, 1),
            trades_to_target_p95=round(ttt_p95, 1),
            equity_at_50=round(equity_at_50, 2),
            equity_at_100=round(equity_at_100, 2),
            equity_at_200=round(equity_at_200, 2),
            frac_terminated_max_trades=round(frac_max_days, 4),
            dd_p50=round(dd_p50, 2),
            dd_p90=round(dd_p90, 2),
            dd_p99=round(dd_p99, 2),
            days_to_target_median=round(dtt_median, 1),
            days_to_target_p5=round(dtt_p5, 1),
            days_to_target_p95=round(dtt_p95, 1),
            max_days_used=max_days,
        )

    # ── Shared equity walk ──────────────────────────────────────────────

    def _walk_and_collect(
        self,
        outcomes_matrix: np.ndarray,
        n: int,
        T: int,
        tpd: int,
        outcomes_pool: np.ndarray,
        nan_input_count: int,
        stress_name: str,
        trade_count_input: int,
        mode: str,
    ) -> SurvivalResult:
        """True path-based equity walk with trailing DD and daily limits.

        For each trade:
            equity += trade_pnl
            peak = max(peak, equity)
            trailing_dd = peak - equity
            if trailing_dd >= MAX_LOSS_LIMIT → ruin

        Daily loss:
            Group trades by trades_per_day blocks.
            Track cumulative daily PnL.
            If daily_pnl <= -DAILY_LOSS_LIMIT → terminate.
        """
        equity = np.zeros(n, dtype=np.float64)
        peak = np.zeros(n, dtype=np.float64)
        max_dd = np.zeros(n, dtype=np.float64)

        streak = np.zeros(n, dtype=np.int64)
        max_streak = np.zeros(n, dtype=np.int64)

        hit_target = np.zeros(n, dtype=bool)
        hit_ruin = np.zeros(n, dtype=bool)
        active = np.ones(n, dtype=bool)

        trades_to_target = np.full(n, T, dtype=np.int64)

        # Daily tracking
        daily_pnl = np.zeros(n, dtype=np.float64)
        daily_breach_ever = np.zeros(n, dtype=bool)

        # Store per-day PnL for consistency-cap checking
        n_days = (T + tpd - 1) // tpd
        day_pnl_matrix = np.zeros((n, n_days), dtype=np.float64)

        # Track worst intraday drawdown
        daily_peak = np.zeros(n, dtype=np.float64)
        worst_intraday = np.zeros(n, dtype=np.float64)

        # Equity snapshots at milestones for profit-path slope
        equity_snapshots: dict[int, np.ndarray] = {}
        _MILESTONES = (50, 100, 200)

        for t in range(T):
            day_idx = t // tpd

            # New day? reset daily counter + daily peak
            if t % tpd == 0:
                daily_pnl[:] = 0.0
                daily_peak[:] = 0.0

            pnl_t = outcomes_matrix[:, t]
            equity[active] += pnl_t[active]
            daily_pnl[active] += pnl_t[active]
            day_pnl_matrix[active, day_idx] += pnl_t[active]

            # Equity milestone snapshots (for profit-path slope)
            trade_num = t + 1
            if trade_num in _MILESTONES:
                equity_snapshots[trade_num] = equity.copy()

            # True path-based trailing drawdown
            peak[active] = np.maximum(peak[active], equity[active])
            dd = peak[active] - equity[active]
            max_dd[active] = np.maximum(max_dd[active], dd)

            # Within-day drawdown tracking
            daily_peak[active] = np.maximum(daily_peak[active], daily_pnl[active])
            intraday_dd = daily_peak[active] - daily_pnl[active]
            worst_intraday[active] = np.maximum(worst_intraday[active], intraday_dd)

            # Losing streak
            is_loss = pnl_t[active] < 0
            s = streak[active].copy()
            s[~is_loss] = 0
            s[is_loss] += 1
            streak[active] = s
            max_streak[active] = np.maximum(max_streak[active], streak[active])

            # Daily loss breach
            daily_breach_now = active.copy()
            daily_breach_now[active] &= np.asarray(
                daily_pnl[active] <= -self.daily_loss_limit  # type: ignore[operator]
            )
            daily_breach_ever |= daily_breach_now

            # Trailing-drawdown ruin (true path-based)
            newly_ruined = active & ((peak - equity) >= self.max_loss_limit)
            hit_ruin |= newly_ruined

            # Target reached
            newly_targeted = active & (equity >= self.profit_target)
            target_this_step = (~hit_target) & newly_targeted
            trades_to_target[target_this_step] = t + 1
            hit_target |= newly_targeted

            # Deactivate: ruin, target, or daily breach
            active &= ~(newly_ruined | newly_targeted | daily_breach_now)

            if not active.any():
                break

        # ── Consistency-cap post-check on target-hitting scenarios ───
        fail_consistency = np.zeros(n, dtype=bool)
        target_mask = hit_target & (~hit_ruin)
        if target_mask.any():
            total_profit = equity[target_mask]
            max_day = np.max(day_pnl_matrix[target_mask], axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(total_profit > 0, max_day / total_profit, 0.0)
            fail_consistency[target_mask] = ratio > self.consistency_cap

        # ── Aggregate metrics ───────────────────────────────────────────
        n_target = int(target_mask.sum())
        n_ruin = int(hit_ruin.sum())
        n_fail_consistency = int(fail_consistency.sum())
        n_daily_breach = int(daily_breach_ever.sum())
        n_max_trades = n - n_target - n_ruin

        p_target = n_target / n
        p_ruin = n_ruin / n
        p_fail_cons = n_fail_consistency / n_target if n_target > 0 else 0.0
        dd_p95 = float(np.percentile(max_dd, 95))
        streak_p95 = float(np.percentile(max_streak, 95))
        p_daily_breach = float(daily_breach_ever.sum() / n)

        median_ttt = float(
            np.median(trades_to_target[target_mask]) if target_mask.any() else 0.0
        )

        # ── Expanded drawdown percentiles ───────────────────────────────
        dd_p50 = float(np.percentile(max_dd, 50))
        dd_p90 = float(np.percentile(max_dd, 90))
        dd_p99 = float(np.percentile(max_dd, 99))

        # ── Trades-to-target distribution ───────────────────────────────
        if target_mask.any():
            ttt_arr = trades_to_target[target_mask]
            ttt_p5 = float(np.percentile(ttt_arr, 5))
            ttt_p25 = float(np.percentile(ttt_arr, 25))
            ttt_p75 = float(np.percentile(ttt_arr, 75))
            ttt_p95 = float(np.percentile(ttt_arr, 95))
        else:
            ttt_p5 = ttt_p25 = ttt_p75 = ttt_p95 = 0.0

        # ── Equity snapshots at trade milestones ────────────────────────
        equity_at_50 = float(np.mean(equity_snapshots[50])) if 50 in equity_snapshots else 0.0
        equity_at_100 = float(np.mean(equity_snapshots[100])) if 100 in equity_snapshots else 0.0
        equity_at_200 = float(np.mean(equity_snapshots[200])) if 200 in equity_snapshots else 0.0

        # ── Fraction terminated by max_trades ───────────────────────────
        frac_max_trades = max(n_max_trades, 0) / n

        # ── Distribution diagnostics ────────────────────────────────────
        avg_trade_pnl = float(np.mean(outcomes_pool))
        std_trade_pnl = float(np.std(outcomes_pool, ddof=1)) if len(outcomes_pool) > 1 else 0.0

        skewness_val = 0.0
        kurtosis_val = 0.0
        if len(outcomes_pool) >= 3:
            try:
                from scipy.stats import skew as _skew, kurtosis as _kurtosis
                sample_std = float(np.std(outcomes_pool, ddof=0))
                if sample_std < 1e-8:
                    skewness_val = 0.0
                    kurtosis_val = 0.0
                else:
                    skewness_val = float(_skew(outcomes_pool))
                    kurtosis_val = float(_kurtosis(outcomes_pool, fisher=True))
            except ImportError:
                # scipy optional — compute manually
                m = np.mean(outcomes_pool)
                s = np.std(outcomes_pool, ddof=0)
                if s > 0:
                    z = (outcomes_pool - m) / s
                    skewness_val = float(np.mean(z**3))
                    kurtosis_val = float(np.mean(z**4) - 3.0)

        worst_loss = float(np.min(outcomes_pool))
        worst_dd = float(np.max(max_dd))
        eq_p1 = float(np.percentile(equity, 1))
        eq_p10 = float(np.percentile(equity, 10))
        eq_p50 = float(np.percentile(equity, 50))
        eq_p99 = float(np.percentile(equity, 99))
        worst_intra = float(np.max(worst_intraday))

        eps = 1e-9
        expected_ttt = self.profit_target / max(avg_trade_pnl, eps) if avg_trade_pnl > eps else 0.0

        notes_parts: list[str] = [
            "True path-based trailing DD (trade-by-trade).",
            f"Consistency cap = {self.consistency_cap:.0%} of total profit.",
            f"avg_trade_pnl_dollars={avg_trade_pnl:.2f}",
            f"expected_trades_to_target≈{expected_ttt:.0f}",
            f"termination: target={n_target} ruin={n_ruin} max_trades={max(n_max_trades, 0)}",
            f"mode={mode}",
        ]
        if stress_name != "base":
            notes_parts.append(f"stress={stress_name}")
        if p_target == 1.0:
            notes_parts.append(
                "Edge saturation likely — verify sample size and stress tests."
            )

        return SurvivalResult(
            n_simulations=n,
            p_target_before_ruin=round(p_target, 4),
            p_ruin=round(p_ruin, 4),
            p_fail_consistency_given_target=round(p_fail_cons, 4),
            dd_p95=round(dd_p95, 2),
            losing_streak_p95=round(streak_p95, 1),
            median_trades_to_target=round(median_ttt, 1),
            p_daily_loss_breach=round(p_daily_breach, 4),
            notes=" | ".join(notes_parts),
            avg_trade_pnl_dollars=round(avg_trade_pnl, 2),
            expected_trades_to_target=round(expected_ttt, 1),
            termination_hit_target=n_target,
            termination_ruin=n_ruin,
            termination_max_trades=max(n_max_trades, 0),
            termination_daily_loss=n_daily_breach,
            nan_trades_skipped=nan_input_count,
            trade_count_input=trade_count_input,
            std_trade_pnl_dollars=round(std_trade_pnl, 2),
            skewness=round(skewness_val, 4),
            kurtosis=round(kurtosis_val, 4),
            worst_sampled_loss=round(worst_loss, 2),
            worst_sampled_drawdown=round(worst_dd, 2),
            equity_p1=round(eq_p1, 2),
            equity_p10=round(eq_p10, 2),
            equity_p50=round(eq_p50, 2),
            equity_p99=round(eq_p99, 2),
            worst_intraday_dd=round(worst_intra, 2),
            mode=mode,
            stress_scenario=stress_name,
            # ── expanded diagnostics ────────────────────────────────────
            trades_to_target_p5=round(ttt_p5, 1),
            trades_to_target_p25=round(ttt_p25, 1),
            trades_to_target_p75=round(ttt_p75, 1),
            trades_to_target_p95=round(ttt_p95, 1),
            equity_at_50=round(equity_at_50, 2),
            equity_at_100=round(equity_at_100, 2),
            equity_at_200=round(equity_at_200, 2),
            frac_terminated_max_trades=round(frac_max_trades, 4),
            dd_p50=round(dd_p50, 2),
            dd_p90=round(dd_p90, 2),
            dd_p99=round(dd_p99, 2),
        )

    # ── Multi-scenario runner (base + stress) ───────────────────────────

    def run_all_scenarios(
        self,
        r_values: list[float] | np.ndarray,
        *,
        seed: int | None = None,
        use_dollar_values: bool = False,
        session_ids: list[str] | None = None,
        timestamps: list[str] | None = None,
        n_batches: int = _cfg.MC_N_BATCHES,
    ) -> dict[str, SurvivalResult]:
        """Run base, mild-stress, severe-stress, and tilt scenarios.

        When ``n_batches > 1``, each scenario is run *n_batches* times with
        different seeds.  The returned ``SurvivalResult`` carries:
        - ``p_target_batch_median/min/max`` — practical batch-spread stability
        - ``p_target_ci_lo/hi`` — Wilson 95% binomial confidence interval

        Returns a dict keyed by scenario name.
        """
        results: dict[str, SurvivalResult] = {}
        for name, preset in STRESS_PRESETS.items():
            stress = {**preset, "_name": name}

            # Tilt mode needs blocks built from *base* PnL
            tilt_frac = preset.get("tilt_frac", 0.0)
            if tilt_frac > 0 and self.mode == "block":
                # Pre-rank blocks by base PnL to define "bad" bucket
                r_arr = np.asarray(r_values, dtype=np.float64)
                valid_mask = np.isfinite(r_arr)
                r_clean = r_arr[valid_mask]
                if use_dollar_values:
                    outcomes = r_clean.copy()
                else:
                    outcomes = r_clean * self.risk_per_trade
                blocks = self._build_blocks(outcomes, session_ids, timestamps, valid_mask)
                bad_indices = self._rank_bad_blocks(blocks, preset.get("tilt_quantile", 0.20))
                stress["_bad_block_indices"] = bad_indices

            if n_batches <= 1:
                results[name] = self.run(
                    r_values,
                    seed=seed,
                    use_dollar_values=use_dollar_values,
                    session_ids=session_ids,
                    timestamps=timestamps,
                    stress=stress if name != "base" else None,
                )
            else:
                results[name] = self._run_multi_batch(
                    r_values,
                    n_batches=n_batches,
                    base_seed=seed or 42,
                    use_dollar_values=use_dollar_values,
                    session_ids=session_ids,
                    timestamps=timestamps,
                    stress=stress if name != "base" else None,
                )

            logger.info(
                "MC scenario '%s': p_target=%.4f  p_ruin=%.4f  dd_p95=$%.2f",
                name,
                results[name].p_target_before_ruin,
                results[name].p_ruin,
                results[name].dd_p95,
            )
        return results

    # ── Multi-batch runner for CI estimation ────────────────────────────

    def _run_multi_batch(
        self,
        r_values: list[float] | np.ndarray,
        *,
        n_batches: int,
        base_seed: int,
        use_dollar_values: bool = False,
        session_ids: list[str] | None = None,
        timestamps: list[str] | None = None,
        stress: dict[str, Any] | None = None,
    ) -> SurvivalResult:
        """Run *n_batches* independent MC runs and aggregate with CI."""
        batch_results: list[SurvivalResult] = []
        for b in range(n_batches):
            batch_seed = base_seed + b * 10_000
            result = self.run(
                r_values,
                seed=batch_seed,
                use_dollar_values=use_dollar_values,
                session_ids=session_ids,
                timestamps=timestamps,
                stress=stress,
            )
            batch_results.append(result)

        # Pool statistics from the "middle" batch (median p_target)
        p_targets = [r.p_target_before_ruin for r in batch_results]
        median_idx = int(np.argsort(p_targets)[len(p_targets) // 2])
        representative = batch_results[median_idx]

        # Batch spread
        batch_median = float(np.median(p_targets))
        batch_min = float(np.min(p_targets))
        batch_max = float(np.max(p_targets))

        # Wilson binomial CI (pooled across all batches)
        total_successes = sum(r.termination_hit_target for r in batch_results)
        total_trials = sum(r.n_simulations for r in batch_results)
        ci_lo, ci_hi = wilson_ci(total_successes, total_trials, _cfg.MC_CI_LEVEL)

        # Build augmented result from representative + CI fields
        # (frozen dataclass — reconstruct)
        d = {f.name: getattr(representative, f.name)
             for f in representative.__dataclass_fields__.values()}
        d["p_target_ci_lo"] = round(ci_lo, 4)
        d["p_target_ci_hi"] = round(ci_hi, 4)
        d["p_target_batch_median"] = round(batch_median, 4)
        d["p_target_batch_min"] = round(batch_min, 4)
        d["p_target_batch_max"] = round(batch_max, 4)
        # Use pooled p_target as the canonical value
        d["p_target_before_ruin"] = round(total_successes / total_trials, 4)

        return SurvivalResult(**d)

    # ── Bad-block ranking for tilt stress ───────────────────────────────

    @staticmethod
    def _rank_bad_blocks(
        blocks: list[np.ndarray], quantile: float
    ) -> list[int]:
        """Return indices of blocks in the bottom *quantile* by total PnL."""
        pnls = [(i, float(b.sum())) for i, b in enumerate(blocks)]
        pnls.sort(key=lambda x: x[1])
        n_bad = max(1, int(len(pnls) * quantile))
        return [idx for idx, _ in pnls[:n_bad]]

    # ── Stress comparison logger ────────────────────────────────────────

    @staticmethod
    def log_stress_comparison(results: dict[str, SurvivalResult]) -> str:
        """Print and return a formatted stress comparison table."""
        scenarios = [n for n in ("base", "mild", "severe", "tilt_bad_week") if n in results]

        header = f"{'Metric':<30}"
        for name in scenarios:
            header += f" {name:>14}"
        sep = "─" * (30 + 15 * len(scenarios))

        rows = [
            ("P(Target)", "p_target_before_ruin", ".2%"),
            ("P(Ruin)", "p_ruin", ".2%"),
            ("Median TTT", "median_trades_to_target", ".0f"),
            ("TTT p95", "trades_to_target_p95", ".0f"),
            ("Frac max_trades", "frac_terminated_max_trades", ".1%"),
            ("Equity @50", "equity_at_50", ",.0f"),
            ("Equity @100", "equity_at_100", ",.0f"),
            ("Equity @200", "equity_at_200", ",.0f"),
            ("DD p50", "dd_p50", ",.0f"),
            ("DD p90", "dd_p90", ",.0f"),
            ("DD p95", "dd_p95", ",.0f"),
            ("DD p99", "dd_p99", ",.0f"),
            ("Hit target", "termination_hit_target", ","),
            ("Ruin", "termination_ruin", ","),
            ("Max trades", "termination_max_trades", ","),
            ("CI lower", "p_target_ci_lo", ".4f"),
            ("CI upper", "p_target_ci_hi", ".4f"),
            ("Batch min", "p_target_batch_min", ".4f"),
            ("Batch max", "p_target_batch_max", ".4f"),
        ]
        # Day-horizon rows (conditional)
        if _cfg.MC_DAY_HORIZON_ENABLED:
            rows.extend([
                ("Days→Target med", "days_to_target_median", ".1f"),
                ("Days→Target p5", "days_to_target_p5", ".1f"),
                ("Days→Target p95", "days_to_target_p95", ".1f"),
                ("Max days used", "max_days_used", ".0f"),
            ])

        lines = [
            "",
            sep,
            "  STRESS COMPARISON TABLE",
            sep,
            header,
            sep,
        ]
        for label, attr, fmt in rows:
            row = f"{label:<30}"
            for name in scenarios:
                val = getattr(results[name], attr, 0)
                if fmt == ".2%":
                    row += f" {val:>13.2%}"
                elif fmt == ".1%":
                    row += f" {val:>13.1%}"
                elif fmt == ".4f":
                    row += f" {val:>14.4f}"
                else:
                    row += f" {val:>14{fmt}}"
            lines.append(row)
        lines.append(sep)
        lines.append("")

        table = "\n".join(lines)
        print(table)
        logger.info("Stress comparison:\n%s", table)
        return table

    # ── I/O ─────────────────────────────────────────────────────────────

    @staticmethod
    def write_results(result: SurvivalResult, output_path: str | Path) -> None:
        """Persist ``SurvivalResult`` as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = MonteCarloSurvivalSimulator._result_to_dict(result)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote MC survival results → %s", path)

    @staticmethod
    def write_all_results(
        results: dict[str, SurvivalResult],
        run_dir: str | Path,
    ) -> dict[str, Path]:
        """Persist all scenario results as separate JSON files.

        Writes:
        - ``mc_results.json``              (base scenario)
        - ``mc_results_stress_mild.json``   (mild stress)
        - ``mc_results_stress_severe.json`` (severe stress)

        Returns dict of scenario_name → output_path.
        """
        run_dir = Path(run_dir)
        paths: dict[str, Path] = {}
        for name, result in results.items():
            if name == "base":
                fname = "mc_results.json"
            else:
                fname = f"mc_results_stress_{name}.json"
            out_path = run_dir / fname
            MonteCarloSurvivalSimulator.write_results(result, out_path)
            paths[name] = out_path
        return paths

    @staticmethod
    def _result_to_dict(result: SurvivalResult) -> dict[str, Any]:
        """Convert SurvivalResult to a JSON-serialisable dict."""
        return {
            "n_simulations": result.n_simulations,
            "p_target_before_ruin": result.p_target_before_ruin,
            "p_ruin": result.p_ruin,
            "p_fail_consistency_given_target": result.p_fail_consistency_given_target,
            "dd_p95": result.dd_p95,
            "losing_streak_p95": result.losing_streak_p95,
            "median_trades_to_target": result.median_trades_to_target,
            "p_daily_loss_breach": result.p_daily_loss_breach,
            "notes": result.notes,
            "avg_trade_pnl_dollars": result.avg_trade_pnl_dollars,
            "expected_trades_to_target": result.expected_trades_to_target,
            "termination_breakdown": {
                "hit_target": result.termination_hit_target,
                "ruin": result.termination_ruin,
                "max_trades": result.termination_max_trades,
                "daily_loss_breach": result.termination_daily_loss,
            },
            "nan_trades_skipped": result.nan_trades_skipped,
            # Distribution diagnostics
            "trade_count_input": result.trade_count_input,
            "std_trade_pnl_dollars": result.std_trade_pnl_dollars,
            "skewness": result.skewness,
            "kurtosis": result.kurtosis,
            "worst_sampled_loss": result.worst_sampled_loss,
            "worst_sampled_drawdown": result.worst_sampled_drawdown,
            "equity_p1": result.equity_p1,
            "equity_p10": result.equity_p10,
            "equity_p50": result.equity_p50,
            "equity_p99": result.equity_p99,
            "worst_intraday_dd": result.worst_intraday_dd,
            # Metadata
            "mode": result.mode,
            "stress_scenario": result.stress_scenario,
            "generated": datetime.now(timezone.utc).isoformat(),
            # ── Expanded diagnostics (Phase 2) ──────────────────────────
            "trades_to_target_distribution": {
                "p5": result.trades_to_target_p5,
                "p25": result.trades_to_target_p25,
                "p50": result.median_trades_to_target,
                "p75": result.trades_to_target_p75,
                "p95": result.trades_to_target_p95,
            },
            "equity_slope": {
                "at_50_trades": result.equity_at_50,
                "at_100_trades": result.equity_at_100,
                "at_200_trades": result.equity_at_200,
            },
            "frac_terminated_max_trades": result.frac_terminated_max_trades,
            "drawdown_distribution": {
                "p50": result.dd_p50,
                "p90": result.dd_p90,
                "p95": result.dd_p95,
                "p99": result.dd_p99,
            },
            # ── Confidence interval (Phase 3) ──────────────────────────
            "p_target_ci": {
                "lower": result.p_target_ci_lo,
                "upper": result.p_target_ci_hi,
            },
            "p_target_batch_spread": {
                "median": result.p_target_batch_median,
                "min": result.p_target_batch_min,
                "max": result.p_target_batch_max,
            },
            # ── Day-horizon metrics ─────────────────────────────────────
            "day_horizon": {
                "enabled": _cfg.MC_DAY_HORIZON_ENABLED,
                "max_days": _cfg.MC_MAX_DAYS if _cfg.MC_DAY_HORIZON_ENABLED else None,
                "days_to_target_median": result.days_to_target_median,
                "days_to_target_p5": result.days_to_target_p5,
                "days_to_target_p95": result.days_to_target_p95,
                "max_days_used": result.max_days_used,
            },
        }
