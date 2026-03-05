"""
risk/monte_carlo.py — Monte Carlo drawdown simulation validator.

Validates that chosen risk parameters (win rate, R:R, position size)
can survive the $2,000 MLL constraint with high probability before
any strategy or API code is engaged.

Usage:
    from risk.monte_carlo import MonteCarloValidator
    mc = MonteCarloValidator()
    result = mc.run(win_rate=0.55, avg_win=30.0, avg_loss=-22.0)
    print(result)
    assert result.accepted
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from config import (
    MAX_LOSS_LIMIT,
    MC_DRAWDOWN_P95_MAX,
    MC_LOSING_STREAK_P95_MAX,
    MC_MAX_TRADES,
    MC_RUIN_THRESHOLD,
    MC_SIMULATIONS,
    MC_TARGET_THRESHOLD,
    PROFIT_TARGET,
)

logger = logging.getLogger(__name__)


# ── Result container ────────────────────────────────────────────────────


@dataclass(frozen=True)
class MonteCarloResult:
    """Immutable container for a single MC validation run."""

    n_simulations: int
    ruin_probability: float
    target_probability: float
    max_drawdown_p95: float
    max_losing_streak_p95: float
    accepted: bool
    rejection_reasons: tuple[str, ...]

    def summary(self) -> str:
        status = "ACCEPTED" if self.accepted else "REJECTED"
        lines = [
            f"Monte Carlo Validation — {status}",
            f"  Simulations       : {self.n_simulations:,}",
            f"  Ruin probability  : {self.ruin_probability:.2%}",
            f"  Target probability: {self.target_probability:.2%}",
            f"  Max DD (p95)      : ${self.max_drawdown_p95:,.2f}",
            f"  Max losing streak : {self.max_losing_streak_p95:.0f}",
        ]
        if self.rejection_reasons:
            lines.append("  Rejection reasons :")
            for r in self.rejection_reasons:
                lines.append(f"    - {r}")
        return "\n".join(lines)


# ── Validator ───────────────────────────────────────────────────────────


class MonteCarloValidator:
    """Vectorised Monte Carlo engine for trade-path simulation."""

    def __init__(
        self,
        starting_capital: float = MAX_LOSS_LIMIT,
        ruin_boundary: float = 0.0,
        target_boundary: float | None = None,
        n_simulations: int = MC_SIMULATIONS,
        max_trades: int = MC_MAX_TRADES,
    ) -> None:
        self.starting_capital = starting_capital
        self.ruin_boundary = ruin_boundary
        self.target_boundary = (
            target_boundary
            if target_boundary is not None
            else starting_capital + PROFIT_TARGET
        )
        self.n_simulations = n_simulations
        self.max_trades = max_trades

    # ── core simulation ─────────────────────────────────────────────────

    def run(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        *,
        seed: int | None = None,
    ) -> MonteCarloResult:
        """Run *n_simulations* randomised trade paths and evaluate acceptance.

        Parameters
        ----------
        win_rate : float   Probability of a winning trade (0–1).
        avg_win  : float   Mean dollar gain on a winning trade (positive).
        avg_loss : float   Mean dollar loss on a losing trade (negative).
        seed     : int     Optional RNG seed for reproducibility.
        """
        if not (0.0 <= win_rate <= 1.0):
            raise ValueError(f"win_rate must be in [0, 1], got {win_rate}")
        if avg_win <= 0:
            raise ValueError(f"avg_win must be positive, got {avg_win}")
        if avg_loss >= 0:
            raise ValueError(f"avg_loss must be negative, got {avg_loss}")

        rng = np.random.default_rng(seed)

        # Pre-generate all outcomes: shape (n_simulations, max_trades)
        is_win = rng.random((self.n_simulations, self.max_trades)) < win_rate
        outcomes = np.where(is_win, avg_win, avg_loss)

        # Walk forward cumulatively
        capital = np.full(self.n_simulations, self.starting_capital, dtype=np.float64)
        peak = capital.copy()

        max_dd = np.zeros(self.n_simulations, dtype=np.float64)
        streak = np.zeros(self.n_simulations, dtype=np.int64)
        max_streak = np.zeros(self.n_simulations, dtype=np.int64)

        hit_ruin = np.zeros(self.n_simulations, dtype=bool)
        hit_target = np.zeros(self.n_simulations, dtype=bool)
        active = np.ones(self.n_simulations, dtype=bool)  # still running

        for t in range(self.max_trades):
            # Apply trade outcome only to active paths
            capital[active] += outcomes[active, t]

            # Peak / drawdown tracking
            peak[active] = np.maximum(peak[active], capital[active])
            dd = peak[active] - capital[active]
            max_dd[active] = np.maximum(max_dd[active], dd)

            # Losing-streak tracking
            is_loss = outcomes[active, t] < 0
            # Reset streak for winners, increment for losers
            streak_active = streak[active]
            streak_active[~is_loss] = 0
            streak_active[is_loss] += 1
            streak[active] = streak_active
            max_streak[active] = np.maximum(max_streak[active], streak[active])

            # Boundary checks
            newly_ruined = active & (capital <= self.ruin_boundary)
            newly_targeted = active & (capital >= self.target_boundary)

            hit_ruin |= newly_ruined
            hit_target |= newly_targeted

            # Deactivate paths that hit a boundary
            active &= ~(newly_ruined | newly_targeted)

            if not active.any():
                break

        # ── metrics ─────────────────────────────────────────────────────
        ruin_prob = hit_ruin.sum() / self.n_simulations
        target_prob = hit_target.sum() / self.n_simulations
        dd_p95 = float(np.percentile(max_dd, 95))
        streak_p95 = float(np.percentile(max_streak, 95))

        # ── acceptance ──────────────────────────────────────────────────
        reasons: list[str] = []
        if ruin_prob > MC_RUIN_THRESHOLD:
            reasons.append(
                f"Ruin probability {ruin_prob:.2%} exceeds {MC_RUIN_THRESHOLD:.0%}"
            )
        if target_prob < MC_TARGET_THRESHOLD:
            reasons.append(
                f"Target probability {target_prob:.2%} below {MC_TARGET_THRESHOLD:.0%}"
            )
        if dd_p95 > MC_DRAWDOWN_P95_MAX:
            reasons.append(
                f"95th pct drawdown ${dd_p95:,.0f} exceeds ${MC_DRAWDOWN_P95_MAX:,.0f}"
            )
        if streak_p95 >= MC_LOSING_STREAK_P95_MAX:
            reasons.append(
                f"95th pct losing streak {streak_p95:.0f} >= {MC_LOSING_STREAK_P95_MAX}"
            )

        result = MonteCarloResult(
            n_simulations=self.n_simulations,
            ruin_probability=ruin_prob,
            target_probability=target_prob,
            max_drawdown_p95=dd_p95,
            max_losing_streak_p95=streak_p95,
            accepted=len(reasons) == 0,
            rejection_reasons=tuple(reasons),
        )

        logger.info(result.summary())
        return result
