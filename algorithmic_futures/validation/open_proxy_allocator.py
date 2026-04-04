"""
validation/open_proxy_allocator.py — Open Proxy v1 Allocator.

Replaces ADX-based trend detection at the open with price-action signals
that are available by 09:45 ET (end of the 15-minute opening range).

Signals used:
  1) opening_range_width_atr  — OR width normalised by ATR
  2) first_3bar_impulse_atr   — |net move over first 3 bars| / ATR
  3) breakout_persistence      — price closes beyond OR high/low for N bars

Decision:
  Route to ORB if ANY configured trend condition fires; else MR.

This module is stateless — call ``decide()`` once per session after the
opening range window closes.

Part of the v1_1 investigation (allocator warmup fix).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class OpenProxyConfig:
    """Tuneable thresholds for open_proxy_v1 allocator."""
    or_width_atr_threshold: float = 2.2    # OR width / ATR >= this → trend signal
    impulse_atr_threshold: float = 0.9     # |first 3-bar net move| / ATR >= this → trend signal
    persist_bars: int = 1                  # consecutive closes beyond OR to confirm breakout
    require_break: bool = False            # if True, breakout_persistence must fire (not just width/impulse)
    enable_orb_selectivity_refinement: bool = config.ALLOC_OPENPROXY_SELECTIVITY_ENABLED
    orb_selectivity_low_atr_threshold: float = config.ALLOC_OPENPROXY_LOW_ATR_THRESHOLD
    orb_selectivity_min_persistence_in_low_atr: int = config.ALLOC_OPENPROXY_MIN_PERSISTENCE_IN_LOW_ATR
    orb_selectivity_high_impulse_threshold: float = config.ALLOC_OPENPROXY_HIGH_IMPULSE_THRESHOLD
    orb_selectivity_min_persistence_when_high_impulse: int = config.ALLOC_OPENPROXY_MIN_PERSISTENCE_WHEN_HIGH_IMPULSE
    enable_medium_impulse_weak_persistence_filter: bool = config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_WEAK_PERSISTENCE_FILTER_ENABLED
    enable_medium_impulse_decay_filter: bool = config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_DECAY_FILTER_ENABLED
    medium_impulse_min_atr: float = config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MIN_ATR
    medium_impulse_max_atr: float = config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MAX_ATR
    medium_impulse_min: float = config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MIN
    medium_impulse_max: float = config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MAX
    medium_impulse_min_persistence: int = config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MIN_PERSISTENCE


# ═══════════════════════════════════════════════════════════════════════
#  Bar accumulator (collects data during open window)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OpenWindowState:
    """Mutable accumulator for open-window bars (09:30–09:45 ET)."""
    or_high: float = float("-inf")
    or_low: float = float("inf")
    bars: list = field(default_factory=list)   # list of (open, high, low, close) tuples
    first_bar_open: float | None = None
    atr_at_decision: float = 0.0
    # Post-OR monitoring (09:45+ bars for persistence check)
    post_or_bars: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
#  Decision result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class OpenProxyDecision:
    """Full diagnostic output of the open_proxy_v1 allocator."""
    decision: str                        # "orb" or "mr"
    reason: str                          # human-readable reason string
    or_high: float = 0.0
    or_low: float = 0.0
    opening_range_width_pts: float = 0.0
    opening_range_width_atr: float = 0.0
    first_3bar_directional_impulse: float = 0.0
    signed_imbalance: float = 0.0        # directional net move (signed)
    breakout_persistence: bool = False
    breakout_direction: str = ""         # "UP", "DOWN", or ""
    persist_bars_observed: int = 0
    atr_at_decision: float = 0.0
    n_or_bars: int = 0
    n_post_or_bars: int = 0
    # Individual trigger flags
    trigger_width: bool = False
    trigger_impulse: bool = False
    trigger_persist: bool = False
    pre_selectivity_decision: str = ""
    pre_selectivity_reason: str = ""
    selectivity_refinement_enabled: bool = False
    selectivity_low_atr_caution: bool = False
    selectivity_high_impulse_caution: bool = False
    selectivity_orb_blocked: bool = False
    selectivity_block_reason: str = ""
    selectivity_medium_impulse_weak_persistence_caution: bool = False
    selectivity_medium_impulse_decay_caution: bool = False
    selectivity_v3_orb_blocked: bool = False
    selectivity_v3_block_reason: str = ""
    pre_v3_selectivity_decision: str = ""
    post_v3_selectivity_decision: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  Core decision logic
# ═══════════════════════════════════════════════════════════════════════

def decide(state: OpenWindowState, cfg: OpenProxyConfig) -> OpenProxyDecision:
    """Evaluate open_proxy_v1 allocator from accumulated bar data.

    Call this once the decision bar is reached (typically the bar after
    ORB_END, i.e. the 4th bar at 09:45 ET).

    Parameters
    ----------
    state : OpenWindowState
        Bars accumulated during 09:30–09:45 *and* any post-OR bars
        collected for persistence checking.
    cfg : OpenProxyConfig
        Tuneable thresholds.

    Returns
    -------
    OpenProxyDecision with full diagnostics.
    """
    result = OpenProxyDecision(decision="mr", reason="DEFAULT_MR")

    # ── 1.  Opening range width ─────────────────────────────────────────
    or_high = state.or_high
    or_low = state.or_low
    result.or_high = or_high
    result.or_low = or_low
    result.n_or_bars = len(state.bars)
    result.n_post_or_bars = len(state.post_or_bars)
    result.atr_at_decision = state.atr_at_decision

    if or_high <= or_low or state.atr_at_decision <= 0:
        result.reason = "INSUFFICIENT_DATA"
        return result

    or_width = or_high - or_low
    or_width_atr = or_width / state.atr_at_decision
    result.opening_range_width_pts = round(or_width, 4)
    result.opening_range_width_atr = round(or_width_atr, 4)

    # ── 2.  First 3-bar directional impulse ─────────────────────────────
    if len(state.bars) >= 3 and state.first_bar_open is not None:
        # Net move = close of bar 3 minus open of bar 1
        third_close = state.bars[2][3]   # (O, H, L, C)[3]
        net_move = third_close - state.first_bar_open
        impulse_abs = abs(net_move) / state.atr_at_decision
        result.first_3bar_directional_impulse = round(impulse_abs, 4)
        result.signed_imbalance = round(net_move / state.atr_at_decision, 4)
    elif len(state.bars) >= 1 and state.first_bar_open is not None:
        # Fewer than 3 bars — use whatever we have
        last_close = state.bars[-1][3]
        net_move = last_close - state.first_bar_open
        impulse_abs = abs(net_move) / state.atr_at_decision
        result.first_3bar_directional_impulse = round(impulse_abs, 4)
        result.signed_imbalance = round(net_move / state.atr_at_decision, 4)

    # ── 3.  Breakout persistence ────────────────────────────────────────
    if state.post_or_bars:
        up_persist = 0
        down_persist = 0
        for (_, _, _, c) in state.post_or_bars:
            if c > or_high:
                up_persist += 1
            else:
                break
        for (_, _, _, c) in state.post_or_bars:
            if c < or_low:
                down_persist += 1
            else:
                break
        best_persist = max(up_persist, down_persist)
        result.persist_bars_observed = best_persist
        if best_persist >= cfg.persist_bars:
            result.breakout_persistence = True
            result.trigger_persist = True
            result.breakout_direction = "UP" if up_persist >= down_persist else "DOWN"

    # ── 4.  Threshold checks ────────────────────────────────────────────
    width_fires = or_width_atr >= cfg.or_width_atr_threshold
    impulse_fires = result.first_3bar_directional_impulse >= cfg.impulse_atr_threshold
    persist_fires = result.breakout_persistence

    result.trigger_width = width_fires
    result.trigger_impulse = impulse_fires

    # Decision logic
    if cfg.require_break:
        # Must have breakout persistence AND at least one of width/impulse
        if persist_fires and (width_fires or impulse_fires):
            result.decision = "orb"
            triggers = []
            if width_fires:
                triggers.append(f"OR_WIDTH_ATR={or_width_atr:.2f}")
            if impulse_fires:
                triggers.append(f"IMPULSE_ATR={result.first_3bar_directional_impulse:.2f}")
            triggers.append(f"PERSIST_{result.breakout_direction}={result.persist_bars_observed}")
            result.reason = "OPEN_PROXY_TREND_" + "+".join(triggers)
        else:
            reasons = []
            if not persist_fires:
                reasons.append("NO_PERSIST")
            if not width_fires:
                reasons.append(f"WIDTH={or_width_atr:.2f}<{cfg.or_width_atr_threshold}")
            if not impulse_fires:
                reasons.append(f"IMPULSE={result.first_3bar_directional_impulse:.2f}<{cfg.impulse_atr_threshold}")
            result.reason = "OPEN_PROXY_RANGE_" + "+".join(reasons)
    else:
        # ANY single trigger fires → ORB
        if width_fires or impulse_fires or persist_fires:
            result.decision = "orb"
            triggers = []
            if width_fires:
                triggers.append(f"OR_WIDTH_ATR={or_width_atr:.2f}")
            if impulse_fires:
                triggers.append(f"IMPULSE_ATR={result.first_3bar_directional_impulse:.2f}")
            if persist_fires:
                triggers.append(f"PERSIST_{result.breakout_direction}={result.persist_bars_observed}")
            result.reason = "OPEN_PROXY_TREND_" + "+".join(triggers)
        else:
            result.reason = (
                f"OPEN_PROXY_RANGE_WIDTH={or_width_atr:.2f}"
                f"+IMPULSE={result.first_3bar_directional_impulse:.2f}"
                f"+PERSIST={result.persist_bars_observed}"
            )

    result.pre_selectivity_decision = result.decision
    result.pre_selectivity_reason = result.reason
    result.selectivity_refinement_enabled = cfg.enable_orb_selectivity_refinement

    if cfg.enable_orb_selectivity_refinement and result.decision == "orb":
        low_atr_caution = (
            result.atr_at_decision < cfg.orb_selectivity_low_atr_threshold
            and result.persist_bars_observed < cfg.orb_selectivity_min_persistence_in_low_atr
        )
        high_impulse_caution = (
            result.first_3bar_directional_impulse >= cfg.orb_selectivity_high_impulse_threshold
            and result.persist_bars_observed < cfg.orb_selectivity_min_persistence_when_high_impulse
        )
        result.selectivity_low_atr_caution = low_atr_caution
        result.selectivity_high_impulse_caution = high_impulse_caution

        if low_atr_caution or high_impulse_caution:
            block_reasons: list[str] = []
            if low_atr_caution:
                block_reasons.append(
                    f"LOW_ATR_{result.atr_at_decision:.2f}<"
                    f"{cfg.orb_selectivity_low_atr_threshold:.2f}_PERSIST_{result.persist_bars_observed}<"
                    f"{cfg.orb_selectivity_min_persistence_in_low_atr}"
                )
            if high_impulse_caution:
                block_reasons.append(
                    f"HIGH_IMPULSE_{result.first_3bar_directional_impulse:.2f}>="
                    f"{cfg.orb_selectivity_high_impulse_threshold:.2f}_PERSIST_{result.persist_bars_observed}<"
                    f"{cfg.orb_selectivity_min_persistence_when_high_impulse}"
                )
            result.selectivity_orb_blocked = True
            result.selectivity_block_reason = "+".join(block_reasons)
            result.decision = "mr"
            result.reason = "OPEN_PROXY_RANGE_SELECTIVITY_" + result.selectivity_block_reason

    result.pre_v3_selectivity_decision = result.decision
    result.post_v3_selectivity_decision = result.decision

    if result.decision == "orb":
        medium_decay_band = (
            cfg.enable_medium_impulse_decay_filter
            and result.atr_at_decision >= cfg.medium_impulse_min_atr
            and result.atr_at_decision < cfg.medium_impulse_max_atr
            and result.first_3bar_directional_impulse >= cfg.medium_impulse_min
            and result.first_3bar_directional_impulse < cfg.medium_impulse_max
            and result.persist_bars_observed < cfg.medium_impulse_min_persistence
        )
        result.selectivity_medium_impulse_decay_caution = medium_decay_band
        if medium_decay_band:
            decay_reason = (
                "OPEN_PROXY_RANGE_SELECTIVITY_V4_MEDIUM_IMPULSE_DECAY "
                f"impulse_atr={result.first_3bar_directional_impulse:.2f} "
                f"atr={result.atr_at_decision:.2f} "
                f"persistence={result.persist_bars_observed}<"
                f"{cfg.medium_impulse_min_persistence} "
                f"atr_band=[{cfg.medium_impulse_min_atr:.2f},{cfg.medium_impulse_max_atr:.2f}) "
                f"impulse_band=[{cfg.medium_impulse_min:.2f},{cfg.medium_impulse_max:.2f})"
            )
            result.selectivity_v3_orb_blocked = True
            result.selectivity_v3_block_reason = decay_reason
            result.selectivity_orb_blocked = True
            result.selectivity_block_reason = decay_reason
            result.decision = "mr"
            result.reason = decay_reason

    if result.decision == "orb":
        medium_band_lower = float(cfg.impulse_atr_threshold)
        medium_band_upper = float(cfg.orb_selectivity_high_impulse_threshold)
        medium_impulse_band = (
            medium_band_upper > medium_band_lower
            and result.first_3bar_directional_impulse >= medium_band_lower
            and result.first_3bar_directional_impulse < medium_band_upper
        )
        weak_persistence = result.persist_bars_observed == 0

        result.selectivity_medium_impulse_weak_persistence_caution = (
            cfg.enable_medium_impulse_weak_persistence_filter
            and medium_impulse_band
            and weak_persistence
        )
        if result.selectivity_medium_impulse_weak_persistence_caution:
            v3_reason = (
                "OPEN_PROXY_RANGE_SELECTIVITY_V3_MEDIUM_IMPULSE_WEAK_PERSISTENCE "
                f"impulse_atr={result.first_3bar_directional_impulse:.2f} "
                f"band=[{medium_band_lower:.2f},{medium_band_upper:.2f}) "
                f"persistence={result.persist_bars_observed}"
            )
            result.selectivity_v3_orb_blocked = True
            result.selectivity_v3_block_reason = v3_reason
            result.selectivity_orb_blocked = True
            result.selectivity_block_reason = v3_reason
            result.decision = "mr"
            result.reason = v3_reason

    result.post_v3_selectivity_decision = result.decision

    logger.info(
        "open_proxy_v1 decision=%s reason=%s  "
        "or_width_atr=%.2f impulse_atr=%.2f persist=%s/%d",
        result.decision, result.reason,
        or_width_atr, result.first_3bar_directional_impulse,
        result.breakout_persistence, result.persist_bars_observed,
    )
    return result
