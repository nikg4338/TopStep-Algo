"""
strategies/mr_signal_engine.py — VWAP Mean Reversion signal engine (v1).

Generates candidate MR signals from 5-minute bars, evaluates entry conditions
against VWAP σ-bands, and produces structured signal objects.  Does NOT
execute trades — this is a pure signal/research layer.

CVD gating is architecture-ready via OrderFlowFilterBase interface but
uses NoOpOrderFlowFilter by default (passes all signals through).

Usage:
    from strategies.mr_signal_engine import MRSignalEngine
    engine = MRSignalEngine()
    engine.on_bar(bar, regime, vwap_state, atr)
    signals = engine.signals  # list[MRSignal]
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from config import (
    LAST_ENTRY_CUTOFF,
    MR_COOLDOWN_BARS,
    MR_MAX_ATTEMPTS_PER_SIDE,
    MR_ATTEMPT_CAP_MODE,
    MR_SOFT_CAP_COOLDOWN_BARS,
    MR_SOFT_CAP_MIN_ZSCORE,
    MR_MIN_DISTANCE_VWAP_TICKS,
    MR_RECLAIM_TICKS,
    MR_SIGMA_ENTRY,
    MR_SIGMA_EXTREME,
    TICK_SIZE,
    TIMEZONE,
    VWAP_STOP_ATR_MULT,
    MR_CLUSTER_RESET_ZSCORE,
    MR_CLUSTER_RESET_ENABLED,
    MR_CLUSTER_RESET_MODE,
    MR_CLUSTER_RETRACE_FRACTION,
    MR_CLUSTER_RESET_MIN_PEAK_Z,
    MR_QUALITY_MIN_EXCURSION_ATR,
    MR_FILTER_DISTANCE_ENABLED,
    MR_QUALITY_VWAP_FLAT_LOOKBACK,
    MR_QUALITY_VWAP_FLAT_MAX_ATR,
    MR_FILTER_VWAP_FLAT_ENABLED,
    MR_QUALITY_RECLAIM_CLOSE_LOC_MIN,
    MR_SOFT_RECLAIM_RANGE_IMPULSE_K,
    MR_FIRST_OUTSIDE_ENABLED,
    MR_TOUCH_LATCH_RESET_BUFFER,
    MR_DEDUPE_WINDOW_BARS,
    MR_DEDUPE_MIN_DELTA_Z,
    MR_REGIME_ADX_BUCKETS,
    MR_FILTER_RECLAIM_STRENGTH_ENABLED,
    MR_EXCURSION_DEDUPE_ENABLED,
    MR_EXCURSION_RESET_ZSCORE,
    MR_EXCURSION_RESET_VWAP_TICKS,
    TREND_CONTAM_ENABLED,
    TREND_CONTAM_ADX_THRESHOLD,
    TREND_CONTAM_VWAP_SLOPE_MIN,
    TREND_CONTAM_SLOPE_LOOKBACK,
)
from data.indicators import VWAPState
from data.market_data import Bar

import pytz

logger = logging.getLogger(__name__)

_ET = pytz.timezone(TIMEZONE)
_UTC = pytz.utc


# ── Signal dataclass ────────────────────────────────────────────────────

@dataclass
class MRSignal:
    """Structured signal object for mean reversion candidates."""
    timestamp: datetime
    side: Literal["BUY", "SELL"]
    signal_type: str = "MR"
    regime_at_signal: str | None = None
    entry_reference_price: float = 0.0
    stop_reference: float = 0.0       # ATR-derived placeholder
    target_reference: float = 0.0     # VWAP or staged target
    band_level_hit: float = 0.0       # 2.5 or 3.0
    vwap_at_signal: float = 0.0
    sigma_at_signal: float = 0.0
    z_at_signal: float = 0.0
    approved: bool = False
    rejection_reason: str = ""
    bar_index: int = 0


# ── Order Flow Filter interface (CVD stub) ──────────────────────────────

class OrderFlowFilterBase(ABC):
    """Interface for order-flow confirmation filters."""

    @abstractmethod
    def allows(self, side: str, bar: Bar) -> tuple[bool, str]:
        """Return (allowed, reason) for a candidate signal."""
        ...


class NoOpOrderFlowFilter(OrderFlowFilterBase):
    """Default pass-through filter (CVD not active)."""

    def allows(self, side: str, bar: Bar) -> tuple[bool, str]:
        return True, ""


class CVDProxyFilter(OrderFlowFilterBase):
    """Stub for CVD-based confirmation.  Not implemented yet."""

    def allows(self, side: str, bar: Bar) -> tuple[bool, str]:
        # TODO: implement CVD divergence check
        return True, "CVD_STUB_PASS"


# ── MR Signal Engine ───────────────────────────────────────────────────

class MRSignalEngine:
    """VWAP Mean Reversion signal generator (v1, no CVD gating).

    Only generates signals when regime == "range".
    Applies cooldown, side limits, distance checks.
    Tracks rejection counters and supports cluster-based attempt reset.
    """

    def __init__(
        self,
        flow_filter: OrderFlowFilterBase | None = None,
        reclaim_mode: Literal["on", "off", "soft", "touch"] = "on",
        sigma_entry: float = MR_SIGMA_ENTRY,
        soft_reclaim_range_impulse_k: float = MR_SOFT_RECLAIM_RANGE_IMPULSE_K,
        cooldown_bars: int = MR_COOLDOWN_BARS,
        max_attempts_per_side: int = MR_MAX_ATTEMPTS_PER_SIDE,
        excursion_dedupe_enabled: bool = MR_EXCURSION_DEDUPE_ENABLED,
        attempt_cap_enabled: bool = True,
        first_outside_enabled: bool = MR_FIRST_OUTSIDE_ENABLED,
        touch_latch_reset_buffer: float = MR_TOUCH_LATCH_RESET_BUFFER,
        dedupe_window_bars: int = MR_DEDUPE_WINDOW_BARS,
        dedupe_min_delta_z: float = MR_DEDUPE_MIN_DELTA_Z,
        regime_enabled: bool = True,
    ) -> None:
        self._flow_filter = flow_filter or NoOpOrderFlowFilter()
        if reclaim_mode not in {"on", "off", "soft", "touch"}:
            raise ValueError("reclaim_mode must be 'on', 'off', 'soft', or 'touch'")
        self._reclaim_mode: Literal["on", "off", "soft", "touch"] = reclaim_mode
        self._sigma_entry: float = max(0.1, float(sigma_entry))
        self._sigma_extreme: float = max(self._sigma_entry, float(MR_SIGMA_EXTREME))
        self._soft_reclaim_range_impulse_k: float = max(0.0, float(soft_reclaim_range_impulse_k))
        self._cooldown_bars: int = max(0, int(cooldown_bars))
        self._max_attempts_per_side: int = max(1, int(max_attempts_per_side))
        self._excursion_dedupe_enabled: bool = bool(excursion_dedupe_enabled)
        self._attempt_cap_enabled: bool = bool(attempt_cap_enabled)
        self._first_outside_enabled: bool = bool(first_outside_enabled)
        self._touch_latch_reset_buffer: float = max(0.0, float(touch_latch_reset_buffer))
        self._dedupe_window_bars: int = max(0, int(dedupe_window_bars))
        self._dedupe_min_delta_z: float = max(0.0, float(dedupe_min_delta_z))
        self._regime_enabled: bool = bool(regime_enabled)
        self._signals: list[MRSignal] = []
        self._bar_index: int = 0
        self._bars_evaluated: int = 0
        self._eligible_session_bars: int = 0
        self._z_cross_inside_to_outside: int = 0
        self._z_cross_outside_to_inside: int = 0
        self._first_eligible_bar_outside: int = 0
        self._seen_first_eligible_bar: bool = False
        self._z_cross_time_buckets: dict[str, int] = {}
        self._z_value_time_buckets: dict[str, list[float]] = {}
        self._session_z_values: list[float] = []
        self._soft_range_impulse_values: list[float] = []
        self._soft_range_impulse_rejections: int = 0
        self._cross_body_impulse_abs_values: list[float] = []
        self._cross_range_impulse_values: list[float] = []

        # Per-session tracking
        self._long_attempts: int = 0
        self._short_attempts: int = 0
        self._last_signal_bar: int = -100  # bar index of last emitted signal
        self._last_long_signal_bar: int = -100
        self._last_short_signal_bar: int = -100

        # Cluster reset tracking
        self._was_in_zone: bool = False  # True when |z| was above entry σ
        self._cluster_peak_long_z: float = 0.0
        self._cluster_peak_short_z: float = 0.0

        # Reclaim + excursion state
        self._prev_z_score: float | None = None
        self._long_excursion_active: bool = False
        self._short_excursion_active: bool = False
        self._long_excursion_traded: bool = False
        self._short_excursion_traded: bool = False
        self._long_touch_latch_armed: bool = True
        self._short_touch_latch_armed: bool = True
        self._last_candidate_bar_by_side: dict[str, int] = {"BUY": -100, "SELL": -100}
        self._last_candidate_abs_z_by_side: dict[str, float] = {"BUY": 0.0, "SELL": 0.0}

        # VWAP slope tracking (rolling VWAP values for slope computation)
        self._vwap_history: list[float] = []

        # ── Rejection counters ──────────────────────────────────────────
        self._rejection_counters: dict[str, int] = {
            "signals_generated": 0,
            "signals_approved": 0,
            "rejected_by_attempt_cap": 0,
            "rejected_by_regime": 0,
            "rejected_by_cooldown": 0,
            "rejected_by_daily_loss_governor": 0,
            "rejected_by_profit_cap": 0,
            "rejected_by_trend_contamination": 0,
            "rejected_by_session_cutoff": 0,
            "rejected_by_flow_filter": 0,
            "rejected_by_excursion_dedupe": 0,
        }

        # Ordered gate funnel (candidate -> approvals)
        self._funnel_gate_order: list[str] = [
            "distance_ticks",
            "distance_atr",
            "vwap_flatness",
            "reclaim_strength",
            "cooldown",
            "excursion_dedupe",
            "attempt_cap",
            "trend_contamination",
            "session_cutoff",
            "flow_filter",
        ]
        self._funnel_passed: dict[str, int] = {g: 0 for g in self._funnel_gate_order}
        self._funnel_fail_reason: dict[str, int] = {}
        self._funnel_candidates_total: int = 0
        self._funnel_approved_trades: int = 0
        self._drop_ledger: dict[str, int] = {
            "bars_evaluated": 0,
            "eligible_session_bars": 0,
            "z_cross_events": 0,
            "dedupe_rejects": 0,
            "attempt_limit_rejects": 0,
            "cooldown_rejects": 0,
            "in_position_rejects": 0,
            "regime_rejects": 0,
            "spread_liquidity_rejects": 0,
            "candidates_formed": 0,
            "orders_submitted": 0,
            "fills": 0,
            "trades": 0,
        }

    def reset(self) -> None:
        """Reset at session open."""
        self._signals.clear()
        self._bar_index = 0
        self._long_attempts = 0
        self._short_attempts = 0
        self._last_signal_bar = -100
        self._last_long_signal_bar = -100
        self._last_short_signal_bar = -100
        self._was_in_zone = False
        self._cluster_peak_long_z = 0.0
        self._cluster_peak_short_z = 0.0
        self._prev_z_score = None
        self._long_excursion_active = False
        self._short_excursion_active = False
        self._long_excursion_traded = False
        self._short_excursion_traded = False
        self._long_touch_latch_armed = True
        self._short_touch_latch_armed = True
        self._last_candidate_bar_by_side = {"BUY": -100, "SELL": -100}
        self._last_candidate_abs_z_by_side = {"BUY": 0.0, "SELL": 0.0}
        self._vwap_history.clear()
        self._rejection_counters = {k: 0 for k in self._rejection_counters}
        self._funnel_passed = {g: 0 for g in self._funnel_gate_order}
        self._funnel_fail_reason = {}
        self._funnel_candidates_total = 0
        self._funnel_approved_trades = 0
        self._bars_evaluated = 0
        self._eligible_session_bars = 0
        self._z_cross_inside_to_outside = 0
        self._z_cross_outside_to_inside = 0
        self._first_eligible_bar_outside = 0
        self._seen_first_eligible_bar = False
        self._z_cross_time_buckets = {}
        self._z_value_time_buckets = {}
        self._session_z_values = []
        self._soft_range_impulse_values = []
        self._soft_range_impulse_rejections = 0
        self._cross_body_impulse_abs_values = []
        self._cross_range_impulse_values = []
        self._drop_ledger = {k: 0 for k in self._drop_ledger}

    @property
    def signals(self) -> list[MRSignal]:
        return self._signals

    @property
    def rejection_counters(self) -> dict[str, int]:
        return dict(self._rejection_counters)

    @property
    def reclaim_mode(self) -> str:
        return self._reclaim_mode

    @property
    def soft_reclaim_range_impulse_k(self) -> float:
        return self._soft_reclaim_range_impulse_k

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = (pct / 100.0) * (len(s) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        w = idx - lo
        return s[lo] * (1 - w) + s[hi] * w

    @staticmethod
    def _time_bucket_et(ts: datetime) -> str:
        if ts.tzinfo is None:
            ts = _UTC.localize(ts)
        et = ts.astimezone(_ET)
        return et.strftime("%H:%M")

    @staticmethod
    def _session_z_stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"min": 0.0, "p50": 0.0, "max": 0.0}
        s = sorted(values)
        mid = len(s) // 2
        if len(s) % 2 == 0:
            p50 = 0.5 * (s[mid - 1] + s[mid])
        else:
            p50 = s[mid]
        return {"min": min(s), "p50": p50, "max": max(s)}

    @staticmethod
    def _adx_bucket(adx: float) -> str:
        for edge in sorted(MR_REGIME_ADX_BUCKETS):
            if adx < edge:
                return f"<{edge:g}"
        if MR_REGIME_ADX_BUCKETS:
            return f">={MR_REGIME_ADX_BUCKETS[-1]:g}"
        return "all"

    @property
    def gate_funnel_report(self) -> dict[str, object]:
        passed_map: dict[str, int] = {}
        running = self._funnel_candidates_total
        for idx, gate in enumerate(self._funnel_gate_order, start=1):
            running = self._funnel_passed.get(gate, 0)
            passed_map[f"passed_gate_{idx}_{gate}"] = running

        pass_rate_map: dict[str, float] = {}
        denom = float(self._funnel_candidates_total) if self._funnel_candidates_total > 0 else 1.0
        for idx, gate in enumerate(self._funnel_gate_order, start=1):
            key = f"pass_rate_gate_{idx}_{gate}"
            pass_rate_map[key] = self._funnel_passed.get(gate, 0) / denom

        top_fail = sorted(self._funnel_fail_reason.items(), key=lambda x: x[1], reverse=True)[:3]

        body_vals = list(self._cross_body_impulse_abs_values)
        range_vals = list(self._cross_range_impulse_values)
        z_bucket_summary = {
            bucket: {
                "count": len(vals),
                "min": min(vals) if vals else 0.0,
                "p50": self._percentile(vals, 50),
                "max": max(vals) if vals else 0.0,
            }
            for bucket, vals in sorted(self._z_value_time_buckets.items())
        }
        z_session_stats = self._session_z_stats(self._session_z_values)
        drop_ledger = dict(self._drop_ledger)
        drop_ledger["orders_submitted"] = self._funnel_approved_trades
        drop_ledger["fills"] = self._funnel_approved_trades
        drop_ledger["trades"] = self._funnel_approved_trades

        return {
            "candidate_mode": self._reclaim_mode,
            "soft_reclaim_range_impulse_k": self._soft_reclaim_range_impulse_k,
            "first_outside_enabled": self._first_outside_enabled,
            "touch_latch_reset_buffer": self._touch_latch_reset_buffer,
            "dedupe_window_bars": self._dedupe_window_bars,
            "dedupe_min_delta_z": self._dedupe_min_delta_z,
            "regime_enabled": self._regime_enabled,
            "bars_evaluated": self._bars_evaluated,
            "eligible_session_bars": self._eligible_session_bars,
            "z_cross_events": self._z_cross_inside_to_outside,
            "z_cross_inside_to_outside": self._z_cross_inside_to_outside,
            "z_cross_outside_to_inside": self._z_cross_outside_to_inside,
            "first_eligible_bar_outside": self._first_eligible_bar_outside,
            "z_cross_time_of_day": dict(sorted(self._z_cross_time_buckets.items())),
            "z_values_time_of_day": z_bucket_summary,
            "session_z_stats": z_session_stats,
            "cross_body_impulse_abs_values": body_vals,
            "cross_range_impulse_values": range_vals,
            "cross_body_impulse_abs_p50": self._percentile(body_vals, 50),
            "cross_body_impulse_abs_p75": self._percentile(body_vals, 75),
            "cross_body_impulse_abs_p90": self._percentile(body_vals, 90),
            "cross_body_impulse_abs_p95": self._percentile(body_vals, 95),
            "cross_body_impulse_abs_max": (max(body_vals) if body_vals else 0.0),
            "cross_range_impulse_p50": self._percentile(range_vals, 50),
            "cross_range_impulse_p75": self._percentile(range_vals, 75),
            "cross_range_impulse_p90": self._percentile(range_vals, 90),
            "cross_range_impulse_p95": self._percentile(range_vals, 95),
            "cross_range_impulse_max": (max(range_vals) if range_vals else 0.0),
            "soft_range_impulse_count": len(self._soft_range_impulse_values),
            "soft_range_impulse_rejections": self._soft_range_impulse_rejections,
            "candidates_total": self._funnel_candidates_total,
            **passed_map,
            "approved_trades": self._funnel_approved_trades,
            **pass_rate_map,
            "top_failure_reasons": [{"reason": r, "count": c} for r, c in top_fail],
            "failure_reason_counts": dict(self._funnel_fail_reason),
            "gate_order": list(self._funnel_gate_order),
            "drop_ledger": drop_ledger,
        }

    def _funnel_record_failure(self, reason: str) -> None:
        self._funnel_fail_reason[reason] = self._funnel_fail_reason.get(reason, 0) + 1

    def on_bar(
        self,
        bar: Bar,
        regime: str | None,
        vwap_state: VWAPState,
        atr: float,
        adx: float = 0.0,
    ) -> MRSignal | None:
        """Evaluate a 5-minute bar for MR signal candidates.

        Returns the signal object if one was generated (approved or rejected),
        or None if no candidate conditions were met.
        """
        self._bar_index += 1
        self._bars_evaluated += 1
        self._drop_ledger["bars_evaluated"] += 1

        # Track VWAP history for slope computation
        if vwap_state.bar_count > 0:
            self._vwap_history.append(vwap_state.vwap)

        # No signals before warmup
        if vwap_state.bar_count < 3 or atr <= 0:
            return None

        bucket = self._time_bucket_et(bar.timestamp)

        # Only in range regime
        if self._regime_enabled and regime != "range":
            self._rejection_counters["rejected_by_regime"] += 1
            self._drop_ledger["regime_rejects"] += 1
            self._funnel_fail_reason["REGIME_FILTER"] = self._funnel_fail_reason.get("REGIME_FILTER", 0) + 1
            tod_key = f"regime_reject_tod_{bucket}"
            self._funnel_fail_reason[tod_key] = self._funnel_fail_reason.get(tod_key, 0) + 1
            adx_key = f"regime_reject_adx_{self._adx_bucket(adx)}"
            self._funnel_fail_reason[adx_key] = self._funnel_fail_reason.get(adx_key, 0) + 1
            return None

        price = bar.close
        vwap = vwap_state.vwap
        sigma = vwap_state.std_dev

        if sigma <= 0:
            return None

        self._eligible_session_bars += 1
        self._drop_ledger["eligible_session_bars"] += 1

        # Check band proximity
        z_score = (price - vwap) / sigma
        prev_z_score = self._prev_z_score
        self._prev_z_score = z_score

        distance_ticks = abs(price - vwap) / TICK_SIZE
        distance_atr = abs(price - vwap) / atr
        self._session_z_values.append(z_score)
        if bucket not in self._z_value_time_buckets:
            self._z_value_time_buckets[bucket] = []
        self._z_value_time_buckets[bucket].append(z_score)

        first_outside_candidate_side: str | None = None
        if not self._seen_first_eligible_bar:
            self._seen_first_eligible_bar = True
            if abs(z_score) >= self._sigma_entry:
                self._first_eligible_bar_outside += 1
                if self._first_outside_enabled:
                    first_outside_candidate_side = "BUY" if z_score <= -self._sigma_entry else "SELL"

        if prev_z_score is not None:
            if abs(prev_z_score) < self._sigma_entry <= abs(z_score):
                self._z_cross_inside_to_outside += 1
                self._drop_ledger["z_cross_events"] += 1
                self._z_cross_time_buckets[bucket] = self._z_cross_time_buckets.get(bucket, 0) + 1
                body_abs = abs(bar.close - bar.open) / atr if atr > 0 else 0.0
                range_imp = (bar.high - bar.low) / atr if atr > 0 else 0.0
                self._cross_body_impulse_abs_values.append(body_abs)
                self._cross_range_impulse_values.append(range_imp)
            elif abs(prev_z_score) >= self._sigma_entry > abs(z_score):
                self._z_cross_outside_to_inside += 1

        # Track excursion activation
        if z_score <= -self._sigma_entry:
            self._long_excursion_active = True
        if z_score >= self._sigma_entry:
            self._short_excursion_active = True

        # Reset excursion states when mean-reverted enough or VWAP touched
        if abs(z_score) <= MR_EXCURSION_RESET_ZSCORE or distance_ticks <= MR_EXCURSION_RESET_VWAP_TICKS:
            self._long_excursion_active = False
            self._short_excursion_active = False
            self._long_excursion_traded = False
            self._short_excursion_traded = False

        reset_threshold = max(0.0, self._sigma_entry - self._touch_latch_reset_buffer)
        if abs(z_score) <= reset_threshold:
            self._long_touch_latch_armed = True
            self._short_touch_latch_armed = True

        # ── Cluster reset: if |z| drops below threshold, reset attempts ──
        if MR_CLUSTER_RESET_ENABLED:
            abs_z = abs(z_score)
            in_zone = abs_z >= self._sigma_entry

            if MR_CLUSTER_RESET_MODE == "retrace":
                if z_score <= -self._sigma_entry:
                    self._cluster_peak_long_z = max(self._cluster_peak_long_z, abs_z)
                if z_score >= self._sigma_entry:
                    self._cluster_peak_short_z = max(self._cluster_peak_short_z, abs_z)

                if (
                    self._long_attempts > 0
                    and self._cluster_peak_long_z >= MR_CLUSTER_RESET_MIN_PEAK_Z
                    and z_score < 0
                ):
                    long_reset_level = self._cluster_peak_long_z * (1.0 - MR_CLUSTER_RETRACE_FRACTION)
                    if abs_z <= long_reset_level:
                        self._long_attempts = 0
                        self._cluster_peak_long_z = abs_z
                        logger.info(
                            "Cluster reset BUY: |z|=%.2f <= %.2f (peak=%.2f)",
                            abs_z,
                            long_reset_level,
                            self._cluster_peak_long_z,
                        )

                if (
                    self._short_attempts > 0
                    and self._cluster_peak_short_z >= MR_CLUSTER_RESET_MIN_PEAK_Z
                    and z_score > 0
                ):
                    short_reset_level = self._cluster_peak_short_z * (1.0 - MR_CLUSTER_RETRACE_FRACTION)
                    if abs_z <= short_reset_level:
                        self._short_attempts = 0
                        self._cluster_peak_short_z = abs_z
                        logger.info(
                            "Cluster reset SELL: |z|=%.2f <= %.2f (peak=%.2f)",
                            abs_z,
                            short_reset_level,
                            self._cluster_peak_short_z,
                        )
            else:
                if self._was_in_zone and not in_zone and abs_z < MR_CLUSTER_RESET_ZSCORE:
                    self._long_attempts = 0
                    self._short_attempts = 0
                    logger.info(
                        "Cluster reset legacy: |z|=%.2f < %.2f, attempts reset",
                        abs_z, MR_CLUSTER_RESET_ZSCORE,
                    )
            self._was_in_zone = in_zone

        candidate_side: str | None = first_outside_candidate_side
        band_hit: float = 0.0

        if candidate_side is None and prev_z_score is not None:
            if self._reclaim_mode == "on":
                # Reclaim ON: signal only when price re-enters from an excursion.
                if prev_z_score <= -self._sigma_entry and z_score > -self._sigma_entry:
                    candidate_side = "BUY"
                    band_hit = self._sigma_extreme if prev_z_score <= -self._sigma_extreme else self._sigma_entry
                elif prev_z_score >= self._sigma_entry and z_score < self._sigma_entry:
                    candidate_side = "SELL"
                    band_hit = self._sigma_extreme if prev_z_score >= self._sigma_extreme else self._sigma_entry
            elif self._reclaim_mode in {"off", "soft"}:
                # Reclaim OFF: threshold cross from inside to outside (no reclaim confirmation).
                if prev_z_score > -self._sigma_entry and z_score <= -self._sigma_entry:
                    candidate_side = "BUY"
                    band_hit = self._sigma_extreme if z_score <= -self._sigma_extreme else self._sigma_entry
                elif prev_z_score < self._sigma_entry and z_score >= self._sigma_entry:
                    candidate_side = "SELL"
                    band_hit = self._sigma_extreme if z_score >= self._sigma_extreme else self._sigma_entry

        if candidate_side is None and self._reclaim_mode == "touch":
            if z_score <= -self._sigma_entry and self._long_touch_latch_armed:
                candidate_side = "BUY"
                band_hit = self._sigma_extreme if z_score <= -self._sigma_extreme else self._sigma_entry
                self._long_touch_latch_armed = False
            elif z_score >= self._sigma_entry and self._short_touch_latch_armed:
                candidate_side = "SELL"
                band_hit = self._sigma_extreme if z_score >= self._sigma_extreme else self._sigma_entry
                self._short_touch_latch_armed = False

        if candidate_side is None:
            return None

        self._drop_ledger["candidates_formed"] += 1

        if self._reclaim_mode == "soft":
            range_impulse = self._soft_reclaim_range_impulse(bar, atr)
            self._soft_range_impulse_values.append(range_impulse)
            if self._fails_soft_reclaim_range_impulse(candidate_side, bar, range_impulse):
                self._soft_range_impulse_rejections += 1
                self._funnel_record_failure("SOFT_RECLAIM_RANGE_IMPULSE")
                logger.info(
                    "Soft-v3 reject: side=%s range_impulse=%.4f k_range=%.4f open=%.2f close=%.2f",
                    candidate_side,
                    range_impulse,
                    self._soft_reclaim_range_impulse_k,
                    bar.open,
                    bar.close,
                )
                return None

        self._funnel_candidates_total += 1

        # Setup quality filters
        if distance_ticks < MR_MIN_DISTANCE_VWAP_TICKS:
            self._funnel_record_failure("DISTANCE_TICKS")
            return None  # too close to VWAP, not a real extreme
        self._funnel_passed["distance_ticks"] += 1

        if MR_FILTER_DISTANCE_ENABLED and distance_atr < MR_QUALITY_MIN_EXCURSION_ATR:
            self._funnel_record_failure("DISTANCE_ATR")
            return None
        self._funnel_passed["distance_atr"] += 1

        vwap_drift_atr = self._compute_vwap_flatness_atr(atr)
        if MR_FILTER_VWAP_FLAT_ENABLED and vwap_drift_atr > MR_QUALITY_VWAP_FLAT_MAX_ATR:
            self._funnel_record_failure("VWAP_FLATNESS")
            return None
        self._funnel_passed["vwap_flatness"] += 1

        if MR_FILTER_RECLAIM_STRENGTH_ENABLED and not self._passes_reclaim_close_location(candidate_side, bar):
            self._funnel_record_failure("RECLAIM_STRENGTH")
            return None
        self._funnel_passed["reclaim_strength"] += 1

        # Build candidate signal
        stop = price - (VWAP_STOP_ATR_MULT * atr) if candidate_side == "BUY" \
            else price + (VWAP_STOP_ATR_MULT * atr)

        signal = MRSignal(
            timestamp=bar.timestamp,
            side=candidate_side,
            regime_at_signal=regime,
            entry_reference_price=price,
            stop_reference=stop,
            target_reference=vwap,
            band_level_hit=band_hit,
            vwap_at_signal=vwap,
            sigma_at_signal=sigma,
            z_at_signal=z_score,
            bar_index=self._bar_index,
        )

        self._rejection_counters["signals_generated"] += 1

        # ── Gate checks ─────────────────────────────────────────────────
        rejection = self._check_gates(signal, bar, adx)
        if rejection:
            signal.approved = False
            signal.rejection_reason = rejection
            self._funnel_record_failure(rejection)
            # Categorize rejection for counters
            self._categorize_rejection(rejection)
        else:
            signal.approved = True
            self._rejection_counters["signals_approved"] += 1
            self._funnel_approved_trades += 1
            self._drop_ledger["orders_submitted"] += 1
            self._drop_ledger["fills"] += 1
            self._drop_ledger["trades"] += 1
            self._last_candidate_bar_by_side[candidate_side] = self._bar_index
            self._last_candidate_abs_z_by_side[candidate_side] = abs(z_score)
            if candidate_side == "BUY":
                self._long_attempts += 1
                self._last_long_signal_bar = self._bar_index
                self._long_excursion_traded = True
            else:
                self._short_attempts += 1
                self._last_short_signal_bar = self._bar_index
                self._short_excursion_traded = True
            self._last_signal_bar = self._bar_index

        self._signals.append(signal)
        logger.info(
            "MR signal: side=%s approved=%s band=%.1fσ price=%.2f vwap=%.2f %s",
            signal.side, signal.approved, signal.band_level_hit,
            signal.entry_reference_price, signal.vwap_at_signal,
            f"reject={signal.rejection_reason}" if not signal.approved else "",
        )
        return signal

    def _categorize_rejection(self, reason: str) -> None:
        """Map rejection reason string to counter bucket."""
        if reason in ("MAX_LONG_ATTEMPTS", "MAX_SHORT_ATTEMPTS"):
            self._rejection_counters["rejected_by_attempt_cap"] += 1
        elif reason == "SOFT_CAP_COOLDOWN":
            self._rejection_counters["rejected_by_attempt_cap"] += 1
        elif reason == "COOLDOWN":
            self._rejection_counters["rejected_by_cooldown"] += 1
        elif reason == "SESSION_CUTOFF":
            self._rejection_counters["rejected_by_session_cutoff"] += 1
        elif reason == "TREND_CONTAMINATION":
            self._rejection_counters["rejected_by_trend_contamination"] += 1
        elif reason == "EXCURSION_ALREADY_TRADED":
            self._rejection_counters["rejected_by_excursion_dedupe"] += 1
        elif reason.startswith("FLOW_FILTER"):
            self._rejection_counters["rejected_by_flow_filter"] += 1
        elif reason.startswith("GOVERNOR:"):
            # Governor rejections are applied externally; this handles internal tagging
            if "DAILY_LOSS" in reason:
                self._rejection_counters["rejected_by_daily_loss_governor"] += 1
            elif "PROFIT" in reason:
                self._rejection_counters["rejected_by_profit_cap"] += 1

    def _compute_vwap_slope(self) -> float:
        """Compute normalised VWAP slope over the lookback window.

        Returns slope in points-per-bar.  Positive = VWAP rising.
        """
        lookback = TREND_CONTAM_SLOPE_LOOKBACK
        if len(self._vwap_history) < lookback:
            return 0.0
        recent = self._vwap_history[-lookback:]
        # Simple linear slope: (last - first) / (n-1)
        return (recent[-1] - recent[0]) / (lookback - 1)

    def _compute_vwap_flatness_atr(self, atr: float) -> float:
        """Compute absolute VWAP drift over lookback, normalised by ATR."""
        lookback = MR_QUALITY_VWAP_FLAT_LOOKBACK
        if len(self._vwap_history) < lookback or atr <= 0:
            return 0.0
        start_vwap = self._vwap_history[-lookback]
        end_vwap = self._vwap_history[-1]
        return abs(end_vwap - start_vwap) / atr

    @staticmethod
    def _passes_reclaim_close_location(side: str, bar: Bar) -> bool:
        """Require reclaim bar close to finish with directional intent."""
        bar_range = bar.high - bar.low
        if bar_range <= 0:
            return False
        close_loc = (bar.close - bar.low) / bar_range
        if side == "BUY":
            return close_loc >= MR_QUALITY_RECLAIM_CLOSE_LOC_MIN
        # SELL: close in bottom (1 - close_loc_min) of bar range
        return close_loc <= (1.0 - MR_QUALITY_RECLAIM_CLOSE_LOC_MIN)

    @staticmethod
    def _soft_reclaim_range_impulse(bar: Bar, atr: float) -> float:
        """Range impulse score on candidate bar, normalised by ATR."""
        if atr <= 0:
            return 0.0
        return (bar.high - bar.low) / atr

    def _fails_soft_reclaim_range_impulse(self, side: str, bar: Bar, range_impulse: float) -> bool:
        """Soft-v3 veto: large-range continuation bars only.

        Reject if range_impulse >= k_range AND candle direction is against fade.
        BUY fade rejects bearish body (close < open).
        SELL fade rejects bullish body (close > open).
        """
        if range_impulse < self._soft_reclaim_range_impulse_k:
            return False
        if side == "BUY":
            return bar.close < bar.open
        return bar.close > bar.open

    def _check_gates(self, signal: MRSignal, bar: Bar, adx: float = 0.0) -> str:
        """Run gate checks. Returns rejection reason or empty string if approved."""
        # Cooldown
        if self._cooldown_bars > 0 and self._bar_index - self._last_signal_bar < self._cooldown_bars:
            self._drop_ledger["cooldown_rejects"] += 1
            return "COOLDOWN"
        self._funnel_passed["cooldown"] += 1

        if self._excursion_dedupe_enabled:
            side = signal.side
            bars_since = self._bar_index - self._last_candidate_bar_by_side.get(side, -100)
            abs_z = abs(signal.z_at_signal)
            prev_abs_z = self._last_candidate_abs_z_by_side.get(side, 0.0)
            same_excursion = (
                (side == "BUY" and self._long_excursion_active)
                or (side == "SELL" and self._short_excursion_active)
            )
            not_progressive = abs_z < (prev_abs_z + self._dedupe_min_delta_z)
            if same_excursion and bars_since <= self._dedupe_window_bars and not_progressive:
                self._drop_ledger["dedupe_rejects"] += 1
                return "EXCURSION_ALREADY_TRADED"
        self._funnel_passed["excursion_dedupe"] += 1

        # Side limits
        if self._attempt_cap_enabled and signal.side == "BUY" and self._long_attempts >= self._max_attempts_per_side:
            if MR_ATTEMPT_CAP_MODE == "hard":
                self._drop_ledger["attempt_limit_rejects"] += 1
                return "MAX_LONG_ATTEMPTS"
            bars_since = self._bar_index - self._last_long_signal_bar
            if bars_since < MR_SOFT_CAP_COOLDOWN_BARS or abs(signal.z_at_signal) < MR_SOFT_CAP_MIN_ZSCORE:
                self._drop_ledger["attempt_limit_rejects"] += 1
                return "SOFT_CAP_COOLDOWN"

        if self._attempt_cap_enabled and signal.side == "SELL" and self._short_attempts >= self._max_attempts_per_side:
            if MR_ATTEMPT_CAP_MODE == "hard":
                self._drop_ledger["attempt_limit_rejects"] += 1
                return "MAX_SHORT_ATTEMPTS"
            bars_since = self._bar_index - self._last_short_signal_bar
            if bars_since < MR_SOFT_CAP_COOLDOWN_BARS or abs(signal.z_at_signal) < MR_SOFT_CAP_MIN_ZSCORE:
                self._drop_ledger["attempt_limit_rejects"] += 1
                return "SOFT_CAP_COOLDOWN"
        self._funnel_passed["attempt_cap"] += 1

        # Trend contamination filter
        if TREND_CONTAM_ENABLED and adx > TREND_CONTAM_ADX_THRESHOLD:
            vwap_slope = self._compute_vwap_slope()
            if abs(vwap_slope) > TREND_CONTAM_VWAP_SLOPE_MIN:
                return "TREND_CONTAMINATION"
        self._funnel_passed["trend_contamination"] += 1

        # Time cutoff  (bar timestamps may be UTC-naive; convert to ET)
        ts = bar.timestamp
        if ts.tzinfo is None:
            ts = _UTC.localize(ts)
        bar_time_str = ts.astimezone(_ET).strftime("%H:%M")
        if bar_time_str >= LAST_ENTRY_CUTOFF:
            return "SESSION_CUTOFF"
        self._funnel_passed["session_cutoff"] += 1

        # Order flow filter (CVD stub)
        allowed, reason = self._flow_filter.allows(signal.side, bar)
        if not allowed:
            return f"FLOW_FILTER:{reason}"
        self._funnel_passed["flow_filter"] += 1

        return ""
