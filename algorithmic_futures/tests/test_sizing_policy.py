"""
Tests for validation.sizing_policy — Dynamic Sizing v1

Covers:
  • Fixed policy always returns configured contracts
  • Dynamic v1 upsize conditions (all must be met)
  • Downshift triggers (each individually)
  • Profit lock persistence
  • Intraday downshift via on_trade
  • apply_sizing_to_trades end-to-end
  • Zero-trade sessions
  • ORB-only mode contracts
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from validation.sizing_policy import (
    DaySizingRecord,
    SizingConfig,
    SizingPolicy,
    apply_sizing_to_trades,
)


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_policy(policy: str = "dynamic_v1", **overrides: Any) -> SizingPolicy:
    """Build a SizingPolicy with sensible defaults."""
    return SizingPolicy(SizingConfig(
        policy=overrides.pop("policy", policy),  # type: ignore[arg-type]
        fixed_contracts=int(overrides.pop("fixed_contracts", 2)),
        up_trail_headroom=float(overrides.pop("up_trail_headroom", 1400.0)),
        up_day_headroom=float(overrides.pop("up_day_headroom", 700.0)),
        down_trail_headroom=float(overrides.pop("down_trail_headroom", 1200.0)),
        down_day_headroom=float(overrides.pop("down_day_headroom", 600.0)),
        loss_streak_up_max=int(overrides.pop("loss_streak_up_max", 1)),
        loss_streak_down_min=int(overrides.pop("loss_streak_down_min", 2)),
        shock_loss_frac=float(overrides.pop("shock_loss_frac", 0.6)),
        profit_lock=float(overrides.pop("profit_lock", 2000.0)),
        daily_loss_limit=float(overrides.pop("daily_loss_limit", 1000.0)),
        trail_dd_limit=float(overrides.pop("trail_dd_limit", 2000.0)),
    ))


def _make_v2_policy(**overrides: Any) -> SizingPolicy:
    """Build a dynamic_v2 SizingPolicy with sensible defaults."""
    return SizingPolicy(SizingConfig(
        policy="dynamic_v2",
        fixed_contracts=int(overrides.pop("fixed_contracts", 2)),
        up_trail_headroom=float(overrides.pop("up_trail_headroom", 1400.0)),
        up_day_headroom=float(overrides.pop("up_day_headroom", 700.0)),
        down_trail_headroom=float(overrides.pop("down_trail_headroom", 1200.0)),
        down_day_headroom=float(overrides.pop("down_day_headroom", 600.0)),
        loss_streak_up_max=int(overrides.pop("loss_streak_up_max", 1)),
        loss_streak_down_min=int(overrides.pop("loss_streak_down_min", 2)),
        shock_loss_frac=float(overrides.pop("shock_loss_frac", 0.6)),
        profit_lock=float(overrides.pop("profit_lock", 2000.0)),
        daily_loss_limit=float(overrides.pop("daily_loss_limit", 1000.0)),
        trail_dd_limit=float(overrides.pop("trail_dd_limit", 2000.0)),
        vol_atr_cap=float(overrides.pop("vol_atr_cap", 14.0)),
        earned_traction=float(overrides.pop("earned_traction", 150.0)),
        earned_giveback=float(overrides.pop("earned_giveback", 50.0)),
    ))


def _make_v3_policy(**overrides: Any) -> SizingPolicy:
    """Build a dynamic_v3 SizingPolicy with sensible defaults."""
    return SizingPolicy(SizingConfig(
        policy="dynamic_v3",
        fixed_contracts=int(overrides.pop("fixed_contracts", 2)),
        up_trail_headroom=float(overrides.pop("up_trail_headroom", 1400.0)),
        up_day_headroom=float(overrides.pop("up_day_headroom", 700.0)),
        down_trail_headroom=float(overrides.pop("down_trail_headroom", 1200.0)),
        down_day_headroom=float(overrides.pop("down_day_headroom", 600.0)),
        loss_streak_up_max=int(overrides.pop("loss_streak_up_max", 1)),
        loss_streak_down_min=int(overrides.pop("loss_streak_down_min", 2)),
        shock_loss_frac=float(overrides.pop("shock_loss_frac", 0.6)),
        profit_lock=float(overrides.pop("profit_lock", 2000.0)),
        daily_loss_limit=float(overrides.pop("daily_loss_limit", 1000.0)),
        trail_dd_limit=float(overrides.pop("trail_dd_limit", 2000.0)),
        v3_earned_traction=float(overrides.pop("v3_earned_traction", 75.0)),
        v3_giveback_floor=float(overrides.pop("v3_giveback_floor", 25.0)),
        v3_orb_upsize_allowed=bool(overrides.pop("v3_orb_upsize_allowed", False)),
        v3_day_headroom_up=float(overrides.pop("v3_day_headroom_up", 800.0)),
        v3_day_headroom_down=float(overrides.pop("v3_day_headroom_down", 600.0)),
        v3_trail_headroom_up=float(overrides.pop("v3_trail_headroom_up", 1400.0)),
        v3_trail_headroom_down=float(overrides.pop("v3_trail_headroom_down", 1200.0)),
        v3_atr_traction_scale_enabled=bool(overrides.pop("v3_atr_traction_scale_enabled", False)),
        v3_atr_traction_baseline=float(overrides.pop("v3_atr_traction_baseline", 12.0)),
        v3_atr_traction_min_scale=float(overrides.pop("v3_atr_traction_min_scale", 0.75)),
        v3_atr_traction_max_scale=float(overrides.pop("v3_atr_traction_max_scale", 1.25)),
        v3_consistency_brake_enabled=bool(overrides.pop("v3_consistency_brake_enabled", False)),
        v3_consistency_cap_pct=float(overrides.pop("v3_consistency_cap_pct", 0.50)),
        v3_consistency_loss_buffer_mult=float(overrides.pop("v3_consistency_loss_buffer_mult", 2.0)),
    ))


def _write_trades_csv(path: Path, trades: list[dict]) -> None:
    """Write a minimal trades.csv with required columns."""
    if not trades:
        return
    fieldnames = list(trades[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trades)


# ═══════════════════════════════════════════════════════════════════════
#  Fixed policy tests
# ═══════════════════════════════════════════════════════════════════════


class TestFixedPolicy:
    def test_fixed_always_returns_configured(self):
        policy = _make_policy("fixed", fixed_contracts=2)
        for i in range(5):
            c = policy.decide_day_start(f"s{i}", "range", "mr", i)
            assert c == 2
            policy.on_trade(-50.0)
            policy.end_of_day()

    def test_fixed_1c(self):
        policy = _make_policy("fixed", fixed_contracts=1)
        c = policy.decide_day_start("s1", "trend", "orb", 1)
        assert c == 1

    def test_fixed_3c(self):
        """Fixed policy can request 3c (it's not restricted to 1↔2)."""
        policy = _make_policy("fixed", fixed_contracts=3)
        c = policy.decide_day_start("s1", "range", "mr", 1)
        assert c == 3


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic v1 — upsize conditions
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicUpsize:
    def test_upsize_all_conditions_met_range_mr(self):
        """Fresh start, trail_headroom=2000, day_headroom=1000, streak=0, range+mr → 2c."""
        policy = _make_policy()
        c = policy.decide_day_start("s1", "range", "mr", 1)
        assert c == 2

    def test_upsize_all_conditions_met_trend_orb(self):
        """Trend regime + ORB engine is also aligned → 2c."""
        policy = _make_policy()
        c = policy.decide_day_start("s1", "trend", "orb", 1)
        assert c == 2

    def test_no_upsize_regime_misaligned_range_orb(self):
        """range + orb is NOT aligned → stays 1c."""
        policy = _make_policy()
        c = policy.decide_day_start("s1", "range", "orb", 1)
        assert c == 1

    def test_no_upsize_regime_misaligned_trend_mr(self):
        """trend + mr is NOT aligned → stays 1c."""
        policy = _make_policy()
        c = policy.decide_day_start("s1", "trend", "mr", 1)
        assert c == 1

    def test_no_upsize_chop_regime(self):
        """chop + any engine is NOT aligned → stays 1c."""
        policy = _make_policy()
        assert policy.decide_day_start("s1", "chop", "mr", 1) == 1
        policy.end_of_day()
        assert policy.decide_day_start("s2", "chop", "orb", 2) == 1

    def test_no_upsize_unknown_regime(self):
        policy = _make_policy()
        assert policy.decide_day_start("s1", "unknown", "mr", 1) == 1

    def test_no_upsize_insufficient_trail_headroom(self):
        """Erode equity so trail_headroom < 1400 → 1c."""
        policy = _make_policy()
        # Start: equity=0, peak=0, trail_headroom = 2000 - (0 - 0) = 2000
        # We need trail_headroom < 1400, i.e. trailing_dd_used > 600
        # trailing_dd_used = peak - equity = 700 - 100 = 600 ... barely not enough
        # Let's set equity to -700 directly (peak stays 0, trailing_dd_used = 700)
        policy.equity = -700.0
        policy.peak_equity = 0.0
        assert policy.trail_headroom == 1300.0  # < 1400
        c = policy.decide_day_start("s1", "range", "mr", 1)
        assert c == 1

    def test_no_upsize_loss_streak_too_high(self):
        """loss_streak=2 > up_max=1 → 1c."""
        policy = _make_policy()
        policy.loss_streak = 2
        c = policy.decide_day_start("s1", "range", "mr", 1)
        assert c == 1


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic v1 — downshift triggers
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicDownshift:
    def test_downshift_trail_headroom(self):
        """After a large loss at 2c, trail_headroom breaches → 1c."""
        policy = _make_policy()
        c = policy.decide_day_start("s1", "range", "mr", 1)
        assert c == 2
        # Lose enough to erode trail_headroom below 1200
        # trail_headroom = trail_dd_limit - (peak - equity) = 2000 - (0 - (-900)) = 1100
        c = policy.on_trade(-900.0)
        assert c == 1
        assert "trail_headroom" in policy._downshift_reason

    def test_downshift_day_headroom(self):
        """Day PnL eroding day_headroom below 600 at 2c → 1c."""
        policy = _make_policy()
        c = policy.decide_day_start("s1", "range", "mr", 1)
        assert c == 2
        # Day headroom = daily_loss_limit + day_pnl = 1000 + (-500) = 500 < 600
        c = policy.on_trade(-500.0)
        assert c == 1
        assert "day_headroom" in policy._downshift_reason

    def test_downshift_loss_streak(self):
        """Two consecutive losses at 2c → 1c."""
        policy = _make_policy()
        c = policy.decide_day_start("s1", "range", "mr", 1)
        assert c == 2
        # First loss — still 2c (streak=1 < 2)
        c = policy.on_trade(-30.0)
        assert c == 2
        # Second loss — streak=2 >= loss_streak_down_min → 1c
        c = policy.on_trade(-30.0)
        assert c == 1
        assert "loss_streak" in policy._downshift_reason

    def test_downshift_shock_loss(self):
        """Single trade loss >= 60% of daily limit → 1c (shock_loss check)."""
        # Use higher daily_loss_limit so day_headroom doesn't fire first
        policy = _make_policy(daily_loss_limit=2000.0)
        c = policy.decide_day_start("s1", "range", "mr", 1)
        assert c == 2
        # shock threshold = 0.6 * 2000 = 1200. Trade pnl = -1200 → triggers shock
        # day_headroom = 2000 + (-1200) = 800 > 600, so day_headroom won't fire
        # trail_headroom = 2000 - (0 - (-1200)) = 800 < 1200 → trail fires first
        # To isolate shock, we need trail_headroom to stay above threshold too
        # Give the policy extra equity headroom
        policy.equity = 500.0
        policy.peak_equity = 500.0
        # trail_headroom = 2000 - (500 - (500-1200)) = 2000 - 1200 = 800 → still fires
        # Let's just make trail_dd_limit very large
        policy2 = _make_policy(daily_loss_limit=2000.0, trail_dd_limit=5000.0)
        c = policy2.decide_day_start("s1", "range", "mr", 1)
        assert c == 2
        # shock threshold = 0.6 * 2000 = 1200
        # loss of -1200 → shock triggers
        # day_headroom = 2000 + (-1200) = 800 >= 600 → no day_headroom trigger
        # trail_headroom = 5000 - (0 - (-1200)) = 3800 >= 1200 → no trail trigger
        # loss_streak = 1 < 2 → no streak trigger
        c = policy2.on_trade(-1200.0)
        assert c == 1
        assert "shock_loss" in policy2._downshift_reason

    def test_no_downshift_at_1c(self):
        """Downshift checks only apply at 2c. 1c stays 1c."""
        policy = _make_policy()
        c = policy.decide_day_start("s1", "chop", "mr", 1)  # Not regime-aligned → 1c
        assert c == 1
        # Even losing a lot, contracts stay 1
        c = policy.on_trade(-600.0)
        assert c == 1

    def test_downshift_is_sticky_intraday(self):
        """Once downshifted to 1c, stays 1c for rest of session."""
        policy = _make_policy()
        c = policy.decide_day_start("s1", "range", "mr", 1)
        assert c == 2
        policy.on_trade(-500.0)  # triggers day_headroom downshift
        assert policy.contracts == 1
        # Win doesn't upsize back intraday
        policy.on_trade(200.0)
        assert policy.contracts == 1


# ═══════════════════════════════════════════════════════════════════════
#  Profit lock
# ═══════════════════════════════════════════════════════════════════════


class TestProfitLock:
    def test_profit_lock_triggers(self):
        """Equity >= 2000 → permanent 1c lock."""
        policy = _make_policy()
        # Accumulate equity
        for i in range(10):
            policy.decide_day_start(f"s{i}", "range", "mr", i)
            policy.on_trade(250.0)  # Total after 10 = 2500
            policy.end_of_day()

        assert policy.profit_lock_triggered
        # Next day must be 1c regardless of conditions
        c = policy.decide_day_start("s10", "range", "mr", 10)
        assert c == 1
        assert policy._downshift_reason == "profit_lock"

    def test_profit_lock_persists_across_days(self):
        """Once triggered, profit lock never clears."""
        policy = _make_policy()
        policy.equity = 2100.0
        policy.peak_equity = 2100.0
        policy.profit_lock_triggered = True

        for i in range(5):
            c = policy.decide_day_start(f"s{i}", "range", "mr", i)
            assert c == 1
            policy.on_trade(-100.0)
            policy.end_of_day()
        assert policy.profit_lock_triggered

    def test_profit_lock_triggers_mid_day(self):
        """A winning trade pushes equity past lock threshold mid-day → 1c."""
        policy = _make_policy()
        policy.equity = 1900.0
        policy.peak_equity = 1900.0
        c = policy.decide_day_start("s1", "range", "mr", 1)
        assert c == 2
        # Win 200 → equity = 2100 → lock
        c = policy.on_trade(200.0)
        assert c == 1
        assert policy.profit_lock_triggered


# ═══════════════════════════════════════════════════════════════════════
#  End-of-day record
# ═══════════════════════════════════════════════════════════════════════


class TestEndOfDay:
    def test_daily_log_append(self):
        policy = _make_policy()
        policy.decide_day_start("s1", "range", "mr", 1)
        policy.on_trade(100.0)
        rec = policy.end_of_day()

        assert isinstance(rec, DaySizingRecord)
        assert rec.session_id == "s1"
        assert rec.equity_after == 100.0
        assert len(policy.daily_log) == 1

    def test_contracts_start_records_upsize(self):
        """When downshifted mid-day, contracts_start should be 2."""
        policy = _make_policy()
        policy.decide_day_start("s1", "range", "mr", 1)
        assert policy.contracts == 2
        policy.on_trade(-500.0)  # Day headroom downshift
        assert policy.contracts == 1
        rec = policy.end_of_day()
        assert rec.contracts_start == 2
        assert rec.contracts_final == 1


# ═══════════════════════════════════════════════════════════════════════
#  apply_sizing_to_trades
# ═══════════════════════════════════════════════════════════════════════


class TestApplySizingToTrades:
    def test_fixed_2c_scales_all_trades(self):
        """Fixed 2c: all trade PnLs doubled."""
        policy = _make_policy("fixed", fixed_contracts=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "trades.csv"
            _write_trades_csv(csv_path, [
                {"entry_time": "10:00", "pnl_dollars": 50.0, "pnl_points": 10.0, "mae_points": -5.0, "mfe_points": 12.0, "exit_reason": "target"},
                {"entry_time": "10:30", "pnl_dollars": -25.0, "pnl_points": -5.0, "mae_points": -8.0, "mfe_points": 3.0, "exit_reason": "stop"},
            ])
            scaled = apply_sizing_to_trades(csv_path, policy, "range", "mr", "s1", 1)

        assert len(scaled) == 2
        assert scaled[0]["pnl_dollars"] == 100.0
        assert scaled[0]["pnl_points"] == 20.0
        assert scaled[0]["contracts"] == 2
        assert scaled[1]["pnl_dollars"] == -50.0
        assert scaled[1]["contracts"] == 2

    def test_dynamic_v1_with_midday_downshift(self):
        """Starts at 2c, second trade triggers downshift, third at 1c."""
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "trades.csv"
            _write_trades_csv(csv_path, [
                {"entry_time": "10:00", "pnl_dollars": 50.0, "pnl_points": 10.0, "mae_points": -3.0, "mfe_points": 12.0, "exit_reason": "target"},
                {"entry_time": "10:30", "pnl_dollars": -350.0, "pnl_points": -70.0, "mae_points": -80.0, "mfe_points": 5.0, "exit_reason": "stop"},
                {"entry_time": "11:00", "pnl_dollars": 30.0, "pnl_points": 6.0, "mae_points": -2.0, "mfe_points": 8.0, "exit_reason": "target"},
            ])
            scaled = apply_sizing_to_trades(csv_path, policy, "range", "mr", "s1", 1)

        # Trade 1 at 2c: 50*2 = 100
        assert scaled[0]["contracts"] == 2
        assert scaled[0]["pnl_dollars"] == 100.0

        # Trade 2 at 2c: -350*2 = -700 → triggers day_headroom downshift
        # day_headroom after trade 1: 1000 + 100 = 1100
        # trade 2 scaled: -700. day_headroom = 1100 - 700 = 400 < 600 → downshift
        assert scaled[1]["contracts"] == 2
        assert scaled[1]["pnl_dollars"] == -700.0

        # Trade 3 at 1c now (downshift happened)
        assert scaled[2]["contracts"] == 1
        assert scaled[2]["pnl_dollars"] == 30.0

    def test_empty_trades_csv(self):
        """Empty CSV registers day but returns no trades."""
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "trades.csv"
            csv_path.write_text("entry_time,pnl_dollars,pnl_points\n")
            scaled = apply_sizing_to_trades(csv_path, policy, "range", "mr", "s1", 1)

        assert scaled == []
        assert len(policy.daily_log) == 1  # Day was still registered

    def test_missing_trades_csv(self):
        """Missing file registers day but returns no trades."""
        policy = _make_policy()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "trades.csv"
            # Don't create the file
            scaled = apply_sizing_to_trades(csv_path, policy, "range", "mr", "s1", 1)

        assert scaled == []
        assert len(policy.daily_log) == 1


# ═══════════════════════════════════════════════════════════════════════
#  Serialization
# ═══════════════════════════════════════════════════════════════════════


class TestSerialization:
    def test_write_daily_log(self):
        policy = _make_policy()
        policy.decide_day_start("s1", "range", "mr", 1)
        policy.on_trade(100.0)
        policy.end_of_day()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sizing_decisions.json"
            policy.write_daily_log(path)
            data = json.loads(path.read_text())

        assert len(data) == 1
        assert data[0]["session_id"] == "s1"
        assert data[0]["equity_after"] == 100.0

    def test_config_snapshot(self):
        policy = _make_policy()
        snap = policy.config_snapshot()
        assert snap["policy"] == "dynamic_v1"
        assert snap["up_trail_headroom"] == 1400.0
        assert snap["trail_dd_limit"] == 2000.0


# ═══════════════════════════════════════════════════════════════════════
#  Multi-day equity tracking
# ═══════════════════════════════════════════════════════════════════════


class TestMultiDayEquity:
    def test_equity_tracks_across_sessions(self):
        """Equity accumulates across sessions."""
        policy = _make_policy()
        policy.decide_day_start("s1", "range", "mr", 1)
        policy.on_trade(200.0)  # 2c scaled = handled by caller, here raw
        policy.end_of_day()

        assert policy.equity == 200.0

        policy.decide_day_start("s2", "range", "mr", 2)
        policy.on_trade(-100.0)
        policy.end_of_day()

        assert policy.equity == 100.0
        assert policy.peak_equity == 200.0
        # trailing_dd_used = 200 - 100 = 100
        assert policy.trailing_dd_used == 100.0

    def test_trail_headroom_erodes_day2_contracts(self):
        """After day 1 losses, day 2 may not qualify for 2c."""
        policy = _make_policy()
        # Day 1: big loss
        policy.decide_day_start("s1", "range", "mr", 1)
        policy.on_trade(-700.0)
        policy.end_of_day()

        # equity = -700, peak = 0, trailing_dd_used = 700
        # trail_headroom = 2000 - 700 = 1300 < 1400
        c = policy.decide_day_start("s2", "range", "mr", 2)
        assert c == 1  # Insufficient trail headroom


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic v2 — volatility throttle
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicV2VolThrottle:
    def test_high_atr_caps_at_1c(self):
        """Session ATR >= vol_atr_cap (14.0) → forced 1c all day."""
        policy = _make_v2_policy()
        c = policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=15.0)
        assert c == 1
        assert policy._vol_throttled

    def test_low_atr_allows_eligibility(self):
        """Session ATR < vol_atr_cap → not throttled, eligible for earned upsize."""
        policy = _make_v2_policy()
        c = policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        assert c == 1  # Still starts at 1c (earned upsize is intraday)
        assert not policy._vol_throttled
        assert policy._v2_upsize_eligible

    def test_exact_atr_threshold_caps(self):
        """ATR exactly at cap → throttled (>= comparison)."""
        policy = _make_v2_policy(vol_atr_cap=14.0)
        c = policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=14.0)
        assert c == 1
        assert policy._vol_throttled

    def test_vol_throttle_prevents_earned_upsize(self):
        """Even with traction, upsize blocked when vol-throttled."""
        policy = _make_v2_policy()
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=16.0)
        assert policy._vol_throttled
        # Big winning trade exceeds traction
        c = policy.on_trade(200.0)
        assert c == 1  # Still 1c, vol-throttled

    def test_vol_throttle_resets_next_day(self):
        """Vol throttle is per-session, resets on next decide_day_start."""
        policy = _make_v2_policy()
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=16.0)
        assert policy._vol_throttled
        policy.end_of_day()

        policy.decide_day_start("s2", "range", "mr", 2, session_atr_median=10.0)
        assert not policy._vol_throttled

    def test_vol_throttle_recorded_in_daily_log(self):
        """DaySizingRecord captures vol_throttled flag."""
        policy = _make_v2_policy()
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=18.0)
        rec = policy.end_of_day()
        assert rec.vol_throttled is True
        assert rec.session_atr_median == 18.0


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic v2 — earned upsize
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicV2EarnedUpsize:
    def test_earned_upsize_on_traction(self):
        """Day PnL >= earned_traction (150) → upgrade to 2c."""
        policy = _make_v2_policy()
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        assert policy.contracts == 1
        # First trade brings day_pnl to 160 (>= 150)
        c = policy.on_trade(160.0)
        assert c == 2
        assert policy._earned_upsize_triggered

    def test_no_upsize_below_traction(self):
        """Day PnL < earned_traction → stays 1c."""
        policy = _make_v2_policy()
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        c = policy.on_trade(100.0)  # < 150
        assert c == 1
        assert not policy._earned_upsize_triggered

    def test_no_upsize_regime_misaligned(self):
        """Even with traction, misaligned regime → stays 1c."""
        policy = _make_v2_policy()
        policy.decide_day_start("s1", "range", "orb", 1, session_atr_median=10.0)
        assert not policy._v2_upsize_eligible
        c = policy.on_trade(200.0)
        assert c == 1

    def test_no_upsize_insufficient_trail_headroom(self):
        """Even with traction, low trail headroom → stays 1c."""
        policy = _make_v2_policy()
        policy.equity = -700.0
        policy.peak_equity = 0.0
        # trail_headroom = 2000 - 700 = 1300 < 1400
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        assert not policy._v2_upsize_eligible
        c = policy.on_trade(200.0)
        assert c == 1

    def test_no_upsize_high_loss_streak(self):
        """Even with traction, high loss streak → stays 1c."""
        policy = _make_v2_policy()
        policy.loss_streak = 2
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        assert not policy._v2_upsize_eligible
        c = policy.on_trade(200.0)
        assert c == 1

    def test_earned_upsize_exact_traction(self):
        """Day PnL exactly at traction → upsize triggers."""
        policy = _make_v2_policy(earned_traction=150.0)
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        c = policy.on_trade(150.0)
        assert c == 2

    def test_earned_upsize_recorded_in_log(self):
        """DaySizingRecord captures earned_upsize_triggered."""
        policy = _make_v2_policy()
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        policy.on_trade(200.0)  # triggers earned upsize
        rec = policy.end_of_day()
        assert rec.earned_upsize_triggered is True
        assert rec.contracts_start == 1
        assert rec.contracts_final == 2


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic v2 — giveback revert
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicV2Giveback:
    def test_giveback_reverts_to_1c(self):
        """After earned upsize, day PnL drops below (traction - giveback) → revert 1c."""
        policy = _make_v2_policy(earned_traction=150.0, earned_giveback=50.0)
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        # Win 160 → earned upsize to 2c
        policy.on_trade(160.0)
        assert policy.contracts == 2
        # Lose 70 at 2c → day_pnl = 160 - 70 = 90 < 100 (150 - 50) → revert
        c = policy.on_trade(-70.0)
        assert c == 1
        assert "earned_giveback" in policy._downshift_reason

    def test_no_giveback_above_floor(self):
        """Day PnL stays above giveback floor → stays 2c."""
        policy = _make_v2_policy(earned_traction=150.0, earned_giveback=50.0)
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        policy.on_trade(160.0)  # upsize to 2c
        assert policy.contracts == 2
        # Small loss: day_pnl = 160 - 30 = 130 >= 100 → stays 2c
        c = policy.on_trade(-30.0)
        assert c == 2

    def test_giveback_is_sticky(self):
        """Once reverted by giveback, stays 1c for rest of session."""
        policy = _make_v2_policy(earned_traction=150.0, earned_giveback=50.0)
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        policy.on_trade(160.0)  # upsize to 2c
        policy.on_trade(-70.0)  # revert to 1c (giveback)
        assert policy.contracts == 1
        # Win again — but giveback already fired, _earned_upsize_triggered is still True
        # However, we're now at 1c and _v2_upsize_eligible should re-check
        # Actually after giveback, downshift_reason is set, so on next on_trade
        # the earned upsize block should still trigger since we're at 1c & eligible
        # BUT the day_pnl = 160 - 70 + 50 = 140 < 150 → no re-upsize
        c = policy.on_trade(50.0)
        assert c == 1  # day_pnl = 140 < 150, stays 1c

    def test_giveback_then_re_earn(self):
        """After giveback revert, can re-earn upsize if day PnL crosses traction again."""
        policy = _make_v2_policy(earned_traction=150.0, earned_giveback=50.0)
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        policy.on_trade(160.0)  # upsize to 2c
        policy.on_trade(-70.0)  # revert to 1c (day_pnl = 90)
        assert policy.contracts == 1
        # Win 65 → day_pnl = 155 >= 150 → re-upsize
        c = policy.on_trade(65.0)
        assert c == 2


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic v2 — downshift at 2c
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicV2Downshift:
    def test_standard_downshift_at_2c(self):
        """After earned upsize to 2c, standard downshift triggers still work."""
        policy = _make_v2_policy(daily_loss_limit=2000.0, trail_dd_limit=5000.0)
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        # Big win to trigger earned upsize
        policy.on_trade(200.0)
        assert policy.contracts == 2
        # Shock loss at 2c: -1200 >= 0.6 * 2000 = 1200 → shock downshift
        c = policy.on_trade(-1200.0)
        assert c == 1
        assert "shock_loss" in policy._downshift_reason

    def test_profit_lock_blocks_v2_upsize(self):
        """Profit lock triggered → v2 stays at 1c."""
        policy = _make_v2_policy()
        policy.equity = 2100.0
        policy.peak_equity = 2100.0
        policy.profit_lock_triggered = True
        c = policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        assert c == 1
        # Even big win doesn't upsize
        c = policy.on_trade(200.0)
        assert c == 1


# ═══════════════════════════════════════════════════════════════════════
#  Dynamic v2 — integration / multi-day
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicV2Integration:
    def test_three_day_scenario(self):
        """Multi-day v2 scenario:
        Day 1: Low ATR, range+mr, traction reached → earned 2c
        Day 2: High ATR → vol-throttled 1c all day
        Day 3: Low ATR but chop → no upsize eligibility
        """
        policy = _make_v2_policy()

        # Day 1: low ATR, range/mr
        c = policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        assert c == 1
        assert not policy._vol_throttled
        assert policy._v2_upsize_eligible
        policy.on_trade(80.0)   # day_pnl = 80 < 150
        assert policy.contracts == 1
        c = policy.on_trade(80.0)   # day_pnl = 160 >= 150
        assert c == 2
        assert policy._earned_upsize_triggered
        policy.on_trade(50.0)   # day_pnl = 210, stays 2c
        assert policy.contracts == 2
        rec1 = policy.end_of_day()
        assert rec1.earned_upsize_triggered
        assert not rec1.vol_throttled
        assert rec1.contracts_start == 1
        assert rec1.contracts_final == 2

        # Day 2: high ATR → vol-throttled
        c = policy.decide_day_start("s2", "range", "mr", 2, session_atr_median=18.0)
        assert c == 1
        assert policy._vol_throttled
        policy.on_trade(200.0)  # Big win but still throttled
        assert policy.contracts == 1
        rec2 = policy.end_of_day()
        assert rec2.vol_throttled
        assert not rec2.earned_upsize_triggered

        # Day 3: low ATR but chop
        c = policy.decide_day_start("s3", "chop", "mr", 3, session_atr_median=8.0)
        assert c == 1
        assert not policy._vol_throttled
        assert not policy._v2_upsize_eligible  # chop+mr not aligned
        policy.on_trade(200.0)
        assert policy.contracts == 1  # No upsize — not eligible
        rec3 = policy.end_of_day()
        assert not rec3.earned_upsize_triggered
        assert not rec3.vol_throttled

    def test_v2_always_starts_1c(self):
        """v2 always starts at 1c regardless of prior day performance."""
        policy = _make_v2_policy()
        # Day 1: big wins
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        policy.on_trade(300.0)
        assert policy.contracts == 2
        policy.end_of_day()

        # Day 2: starts fresh at 1c
        c = policy.decide_day_start("s2", "range", "mr", 2, session_atr_median=10.0)
        assert c == 1

    def test_apply_sizing_v2_with_earned_upsize(self):
        """apply_sizing_to_trades integration with v2 earned upsize."""


class TestDynamicV3Enhancements:
    def test_atr_scaled_traction_reduces_threshold_on_low_atr_days(self):
        policy = _make_v3_policy(
            v3_earned_traction=80.0,
            v3_atr_traction_scale_enabled=True,
            v3_atr_traction_baseline=12.0,
            v3_atr_traction_min_scale=0.5,
            v3_atr_traction_max_scale=1.5,
        )
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=6.0)
        assert policy._v3_effective_traction == 40.0
        c = policy.on_trade(45.0)
        assert c == 2
        assert policy._v3_upsize_trigger == "traction"

    def test_consistency_brake_blocks_first_trade_win_upsize(self):
        policy = _make_v3_policy(
            v3_consistency_brake_enabled=True,
            v3_consistency_cap_pct=0.5,
            v3_consistency_loss_buffer_mult=3.0,
            v3_giveback_floor=25.0,
        )
        policy.equity = 200.0
        policy.peak_equity = 200.0
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        c = policy.on_trade(150.0)
        assert c == 1
        assert policy._v3_consistency_brake_blocked is True
        assert policy._downshift_reason == "v3_consistency_brake"

    def test_consistency_brake_flag_is_recorded_in_log(self):
        policy = _make_v3_policy(v3_consistency_brake_enabled=True, v3_consistency_loss_buffer_mult=3.0)
        policy.equity = 200.0
        policy.peak_equity = 200.0
        policy.decide_day_start("s1", "range", "mr", 1, session_atr_median=10.0)
        policy.on_trade(150.0)
        rec = policy.end_of_day()
        assert rec.v3_consistency_brake_blocked is True
        policy = _make_v2_policy(earned_traction=100.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "trades.csv"
            _write_trades_csv(csv_path, [
                {"entry_time": "10:00", "pnl_dollars": 60.0, "pnl_points": 12.0,
                 "mae_points": -3.0, "mfe_points": 14.0, "exit_reason": "target"},
                {"entry_time": "10:30", "pnl_dollars": 60.0, "pnl_points": 12.0,
                 "mae_points": -2.0, "mfe_points": 14.0, "exit_reason": "target"},
                {"entry_time": "11:00", "pnl_dollars": 40.0, "pnl_points": 8.0,
                 "mae_points": -1.0, "mfe_points": 10.0, "exit_reason": "target"},
            ])
            scaled = apply_sizing_to_trades(
                csv_path, policy, "range", "mr", "s1", 1,
                session_atr_median=10.0,
            )

        # Trade 1 at 1c: 60 * 1 = 60. day_pnl = 60 < 100 → stays 1c
        assert scaled[0]["contracts"] == 1
        assert scaled[0]["pnl_dollars"] == 60.0

        # Trade 2 at 1c: 60 * 1 = 60. day_pnl = 60 + 60 = 120 >= 100 → upsize to 2c
        assert scaled[1]["contracts"] == 1
        assert scaled[1]["pnl_dollars"] == 60.0

        # Trade 3 now at 2c: 40 * 2 = 80
        assert scaled[2]["contracts"] == 2
        assert scaled[2]["pnl_dollars"] == 80.0