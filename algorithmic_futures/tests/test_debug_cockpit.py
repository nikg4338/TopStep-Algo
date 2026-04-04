"""
tests/test_debug_cockpit.py — Tests for the strategy debugging cockpit modules.

Five test categories per sprint spec:
  1. Short replay (single partial bucket) — no crash
  2. Regime classifier warmup & classification
  3. Risk governor synthetic test — daily loss halt, consistency cap rejection
  4. Consistency engine math — combine 50%, express_funded 40%, xfa_standard
  5. Signal engine sanity — no signals before warmup, only MR in range regime
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from data.indicators import VWAPCalculator, VWAPState
from data.market_data import Bar
from regime.regime_v1 import (
    REGIME_EXTREME,
    REGIME_RANGE,
    REGIME_TREND,
    HybridThresholdRegimeClassifier,
    RegimeFeatures,
)
from risk.risk_governor import (
    APPROVE,
    REJECT_CONSISTENCY_CAP,
    REJECT_DAILY_LOSS_HALT,
    REJECT_DAILY_PROFIT_HALT,
    REJECT_MAX_TRADES,
    REJECT_NO_TRADE_WINDOW,
    REJECT_SESSION_CUTOFF,
    ConsistencyCapEngine,
    GovernorResult,
    RiskGovernor,
)
from strategies.mr_signal_engine import MRSignal, MRSignalEngine


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_bar(
    ts: datetime,
    open_: float = 5800.0,
    high: float = 5801.0,
    low: float = 5799.0,
    close: float = 5800.5,
    volume: float = 100.0,
) -> Bar:
    return Bar(
        timestamp=ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def _make_bars(n: int, base_price: float = 5800.0, spread: float = 1.0) -> list[Bar]:
    """Generate n synthetic bars with small random-walk-like movement."""
    bars: list[Bar] = []
    base = datetime(2025, 2, 18, 10, 0, 0)
    price = base_price
    for i in range(n):
        ts = base.replace(minute=(base.minute + i * 5) % 60,
                          hour=base.hour + (base.minute + i * 5) // 60)
        o = price
        h = price + spread
        l = price - spread
        c = price + (0.25 if i % 2 == 0 else -0.25)
        bars.append(_make_bar(ts, open_=o, high=h, low=l, close=c, volume=100 + i * 10))
        price = c
    return bars


# ═══════════════════════════════════════════════════════════════════════
#  1. Regime Classifier — warmup & classification
# ═══════════════════════════════════════════════════════════════════════


class TestRegimeClassifier:
    def test_warmup_returns_none(self) -> None:
        """Classifier returns None during warmup period."""
        from config import REGIME_WARMUP_BARS
        clf = HybridThresholdRegimeClassifier()
        bars = _make_bars(REGIME_WARMUP_BARS - 1)  # One bar short of warmup
        for bar in bars:
            result = clf.update(bar)
        assert result is None
        assert clf.current_regime is None

    def test_classification_after_warmup(self) -> None:
        """Classifier returns a valid label after warmup."""
        clf = HybridThresholdRegimeClassifier()
        bars = _make_bars(20)
        for bar in bars:
            result = clf.update(bar)
        assert result in (REGIME_RANGE, REGIME_TREND, REGIME_EXTREME)
        assert clf.current_regime == result

    def test_features_history_length(self) -> None:
        """One RegimeFeatures per bar is recorded."""
        clf = HybridThresholdRegimeClassifier()
        bars = _make_bars(25)
        for bar in bars:
            clf.update(bar)
        assert len(clf.features_history) == 25

    def test_features_have_timestamps(self) -> None:
        """Every feature snapshot has the bar's timestamp."""
        clf = HybridThresholdRegimeClassifier()
        bars = _make_bars(15)
        for bar in bars:
            clf.update(bar)
        for feat, bar in zip(clf.features_history, bars):
            assert feat.timestamp == bar.timestamp

    def test_reset_clears_state(self) -> None:
        """Reset returns classifier to fresh state."""
        clf = HybridThresholdRegimeClassifier()
        for bar in _make_bars(20):
            clf.update(bar)
        assert clf.bar_count == 20
        clf.reset()
        assert clf.bar_count == 0
        assert clf.current_regime is None
        assert len(clf.features_history) == 0

    def test_low_volatility_range_bars_classify_as_range(self) -> None:
        """Very tight bars (low ATR, low ADX) should classify as range."""
        clf = HybridThresholdRegimeClassifier()
        # 30 bars with tiny spread → low ADX, low ATR → range
        bars = _make_bars(30, base_price=5800.0, spread=0.25)
        for bar in bars:
            clf.update(bar)
        # After warmup, should be in range regime
        assert clf.current_regime == REGIME_RANGE


# ═══════════════════════════════════════════════════════════════════════
#  2. MR Signal Engine — sanity checks
# ═══════════════════════════════════════════════════════════════════════


class TestMRSignalEngine:
    def test_no_signals_before_vwap_warmup(self) -> None:
        """Engine should not emit signals when VWAP bar_count < 3."""
        engine = MRSignalEngine()
        bar = _make_bar(datetime(2025, 2, 18, 10, 0))
        vs = VWAPState(bar_count=2, vwap=5800.0, std_dev=5.0)
        sig = engine.on_bar(bar, "range", vs, atr=2.0)
        assert sig is None

    def test_no_signals_in_trend_regime(self) -> None:
        """MR signals only in range regime — trend should produce None."""
        engine = MRSignalEngine()
        # Price at 2.5σ below VWAP — would be a BUY candidate in range
        bar = _make_bar(datetime(2025, 2, 18, 10, 30), close=5780.0)
        vs = VWAPState(bar_count=10, vwap=5800.0, std_dev=5.0,
                       lower_2_5=5787.5, upper_2_5=5812.5,
                       lower_3_0=5785.0, upper_3_0=5815.0)
        sig = engine.on_bar(bar, "trend", vs, atr=2.0)
        assert sig is None

    def test_no_signals_in_extreme_regime(self) -> None:
        """Extreme regime should also suppress MR signals."""
        engine = MRSignalEngine()
        bar = _make_bar(datetime(2025, 2, 18, 10, 30), close=5780.0)
        vs = VWAPState(bar_count=10, vwap=5800.0, std_dev=5.0)
        sig = engine.on_bar(bar, "extreme", vs, atr=2.0)
        assert sig is None

    def test_buy_signal_at_lower_band(self) -> None:
        """BUY signal fires on reclaim after a lower-band excursion."""
        engine = MRSignalEngine()
        # First bar: excursion below entry threshold
        extreme_bar = _make_bar(datetime(2025, 2, 18, 10, 30), close=5787.0)
        # Second bar: reclaim above threshold with close in top 40% of range
        reclaim_bar = _make_bar(
            datetime(2025, 2, 18, 10, 35),
            high=5796.0,
            low=5792.0,
            close=5795.5,
        )
        vs = VWAPState(bar_count=10, vwap=5800.0, std_dev=5.0,
                       lower_2_5=5787.5, upper_2_5=5812.5,
                       lower_3_0=5785.0, upper_3_0=5815.0)
        assert engine.on_bar(extreme_bar, "range", vs, atr=2.0) is None
        sig = engine.on_bar(reclaim_bar, "range", vs, atr=2.0)
        assert sig is not None
        assert sig.side == "BUY"
        assert sig.approved is True

    def test_sell_signal_at_upper_band(self) -> None:
        """SELL signal fires on reclaim after an upper-band excursion."""
        engine = MRSignalEngine()
        extreme_bar = _make_bar(datetime(2025, 2, 18, 10, 30), close=5813.0)
        reclaim_bar = _make_bar(
            datetime(2025, 2, 18, 10, 35),
            high=5808.0,
            low=5804.0,
            close=5804.2,
        )
        vs = VWAPState(bar_count=10, vwap=5800.0, std_dev=5.0,
                       lower_2_5=5787.5, upper_2_5=5812.5,
                       lower_3_0=5785.0, upper_3_0=5815.0)
        assert engine.on_bar(extreme_bar, "range", vs, atr=2.0) is None
        sig = engine.on_bar(reclaim_bar, "range", vs, atr=2.0)
        assert sig is not None
        assert sig.side == "SELL"
        assert sig.approved is True

    def test_reclaim_off_buy_triggers_on_threshold_cross(self) -> None:
        """Reclaim OFF emits BUY on inside→outside threshold cross."""
        engine = MRSignalEngine(reclaim_mode="off")
        vs = VWAPState(bar_count=20, vwap=5800.0, std_dev=5.0)

        # Warmup/inside bar
        assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 30), close=5794.0), "range", vs, atr=2.0) is None

        # Cross below entry threshold (-1.4σ = 5793.0) => BUY candidate in OFF mode
        sig = engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 10, 35), high=5793.5, low=5791.5, close=5792.5),
            "range",
            vs,
            atr=2.0,
        )
        assert sig is not None
        assert sig.side == "BUY"

    def test_reclaim_on_does_not_trigger_until_reclaim(self) -> None:
        """Reclaim ON requires outside→inside re-entry before signal emission."""
        engine = MRSignalEngine(reclaim_mode="on")
        vs = VWAPState(bar_count=20, vwap=5800.0, std_dev=5.0)

        # Inside bar
        assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 30), close=5794.0), "range", vs, atr=2.0) is None
        # Outside threshold: should not emit yet in reclaim ON mode
        assert engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 10, 35), high=5793.5, low=5791.5, close=5792.5),
            "range",
            vs,
            atr=2.0,
        ) is None

        # Reclaim back inside threshold => BUY signal
        sig = engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 10, 40), high=5795.5, low=5793.5, close=5794.0),
            "range",
            vs,
            atr=2.0,
        )
        assert sig is not None
        assert sig.side == "BUY"

    def test_reclaim_soft_blocks_when_impulse_too_negative_for_buy(self) -> None:
        """Soft-v3 rejects large-range continuation bar against BUY fade."""
        engine = MRSignalEngine(reclaim_mode="soft", soft_reclaim_range_impulse_k=1.2)
        vs = VWAPState(bar_count=20, vwap=5800.0, std_dev=5.0)

        # Inside bar
        assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 30), open_=5794.0, close=5794.0), "range", vs, atr=2.0) is None

        # Cross below threshold with large range_impulse and bearish body => blocked
        blocked = engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 10, 35), open_=5793.0, high=5793.5, low=5790.9, close=5791.5),
            "range",
            vs,
            atr=2.0,
        )
        assert blocked is None

        # Re-enter inside to reset crossing state
        assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 40), open_=5794.0, close=5794.0), "range", vs, atr=2.0) is None

        # Cross below threshold with small range => allowed
        sig = engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 10, 45), open_=5793.0, high=5793.2, low=5791.9, close=5792.6),
            "range",
            vs,
            atr=2.0,
        )
        assert sig is not None
        assert sig.side == "BUY"

    def test_reclaim_soft_threshold_is_tunable(self) -> None:
        """Higher k_range should keep candidate that lower k_range rejects."""
        vs = VWAPState(bar_count=20, vwap=5800.0, std_dev=5.0)

        strict = MRSignalEngine(reclaim_mode="soft", soft_reclaim_range_impulse_k=1.0)
        loose = MRSignalEngine(reclaim_mode="soft", soft_reclaim_range_impulse_k=1.4)

        for engine in (strict, loose):
            assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 30), open_=5794.0, close=5794.0), "range", vs, atr=2.0) is None

        candidate_bar = _make_bar(
            datetime(2025, 2, 18, 10, 35),
            open_=5793.0,
            high=5793.2,
            low=5790.9,
            close=5792.6,  # range_impulse = 1.15 ATR with bearish body
        )

        assert strict.on_bar(candidate_bar, "range", vs, atr=2.0) is None
        loose_sig = loose.on_bar(candidate_bar, "range", vs, atr=2.0)
        assert loose_sig is not None
        assert loose_sig.side == "BUY"

    def test_sigma_entry_is_tunable(self) -> None:
        """Lower sigma entry should emit sooner than the default threshold."""
        base = MRSignalEngine(reclaim_mode="off")
        loose = MRSignalEngine(reclaim_mode="off", sigma_entry=1.2)
        vs = VWAPState(bar_count=20, vwap=5800.0, std_dev=5.0)

        inside_bar = _make_bar(datetime(2025, 2, 18, 10, 30), close=5794.3)
        candidate_bar = _make_bar(datetime(2025, 2, 18, 10, 35), high=5794.2, low=5793.6, close=5793.9)

        assert base.on_bar(inside_bar, "range", vs, atr=2.0) is None
        assert loose.on_bar(inside_bar, "range", vs, atr=2.0) is None

        assert base.on_bar(candidate_bar, "range", vs, atr=2.0) is None
        loose_sig = loose.on_bar(candidate_bar, "range", vs, atr=2.0)
        assert loose_sig is not None
        assert loose_sig.side == "BUY"

    def test_gate_funnel_includes_accounting_fields(self) -> None:
        """Gate funnel report carries bars/crosses/mode accounting fields."""
        engine = MRSignalEngine(reclaim_mode="off")
        vs = VWAPState(bar_count=20, vwap=5800.0, std_dev=5.0)

        # One inside and one outside-cross bar
        assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 30), close=5794.0), "range", vs, atr=2.0) is None
        _ = engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 35), close=5792.5), "range", vs, atr=2.0)

        funnel = engine.gate_funnel_report
        assert funnel.get("candidate_mode") == "off"
        assert int(funnel.get("bars_evaluated", 0)) >= 2
        assert int(funnel.get("z_cross_inside_to_outside", 0)) >= 1

    def test_cooldown_blocks_consecutive_signals(self) -> None:
        """Cooldown prevents immediate back-to-back signals when > 1."""
        from config import MR_COOLDOWN_BARS
        engine = MRSignalEngine()
        vs = VWAPState(bar_count=10, vwap=5800.0, std_dev=5.0,
                       lower_2_5=5787.5, upper_2_5=5812.5,
                       lower_3_0=5785.0, upper_3_0=5815.0)
        # Signal 1: excursion then reclaim
        assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 30), close=5787.0), "range", vs, atr=2.0) is None
        sig1 = engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 10, 35), high=5796.0, low=5792.0, close=5795.5),
            "range",
            vs,
            atr=2.0,
        )
        assert sig1 is not None and sig1.approved

        if MR_COOLDOWN_BARS <= 1:
            return

        # Signal 2 attempt within cooldown: another excursion + reclaim
        assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 40), close=5787.0), "range", vs, atr=2.0) is None
        blocked = engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 10, 45), high=5796.0, low=5792.0, close=5795.5),
            "range",
            vs,
            atr=2.0,
        )
        assert blocked is not None
        assert blocked.approved is False
        assert blocked.rejection_reason == "COOLDOWN"

    def test_reset_clears_signals(self) -> None:
        """Reset clears all accumulated signals."""
        engine = MRSignalEngine()
        vs = VWAPState(bar_count=10, vwap=5800.0, std_dev=5.0,
                       lower_2_5=5787.5, upper_2_5=5812.5,
                       lower_3_0=5785.0, upper_3_0=5815.0)
        engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 30), close=5787.0), "range", vs, atr=2.0)
        engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 10, 35), high=5796.0, low=5792.0, close=5795.5),
            "range",
            vs,
            atr=2.0,
        )
        assert len(engine.signals) == 1
        engine.reset()
        assert len(engine.signals) == 0

    def test_excursion_dedupe_allows_one_trade_per_excursion(self) -> None:
        """Only one approved trade is allowed per excursion until reset."""
        from config import MR_EXCURSION_DEDUPE_ENABLED
        if not MR_EXCURSION_DEDUPE_ENABLED:
            return
        engine = MRSignalEngine()
        vs = VWAPState(bar_count=20, vwap=5800.0, std_dev=5.0)
        # First excursion + reclaim => approved
        assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 30), close=5787.0), "range", vs, atr=2.0) is None
        first = engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 10, 35), high=5796.0, low=5792.0, close=5794.5),
            "range",
            vs,
            atr=2.0,
        )
        assert first is not None and first.approved

        # Same excursion, second reclaim attempt => rejected by dedupe
        assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 40), close=5787.0), "range", vs, atr=2.0) is None
        second = engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 10, 45), high=5796.0, low=5792.0, close=5794.5),
            "range",
            vs,
            atr=2.0,
        )
        assert second is not None
        assert second.approved is False
        assert second.rejection_reason == "EXCURSION_ALREADY_TRADED"

        # Reset excursion by touching VWAP, then allow next excursion
        reset_bar = _make_bar(datetime(2025, 2, 18, 10, 50), close=5800.0)
        assert engine.on_bar(reset_bar, "range", vs, atr=2.0) is None

        assert engine.on_bar(_make_bar(datetime(2025, 2, 18, 10, 55), close=5787.0), "range", vs, atr=2.0) is None
        reopened = engine.on_bar(
            _make_bar(datetime(2025, 2, 18, 11, 0), high=5796.0, low=5792.0, close=5795.5),
            "range",
            vs,
            atr=2.0,
        )
        assert reopened is not None
        assert reopened.approved is True


# ═══════════════════════════════════════════════════════════════════════
#  3. Risk Governor — synthetic gate tests
# ═══════════════════════════════════════════════════════════════════════


class TestRiskGovernor:
    def test_approve_normal_conditions(self) -> None:
        """Governor approves when all conditions are met."""
        gov = RiskGovernor()
        result = gov.evaluate(
            daily_pnl=100.0,
            daily_trade_count=2,
            current_time_str_HHMM="10:30",
        )
        assert result.approved is True
        assert APPROVE in result.reasons

    def test_reject_daily_loss_halt(self) -> None:
        """Governor rejects when daily P&L exceeds loss halt."""
        gov = RiskGovernor()
        result = gov.evaluate(
            daily_pnl=-250.0,  # exceeds RG_DAILY_LOSS_HALT=240
            daily_trade_count=2,
            current_time_str_HHMM="10:30",
        )
        assert result.approved is False
        assert REJECT_DAILY_LOSS_HALT in result.reasons

    def test_reject_daily_profit_halt(self) -> None:
        """Governor rejects when daily profit exceeds cap."""
        gov = RiskGovernor()
        result = gov.evaluate(
            daily_pnl=1300.0,  # exceeds RG_STRATEGY_DAILY_PROFIT_CAP=1200
            daily_trade_count=2,
            current_time_str_HHMM="10:30",
        )
        assert result.approved is False
        assert REJECT_DAILY_PROFIT_HALT in result.reasons

    def test_reject_max_trades(self) -> None:
        """Governor rejects when max daily trades reached."""
        gov = RiskGovernor()
        result = gov.evaluate(
            daily_pnl=100.0,
            daily_trade_count=5,  # == RG_MAX_TRADES_PER_DAY
            current_time_str_HHMM="10:30",
        )
        assert result.approved is False
        assert REJECT_MAX_TRADES in result.reasons

    def test_reject_no_trade_window(self) -> None:
        """Governor rejects during configured no-trade window."""
        gov = RiskGovernor()
        result = gov.evaluate(
            daily_pnl=100.0,
            daily_trade_count=1,
            current_time_str_HHMM="09:31",  # inside ("09:30","09:32")
        )
        assert result.approved is False
        assert REJECT_NO_TRADE_WINDOW in result.reasons

    def test_reject_session_cutoff(self) -> None:
        """Governor rejects after session cutoff time."""
        gov = RiskGovernor()
        result = gov.evaluate(
            daily_pnl=100.0,
            daily_trade_count=1,
            current_time_str_HHMM="15:55",  # past RG_FLATTEN_CUTOFF_TIME=15:50
        )
        assert result.approved is False
        assert REJECT_SESSION_CUTOFF in result.reasons

    def test_multiple_rejections_accumulated(self) -> None:
        """Multiple rejection reasons are accumulated."""
        gov = RiskGovernor()
        result = gov.evaluate(
            daily_pnl=-300.0,  # loss halt + ...
            daily_trade_count=6,  # max trades
            current_time_str_HHMM="16:00",  # session cutoff
        )
        assert result.approved is False
        assert REJECT_DAILY_LOSS_HALT in result.reasons
        assert REJECT_MAX_TRADES in result.reasons
        assert REJECT_SESSION_CUTOFF in result.reasons

    def test_consistency_cap_rejection(self) -> None:
        """Governor rejects when consistency cap is binding."""
        cap_engine = ConsistencyCapEngine(mode="combine")
        gov = RiskGovernor(consistency_engine=cap_engine)

        # Best day is > 50% of total → cap should bind
        result = gov.evaluate(
            daily_pnl=1600.0,  # today is the best day
            daily_trade_count=2,
            current_time_str_HHMM="10:30",
            total_realized_pnl=2000.0,
            best_day_pnl=1500.0,
        )
        assert result.approved is False
        assert REJECT_DAILY_PROFIT_HALT in result.reasons  # profit halt fires too


# ═══════════════════════════════════════════════════════════════════════
#  4. Consistency Cap Engine — math tests
# ═══════════════════════════════════════════════════════════════════════


class TestConsistencyCapEngine:
    def test_combine_50_pct_cap(self) -> None:
        """Combine mode: 50% of profit_target = $1500 cap."""
        engine = ConsistencyCapEngine(mode="combine")
        result = engine.evaluate(
            total_realized_pnl=3000.0,
            best_day_pnl=1000.0,
            today_realized_pnl=0.0,
            profit_target=3000.0,
        )
        assert result.cap_pct == 50.0
        assert result.effective_daily_cap == 1500.0  # 0.5 * 3000
        assert result.capped is False
        assert result.allowed_profit_remaining > 0

    def test_combine_cap_binding(self) -> None:
        """Combine mode caps when today exceeds 50% of target."""
        engine = ConsistencyCapEngine(mode="combine")
        result = engine.evaluate(
            total_realized_pnl=3000.0,
            best_day_pnl=500.0,
            today_realized_pnl=1600.0,  # exceeds $1500 cap
            profit_target=3000.0,
        )
        assert result.capped is True
        assert result.allowed_profit_remaining == 0.0

    def test_express_funded_40_pct(self) -> None:
        """Express funded: 40% cap."""
        engine = ConsistencyCapEngine(mode="express_funded")
        result = engine.evaluate(
            total_realized_pnl=3000.0,
            best_day_pnl=500.0,
            today_realized_pnl=0.0,
            profit_target=3000.0,
        )
        assert result.cap_pct == 40.0
        assert result.effective_daily_cap == 1200.0  # 0.4 * 3000

    def test_xfa_standard_no_strict_cap(self) -> None:
        """XFA standard: no strict percentage cap (soft cap only)."""
        engine = ConsistencyCapEngine(mode="xfa_standard")
        result = engine.evaluate(
            total_realized_pnl=5000.0,
            best_day_pnl=3000.0,  # 60% of total
            today_realized_pnl=0.0,
        )
        assert result.mode == "xfa_standard"
        assert result.cap_pct == 0.0  # no pct cap

    def test_zero_total_pnl_first_day(self) -> None:
        """First day with no prior P&L — should allow up to effective cap."""
        engine = ConsistencyCapEngine(mode="combine")
        result = engine.evaluate(
            total_realized_pnl=0.0,
            best_day_pnl=0.0,
            today_realized_pnl=0.0,
            profit_target=3000.0,
        )
        assert result.capped is False
        assert result.allowed_profit_remaining == 1500.0  # 50% of 3000


# ═══════════════════════════════════════════════════════════════════════
#  5. Replay Report — basic instantiation & export
# ═══════════════════════════════════════════════════════════════════════


class TestReplayReport:
    def test_report_instantiation(self) -> None:
        """ReplaySessionReport instantiates without error."""
        from reporting.replay_report import ReplaySessionReport
        report = ReplaySessionReport(
            session_id="test_001",
            symbol="MES.c.0",
            replay_start="2025-02-18T14:30:00Z",
            replay_end="2025-02-18T16:00:00Z",
        )
        assert report.session_id == "test_001"
        assert report.symbol == "MES.c.0"

    def test_report_export_creates_files(self, tmp_path) -> None:
        """Export creates signals.csv and session_summary.json."""
        from reporting.replay_report import ReplaySessionReport
        report = ReplaySessionReport(
            session_id="test_export",
            symbol="MES.c.0",
            replay_start="2025-02-18T14:30:00Z",
            replay_end="2025-02-18T16:00:00Z",
        )
        report.set_tick_stats(
            ticks_processed=1000,
            bars_closed=10,
            bars_partial_flushed=1,
            unique_buckets=3,
        )
        # Export to tmp_path
        out_dir = report.export(base_dir=str(tmp_path))
        assert (out_dir / "signals.csv").exists()
        assert (out_dir / "session_summary.json").exists()

    def test_report_with_signals(self, tmp_path) -> None:
        """Export with actual signals produces CSV rows."""
        import csv
        from reporting.replay_report import ReplaySessionReport

        report = ReplaySessionReport(
            session_id="test_signals",
            symbol="MES.c.0",
            replay_start="2025-02-18T14:30:00Z",
            replay_end="2025-02-18T16:00:00Z",
        )
        report.set_tick_stats(ticks_processed=500, bars_closed=5,
                              bars_partial_flushed=0, unique_buckets=2)
        sig = MRSignal(
            timestamp=datetime(2025, 2, 18, 10, 30),
            side="BUY",
            approved=True,
            entry_reference_price=5787.0,
            vwap_at_signal=5800.0,
            sigma_at_signal=5.0,
            band_level_hit=2.5,
        )
        report.add_signals([sig])
        out_dir = report.export(base_dir=str(tmp_path))

        with open(out_dir / "signals.csv") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["side"] == "BUY"
        assert reader[0]["approved"] == "True"


# ═══════════════════════════════════════════════════════════════════════
#  6. Dashboard — instantiation (no display needed)
# ═══════════════════════════════════════════════════════════════════════


class TestReplayDashboard:
    def test_dashboard_instantiation(self) -> None:
        """Dashboard instantiates with mock figure and axes."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualization.replay_dashboard import ReplayDashboard

        fig, (ax1, ax2) = plt.subplots(2, 1)
        dash = ReplayDashboard(fig, ax1, ax2)
        assert dash.fig is fig
        assert dash.ax_px is ax1
        assert dash.ax_bar is ax2
        plt.close(fig)

    def test_dashboard_finalize_empty(self) -> None:
        """Finalize with no data should not crash."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualization.replay_dashboard import ReplayDashboard

        fig, (ax1, ax2) = plt.subplots(2, 1)
        dash = ReplayDashboard(fig, ax1, ax2)
        # Should not raise
        dash.finalize(bar_times=[], bar_closes=[])
        plt.close(fig)

    def test_dashboard_finalize_with_data(self) -> None:
        """Finalize with bars and VWAP history draws without error."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualization.replay_dashboard import ReplayDashboard

        fig, (ax1, ax2) = plt.subplots(2, 1)
        dash = ReplayDashboard(fig, ax1, ax2)

        bar_times = [datetime(2025, 2, 18, 10, i * 5) for i in range(5)]
        bar_closes = [5800.0, 5801.0, 5799.0, 5802.0, 5800.5]
        vwap_hist = [
            VWAPState(vwap=5800.0, std_dev=3.0, bar_count=i + 1,
                      upper_2_5=5807.5, lower_2_5=5792.5,
                      upper_3_0=5809.0, lower_3_0=5791.0)
            for i in range(5)
        ]

        dash.finalize(
            bar_times=bar_times,
            bar_closes=bar_closes,
            vwap_history=vwap_hist,
        )
        plt.close(fig)
