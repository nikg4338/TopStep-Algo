"""
tests/test_indicators.py — Tests for data/indicators.py (VWAP, ATR, CVD).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from data.indicators import VWAPCalculator, VWAPState, ATRCalculator, CVDCalculator


# ═══════════════════════════════════════════════════════════════════════
#  VWAP Calculator Tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def vwap():
    return VWAPCalculator()


class TestVWAPSingleBar:
    def test_single_bar_vwap_equals_typical_price(self, vwap):
        """Single bar → VWAP = typical price = (H+L+C)/3."""
        h, l, c, v = 100.0, 98.0, 99.0, 1000
        state = vwap.update(h, l, c, v)

        expected_tp = (h + l + c) / 3.0
        assert state.vwap == pytest.approx(expected_tp)

    def test_single_bar_std_dev_zero(self, vwap):
        """Only one bar → no deviation from the mean."""
        vwap.update(100.0, 98.0, 99.0, 1000)
        assert vwap.state.std_dev == pytest.approx(0.0)

    def test_single_bar_count(self, vwap):
        vwap.update(100.0, 98.0, 99.0, 1000)
        assert vwap.state.bar_count == 1


class TestVWAPMultipleBars:
    def test_20_bars_vwap_matches_manual(self, vwap):
        """Feed 20 known bars and verify VWAP = Σ(TP×V) / Σ(V)."""
        bars = []
        cum_tpv = 0.0
        cum_vol = 0.0

        for i in range(20):
            h = 100.0 + i * 0.5
            l = 98.0 + i * 0.3
            c = 99.0 + i * 0.4
            v = 1000 + i * 100

            tp = (h + l + c) / 3.0
            cum_tpv += tp * v
            cum_vol += v
            bars.append((h, l, c, v))

        for h, l, c, v in bars:
            vwap.update(h, l, c, v)

        expected_vwap = cum_tpv / cum_vol
        assert vwap.state.vwap == pytest.approx(expected_vwap, rel=1e-9)
        assert vwap.state.bar_count == 20

    def test_sd_bands_expand_with_price_deviation(self, vwap):
        """As prices deviate from VWAP, standard deviation should increase."""
        # Tight-range bars
        for _ in range(5):
            vwap.update(100.0, 99.5, 99.75, 1000)
        sd_tight = vwap.state.std_dev

        # Now introduce wide-range bars
        for _ in range(5):
            vwap.update(110.0, 90.0, 100.0, 1000)
        sd_wide = vwap.state.std_dev

        assert sd_wide > sd_tight

    def test_sd_bands_symmetric(self, vwap):
        """Upper and lower bands should be symmetric around VWAP."""
        for i in range(10):
            vwap.update(100.0 + i, 98.0 + i, 99.0 + i, 500)

        s = vwap.state
        # upper_2_5 - vwap should equal vwap - lower_2_5
        assert (s.upper_2_5 - s.vwap) == pytest.approx(s.vwap - s.lower_2_5, rel=1e-9)
        assert (s.upper_3_0 - s.vwap) == pytest.approx(s.vwap - s.lower_3_0, rel=1e-9)


class TestVWAPReset:
    def test_reset_clears_all_state(self, vwap):
        vwap.update(100.0, 98.0, 99.0, 1000)
        assert vwap.state.bar_count == 1

        vwap.reset()

        assert vwap.state.bar_count == 0
        assert vwap.state.cum_volume == 0.0
        assert vwap.state.vwap == 0.0
        assert vwap.state.std_dev == 0.0


class TestVWAPExtremes:
    def test_is_at_lower_extreme(self, vwap):
        """Price in the -2.5σ to -3.0σ zone returns True."""
        # Build enough bars to establish bands
        for i in range(20):
            vwap.update(100.0 + i * 0.1, 99.0 + i * 0.1, 99.5 + i * 0.1, 1000)

        s = vwap.state
        # A price right between lower_3_0 and lower_2_5 should be extreme
        if s.std_dev > 0:
            test_price = (s.lower_2_5 + s.lower_3_0) / 2.0
            assert vwap.is_at_lower_extreme(test_price) is True

    def test_is_at_upper_extreme(self, vwap):
        """Price in the +2.5σ to +3.0σ zone returns True."""
        for i in range(20):
            vwap.update(100.0 + i * 0.1, 99.0 + i * 0.1, 99.5 + i * 0.1, 1000)

        s = vwap.state
        if s.std_dev > 0:
            test_price = (s.upper_2_5 + s.upper_3_0) / 2.0
            assert vwap.is_at_upper_extreme(test_price) is True

    def test_not_extreme_at_vwap(self, vwap):
        """Price at VWAP should not be at any extreme."""
        for i in range(20):
            vwap.update(100.0 + i * 0.1, 99.0 + i * 0.1, 99.5 + i * 0.1, 1000)

        assert vwap.is_at_lower_extreme(vwap.state.vwap) is False
        assert vwap.is_at_upper_extreme(vwap.state.vwap) is False

    def test_no_bars_no_extreme(self, vwap):
        """With zero bars, extreme checks should return False."""
        assert vwap.is_at_lower_extreme(100.0) is False
        assert vwap.is_at_upper_extreme(100.0) is False

    def test_zero_volume_bar_ignored(self, vwap):
        """A bar with 0 volume should not update state."""
        vwap.update(100.0, 98.0, 99.0, 0)
        assert vwap.state.bar_count == 0


# ═══════════════════════════════════════════════════════════════════════
#  ATR Calculator Tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def atr():
    return ATRCalculator(period=14)


@pytest.fixture
def atr2():
    """ATR with period=2 for short-sequence testing."""
    return ATRCalculator(period=2)


class TestATRBasic:
    def test_first_bar_true_range(self, atr):
        """First bar: TR = high - low (no previous close)."""
        result = atr.update(high=105.0, low=100.0, close=103.0)
        assert result == pytest.approx(5.0)

    def test_second_bar_with_gap(self, atr):
        """Second bar considers previous close for true range."""
        atr.update(high=105.0, low=100.0, close=103.0)
        result = atr.update(high=108.0, low=104.0, close=106.0)

        # TR = max(108-104, |108-103|, |104-103|) = max(4, 5, 1) = 5
        expected_atr = (5.0 + 5.0) / 2.0  # SMA of 2 bars
        assert result == pytest.approx(expected_atr)


class TestATRWilderSmoothing:
    def test_period_2_wilder_smoothing(self, atr2):
        """Period=2: first 2 bars → SMA seed, 3rd bar → Wilder smoothing."""
        # Bar 1: TR = 10 - 8 = 2
        atr2.update(high=10.0, low=8.0, close=9.0)

        # Bar 2: TR = max(12-9, |12-9|, |9-9|) = max(3, 3, 0) = 3
        val2 = atr2.update(high=12.0, low=9.0, close=11.0)
        # SMA seed = (2 + 3) / 2 = 2.5
        assert val2 == pytest.approx(2.5)

        # Bar 3: TR = max(14-11, |14-11|, |11-11|) = max(3, 3, 0) = 3
        val3 = atr2.update(high=14.0, low=11.0, close=13.0)
        # Wilder: (2.5 * (2-1) + 3) / 2 = (2.5 + 3) / 2 = 2.75
        assert val3 == pytest.approx(2.75)


class TestATRReset:
    def test_reset_clears(self, atr):
        atr.update(100.0, 95.0, 98.0)
        atr.reset()
        assert atr.atr == 0.0
        assert atr._prev_close is None


# ═══════════════════════════════════════════════════════════════════════
#  CVD Calculator Tests
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def cvd():
    return CVDCalculator()


class TestCVDBarLevel:
    def test_bullish_bar_adds_volume(self, cvd):
        """Close > open → volume added to CVD."""
        cvd.update_bar(open_=100.0, close=102.0, volume=500)
        assert cvd.cumulative_delta == 500
        assert len(cvd.bar_deltas) == 1

    def test_bearish_bar_subtracts_volume(self, cvd):
        """Close < open → volume subtracted from CVD."""
        cvd.update_bar(open_=102.0, close=100.0, volume=500)
        assert cvd.cumulative_delta == -500

    def test_flat_bar_adds_volume(self, cvd):
        """Close == open → treated as buying (close >= open)."""
        cvd.update_bar(open_=100.0, close=100.0, volume=300)
        assert cvd.cumulative_delta == 300


class TestCVDDivergence:
    def test_bullish_divergence(self, cvd):
        """Price lower low + CVD higher low → 'BULLISH'."""
        # Build bar deltas: CVD going up (higher lows)
        cvd.update_bar(100.0, 101.0, 500)   # +500
        cvd.update_bar(100.5, 101.5, 600)   # +1100
        cvd.update_bar(101.0, 102.0, 700)   # +1800

        # Price lows: descending (lower lows)
        price_lows = [50.0, 49.0, 48.0]
        price_highs = [55.0, 54.0, 53.0]

        result = cvd.detect_divergence(price_lows, price_highs, lookback=3)
        assert result == "BULLISH"

    def test_bearish_divergence(self, cvd):
        """Price higher high + CVD lower high → 'BEARISH'."""
        # Build bar deltas: CVD going down (lower highs in CVD terms)
        cvd.update_bar(102.0, 100.0, 500)   # -500
        cvd.update_bar(103.0, 101.0, 600)   # -1100
        cvd.update_bar(104.0, 102.0, 700)   # -1800

        # Price highs: ascending (higher highs)
        price_lows = [48.0, 49.0, 50.0]
        price_highs = [53.0, 54.0, 55.0]

        result = cvd.detect_divergence(price_lows, price_highs, lookback=3)
        assert result == "BEARISH"

    def test_no_divergence_same_direction(self, cvd):
        """Both price and CVD move same direction → None."""
        # Both price and CVD trending up
        cvd.update_bar(100.0, 102.0, 500)
        cvd.update_bar(102.0, 104.0, 600)
        cvd.update_bar(104.0, 106.0, 700)

        # Price lows ascending, CVD ascending → no divergence
        price_lows = [48.0, 49.0, 50.0]
        price_highs = [53.0, 54.0, 55.0]

        result = cvd.detect_divergence(price_lows, price_highs, lookback=3)
        assert result is None

    def test_insufficient_data_returns_none(self, cvd):
        """Not enough bars for lookback → None."""
        cvd.update_bar(100.0, 102.0, 500)
        # Only 1 bar, lookback=3
        result = cvd.detect_divergence([50.0], [55.0], lookback=3)
        assert result is None

    def test_insufficient_price_data(self, cvd):
        """Enough CVD bars but not enough price data → None."""
        for _ in range(5):
            cvd.update_bar(100.0, 102.0, 500)

        result = cvd.detect_divergence([50.0], [55.0], lookback=3)
        assert result is None

    def test_empty_state_returns_none(self, cvd):
        """No bars at all → None."""
        result = cvd.detect_divergence([], [], lookback=3)
        assert result is None


class TestCVDReset:
    def test_reset_clears_state(self, cvd):
        cvd.update_bar(100.0, 102.0, 500)
        cvd.update_bar(102.0, 104.0, 600)

        assert cvd.cumulative_delta != 0
        assert len(cvd.bar_deltas) == 2

        cvd.reset()

        assert cvd.cumulative_delta == 0.0
        assert len(cvd.bar_deltas) == 0
        assert cvd._last_price is None

    def test_divergence_after_reset_returns_none(self, cvd):
        """After reset, no data → divergence detection returns None."""
        cvd.update_bar(100.0, 102.0, 500)
        cvd.update_bar(102.0, 104.0, 600)
        cvd.update_bar(104.0, 106.0, 700)
        cvd.reset()

        result = cvd.detect_divergence([48.0, 49.0, 50.0], [53.0, 54.0, 55.0], lookback=3)
        assert result is None
