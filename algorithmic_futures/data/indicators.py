"""
data/indicators.py — Technical indicators for intraday strategy execution.

Provides:
  - VWAPCalculator   : anchored VWAP + σ bands from RTH open
  - ATRCalculator    : rolling ATR on arbitrary bar size
  - CVDCalculator    : cumulative volume delta with divergence detection (proxy mode)
  - FeatureBuilder   : daily HMM feature construction (log return, ATR percentile, VIX slope)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import (
    CVD_DIVERGENCE_LOOKBACK,
    VWAP_SD_ENTRY_MAX,
    VWAP_SD_ENTRY_MIN,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  VWAP Calculator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class VWAPState:
    """Running state for anchored VWAP computation."""
    cum_volume: float = 0.0
    cum_vwap_num: float = 0.0      # Σ (typical_price × volume)
    cum_vwap_sq_num: float = 0.0   # Σ (typical_price² × volume)
    vwap: float = 0.0
    std_dev: float = 0.0
    upper_2_5: float = 0.0
    upper_3_0: float = 0.0
    lower_2_5: float = 0.0
    lower_3_0: float = 0.0
    bar_count: int = 0


class VWAPCalculator:
    """Anchored VWAP with rolling standard deviation bands.

    Reset at RTH open each session.  Updated on every new bar close.
    """

    def __init__(self) -> None:
        self._state = VWAPState()

    @property
    def state(self) -> VWAPState:
        return self._state

    def reset(self) -> None:
        """Call at RTH open to re-anchor VWAP."""
        self._state = VWAPState()

    def update(self, high: float, low: float, close: float, volume: float) -> VWAPState:
        """Feed a new OHLCV bar and return updated VWAP state."""
        if volume <= 0:
            return self._state

        typical_price = (high + low + close) / 3.0

        s = self._state
        s.cum_volume += volume
        s.cum_vwap_num += typical_price * volume
        s.cum_vwap_sq_num += (typical_price ** 2) * volume
        s.bar_count += 1

        s.vwap = s.cum_vwap_num / s.cum_volume

        # Population variance: E[X²] - E[X]²
        variance = max(0.0, (s.cum_vwap_sq_num / s.cum_volume) - s.vwap ** 2)
        s.std_dev = variance ** 0.5

        # SD bands
        s.upper_2_5 = s.vwap + VWAP_SD_ENTRY_MIN * s.std_dev
        s.upper_3_0 = s.vwap + VWAP_SD_ENTRY_MAX * s.std_dev
        s.lower_2_5 = s.vwap - VWAP_SD_ENTRY_MIN * s.std_dev
        s.lower_3_0 = s.vwap - VWAP_SD_ENTRY_MAX * s.std_dev

        return self._state

    def is_at_lower_extreme(self, price: float) -> bool:
        """True when price is in the -2.5σ to -3.0σ zone."""
        s = self._state
        return s.bar_count > 0 and s.lower_3_0 <= price <= s.lower_2_5

    def is_at_upper_extreme(self, price: float) -> bool:
        """True when price is in the +2.5σ to +3.0σ zone."""
        s = self._state
        return s.bar_count > 0 and s.upper_2_5 <= price <= s.upper_3_0


# ═══════════════════════════════════════════════════════════════════════
#  ATR Calculator
# ═══════════════════════════════════════════════════════════════════════


class ATRCalculator:
    """Rolling Average True Range on intraday bars."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._true_ranges: list[float] = []
        self._prev_close: float | None = None
        self.atr: float = 0.0

    def reset(self) -> None:
        self._true_ranges.clear()
        self._prev_close = None
        self.atr = 0.0

    def update(self, high: float, low: float, close: float) -> float:
        """Feed a new bar; returns current ATR value."""
        if self._prev_close is not None:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )
        else:
            tr = high - low

        self._prev_close = close
        self._true_ranges.append(tr)

        if len(self._true_ranges) >= self.period:
            # Wilder smoothing after initial SMA seed
            if len(self._true_ranges) == self.period:
                self.atr = float(np.mean(self._true_ranges[-self.period:]))
            else:
                self.atr = (self.atr * (self.period - 1) + tr) / self.period
        elif self._true_ranges:
            self.atr = float(np.mean(self._true_ranges))

        return self.atr


# ═══════════════════════════════════════════════════════════════════════
#  CVD Calculator (proxy mode)
# ═══════════════════════════════════════════════════════════════════════


class CVDCalculator:
    """Cumulative Volume Delta — proxy implementation.

    In client_fallback mode, trade direction is inferred from price
    movement (tick rule): uptick = buy-initiated, downtick = sell-initiated.

    Signals from this calculator are labelled as *proxy-derived*
    per project convention.  When projectx_native tick classification
    becomes available, swap in a native CVDProvider that receives
    actual bid/ask aggressor tags.
    """

    def __init__(self) -> None:
        self.cumulative_delta: float = 0.0
        self.bar_deltas: list[float] = []
        self._last_price: float | None = None

    def reset(self) -> None:
        self.cumulative_delta = 0.0
        self.bar_deltas.clear()
        self._last_price = None

    # -- tick-level feed --------------------------------------------------

    def on_tick(self, price: float, size: float, side: str | None = None) -> None:
        """Process a single tick.

        Parameters
        ----------
        price : trade price
        size  : trade size (contracts)
        side  : explicit 'buy'/'sell' if available; otherwise tick rule is applied.
        """
        if side is not None:
            delta = size if side.lower() == "buy" else -size
        else:
            # Tick-rule proxy
            if self._last_price is not None:
                if price > self._last_price:
                    delta = size
                elif price < self._last_price:
                    delta = -size
                else:
                    delta = 0.0
            else:
                delta = 0.0
            self._last_price = price

        self.cumulative_delta += delta

    def close_bar(self) -> None:
        """Snapshot CVD at bar close."""
        self.bar_deltas.append(self.cumulative_delta)

    # -- bar-level feed (simplified proxy from OHLCV) ---------------------

    def update_bar(self, open_: float, close: float, volume: float) -> None:
        """Proxy CVD from a completed OHLCV bar.

        If close > open, volume attributed as buying; else selling.
        This is a coarse approximation acceptable for v1 fallback.
        """
        if close >= open_:
            self.cumulative_delta += volume
        else:
            self.cumulative_delta -= volume
        self.bar_deltas.append(self.cumulative_delta)

    # -- divergence detection ---------------------------------------------

    def detect_divergence(
        self,
        price_lows: list[float] | np.ndarray,
        price_highs: list[float] | np.ndarray,
        lookback: int = CVD_DIVERGENCE_LOOKBACK,
    ) -> str | None:
        """Check for CVD divergence in the last *lookback* bars.

        Returns
        -------
        'BULLISH'  — price lower low + CVD higher low (buy absorption)
        'BEARISH'  — price higher high + CVD lower high (sell absorption)
        None       — no divergence detected
        """
        if len(self.bar_deltas) < lookback or len(price_lows) < lookback:
            return None

        p_lows = list(price_lows)[-lookback:]
        p_highs = list(price_highs)[-lookback:]
        cvd_recent = self.bar_deltas[-lookback:]

        # Bullish: price made lower low, but CVD made higher low
        if p_lows[-1] < p_lows[0] and cvd_recent[-1] > cvd_recent[0]:
            return "BULLISH"

        # Bearish: price made higher high, but CVD made lower high
        if p_highs[-1] > p_highs[0] and cvd_recent[-1] < cvd_recent[0]:
            return "BEARISH"

        return None


# ═══════════════════════════════════════════════════════════════════════
#  Daily HMM Feature Builder
# ═══════════════════════════════════════════════════════════════════════


class FeatureBuilder:
    """Construct the 3-feature DataFrame required by the HMM classifier.

    Expects daily OHLCV for ES/SPY and closing values for VIX + VIX3M.
    """

    ATR_PERIOD = 14
    ATR_RANK_WINDOW = 252  # 1 trading year for percentile ranking

    @staticmethod
    def build(
        daily_df: pd.DataFrame,
        vix_series: pd.Series,
        vix3m_series: pd.Series,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns: log_return, atr_percentile, vix_term_slope.

        Parameters
        ----------
        daily_df   : Daily OHLCV with columns High, Low, Close (index = date).
        vix_series : Daily VIX close, aligned to same dates.
        vix3m_series : Daily VIX3M close, aligned to same dates.
        """
        df = daily_df.copy()

        # 1. Log return
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # 2. ATR percentile (14-day ATR ranked over 252-day window)
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - df["Close"].shift(1)).abs(),
                (df["Low"] - df["Close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(window=FeatureBuilder.ATR_PERIOD).mean()
        df["atr_percentile"] = atr.rolling(window=FeatureBuilder.ATR_RANK_WINDOW).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # 3. VIX term structure slope: (VIX3M - VIX) / VIX
        vix_aligned = vix_series.reindex(df.index).ffill()
        vix3m_aligned = vix3m_series.reindex(df.index).ffill()
        df["vix_term_slope"] = (vix3m_aligned - vix_aligned) / vix_aligned.replace(0, np.nan)

        features = df[["log_return", "atr_percentile", "vix_term_slope"]].dropna()
        return features
