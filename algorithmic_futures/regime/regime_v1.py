"""
regime/regime_v1.py — Hybrid threshold regime classifier (v1).

Classifies the current market state as one of:
  - "range"   — low trend strength, normal volatility
  - "trend"   — high trend strength (ADX above threshold)
  - "extreme" — abnormally high volatility or realized vol

Uses 5-minute bars and incremental feature computation.
Interface is designed for easy swap to HMM-based classifier later.

Usage:
    from regime.regime_v1 import HybridThresholdRegimeClassifier
    clf = HybridThresholdRegimeClassifier()
    clf.update(bar)
    state = clf.current_regime  # "range" | "trend" | "extreme" | None
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from config import (
    REGIME_ADX_TREND_THRESHOLD,
    REGIME_ATR_EXTREME_PCTILE,
    REGIME_VOL_EXTREME_PCTILE,
    REGIME_WARMUP_BARS,
)
from data.market_data import Bar

logger = logging.getLogger(__name__)

# ── Regime labels ───────────────────────────────────────────────────────
REGIME_RANGE = "range"
REGIME_TREND = "trend"
REGIME_EXTREME = "extreme"
REGIME_UNKNOWN = None  # Not enough data yet


# ── Feature snapshot per bar ────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeFeatures:
    """Snapshot of regime-relevant features at a given bar."""
    timestamp: datetime
    adx: float
    atr: float
    atr_percentile: float       # percentile of current ATR vs rolling window
    realized_vol: float         # rolling std of log returns
    vol_percentile: float       # percentile of realized vol vs rolling window
    regime: str | None          # classified regime label


# ── Abstract base ───────────────────────────────────────────────────────

class RegimeClassifierBase(ABC):
    """Interface for regime classifiers.  All implementations must
    provide ``update(bar)`` and ``current_regime``."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, bar: Bar) -> str | None: ...

    @property
    @abstractmethod
    def current_regime(self) -> str | None: ...

    @property
    @abstractmethod
    def features_history(self) -> list[RegimeFeatures]: ...


# ── Directional Movement / ADX calculator ───────────────────────────────

class _ADXCalculator:
    """Wilder-smoothed ADX from bar data."""

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._prev_high: float | None = None
        self._prev_low: float | None = None
        self._prev_close: float | None = None
        self._tr_buffer: list[float] = []
        self._plus_dm_buffer: list[float] = []
        self._minus_dm_buffer: list[float] = []
        self._smoothed_tr: float = 0.0
        self._smoothed_plus_dm: float = 0.0
        self._smoothed_minus_dm: float = 0.0
        self._dx_buffer: list[float] = []
        self._adx: float = 0.0
        self._count: int = 0

    def reset(self) -> None:
        self.__init__(self.period)  # type: ignore[misc]

    @property
    def adx(self) -> float:
        return self._adx

    def update(self, high: float, low: float, close: float) -> float:
        if self._prev_high is not None:
            assert self._prev_low is not None
            assert self._prev_close is not None
            tr = max(high - low,
                     abs(high - self._prev_close),
                     abs(low - self._prev_close))
            plus_dm = max(high - self._prev_high, 0.0) \
                if (high - self._prev_high) > (self._prev_low - low) else 0.0
            minus_dm = max(self._prev_low - low, 0.0) \
                if (self._prev_low - low) > (high - self._prev_high) else 0.0
        else:
            tr = high - low
            plus_dm = 0.0
            minus_dm = 0.0

        self._prev_high = high
        self._prev_low = low
        self._prev_close = close
        self._count += 1

        p = self.period

        if self._count <= p:
            self._tr_buffer.append(tr)
            self._plus_dm_buffer.append(plus_dm)
            self._minus_dm_buffer.append(minus_dm)
            if self._count == p:
                self._smoothed_tr = sum(self._tr_buffer)
                self._smoothed_plus_dm = sum(self._plus_dm_buffer)
                self._smoothed_minus_dm = sum(self._minus_dm_buffer)
            return 0.0

        # Wilder smoothing
        self._smoothed_tr = self._smoothed_tr - (self._smoothed_tr / p) + tr
        self._smoothed_plus_dm = self._smoothed_plus_dm - (self._smoothed_plus_dm / p) + plus_dm
        self._smoothed_minus_dm = self._smoothed_minus_dm - (self._smoothed_minus_dm / p) + minus_dm

        if self._smoothed_tr == 0:
            return self._adx

        plus_di = 100.0 * self._smoothed_plus_dm / self._smoothed_tr
        minus_di = 100.0 * self._smoothed_minus_dm / self._smoothed_tr
        di_sum = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0.0

        self._dx_buffer.append(dx)

        if len(self._dx_buffer) < p:
            return 0.0
        elif len(self._dx_buffer) == p:
            self._adx = float(np.mean(self._dx_buffer))
        else:
            self._adx = (self._adx * (p - 1) + dx) / p

        return self._adx


# ── Hybrid Threshold Classifier ────────────────────────────────────────

class HybridThresholdRegimeClassifier(RegimeClassifierBase):
    """V1 regime classifier using ADX + ATR/vol percentiles.

    Classification logic:
      1. If ATR percentile > extreme threshold OR vol percentile > extreme
         threshold → "extreme"
      2. If ADX > trend threshold → "trend"
      3. Otherwise → "range"
    """

    def __init__(
        self,
        adx_period: int = 14,
        atr_period: int = 14,
        vol_window: int = 20,
        percentile_window: int = 100,
    ) -> None:
        self._adx_period = adx_period
        self._atr_period = atr_period
        self._vol_window = vol_window
        self._pctile_window = percentile_window

        self._adx_calc = _ADXCalculator(period=adx_period)
        self._atr_values: list[float] = []
        self._log_returns: list[float] = []
        self._prev_close: float | None = None
        self._bar_count: int = 0
        self._regime: str | None = None
        self._history: list[RegimeFeatures] = []

        # Current ATR via simple rolling mean of TR
        self._tr_buffer: list[float] = []
        self._current_atr: float = 0.0

    def reset(self) -> None:
        self._adx_calc.reset()
        self._atr_values.clear()
        self._log_returns.clear()
        self._prev_close = None
        self._bar_count = 0
        self._regime = None
        self._history.clear()
        self._tr_buffer.clear()
        self._current_atr = 0.0

    @property
    def current_regime(self) -> str | None:
        return self._regime

    @property
    def features_history(self) -> list[RegimeFeatures]:
        return self._history

    @property
    def bar_count(self) -> int:
        return self._bar_count

    def update(self, bar: Bar) -> str | None:
        """Feed a 5-minute bar and return the updated regime label."""
        h, l, c = bar.high, bar.low, bar.close
        self._bar_count += 1

        # ADX
        adx = self._adx_calc.update(h, l, c)

        # ATR (simple rolling mean of TR)
        if self._prev_close is not None:
            tr = max(h - l, abs(h - self._prev_close), abs(l - self._prev_close))
        else:
            tr = h - l
        self._tr_buffer.append(tr)
        if len(self._tr_buffer) >= self._atr_period:
            self._current_atr = float(np.mean(self._tr_buffer[-self._atr_period:]))
        else:
            self._current_atr = float(np.mean(self._tr_buffer))
        self._atr_values.append(self._current_atr)

        # Log return for realized vol
        if self._prev_close is not None and self._prev_close > 0:
            lr = np.log(c / self._prev_close)
        else:
            lr = 0.0
        self._log_returns.append(lr)
        self._prev_close = c

        # Not enough data → unknown
        if self._bar_count < REGIME_WARMUP_BARS:
            feat = RegimeFeatures(
                timestamp=bar.timestamp, adx=adx, atr=self._current_atr,
                atr_percentile=0.0, realized_vol=0.0, vol_percentile=0.0,
                regime=None,
            )
            self._history.append(feat)
            return None

        # ATR percentile
        window = self._atr_values[-self._pctile_window:]
        atr_pctile = float(np.searchsorted(np.sort(window), self._current_atr)
                           / len(window) * 100.0)

        # Realized vol (rolling std of log returns)
        vol_slice = self._log_returns[-self._vol_window:]
        realized_vol = float(np.std(vol_slice)) if len(vol_slice) >= 2 else 0.0

        # Vol percentile (over all available realized vols)
        if len(self._log_returns) >= self._vol_window:
            all_vols: list[float] = []
            for i in range(self._vol_window, len(self._log_returns) + 1):
                chunk = self._log_returns[max(0, i - self._vol_window):i]
                all_vols.append(float(np.std(chunk)))
            vol_pctile = float(np.searchsorted(np.sort(all_vols), realized_vol)
                               / len(all_vols) * 100.0)
        else:
            vol_pctile = 50.0  # default until enough data

        # Classification
        if atr_pctile > REGIME_ATR_EXTREME_PCTILE or vol_pctile > REGIME_VOL_EXTREME_PCTILE:
            regime = REGIME_EXTREME
        elif adx > REGIME_ADX_TREND_THRESHOLD:
            regime = REGIME_TREND
        else:
            regime = REGIME_RANGE

        self._regime = regime

        feat = RegimeFeatures(
            timestamp=bar.timestamp, adx=adx, atr=self._current_atr,
            atr_percentile=atr_pctile, realized_vol=realized_vol,
            vol_percentile=vol_pctile, regime=regime,
        )
        self._history.append(feat)

        return regime
