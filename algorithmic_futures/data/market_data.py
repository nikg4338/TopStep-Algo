"""
data/market_data.py — Market data sourcing and bar aggregation.

Provides:
  - DailyDataProvider  : fetches daily OHLCV + VIX/VIX3M for HMM training
  - IntradayBarAggregator : aggregates tick stream into N-minute OHLCV bars
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Daily Data Provider  (for HMM nightly refit)
# ═══════════════════════════════════════════════════════════════════════


class DailyDataProvider:
    """Fetch daily OHLCV and VIX term-structure data for HMM features.

    Uses yfinance as the default free source.  If yfinance is unavailable,
    a CSV fallback path can be provided.
    """

    DEFAULT_EQUITY_SYMBOL = "^GSPC"  # S&P 500 proxy (SPY also acceptable)
    VIX_SYMBOL = "^VIX"
    VIX3M_SYMBOL = "^VIX3M"

    def __init__(self, csv_dir: str | None = None) -> None:
        self._csv_dir = csv_dir

    def fetch(
        self,
        lookback_days: int = 756,  # 3 years to ensure 504+ bars after dropna
        end_date: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Return (daily_ohlcv, vix_close, vix3m_close) aligned by date.

        ``daily_ohlcv`` has columns: Open, High, Low, Close, Volume.
        """
        try:
            return self._fetch_yfinance(lookback_days, end_date)
        except Exception:
            logger.exception("yfinance fetch failed — attempting CSV fallback")
            if self._csv_dir:
                return self._fetch_csv()
            raise

    # -- yfinance implementation ------------------------------------------

    @staticmethod
    def _fetch_yfinance(
        lookback_days: int, end_date: str | None
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        import yfinance as yf  # type: ignore[import-untyped]

        end = end_date or datetime.now().strftime("%Y-%m-%d")
        start_dt = pd.Timestamp(end) - pd.Timedelta(days=int(lookback_days * 1.5))
        start = start_dt.strftime("%Y-%m-%d")

        equity = yf.download(
            DailyDataProvider.DEFAULT_EQUITY_SYMBOL,
            start=start,
            end=end,
            progress=False,
        )
        vix = yf.download(DailyDataProvider.VIX_SYMBOL, start=start, end=end, progress=False)
        vix3m = yf.download(DailyDataProvider.VIX3M_SYMBOL, start=start, end=end, progress=False)

        # Flatten MultiIndex columns if yfinance returns them
        for df in (equity, vix, vix3m):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        daily_ohlcv = equity[["Open", "High", "Low", "Close", "Volume"]]
        vix_close = vix["Close"].rename("VIX")
        vix3m_close = vix3m["Close"].rename("VIX3M")

        logger.info(
            "Fetched %d equity bars, %d VIX bars, %d VIX3M bars",
            len(daily_ohlcv),
            len(vix_close),
            len(vix3m_close),
        )
        return daily_ohlcv, vix_close, vix3m_close

    def _fetch_csv(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load daily data from local CSV files as a fallback."""
        from pathlib import Path

        base = Path(self._csv_dir)  # type: ignore[arg-type]
        equity = pd.read_csv(base / "equity.csv", index_col=0, parse_dates=True)
        vix = pd.read_csv(base / "vix.csv", index_col=0, parse_dates=True)["Close"]
        vix3m = pd.read_csv(base / "vix3m.csv", index_col=0, parse_dates=True)["Close"]
        return equity, vix, vix3m


# ═══════════════════════════════════════════════════════════════════════
#  Intraday Bar Aggregator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Bar:
    """Completed OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class IntradayBarAggregator:
    """Aggregates a tick stream into fixed-interval OHLCV bars.

    On each completed bar, calls ``on_bar_callback(bar)`` so strategies
    and indicators can react.
    """

    def __init__(
        self,
        interval_minutes: int = 5,
        on_bar_callback: Callable[[Bar], None] | None = None,
    ) -> None:
        self.interval_minutes = interval_minutes
        self.on_bar = on_bar_callback

        self._current_open: float | None = None
        self._current_high: float = -np.inf
        self._current_low: float = np.inf
        self._current_volume: float = 0.0
        self._bar_start: datetime | None = None

        self.bars: list[Bar] = []

    def reset(self) -> None:
        self._current_open = None
        self._current_high = -np.inf
        self._current_low = np.inf
        self._current_volume = 0.0
        self._bar_start = None
        self.bars.clear()

    def on_tick(self, timestamp: datetime, price: float, size: float = 1.0) -> None:
        """Feed a single tick; automatically emits bars when the interval elapses."""
        # Determine the bar window this tick belongs to
        bar_start = timestamp.replace(
            minute=(timestamp.minute // self.interval_minutes) * self.interval_minutes,
            second=0,
            microsecond=0,
        )

        # If we've moved into a new bar window, close the previous bar first
        if self._bar_start is not None and bar_start > self._bar_start:
            self._emit_bar()

        # Start a new bar if needed
        if self._bar_start is None or bar_start > self._bar_start:
            self._bar_start = bar_start
            self._current_open = price
            self._current_high = price
            self._current_low = price
            self._current_volume = size
        else:
            self._current_high = max(self._current_high, price)
            self._current_low = min(self._current_low, price)
            self._current_volume += size

        # close always tracks latest tick
        self._last_price = price

    def flush(self) -> None:
        """Force-emit the current incomplete bar (e.g., at EOD)."""
        if self._bar_start is not None and self._current_open is not None:
            self._emit_bar()

    def _emit_bar(self) -> None:
        bar = Bar(
            timestamp=self._bar_start,  # type: ignore[arg-type]
            open=self._current_open,    # type: ignore[arg-type]
            high=self._current_high,
            low=self._current_low,
            close=self._last_price,
            volume=self._current_volume,
        )
        self.bars.append(bar)
        if self.on_bar:
            self.on_bar(bar)

        # Reset for next bar
        self._current_open = None
        self._current_high = -np.inf
        self._current_low = np.inf
        self._current_volume = 0.0
