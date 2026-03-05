"""
visualization/replay_dashboard.py — Replay Strategy Debug Dashboard.

Overlay layer for the existing replay chart in visualize_live.py.
Adds VWAP, σ-bands, ORB levels, MR signal markers, and regime background
shading without modifying the original replay code.

Usage (from existing replay flow):
    dashboard = ReplayDashboard(fig, ax_px, ax_bar)
    # During replay loop:
    dashboard.update_overlays(bar_times, bar_closes, vwap_history,
                              orb_levels, regime_history, signals)
    # At end:
    dashboard.finalize(...)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from config import (
    ORB_END,
    REPLAY_SHOW_ORB,
    REPLAY_SHOW_REGIME_LABEL,
    REPLAY_SHOW_SIGMA_BANDS,
    REPLAY_SHOW_SIGNAL_MARKERS,
    REPLAY_SHOW_VWAP,
    RTH_OPEN,
    VWAP_SD_ENTRY_MAX,
    VWAP_SD_ENTRY_MIN,
)
from data.indicators import VWAPState
from regime.regime_v1 import RegimeFeatures
from strategies.mr_signal_engine import MRSignal

logger = logging.getLogger(__name__)

# ── Colour / style constants ───────────────────────────────────────────
_VWAP_COLOR = "green"
_VWAP_STYLE = "--"
_VWAP_LW = 1.5

_BAND_2_5_COLOR = "orange"
_BAND_2_5_ALPHA = 0.3
_BAND_3_0_COLOR = "red"
_BAND_3_0_ALPHA = 0.2

_ORB_COLOR = "orange"
_ORB_STYLE = "--"
_ORB_LW = 1.2

_SIGNAL_APPROVED_BUY_COLOR = "#00e676"   # bright green
_SIGNAL_APPROVED_SELL_COLOR = "#ff1744"  # bright red
_SIGNAL_REJECTED_ALPHA = 0.3
_SIGNAL_APPROVED_SIZE = 100
_SIGNAL_REJECTED_SIZE = 60

_REGIME_COLORS: dict[str | None, str] = {
    "range": "green",
    "trend": "blue",
    "extreme": "red",
}
_REGIME_ALPHA = 0.08


class ReplayDashboard:
    """Overlay manager for the replay debug chart.

    Accepts the existing figure and axes created by ``visualize_live.py``
    and draws strategy overlays on top of them.
    """

    def __init__(self, fig: Figure, ax_px: Axes, ax_bar: Axes) -> None:
        self.fig = fig
        self.ax_px = ax_px
        self.ax_bar = ax_bar

        # Track drawn artists to avoid duplicate legend entries
        self._legend_labels_px: set[str] = set()
        self._legend_labels_bar: set[str] = set()

    # ── Public API ──────────────────────────────────────────────────────

    def update_overlays(
        self,
        bar_times: Sequence[datetime],
        bar_closes: Sequence[float],
        vwap_history: Sequence[VWAPState] | None = None,
        orb_levels: dict | None = None,
        regime_history: Sequence[RegimeFeatures] | None = None,
        signals: Sequence[MRSignal] | None = None,
    ) -> None:
        """Incrementally redraw all enabled overlays.

        Parameters
        ----------
        bar_times : timestamps aligned with bar_closes
        bar_closes : close prices per bar (for axis alignment)
        vwap_history : one VWAPState per bar
        orb_levels : {"high": float, "low": float,
                      "start": datetime, "end": datetime}
        regime_history : RegimeFeatures per bar
        signals : MRSignal list emitted so far
        """
        if not bar_times:
            return

        times = list(bar_times)

        # VWAP + sigma bands
        if vwap_history:
            vwap_vals = [s.vwap for s in vwap_history]
            self.plot_vwap(self.ax_bar, times, vwap_vals)

            upper_2_5 = [s.upper_2_5 for s in vwap_history]
            lower_2_5 = [s.lower_2_5 for s in vwap_history]
            upper_3_0 = [s.upper_3_0 for s in vwap_history]
            lower_3_0 = [s.lower_3_0 for s in vwap_history]
            self.plot_sigma_bands(
                self.ax_bar, times,
                upper_2_5, lower_2_5, upper_3_0, lower_3_0,
            )

        # ORB
        if orb_levels:
            self.plot_orb_levels(
                self.ax_bar,
                orb_levels.get("high"),
                orb_levels.get("low"),
                orb_levels.get("start"),
                orb_levels.get("end"),
            )

        # Regime background
        if regime_history:
            self.plot_regime_background(self.ax_bar, regime_history)

        # Signal markers
        if signals:
            self.plot_signals(self.ax_bar, signals)

    def finalize(
        self,
        bar_times: Sequence[datetime] | None = None,
        bar_closes: Sequence[float] | None = None,
        vwap_history: Sequence[VWAPState] | None = None,
        orb_levels: dict | None = None,
        regime_history: Sequence[RegimeFeatures] | None = None,
        signals: Sequence[MRSignal] | None = None,
        extra_title: str = "",
    ) -> None:
        """Apply all overlays and set summary titles.

        Safe for 0-bar, 1-bar, and multi-bar cases.
        """
        n_bars = len(bar_times) if bar_times else 0
        bar_times_list = list(bar_times) if bar_times else []
        bar_closes_list = list(bar_closes) if bar_closes else []

        # Draw overlays one final time
        self.update_overlays(
            bar_times_list, bar_closes_list,
            vwap_history, orb_levels, regime_history, signals,
        )

        # Build summary title
        parts: list[str] = []
        if extra_title:
            parts.append(extra_title)
        parts.append(f"bars={n_bars}")

        # Regime summary
        if regime_history and REPLAY_SHOW_REGIME_LABEL:
            regime_counts: dict[str, int] = {}
            for rf in regime_history:
                label = rf.regime or "unknown"
                regime_counts[label] = regime_counts.get(label, 0) + 1
            regime_str = " | ".join(
                f"{k}:{v}" for k, v in sorted(regime_counts.items())
            )
            parts.append(f"regimes=[{regime_str}]")

        # Signal summary
        if signals and REPLAY_SHOW_SIGNAL_MARKERS:
            approved = sum(1 for s in signals if s.approved)
            rejected = len(signals) - approved
            parts.append(f"signals={len(signals)} (ok={approved} rej={rejected})")

        title = "  ·  ".join(parts)
        self.ax_bar.set_title(title, fontsize=9, loc="left")

        # Collect legend on bar axis (avoid duplicates)
        handles, labels = self.ax_bar.get_legend_handles_labels()
        if handles:
            seen: set[str] = set()
            unique_h, unique_l = [], []
            for h, lbl in zip(handles, labels):
                if lbl not in seen:
                    seen.add(lbl)
                    unique_h.append(h)
                    unique_l.append(lbl)
            self.ax_bar.legend(
                unique_h, unique_l,
                fontsize=7, loc="upper left", framealpha=0.6,
            )

        self.fig.tight_layout()

    # ── Overlay methods ─────────────────────────────────────────────────

    def plot_vwap(
        self,
        ax: Axes,
        bar_times: Sequence[datetime],
        vwap_values: Sequence[float],
    ) -> None:
        """Dark-green dashed VWAP line on *ax*."""
        if not REPLAY_SHOW_VWAP:
            return
        if not bar_times or not vwap_values:
            return
        # Filter out zero VWAP entries (pre-first-bar state)
        times, vals = [], []
        for t, v in zip(bar_times, vwap_values):
            if v > 0:
                times.append(t)
                vals.append(v)
        if not times:
            return

        label = "VWAP" if "VWAP" not in self._legend_labels_bar else ""
        ax.plot(
            times, vals,
            color=_VWAP_COLOR, linestyle=_VWAP_STYLE, linewidth=_VWAP_LW,
            label=label, zorder=3,
        )
        self._legend_labels_bar.add("VWAP")

    def plot_sigma_bands(
        self,
        ax: Axes,
        bar_times: Sequence[datetime],
        upper_2_5: Sequence[float],
        lower_2_5: Sequence[float],
        upper_3_0: Sequence[float],
        lower_3_0: Sequence[float],
    ) -> None:
        """Draw 2.5σ and 3.0σ band lines with shaded fills."""
        if not REPLAY_SHOW_SIGMA_BANDS:
            return
        if not bar_times:
            return

        times = mdates.date2num(list(bar_times))

        # 2.5σ bands
        u25 = np.asarray(upper_2_5, dtype=float)
        l25 = np.asarray(lower_2_5, dtype=float)
        lbl_25 = f"{VWAP_SD_ENTRY_MIN}σ" if f"{VWAP_SD_ENTRY_MIN}σ" not in self._legend_labels_bar else ""
        ax.plot(times, u25, color=_BAND_2_5_COLOR, linewidth=0.8,
                alpha=_BAND_2_5_ALPHA + 0.2, label=lbl_25, zorder=2)
        ax.plot(times, l25, color=_BAND_2_5_COLOR, linewidth=0.8,
                alpha=_BAND_2_5_ALPHA + 0.2, zorder=2)
        self._legend_labels_bar.add(f"{VWAP_SD_ENTRY_MIN}σ")

        # 3.0σ bands
        u30 = np.asarray(upper_3_0, dtype=float)
        l30 = np.asarray(lower_3_0, dtype=float)
        lbl_30 = f"{VWAP_SD_ENTRY_MAX}σ" if f"{VWAP_SD_ENTRY_MAX}σ" not in self._legend_labels_bar else ""
        ax.plot(times, u30, color=_BAND_3_0_COLOR, linewidth=0.8,
                alpha=_BAND_3_0_ALPHA + 0.2, label=lbl_30, zorder=2)
        ax.plot(times, l30, color=_BAND_3_0_COLOR, linewidth=0.8,
                alpha=_BAND_3_0_ALPHA + 0.2, zorder=2)
        self._legend_labels_bar.add(f"{VWAP_SD_ENTRY_MAX}σ")

        # Shaded fills between bands
        ax.fill_between(times, u25, u30, color=_BAND_2_5_COLOR,
                         alpha=_BAND_2_5_ALPHA * 0.5, zorder=1)
        ax.fill_between(times, l30, l25, color=_BAND_2_5_COLOR,
                         alpha=_BAND_2_5_ALPHA * 0.5, zorder=1)
        ax.fill_between(times, u30, u30 + (u30 - u25) * 0.2,
                         color=_BAND_3_0_COLOR, alpha=_BAND_3_0_ALPHA * 0.5,
                         zorder=1)
        ax.fill_between(times, l30 - (u30 - u25) * 0.2, l30,
                         color=_BAND_3_0_COLOR, alpha=_BAND_3_0_ALPHA * 0.5,
                         zorder=1)

    def plot_orb_levels(
        self,
        ax: Axes,
        orb_high: float | None,
        orb_low: float | None,
        orb_start_time: datetime | None,
        orb_end_time: datetime | None,
    ) -> None:
        """Orange dashed horizontal lines for the Opening Range."""
        if not REPLAY_SHOW_ORB:
            return
        if orb_high is None or orb_low is None:
            return

        lbl_h = "ORB High" if "ORB High" not in self._legend_labels_bar else ""
        lbl_l = "ORB Low" if "ORB Low" not in self._legend_labels_bar else ""

        ax.axhline(
            orb_high, color=_ORB_COLOR, linestyle=_ORB_STYLE,
            linewidth=_ORB_LW, label=lbl_h, zorder=2,
        )
        ax.axhline(
            orb_low, color=_ORB_COLOR, linestyle=_ORB_STYLE,
            linewidth=_ORB_LW, label=lbl_l, zorder=2,
        )
        self._legend_labels_bar.add("ORB High")
        self._legend_labels_bar.add("ORB Low")

        # Light vertical span showing the ORB window
        if orb_start_time and orb_end_time:
            ax.axvspan(
                float(mdates.date2num(orb_start_time)),
                float(mdates.date2num(orb_end_time)),
                color=_ORB_COLOR, alpha=0.04, zorder=0,
            )

    def plot_signals(
        self, ax: Axes, signals: Sequence[MRSignal],
    ) -> None:
        """Triangle markers for MR signals on *ax*.

        Approved: bright colour, size=100.
        Rejected: same shape, alpha=0.3, size=60.
        """
        if not REPLAY_SHOW_SIGNAL_MARKERS:
            return
        if not signals:
            return

        for sig in signals:
            is_buy = sig.side == "BUY"
            marker = "^" if is_buy else "v"
            base_color = _SIGNAL_APPROVED_BUY_COLOR if is_buy else _SIGNAL_APPROVED_SELL_COLOR

            if sig.approved:
                alpha = 1.0
                size = _SIGNAL_APPROVED_SIZE
                label_key = f"{'BUY' if is_buy else 'SELL'} (ok)"
            else:
                alpha = _SIGNAL_REJECTED_ALPHA
                size = _SIGNAL_REJECTED_SIZE
                label_key = f"{'BUY' if is_buy else 'SELL'} (rej)"

            label = label_key if label_key not in self._legend_labels_bar else ""

            price = sig.entry_reference_price if sig.entry_reference_price else sig.vwap_at_signal
            ax.scatter(
                mdates.date2num([sig.timestamp]), [price],
                marker=marker, s=size, color=base_color,
                alpha=alpha, edgecolors="none", zorder=5,
                label=label,
            )
            self._legend_labels_bar.add(label_key)

    def plot_regime_background(
        self, ax: Axes, regime_history: Sequence[RegimeFeatures],
    ) -> None:
        """Light background shading by regime class.

        Contiguous spans of the same regime are merged into single
        ``axvspan`` calls.
        """
        if not REPLAY_SHOW_REGIME_LABEL:
            return
        if not regime_history:
            return

        # Build contiguous spans: (start_time, end_time, regime_label)
        spans: list[tuple[datetime, datetime, str | None]] = []
        current_regime = regime_history[0].regime
        span_start = regime_history[0].timestamp

        for i in range(1, len(regime_history)):
            rf = regime_history[i]
            if rf.regime != current_regime:
                spans.append((span_start, regime_history[i - 1].timestamp, current_regime))
                current_regime = rf.regime
                span_start = rf.timestamp
        # Close final span
        spans.append((span_start, regime_history[-1].timestamp, current_regime))

        for start, end, regime in spans:
            color = _REGIME_COLORS.get(regime, "gray")
            label_key = f"regime:{regime or 'unknown'}"
            label = label_key if label_key not in self._legend_labels_bar else ""
            ax.axvspan(
                float(mdates.date2num(start)), float(mdates.date2num(end)),
                color=color, alpha=_REGIME_ALPHA,
                zorder=0, label=label,
            )
            self._legend_labels_bar.add(label_key)

    def add_status_text(self, ax: Axes, text: str) -> None:
        """Small annotation in the upper-right corner of *ax*."""
        ax.text(
            0.98, 0.97, text,
            transform=ax.transAxes,
            fontsize=7, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            zorder=10,
        )
