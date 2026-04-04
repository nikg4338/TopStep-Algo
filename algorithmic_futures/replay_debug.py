"""
replay_debug.py — Integrated Replay Strategy Debugging Cockpit.

Wires together all v1 modules for visual + data analysis of MES replay sessions:
  • IntradayBarAggregator  → 5-minute bars from Databento tick data
  • VWAPCalculator         → anchored VWAP + σ-bands
  • ATRCalculator          → rolling 14-bar ATR
  • HybridThresholdRegimeClassifier → range / trend / extreme
  • MRSignalEngine         → candidate mean-reversion signals
  • RiskGovernor           → pre-trade gating (daily loss, profit cap, consistency)
  • ReplayDashboard        → chart overlays (VWAP, bands, ORB, signals, regime)
  • ReplaySessionReport    → export signals.csv, session_summary.json, features_snapshot.csv

Usage:
  python replay_debug.py \\
    --start 2025-02-18T14:30:00Z \\
    --end   2025-02-18T16:00:00Z \\
    --session-id run_001

  # Headless (no chart window, save image)
  python replay_debug.py \\
    --start 2025-02-18T14:30:00Z \\
    --end   2025-02-18T16:00:00Z \\
    --no-show --save-path artifacts/debug_chart.png

  # Limit ticks for quick smoke test
  python replay_debug.py \\
    --start 2025-02-18T14:30:00Z \\
    --end   2025-02-18T15:00:00Z \\
    --max-ticks 5000 --session-id smoke_test
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pytz
from dotenv import load_dotenv

import config
from validation.preset_utils import normalize_allocator_policy
from data.databento_provider import DatabentoReplayProvider
from data.indicators import ATRCalculator, VWAPCalculator, VWAPState
from data.market_data import Bar, IntradayBarAggregator
from regime.regime_v1 import HybridThresholdRegimeClassifier, RegimeFeatures
from reporting.replay_report import ReplaySessionReport
from risk.risk_governor import ConsistencyCapEngine, GovernorResult, RiskGovernor
from strategies.mr_signal_engine import MRSignal, MRSignalEngine
from validation.open_proxy_allocator import (
    OpenProxyConfig,
    OpenProxyDecision,
    OpenWindowState,
    decide as open_proxy_decide,
)
from visualization.replay_dashboard import ReplayDashboard


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Replay strategy debugging cockpit — VWAP MR + regime + risk overlays"
    )
    p.add_argument("--start", required=True, help="Replay start (ISO-8601)")
    p.add_argument("--end", required=True, help="Replay end (ISO-8601)")
    p.add_argument("--symbol", default="MES.c.0", help="Databento symbol")
    p.add_argument("--max-ticks", type=int, default=None, help="Cap ticks for quick tests")
    p.add_argument("--update-every", type=int, default=200, help="Chart refresh interval (ticks)")
    p.add_argument("--pause", type=float, default=0.001, help="Matplotlib pause per update")
    p.add_argument("--no-show", action="store_true", help="Headless mode (no window)")
    p.add_argument("--no-dashboard", action="store_true", help="Skip dashboard rendering for batch runs")
    p.add_argument("--save-path", default="", help="Save final chart image to path")
    p.add_argument(
        "--session-id",
        default="",
        help="Session ID for report export (default: auto timestamp)",
    )
    p.add_argument("--no-report", action="store_true", help="Skip report export")
    p.add_argument(
        "--mr-reclaim-mode",
        choices=("on", "off", "soft", "touch"),
        default="on",
        help="MR candidate mode: 'on' requires reclaim, 'off' threshold-cross, 'soft' threshold-cross + light momentum confirm",
    )
    p.add_argument(
        "--mr-sigma-entry",
        type=float,
        default=config.MR_SIGMA_ENTRY,
        help="MR entry threshold in z-score units",
    )
    p.add_argument(
        "--mr-soft-range-impulse-k",
        type=float,
        default=config.MR_SOFT_RECLAIM_RANGE_IMPULSE_K,
        help="Soft-v3 range impulse threshold k_range in ATR units",
    )
    p.add_argument(
        "--mr-soft-impulse-k",
        type=float,
        default=None,
        help="Deprecated alias for --mr-soft-range-impulse-k",
    )
    p.add_argument(
        "--mr-dedupe-enabled",
        choices=("on", "off"),
        default=("on" if config.MR_EXCURSION_DEDUPE_ENABLED else "off"),
        help="Enable/disable MR excursion dedupe gate",
    )
    p.add_argument(
        "--mr-attempt-cap-enabled",
        choices=("on", "off"),
        default="on",
        help="Enable/disable MR per-side attempt cap gate",
    )
    p.add_argument(
        "--mr-cooldown-bars",
        type=int,
        default=config.MR_COOLDOWN_BARS,
        help="MR cooldown bars between approved entries",
    )
    p.add_argument(
        "--mr-first-outside-enabled",
        choices=("on", "off"),
        default=("on" if config.MR_FIRST_OUTSIDE_ENABLED else "off"),
        help="Enable first-eligible outside candidate salvage rule",
    )
    p.add_argument(
        "--mr-touch-latch-reset-buffer",
        type=float,
        default=config.MR_TOUCH_LATCH_RESET_BUFFER,
        help="Touch mode latch reset buffer in z units",
    )
    p.add_argument(
        "--mr-dedupe-window-bars",
        type=int,
        default=config.MR_DEDUPE_WINDOW_BARS,
        help="Smarter dedupe window in bars",
    )
    p.add_argument(
        "--mr-dedupe-min-delta-z",
        type=float,
        default=config.MR_DEDUPE_MIN_DELTA_Z,
        help="Smarter dedupe minimum |z| progression required",
    )
    p.add_argument(
        "--mr-regime-enabled",
        choices=("on", "off"),
        default="on",
        help="Enable/disable MR range-regime gate",
    )
    p.add_argument(
        "--engine-mode",
        choices=("mr", "orb", "both"),
        default="both",
        help="Strategy engine mode: mr-only, orb-only, or both",
    )
    p.add_argument(
        "--allocator-policy",
        choices=("none", "v1", "v2", "open_proxy_v1"),
        default="none",
        help="Day-level allocator policy (applies only when --engine-mode both)",
    )
    # ── open_proxy_v1 allocator flags (calibrated to ~53% ORB routing) ──
    p.add_argument(
        "--alloc-openproxy-or-width-atr",
        type=float,
        default=2.2,
        help="open_proxy_v1: OR width / ATR threshold for trend signal (calibrated: 2.2)",
    )
    p.add_argument(
        "--alloc-openproxy-impulse-atr",
        type=float,
        default=0.9,
        help="open_proxy_v1: |first 3-bar net move| / ATR threshold",
    )
    p.add_argument(
        "--alloc-openproxy-persist-bars",
        type=int,
        default=1,
        help="open_proxy_v1: consecutive closes beyond OR for persistence",
    )
    p.add_argument(
        "--alloc-openproxy-require-break",
        choices=("on", "off"),
        default="off",
        help="open_proxy_v1: require breakout persistence (not just width/impulse)",
    )
    p.add_argument(
        "--alloc-openproxy-enable-orb-selectivity-refinement",
        choices=("on", "off"),
        default=("on" if config.ALLOC_OPENPROXY_SELECTIVITY_ENABLED else "off"),
        help="open_proxy_v1: enable research-only ORB selectivity refinement",
    )
    p.add_argument(
        "--alloc-openproxy-low-atr-threshold",
        type=float,
        default=config.ALLOC_OPENPROXY_LOW_ATR_THRESHOLD,
        help="open_proxy_v1: low-ATR threshold below which ORB requires stronger persistence",
    )
    p.add_argument(
        "--alloc-openproxy-min-persistence-in-low-atr",
        type=int,
        default=config.ALLOC_OPENPROXY_MIN_PERSISTENCE_IN_LOW_ATR,
        help="open_proxy_v1: minimum persistence required in low-ATR contexts",
    )
    p.add_argument(
        "--alloc-openproxy-high-impulse-threshold",
        type=float,
        default=config.ALLOC_OPENPROXY_HIGH_IMPULSE_THRESHOLD,
        help="open_proxy_v1: impulse threshold above which weak persistence blocks ORB routing",
    )
    p.add_argument(
        "--alloc-openproxy-min-persistence-when-high-impulse",
        type=int,
        default=config.ALLOC_OPENPROXY_MIN_PERSISTENCE_WHEN_HIGH_IMPULSE,
        help="open_proxy_v1: minimum persistence required when impulse exceeds the high-impulse threshold",
    )
    p.add_argument(
        "--alloc-openproxy-medium-impulse-weak-persistence-filter-enabled",
        choices=("on", "off"),
        default=("on" if config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_WEAK_PERSISTENCE_FILTER_ENABLED else "off"),
        help="open_proxy_v1: enable research-only filter for medium-impulse weak-persistence ORB conditions",
    )
    p.add_argument(
        "--allocator-v1-adx-threshold",
        type=float,
        default=25.0,
        help="Allocator v1 trend threshold: ADX >= threshold -> ORB-only day",
    )
    p.add_argument(
        "--allocator-v2-trend-open-threshold",
        type=float,
        default=25.0,
        help="Allocator v2 trend condition at open window",
    )
    p.add_argument(
        "--allocator-v2-rising-threshold",
        type=float,
        default=20.0,
        help="Allocator v2 rising-ADX floor",
    )
    p.add_argument(
        "--allocator-v2-rising-bars",
        type=int,
        default=3,
        help="Allocator v2 rising-ADX consecutive bars requirement",
    )
    p.add_argument(
        "--allocator-v2-range-threshold",
        type=float,
        default=18.0,
        help="Allocator v2 range condition ceiling",
    )
    p.add_argument(
        "--allocator-v2-range-bars",
        type=int,
        default=3,
        help="Allocator v2 range-ADX consecutive bars requirement",
    )
    p.add_argument(
        "--orb-enabled",
        choices=("on", "off"),
        default="off",
        help="Enable ORB Engine 2 scaffold signals in replay",
    )
    p.add_argument(
        "--orb-trigger-mode",
        choices=("break", "pullback", "either", "pullback_v3"),
        default=config.ORB_TRIGGER_MODE,
        help="ORB trigger mode",
    )
    p.add_argument(
        "--orb-pullback-confirm-bars",
        type=int,
        default=config.ORB_PULLBACK_CONFIRM_BARS,
        help="Max bars to wait for ORB pullback confirmation after break",
    )
    p.add_argument(
        "--orb-pullback-max-bars",
        type=int,
        default=config.ORB_PULLBACK_V3_MAX_BARS,
        help="pullback_v3: max bars after breakout to wait for pullback",
    )
    p.add_argument(
        "--orb-pullback-tolerance-pts",
        type=float,
        default=config.ORB_PULLBACK_V3_TOLERANCE_PTS,
        help="pullback_v3: points from OR level for pullback detection",
    )
    p.add_argument(
        "--orb-pullback-entry-mode",
        choices=("touch_only", "touch_recovery"),
        default=config.ORB_PULLBACK_V3_ENTRY_MODE,
        help="pullback_v3: entry sub-mode",
    )
    return p


# ═══════════════════════════════════════════════════════════════════════
#  Config snapshot helper
# ═══════════════════════════════════════════════════════════════════════


def _config_snapshot(runtime_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Capture relevant config values for provenance."""
    snapshot = {
        k: getattr(config, k)
        for k in sorted(dir(config))
        if k.isupper() and not k.startswith("_")
    }
    if runtime_overrides:
        snapshot["_runtime_overrides"] = dict(runtime_overrides)
    return snapshot


# ═══════════════════════════════════════════════════════════════════════
#  ORB tracker (simple high/low during the ORB window)
# ═══════════════════════════════════════════════════════════════════════


class _ORBTracker:
    """Track ORB high/low from bars within the ORB window.

    Bar timestamps may be UTC-naive; we convert to ET for comparison
    against config.RTH_OPEN / config.ORB_END.
    """

    def __init__(self) -> None:
        self.high: float | None = None
        self.low: float | None = None
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self._finalized = False
        self._et = pytz.timezone(config.TIMEZONE)
        self._utc = pytz.utc

    def _to_et(self, ts: datetime) -> datetime:
        """Convert a (possibly naive-UTC) timestamp to ET."""
        if ts.tzinfo is None:
            ts = self._utc.localize(ts)
        return ts.astimezone(self._et)

    def on_bar(self, bar: Bar) -> None:
        if self._finalized:
            return
        bar_et = self._to_et(bar.timestamp)
        bar_time_str = bar_et.strftime("%H:%M")
        if bar_time_str < config.RTH_OPEN:
            return
        if bar_time_str >= config.ORB_END:
            self._finalized = True
            return
        # Within ORB window
        if self.start_time is None:
            self.start_time = bar.timestamp
        self.end_time = bar.timestamp
        if self.high is None or bar.high > self.high:
            self.high = bar.high
        if self.low is None or bar.low < self.low:
            self.low = bar.low

    @property
    def levels(self) -> dict | None:
        if self.high is None or self.low is None:
            return None
        return {
            "high": self.high,
            "low": self.low,
            "start": self.start_time,
            "end": self.end_time,
        }


# ═══════════════════════════════════════════════════════════════════════
#  Signal diagnostic helper
# ═══════════════════════════════════════════════════════════════════════


def _print_signal_diagnostic(
    features: list,
    vwap_history: list,
    bar_closes: list[float],
    bars_closed: int,
    warmup_bars: int,
    orb_tracker: "_ORBTracker",
    mr_sigma_entry: float,
    mr_cooldown_bars: int,
) -> None:
    """Print a per-session diagnostic explaining *why* signals did/didn't fire."""
    from config import TIMEZONE

    et = pytz.timezone(TIMEZONE)

    print(f"\n{'─'*70}")
    print("  SIGNAL DIAGNOSTIC — why no signals?")
    print(f"{'─'*70}")

    # Bars eligible after warmup
    armed_bars = max(0, bars_closed - warmup_bars)
    print(f"  Total bars closed       : {bars_closed}")
    print(f"  Warmup bars required    : {warmup_bars}")
    print(f"  Armed bars (post warmup): {armed_bars}")

    # How many bars had σ available
    bars_with_sigma = sum(
        1 for vs in vwap_history
        if vs.std_dev > 0 and vs.bar_count >= 3
    )
    print(f"  Bars with σ > 0          : {bars_with_sigma}")

    # Z-scores: compute peak |z| and band touches
    z_scores: list[float] = []
    for i, vs in enumerate(vwap_history):
        if vs.std_dev > 0 and i < len(bar_closes):
            z = (bar_closes[i] - vs.vwap) / vs.std_dev
            z_scores.append(z)
        else:
            z_scores.append(0.0)

    if z_scores:
        abs_z = [abs(z) for z in z_scores]
        peak_z = max(abs_z)
        touch_2_0 = sum(1 for z in abs_z if z >= 2.0)
        touch_2_5 = sum(1 for z in abs_z if z >= 2.5)
        touch_3_0 = sum(1 for z in abs_z if z >= 3.0)
        print(f"  Peak |z-score|           : {peak_z:.2f}")
        print(f"  Band touches ≥ 2.0σ      : {touch_2_0}")
        print(f"  Band touches ≥ 2.5σ      : {touch_2_5}")
        print(f"  Band touches ≥ 3.0σ      : {touch_3_0}")
        print(f"  Current σ entry threshold: {mr_sigma_entry}")
    else:
        print("  Peak |z-score|           : N/A (no bars with σ)")

    # Regime filter: how many bars were blocked by regime != "range"
    post_warmup_features = features[warmup_bars:] if len(features) > warmup_bars else []
    regime_blocked = sum(1 for f in post_warmup_features if f.regime != "range")
    regime_range = sum(1 for f in post_warmup_features if f.regime == "range")
    regime_dist = {}
    for f in post_warmup_features:
        r = f.regime or "warmup"
        regime_dist[r] = regime_dist.get(r, 0) + 1
    print(f"  Post-warmup regime dist  : {regime_dist}")
    print(f"  Bars blocked by regime   : {regime_blocked} (non-range)")
    print(f"  Bars eligible (range)    : {regime_range}")

    # ADX values
    adx_values = [f.adx for f in features if f.adx > 0]
    if adx_values:
        print(f"  ADX range                : {min(adx_values):.1f} – {max(adx_values):.1f}")
        print(f"  ADX (final)              : {features[-1].adx:.1f}")
    else:
        print(f"  ADX                      : 0.0 (never fired — need {warmup_bars * 2}+ bars for period-{warmup_bars} ADX)")

    # Cooldown info
    print(f"  Cooldown bars            : {mr_cooldown_bars}")

    # ORB diagnostic
    orb = orb_tracker.levels
    if orb:
        print(f"  ORB levels               : H={orb['high']:.2f} L={orb['low']:.2f}")
        print(f"  ORB window               : {orb['start']} → {orb['end']}")
    else:
        print("  ORB levels               : None (no bars landed in ORB window)")
        if features:
            first_ts = features[0].timestamp
            if first_ts.tzinfo is None:
                first_et = pytz.utc.localize(first_ts).astimezone(et)
            else:
                first_et = first_ts.astimezone(et)
            print(f"  First bar (ET)           : {first_et.strftime('%H:%M')}")
            print(f"  ORB window config        : {config.RTH_OPEN} → {config.ORB_END}")

    print(f"{'─'*70}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Main replay runner
# ═══════════════════════════════════════════════════════════════════════


def run_debug_replay(args: argparse.Namespace) -> int:
    dashboard_enabled = not getattr(args, "no_dashboard", False)

    # ── Matplotlib setup ────────────────────────────────────────────────
    plt: Any = None
    mdates: Any = None
    if dashboard_enabled:
        if args.no_show:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.dates as mdates  # type: ignore[assignment]
        import matplotlib.pyplot as plt  # type: ignore[assignment]

    # ── Session ID ──────────────────────────────────────────────────────
    session_id = args.session_id or datetime.now().strftime("debug_%Y%m%d_%H%M%S")

    # ── Data fetch ──────────────────────────────────────────────────────
    print(f"[replay_debug] session_id={session_id}")
    print(f"[replay_debug] Fetching trades: {args.start} → {args.end} ({args.symbol})")
    provider = DatabentoReplayProvider()
    trades = provider.fetch_trades(start=args.start, end=args.end, symbol=args.symbol)
    if args.max_ticks is not None:
        trades = trades.head(args.max_ticks)
    print(f"[replay_debug] {len(trades)} ticks loaded")

    # ── Module instantiation ────────────────────────────────────────────
    vwap_calc = VWAPCalculator()
    atr_calc = ATRCalculator(period=14)
    regime_clf = HybridThresholdRegimeClassifier()
    reclaim_mode_raw = getattr(args, "mr_reclaim_mode", "on")
    reclaim_mode = reclaim_mode_raw if reclaim_mode_raw in ("on", "off", "soft", "touch") else "on"
    mr_sigma_entry = max(0.1, float(getattr(args, "mr_sigma_entry", config.MR_SIGMA_ENTRY)))
    soft_range_k = getattr(args, "mr_soft_range_impulse_k", config.MR_SOFT_RECLAIM_RANGE_IMPULSE_K)
    soft_alias = getattr(args, "mr_soft_impulse_k", None)
    if soft_alias is not None:
        soft_range_k = soft_alias
    soft_range_k = max(0.0, float(soft_range_k))
    dedupe_enabled = getattr(args, "mr_dedupe_enabled", "off") == "on"
    attempt_cap_enabled = getattr(args, "mr_attempt_cap_enabled", "on") == "on"
    cooldown_bars = max(0, int(getattr(args, "mr_cooldown_bars", config.MR_COOLDOWN_BARS)))
    first_outside_enabled = getattr(args, "mr_first_outside_enabled", "off") == "on"
    touch_latch_reset_buffer = max(0.0, float(getattr(args, "mr_touch_latch_reset_buffer", config.MR_TOUCH_LATCH_RESET_BUFFER)))
    dedupe_window_bars = max(0, int(getattr(args, "mr_dedupe_window_bars", config.MR_DEDUPE_WINDOW_BARS)))
    dedupe_min_delta_z = max(0.0, float(getattr(args, "mr_dedupe_min_delta_z", config.MR_DEDUPE_MIN_DELTA_Z)))
    regime_enabled = getattr(args, "mr_regime_enabled", "on") == "on"
    signal_engine = MRSignalEngine(
        reclaim_mode=reclaim_mode,
        sigma_entry=mr_sigma_entry,
        soft_reclaim_range_impulse_k=soft_range_k,
        cooldown_bars=cooldown_bars,
        max_attempts_per_side=config.MR_MAX_ATTEMPTS_PER_SIDE,
        excursion_dedupe_enabled=dedupe_enabled,
        attempt_cap_enabled=attempt_cap_enabled,
        first_outside_enabled=first_outside_enabled,
        touch_latch_reset_buffer=touch_latch_reset_buffer,
        dedupe_window_bars=dedupe_window_bars,
        dedupe_min_delta_z=dedupe_min_delta_z,
        regime_enabled=regime_enabled,
    )
    consistency_engine = ConsistencyCapEngine(mode=config.ACCOUNT_MODE)
    risk_gov = RiskGovernor(consistency_engine=consistency_engine)
    orb_tracker = _ORBTracker()

    # ── Histories for dashboard & report ────────────────────────────────
    bar_times: list[datetime] = []
    bar_closes: list[float] = []
    bar_is_partial: list[bool] = []
    vwap_history: list[VWAPState] = []
    all_signals: list[MRSignal] = []
    engine_mode = getattr(args, "engine_mode", "both")
    if engine_mode not in {"mr", "orb", "both"}:
        engine_mode = "both"
    base_orb_enabled = getattr(args, "orb_enabled", "off") == "on"
    mr_runtime_enabled = engine_mode in {"mr", "both"}
    if engine_mode == "orb":
        orb_enabled = True
    elif engine_mode == "mr":
        orb_enabled = False
    else:
        orb_enabled = base_orb_enabled

    try:
        allocator_policy = normalize_allocator_policy(getattr(args, "allocator_policy", "none"))
    except ValueError:
        allocator_policy = "none"
    allocator_v1_adx_threshold = float(getattr(args, "allocator_v1_adx_threshold", 25.0))
    allocator_v2_trend_open_threshold = float(getattr(args, "allocator_v2_trend_open_threshold", 25.0))
    allocator_v2_rising_threshold = float(getattr(args, "allocator_v2_rising_threshold", 20.0))
    allocator_v2_rising_bars = max(1, int(getattr(args, "allocator_v2_rising_bars", 3)))
    allocator_v2_range_threshold = float(getattr(args, "allocator_v2_range_threshold", 18.0))
    allocator_v2_range_bars = max(1, int(getattr(args, "allocator_v2_range_bars", 3)))
    allocator_active = (engine_mode == "both" and allocator_policy != "none")
    allocator_open_window_adx: list[float] = []
    allocator_decision: str | None = None
    allocator_reason = ""

    # ── open_proxy_v1 state ──────────────────────────────────────────────
    open_proxy_cfg = OpenProxyConfig(
        or_width_atr_threshold=float(getattr(args, "alloc_openproxy_or_width_atr", 0.8)),
        impulse_atr_threshold=float(getattr(args, "alloc_openproxy_impulse_atr", 0.9)),
        persist_bars=max(0, int(getattr(args, "alloc_openproxy_persist_bars", 1))),
        require_break=(getattr(args, "alloc_openproxy_require_break", "off") == "on"),
        enable_orb_selectivity_refinement=(getattr(args, "alloc_openproxy_enable_orb_selectivity_refinement", "off") == "on"),
        orb_selectivity_low_atr_threshold=float(getattr(args, "alloc_openproxy_low_atr_threshold", config.ALLOC_OPENPROXY_LOW_ATR_THRESHOLD)),
        orb_selectivity_min_persistence_in_low_atr=max(0, int(getattr(args, "alloc_openproxy_min_persistence_in_low_atr", config.ALLOC_OPENPROXY_MIN_PERSISTENCE_IN_LOW_ATR))),
        orb_selectivity_high_impulse_threshold=float(getattr(args, "alloc_openproxy_high_impulse_threshold", config.ALLOC_OPENPROXY_HIGH_IMPULSE_THRESHOLD)),
        orb_selectivity_min_persistence_when_high_impulse=max(0, int(getattr(args, "alloc_openproxy_min_persistence_when_high_impulse", config.ALLOC_OPENPROXY_MIN_PERSISTENCE_WHEN_HIGH_IMPULSE))),
        enable_medium_impulse_weak_persistence_filter=(getattr(args, "alloc_openproxy_medium_impulse_weak_persistence_filter_enabled", "off") == "on"),
        enable_medium_impulse_decay_filter=(getattr(args, "alloc_openproxy_medium_impulse_decay_filter_enabled", "off") == "on"),
        medium_impulse_min_atr=float(getattr(args, "alloc_openproxy_medium_impulse_min_atr", config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MIN_ATR)),
        medium_impulse_max_atr=float(getattr(args, "alloc_openproxy_medium_impulse_max_atr", config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MAX_ATR)),
        medium_impulse_min=float(getattr(args, "alloc_openproxy_medium_impulse_min", config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MIN)),
        medium_impulse_max=float(getattr(args, "alloc_openproxy_medium_impulse_max", config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MAX)),
        medium_impulse_min_persistence=max(0, int(getattr(args, "alloc_openproxy_medium_impulse_min_persistence", config.ALLOC_OPENPROXY_MEDIUM_IMPULSE_MIN_PERSISTENCE))),
    )
    open_proxy_state = OpenWindowState()
    open_proxy_result: OpenProxyDecision | None = None

    orb_trigger_mode = getattr(args, "orb_trigger_mode", config.ORB_TRIGGER_MODE)
    if orb_trigger_mode not in {"break", "pullback", "either", "pullback_v3"}:
        orb_trigger_mode = "either"
    orb_pullback_confirm_bars = max(1, int(getattr(args, "orb_pullback_confirm_bars", config.ORB_PULLBACK_CONFIRM_BARS)))
    orb_pb3_max_bars = max(1, int(getattr(args, "orb_pullback_max_bars", config.ORB_PULLBACK_V3_MAX_BARS)))
    orb_pb3_tolerance_pts = max(0.0, float(getattr(args, "orb_pullback_tolerance_pts", config.ORB_PULLBACK_V3_TOLERANCE_PTS)))
    orb_pb3_entry_mode = getattr(args, "orb_pullback_entry_mode", config.ORB_PULLBACK_V3_ENTRY_MODE)
    if orb_pb3_entry_mode not in {"touch_only", "touch_recovery"}:
        orb_pb3_entry_mode = "touch_only"
    orb_pending_side: str | None = None
    orb_pending_age: int = 0
    orb_signals_emitted: int = 0
    orb_break_count: int = 0
    orb_confirmation_count: int = 0
    orb_or_constructed: int = 0

    # ── pullback_v3 state ───────────────────────────────────────────────
    orb_pb3_active: bool = False       # True once breakout detected
    orb_pb3_direction: str = ""        # "BUY" or "SELL"
    orb_pb3_breakout_bar: int = 0      # bar index of breakout
    orb_pb3_breakout_level: float = 0.0  # orb_high or orb_low
    orb_pb3_or_high: float = 0.0
    orb_pb3_or_low: float = 0.0
    orb_pb3_pullback_detected: bool = False
    orb_pb3_recovery_pending: bool = False  # for touch_recovery mode
    orb_pb3_diagnostics: list[dict] = []  # per-breakout diagnostics

    bars_closed = 0
    bars_partial_flushed = 0
    unique_buckets: set[str] = set()

    # Simulated session P&L tracking (no real execution)
    sim_daily_pnl: float = 0.0
    sim_trade_count: int = 0

    # ── Bar callback ────────────────────────────────────────────────────
    def on_bar(bar: Bar, *, is_flush: bool = False) -> None:
        nonlocal bars_closed, bars_partial_flushed, orb_pending_side, orb_signals_emitted, orb_pending_age, orb_break_count, orb_confirmation_count, orb_or_constructed
        nonlocal allocator_decision, allocator_reason, open_proxy_result
        nonlocal orb_pb3_active, orb_pb3_direction, orb_pb3_breakout_bar, orb_pb3_breakout_level, orb_pb3_pullback_detected, orb_pb3_recovery_pending, orb_pb3_or_high, orb_pb3_or_low

        if is_flush:
            bars_partial_flushed += 1
        else:
            bars_closed += 1

        bar_times.append(bar.timestamp)
        bar_closes.append(bar.close)
        bar_is_partial.append(is_flush)

        label = "PARTIAL-FLUSH" if is_flush else "CLOSED"
        n = bars_closed + bars_partial_flushed
        print(
            f"  [BAR #{n} {label}] "
            f"ts={bar.timestamp}  O={bar.open:.2f} H={bar.high:.2f} "
            f"L={bar.low:.2f} C={bar.close:.2f} V={bar.volume:.0f}"
        )

        # 1) VWAP
        vs = vwap_calc.update(bar.high, bar.low, bar.close, bar.volume)
        # Snapshot the state (frozen copy)
        vwap_snap = VWAPState(
            cum_volume=vs.cum_volume,
            cum_vwap_num=vs.cum_vwap_num,
            cum_vwap_sq_num=vs.cum_vwap_sq_num,
            vwap=vs.vwap,
            std_dev=vs.std_dev,
            upper_2_5=vs.upper_2_5,
            upper_3_0=vs.upper_3_0,
            lower_2_5=vs.lower_2_5,
            lower_3_0=vs.lower_3_0,
            bar_count=vs.bar_count,
        )
        vwap_history.append(vwap_snap)

        # 2) ATR
        atr = atr_calc.update(bar.high, bar.low, bar.close)

        # 3) Regime
        regime = regime_clf.update(bar)
        current_adx = float(regime_clf._adx_calc.adx)

        _ts = bar.timestamp
        if _ts.tzinfo is None:
            _ts = pytz.utc.localize(_ts)
        bar_time_str = _ts.astimezone(pytz.timezone(config.TIMEZONE)).strftime("%H:%M")

        if config.RTH_OPEN <= bar_time_str < config.ORB_END and current_adx > 0:
            allocator_open_window_adx.append(current_adx)

        # ── open_proxy_v1: accumulate bars for open-window decision ──────
        if allocator_policy == "open_proxy_v1" and allocator_decision is None:
            bar_tuple = (bar.open, bar.high, bar.low, bar.close)
            if config.RTH_OPEN <= bar_time_str < config.ORB_END:
                # OR-building bars (09:30–09:45)
                open_proxy_state.bars.append(bar_tuple)
                if open_proxy_state.first_bar_open is None:
                    open_proxy_state.first_bar_open = bar.open
                open_proxy_state.or_high = max(open_proxy_state.or_high, bar.high)
                open_proxy_state.or_low = min(open_proxy_state.or_low, bar.low)
            elif bar_time_str >= config.ORB_END:
                # Post-OR bars (for persistence check)
                open_proxy_state.post_or_bars.append(bar_tuple)

        if allocator_active and allocator_decision is None and bar_time_str >= config.ORB_END:
            decision = "mr"
            reason = "ALLOCATOR_DEFAULT_MR"
            if allocator_policy == "v1":
                open_adx = allocator_open_window_adx[-1] if allocator_open_window_adx else 0.0
                if open_adx >= allocator_v1_adx_threshold:
                    decision = "orb"
                    reason = f"V1_TREND_OPEN_ADX_{open_adx:.2f}"
                else:
                    reason = f"V1_RANGE_OPEN_ADX_{open_adx:.2f}"
            elif allocator_policy == "v2":
                adx_series = allocator_open_window_adx[:]
                trend_open = any(v >= allocator_v2_trend_open_threshold for v in adx_series)
                recent_for_rising = adx_series[-allocator_v2_rising_bars:]
                rising_ok = (
                    len(recent_for_rising) >= allocator_v2_rising_bars
                    and all(v > allocator_v2_rising_threshold for v in recent_for_rising)
                    and all(recent_for_rising[i] < recent_for_rising[i + 1] for i in range(len(recent_for_rising) - 1))
                )
                recent_for_range = adx_series[-allocator_v2_range_bars:]
                range_ok = (
                    len(recent_for_range) >= allocator_v2_range_bars
                    and all(v <= allocator_v2_range_threshold for v in recent_for_range)
                )
                if trend_open or rising_ok:
                    decision = "orb"
                    reason = "V2_TREND"
                elif range_ok:
                    decision = "mr"
                    reason = "V2_RANGE"
                else:
                    decision = "mr"
                    reason = "V2_UNCLEAR_DEFAULT_MR"
            elif allocator_policy == "open_proxy_v1":
                # Open proxy: wait for at least persist_bars post-OR bars
                needed = max(1, open_proxy_cfg.persist_bars)
                if len(open_proxy_state.post_or_bars) >= needed:
                    open_proxy_state.atr_at_decision = atr if atr > 0 else 1.0
                    open_proxy_result = open_proxy_decide(open_proxy_state, open_proxy_cfg)
                    decision = open_proxy_result.decision
                    reason = open_proxy_result.reason
                else:
                    return  # wait for more post-OR bars
            allocator_decision = decision
            allocator_reason = reason
            print(f"  [ALLOCATOR] policy={allocator_policy} decision={allocator_decision} reason={allocator_reason}")
            if open_proxy_result is not None:
                print(
                    f"    [OPEN_PROXY] or_width={open_proxy_result.opening_range_width_pts:.2f}pts "
                    f"({open_proxy_result.opening_range_width_atr:.2f}×ATR)  "
                    f"impulse={open_proxy_result.first_3bar_directional_impulse:.2f}×ATR  "
                    f"persist={open_proxy_result.persist_bars_observed}  "
                    f"triggers: width={open_proxy_result.trigger_width} "
                    f"impulse={open_proxy_result.trigger_impulse} "
                    f"persist={open_proxy_result.trigger_persist}"
                )
                if open_proxy_result.selectivity_orb_blocked:
                    print(
                        f"    [OPEN_PROXY_SELECTIVITY] blocked=True explanation={open_proxy_result.selectivity_block_reason} "
                        f"pre_decision={open_proxy_result.pre_selectivity_decision}"
                    )
                if open_proxy_result.selectivity_v3_orb_blocked:
                    print(
                        f"    [OPEN_PROXY_SELECTIVITY_V3] blocked=True explanation={open_proxy_result.selectivity_v3_block_reason} "
                        f"pre_decision={open_proxy_result.pre_v3_selectivity_decision} "
                        f"post_decision={open_proxy_result.post_v3_selectivity_decision}"
                    )

        # 4) ORB
        orb_tracker.on_bar(bar)
        levels = orb_tracker.levels
        if levels is not None:
            orb_or_constructed = 1

        # 5) Signal engine (skip partial-flushed bars)
        if is_flush:
            return

        if allocator_active and allocator_decision is None:
            return

        active_engine = allocator_decision if allocator_active else engine_mode
        mr_generation_enabled = mr_runtime_enabled and active_engine in {"mr", "both"}
        orb_generation_enabled = orb_enabled and active_engine in {"orb", "both"}

        sig = None
        if mr_generation_enabled:
            sig = signal_engine.on_bar(bar, regime, vs, atr, adx=current_adx)
        if sig is not None:
            # 6) Risk governor gate — convert UTC-naive → ET for time checks
            gov_result: GovernorResult = risk_gov.evaluate(
                daily_pnl=sim_daily_pnl,
                daily_trade_count=sim_trade_count,
                current_time_str_HHMM=bar_time_str,
                total_realized_pnl=0.0,  # no real trades yet
                best_day_pnl=0.0,
            )
            if not gov_result.approved and sig.approved:
                # Governor overrides signal engine approval
                sig = MRSignal(
                    timestamp=sig.timestamp,
                    side=sig.side,
                    signal_type=sig.signal_type,
                    regime_at_signal=sig.regime_at_signal,
                    entry_reference_price=sig.entry_reference_price,
                    stop_reference=sig.stop_reference,
                    target_reference=sig.target_reference,
                    band_level_hit=sig.band_level_hit,
                    vwap_at_signal=sig.vwap_at_signal,
                    sigma_at_signal=sig.sigma_at_signal,
                    approved=False,
                    rejection_reason=f"GOVERNOR:{','.join(gov_result.reasons)}",
                    bar_index=sig.bar_index,
                )
                # Update engine's rejection counters for governor overrides
                signal_engine._rejection_counters["signals_approved"] -= 1
                gov_reasons = ",".join(gov_result.reasons)
                if "DAILY_LOSS" in gov_reasons:
                    signal_engine._rejection_counters["rejected_by_daily_loss_governor"] += 1
                elif "PROFIT" in gov_reasons:
                    signal_engine._rejection_counters["rejected_by_profit_cap"] += 1

            status = "✓ APPROVED" if sig.approved else f"✗ REJECTED ({sig.rejection_reason})"
            print(
                f"  [SIGNAL] {sig.side} @ {sig.entry_reference_price:.2f} "
                f"band={sig.band_level_hit:.1f}σ regime={sig.regime_at_signal} {status}"
            )
            all_signals.append(sig)

        # 6) ORB Engine 2 scaffold (break + pullback confirmation) — legacy modes only
        if orb_trigger_mode != "pullback_v3" and orb_generation_enabled and not is_flush:
            if levels is not None and orb_signals_emitted < config.ORB_TRADES_PER_DAY:
                if (
                    config.ORB_END <= bar_time_str < config.ORB_STALE_CUTOFF
                    and bar_time_str < config.LAST_ENTRY_CUTOFF
                    and regime in ("trend", "extreme")
                ):
                    orb_high = float(levels["high"])
                    orb_low = float(levels["low"])

                    broke_up = bar.close > orb_high
                    broke_down = bar.close < orb_low

                    if broke_up or broke_down:
                        orb_break_count += 1

                    if orb_pending_side is None:
                        if broke_up:
                            orb_pending_side = "BUY"
                            orb_pending_age = 0
                            if orb_trigger_mode in {"break", "either"}:
                                orb_confirmation_count += 1
                        elif broke_down:
                            orb_pending_side = "SELL"
                            orb_pending_age = 0
                            if orb_trigger_mode in {"break", "either"}:
                                orb_confirmation_count += 1
                    else:
                        orb_pending_age += 1

                    if orb_pending_age > orb_pullback_confirm_bars:
                        orb_pending_side = None
                        orb_pending_age = 0

                    if orb_pending_side == "BUY":
                        confirmed = (
                            (orb_trigger_mode == "break" and broke_up)
                            or (orb_trigger_mode == "pullback" and bar.low <= orb_high and bar.close > orb_high)
                            or (orb_trigger_mode == "either" and (broke_up or (bar.low <= orb_high and bar.close > orb_high)))
                        )
                        if confirmed:
                            if orb_trigger_mode in {"pullback", "either"} and (bar.low <= orb_high and bar.close > orb_high):
                                orb_confirmation_count += 1
                            entry_price = bar.close
                            stop_price = orb_low
                            risk = entry_price - stop_price
                            if risk > 0:
                                orb_sig = MRSignal(
                                    timestamp=bar.timestamp,
                                    side="BUY",
                                    signal_type="ORB",
                                    regime_at_signal=regime,
                                    entry_reference_price=entry_price,
                                    stop_reference=stop_price,
                                    target_reference=entry_price + risk,
                                    band_level_hit=0.0,
                                    vwap_at_signal=entry_price + risk,
                                    sigma_at_signal=0.0,
                                    z_at_signal=0.0,
                                    approved=True,
                                    rejection_reason="",
                                    bar_index=0,
                                )
                                gov_result = risk_gov.evaluate(
                                    daily_pnl=sim_daily_pnl,
                                    daily_trade_count=sim_trade_count,
                                    current_time_str_HHMM=bar_time_str,
                                    total_realized_pnl=0.0,
                                    best_day_pnl=0.0,
                                )
                                if not gov_result.approved:
                                    orb_sig.approved = False
                                    orb_sig.rejection_reason = f"GOVERNOR:{','.join(gov_result.reasons)}"
                                all_signals.append(orb_sig)
                                orb_signals_emitted += 1
                            orb_pending_side = None
                            orb_pending_age = 0

                    if orb_pending_side == "SELL":
                        confirmed = (
                            (orb_trigger_mode == "break" and broke_down)
                            or (orb_trigger_mode == "pullback" and bar.high >= orb_low and bar.close < orb_low)
                            or (orb_trigger_mode == "either" and (broke_down or (bar.high >= orb_low and bar.close < orb_low)))
                        )
                        if confirmed:
                            if orb_trigger_mode in {"pullback", "either"} and (bar.high >= orb_low and bar.close < orb_low):
                                orb_confirmation_count += 1
                            entry_price = bar.close
                            stop_price = orb_high
                            risk = stop_price - entry_price
                            if risk > 0:
                                orb_sig = MRSignal(
                                    timestamp=bar.timestamp,
                                    side="SELL",
                                    signal_type="ORB",
                                    regime_at_signal=regime,
                                    entry_reference_price=entry_price,
                                    stop_reference=stop_price,
                                    target_reference=entry_price - risk,
                                    band_level_hit=0.0,
                                    vwap_at_signal=entry_price - risk,
                                    sigma_at_signal=0.0,
                                    z_at_signal=0.0,
                                    approved=True,
                                    rejection_reason="",
                                    bar_index=0,
                                )
                                gov_result = risk_gov.evaluate(
                                    daily_pnl=sim_daily_pnl,
                                    daily_trade_count=sim_trade_count,
                                    current_time_str_HHMM=bar_time_str,
                                    total_realized_pnl=0.0,
                                    best_day_pnl=0.0,
                                )
                                if not gov_result.approved:
                                    orb_sig.approved = False
                                    orb_sig.rejection_reason = f"GOVERNOR:{','.join(gov_result.reasons)}"
                                all_signals.append(orb_sig)
                                orb_signals_emitted += 1
                            orb_pending_side = None
                            orb_pending_age = 0

        # 7) ORB pullback_v3 engine — breakout → pullback → entry
        if orb_trigger_mode == "pullback_v3" and orb_generation_enabled and not is_flush:
            if levels is not None and orb_signals_emitted < config.ORB_TRADES_PER_DAY:
                orb_high = float(levels["high"])
                orb_low = float(levels["low"])

                # ─── Phase 1: Detect breakout ──────────────────────
                # Regime gate removed: trend sessions are pre-filtered
                # in the validation pack; regime warmup (10 bars) would
                # otherwise shrink the detection window to bars 10-18.
                if not orb_pb3_active:
                    if (
                        config.ORB_END <= bar_time_str < config.ORB_STALE_CUTOFF
                        and bar_time_str < config.LAST_ENTRY_CUTOFF
                    ):
                        broke_up = bar.close > orb_high
                        broke_down = bar.close < orb_low
                        if broke_up or broke_down:
                            orb_break_count += 1
                            orb_pb3_active = True
                            orb_pb3_direction = "BUY" if broke_up else "SELL"
                            orb_pb3_breakout_bar = bars_closed
                            orb_pb3_breakout_level = orb_high if broke_up else orb_low
                            orb_pb3_or_high = orb_high
                            orb_pb3_or_low = orb_low
                            orb_pb3_pullback_detected = False
                            orb_pb3_recovery_pending = False
                            print(
                                f"  [ORB_PB3] BREAKOUT {orb_pb3_direction} "
                                f"level={orb_pb3_breakout_level:.2f} bar={bars_closed}"
                            )

                # ─── Phase 2: Wait for pullback & entry (regime-free) ──
                elif orb_pb3_active:
                    bars_since = bars_closed - orb_pb3_breakout_bar
                    # Only enforce LAST_ENTRY_CUTOFF, NOT ORB_STALE_CUTOFF.
                    # Once a breakout is detected, give it the full max_bars
                    # window for pullback — stale cutoff gates Phase 1 only.
                    stale = bar_time_str >= config.LAST_ENTRY_CUTOFF
                    # Expiry check
                    if bars_since > orb_pb3_max_bars or stale:
                        diag = {
                            "breakout_direction": orb_pb3_direction,
                            "breakout_bar": orb_pb3_breakout_bar,
                            "breakout_level": orb_pb3_breakout_level,
                            "pullback_detected": orb_pb3_pullback_detected,
                            "pullback_depth_pts": 0.0,
                            "bars_to_pullback": bars_since,
                            "entry_triggered": False,
                            "entry_mode_used": orb_pb3_entry_mode,
                            "expired": True,
                        }
                        orb_pb3_diagnostics.append(diag)
                        print(
                            f"  [ORB_PB3] EXPIRED {orb_pb3_direction} "
                            f"after {bars_since} bars (max={orb_pb3_max_bars} stale={stale})"
                        )
                        orb_pb3_active = False
                        orb_pb3_recovery_pending = False
                    else:
                        # Check for pullback touch
                        pullback_touch = False
                        pullback_depth = 0.0
                        if orb_pb3_direction == "BUY":
                            # Long: pullback if low comes within tolerance of orb_high
                            if bar.low <= orb_pb3_breakout_level + orb_pb3_tolerance_pts:
                                pullback_touch = True
                                pullback_depth = max(0.0, orb_pb3_breakout_level - bar.low)
                        else:
                            # Short: pullback if high comes within tolerance of orb_low
                            if bar.high >= orb_pb3_breakout_level - orb_pb3_tolerance_pts:
                                pullback_touch = True
                                pullback_depth = max(0.0, bar.high - orb_pb3_breakout_level)

                        if pullback_touch and not orb_pb3_pullback_detected:
                            orb_pb3_pullback_detected = True
                            orb_confirmation_count += 1
                            print(
                                f"  [ORB_PB3] PULLBACK detected {orb_pb3_direction} "
                                f"depth={pullback_depth:.2f}pts bar={bars_closed}"
                            )

                        # Entry decision
                        enter_now = False
                        if orb_pb3_pullback_detected:
                            if orb_pb3_entry_mode == "touch_only":
                                enter_now = True
                            elif orb_pb3_entry_mode == "touch_recovery":
                                if orb_pb3_recovery_pending:
                                    if orb_pb3_direction == "BUY" and bar.close > orb_pb3_breakout_level:
                                        enter_now = True
                                    elif orb_pb3_direction == "SELL" and bar.close < orb_pb3_breakout_level:
                                        enter_now = True
                                else:
                                    if orb_pb3_direction == "BUY" and bar.close > orb_pb3_breakout_level:
                                        enter_now = True
                                    elif orb_pb3_direction == "SELL" and bar.close < orb_pb3_breakout_level:
                                        enter_now = True
                                    else:
                                        orb_pb3_recovery_pending = True

                        if enter_now:
                            entry_price = bar.close
                            if orb_pb3_direction == "BUY":
                                stop_price = orb_pb3_or_low
                                risk = entry_price - stop_price
                            else:
                                stop_price = orb_pb3_or_high
                                risk = stop_price - entry_price

                            if risk > 0:
                                orb_sig = MRSignal(
                                    timestamp=bar.timestamp,
                                    side=orb_pb3_direction,  # type: ignore[arg-type]
                                    signal_type="ORB",
                                    regime_at_signal=regime,
                                    entry_reference_price=entry_price,
                                    stop_reference=stop_price,
                                    target_reference=(entry_price + risk if orb_pb3_direction == "BUY" else entry_price - risk),
                                    band_level_hit=0.0,
                                    vwap_at_signal=(entry_price + risk if orb_pb3_direction == "BUY" else entry_price - risk),
                                    sigma_at_signal=0.0,
                                    z_at_signal=0.0,
                                    approved=True,
                                    rejection_reason="",
                                    bar_index=0,
                                )
                                gov_result = risk_gov.evaluate(
                                    daily_pnl=sim_daily_pnl,
                                    daily_trade_count=sim_trade_count,
                                    current_time_str_HHMM=bar_time_str,
                                    total_realized_pnl=0.0,
                                    best_day_pnl=0.0,
                                )
                                if not gov_result.approved:
                                    orb_sig.approved = False
                                    orb_sig.rejection_reason = f"GOVERNOR:{','.join(gov_result.reasons)}"
                                all_signals.append(orb_sig)
                                orb_signals_emitted += 1
                                print(
                                    f"  [ORB_PB3] ENTRY {orb_pb3_direction} @ {entry_price:.2f} "
                                    f"stop={stop_price:.2f} risk={risk:.2f}"
                                )

                            # Log diagnostic
                            diag = {
                                "breakout_direction": orb_pb3_direction,
                                "breakout_bar": orb_pb3_breakout_bar,
                                "breakout_level": orb_pb3_breakout_level,
                                "pullback_detected": True,
                                "pullback_depth_pts": pullback_depth if orb_pb3_pullback_detected else 0.0,
                                "bars_to_pullback": bars_since,
                                "entry_triggered": True,
                                "entry_mode_used": orb_pb3_entry_mode,
                                "entry_price": entry_price,
                                "expired": False,
                            }
                            orb_pb3_diagnostics.append(diag)

                            # Reset for next breakout
                            orb_pb3_active = False
                            orb_pb3_recovery_pending = False

    # ── Aggregator ──────────────────────────────────────────────────────
    agg = IntradayBarAggregator(interval_minutes=5, on_bar_callback=on_bar)

    # ── Chart setup ─────────────────────────────────────────────────────
    fig: Any = None
    ax_px: Any = None
    ax_bar: Any = None
    line_px: Any = None
    dashboard: Any = None
    if dashboard_enabled:
        fig, (ax_px, ax_bar) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
        line_px, = ax_px.plot([], [], color="tab:blue", linewidth=0.6, label="Trade Price")

        for ax in (ax_px, ax_bar):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax.tick_params(axis="x", rotation=30, labelsize=8)

        ax_px.set_title("Replay Debug Cockpit — Tick Prices")
        ax_px.set_ylabel("Price")
        ax_px.legend(loc="best", fontsize=7)
        ax_bar.set_ylabel("Price / VWAP")

        dashboard = ReplayDashboard(fig, ax_px, ax_bar)

        if not args.no_show:
            plt.ion()
            plt.show(block=False)

    # ── Tick loop ───────────────────────────────────────────────────────
    times: list[datetime] = []
    prices: list[float] = []

    ts_series = np.array(trades["timestamp"], dtype="datetime64[us]")
    px_series = np.asarray(trades["price"], dtype=float)
    sz_series = np.asarray(trades["size"], dtype=float)
    total_ticks = len(trades)

    t_start = time.monotonic()

    for idx, (ts64, px, sz) in enumerate(zip(ts_series, px_series, sz_series), start=1):
        ts = np.datetime64(ts64, "us").astype("datetime64[us]").astype(object)

        # Track unique 5-min bucket keys
        bucket_key = ts.replace(
            minute=(ts.minute // 5) * 5, second=0, microsecond=0
        ).strftime("%H:%M")
        unique_buckets.add(bucket_key)

        agg.on_tick(ts, float(px), float(sz))

        times.append(ts)
        prices.append(float(px))

        # Periodic chart update
        if dashboard_enabled and (idx % args.update_every == 0 or idx == total_ticks):
            line_px.set_data(times, prices)
            ax_px.relim()
            ax_px.autoscale_view()

            # Draw bar close line
            if bar_closes:
                ax_bar.clear()
                ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                ax_bar.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                ax_bar.tick_params(axis="x", rotation=30, labelsize=8)
                ax_bar.set_ylabel("Price / VWAP")
                ax_bar.plot(bar_times, bar_closes, color="tab:purple",
                            linewidth=1.2, label="5m Close", zorder=4)
                # Redraw all overlays
                dashboard._legend_labels_bar.clear()
                dashboard.update_overlays(
                    bar_times, bar_closes,
                    vwap_history=vwap_history,
                    orb_levels=orb_tracker.levels,
                    regime_history=regime_clf.features_history,
                    signals=all_signals,
                )

            elapsed = time.monotonic() - t_start
            ax_px.set_title(
                f"Replay Debug Cockpit | "
                f"ticks={idx}/{total_ticks}  bars={bars_closed}  "
                f"regime={regime_clf.current_regime or 'warmup'}  "
                f"signals={len(all_signals)}  "
                f"elapsed={elapsed:.1f}s",
                fontsize=9,
            )

            if not args.no_show:
                plt.pause(args.pause)

    # ── Flush partial bar ───────────────────────────────────────────────
    has_partial = agg._bar_start is not None and agg._current_open is not None
    print(f"\n  [FLUSH] partial_bar_exists={has_partial}")
    if has_partial:
        _orig_cb = agg.on_bar
        agg.on_bar = lambda bar: on_bar(bar, is_flush=True)
        agg.flush()
        agg.on_bar = _orig_cb
    else:
        agg.flush()

    # ── Final chart ─────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t_start
    if dashboard_enabled:
        # Redraw bar axis cleanly
        ax_bar.clear()
        ax_bar.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax_bar.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax_bar.tick_params(axis="x", rotation=30, labelsize=8)
        ax_bar.set_ylabel("Price / VWAP")

        if bar_closes:
            ax_bar.plot(bar_times, bar_closes, color="tab:purple",
                        linewidth=1.2, label="5m Close", zorder=4)

        # Reset legend tracking for clean final draw
        dashboard._legend_labels_bar.clear()

        dashboard.finalize(
            bar_times=bar_times,
            bar_closes=bar_closes,
            vwap_history=vwap_history,
            orb_levels=orb_tracker.levels,
            regime_history=regime_clf.features_history,
            signals=all_signals,
            extra_title=f"Session: {session_id}",
        )

        # Tick chart final update
        line_px.set_data(times, prices)
        ax_px.relim()
        ax_px.autoscale_view()
        ax_px.set_title(
            f"Replay Debug Cockpit — FINAL | "
            f"ticks={total_ticks}  bars_closed={bars_closed}  "
            f"bars_partial={bars_partial_flushed}  "
            f"regime={regime_clf.current_regime or 'warmup'}  "
            f"signals={len(all_signals)}  "
            f"elapsed={total_elapsed:.1f}s",
            fontsize=9,
        )

    # ── Console summary ────────────────────────────────────────────────
    approved = [s for s in all_signals if s.approved]
    rejected = [s for s in all_signals if not s.approved]
    print(f"\n{'='*70}")
    print(f"  REPLAY DEBUG SUMMARY — {session_id}")
    print(f"{'='*70}")
    print(f"  Ticks processed      : {total_ticks:,}")
    print(f"  Bars closed          : {bars_closed}")
    print(f"  Bars partial flushed : {bars_partial_flushed}")
    print(f"  Unique 5m buckets    : {len(unique_buckets)}")
    print(f"  Regime (final)       : {regime_clf.current_regime or 'warmup'}")
    print(f"  Regime warmup bars   : {config.REGIME_WARMUP_BARS}")
    print(f"  Classifier bars seen : {regime_clf.bar_count}")
    print(f"  VWAP (final)         : {vwap_history[-1].vwap:.2f}" if vwap_history else "  VWAP: n/a")
    print(f"  ATR (final)          : {atr_calc.atr:.4f}")
    print(f"  ORB levels           : {orb_tracker.levels}")
    print(f"  Total signals        : {len(all_signals)}")
    print(f"    Approved           : {len(approved)}")
    print(f"    Rejected           : {len(rejected)}")
    for sig in all_signals:
        status = "OK" if sig.approved else f"REJ:{sig.rejection_reason}"
        print(
            f"    [{status}] {sig.side} @ {sig.entry_reference_price:.2f} "
            f"band={sig.band_level_hit:.1f}σ ts={sig.timestamp}"
        )

    # ── Rejection counter breakdown ─────────────────────────────────────
    rej_counters = signal_engine.rejection_counters
    print(f"  Signal counters:")
    for k, v in rej_counters.items():
        print(f"    {k:<38}: {v}")

    gate_funnel = signal_engine.gate_funnel_report
    print("  Gate funnel:")
    print(f"    candidates_total                     : {gate_funnel.get('candidates_total', 0)}")
    for key, val in gate_funnel.items():
        if key.startswith("passed_gate_"):
            print(f"    {key:<38}: {val}")
    print(f"    approved_trades                      : {gate_funnel.get('approved_trades', 0)}")
    top_fail = gate_funnel.get("top_failure_reasons", [])
    if isinstance(top_fail, list) and top_fail:
        print("    top_failure_reasons:")
        for item in top_fail:
            print(f"      - {item.get('reason')}: {item.get('count')}")

    print(f"  Elapsed              : {total_elapsed:.1f}s")
    print(f"{'='*70}")

    # ── Signal diagnostic: "why no signals?" ────────────────────────────
    _print_signal_diagnostic(
        regime_clf.features_history, vwap_history, bar_closes,
        bars_closed, config.REGIME_WARMUP_BARS, orb_tracker,
        mr_sigma_entry, cooldown_bars,
    )

    # ── Report export ───────────────────────────────────────────────────
    if not args.no_report:
        report = ReplaySessionReport(
            session_id=session_id,
            symbol=args.symbol,
            replay_start=args.start,
            replay_end=args.end,
        )
        report.set_tick_stats(
            ticks_processed=total_ticks,
            bars_closed=bars_closed,
            bars_partial_flushed=bars_partial_flushed,
            unique_buckets=len(unique_buckets),
        )
        report.add_signals(all_signals)
        report.add_features(regime_clf.features_history)
        report.set_config_snapshot(
            _config_snapshot(
                {
                    "MR_SIGMA_ENTRY": mr_sigma_entry,
                    "MR_RECLAIM_MODE": reclaim_mode,
                    "MR_SOFT_RECLAIM_RANGE_IMPULSE_K": soft_range_k,
                    "MR_SOFT_RECLAIM_IMPULSE_K": soft_range_k,
                    "MR_EXCURSION_DEDUPE_ENABLED": dedupe_enabled,
                    "MR_ATTEMPT_CAP_ENABLED": attempt_cap_enabled,
                    "MR_COOLDOWN_BARS": cooldown_bars,
                    "MR_FIRST_OUTSIDE_ENABLED": first_outside_enabled,
                    "MR_TOUCH_LATCH_RESET_BUFFER": touch_latch_reset_buffer,
                    "MR_DEDUPE_WINDOW_BARS": dedupe_window_bars,
                    "MR_DEDUPE_MIN_DELTA_Z": dedupe_min_delta_z,
                    "MR_REGIME_ENABLED": regime_enabled,
                    "ENGINE_MODE": engine_mode,
                    "ALLOCATOR_POLICY": allocator_policy,
                    "ALLOCATOR_V1_ADX_THRESHOLD": allocator_v1_adx_threshold,
                    "ALLOCATOR_V2_TREND_OPEN_THRESHOLD": allocator_v2_trend_open_threshold,
                    "ALLOCATOR_V2_RISING_THRESHOLD": allocator_v2_rising_threshold,
                    "ALLOCATOR_V2_RISING_BARS": allocator_v2_rising_bars,
                    "ALLOCATOR_V2_RANGE_THRESHOLD": allocator_v2_range_threshold,
                    "ALLOCATOR_V2_RANGE_BARS": allocator_v2_range_bars,
                    "ALLOC_OPENPROXY_OR_WIDTH_ATR": open_proxy_cfg.or_width_atr_threshold,
                    "ALLOC_OPENPROXY_IMPULSE_ATR": open_proxy_cfg.impulse_atr_threshold,
                    "ALLOC_OPENPROXY_PERSIST_BARS": open_proxy_cfg.persist_bars,
                    "ALLOC_OPENPROXY_REQUIRE_BREAK": open_proxy_cfg.require_break,
                    "ALLOC_OPENPROXY_ENABLE_ORB_SELECTIVITY_REFINEMENT": open_proxy_cfg.enable_orb_selectivity_refinement,
                    "ALLOC_OPENPROXY_LOW_ATR_THRESHOLD": open_proxy_cfg.orb_selectivity_low_atr_threshold,
                    "ALLOC_OPENPROXY_MIN_PERSISTENCE_IN_LOW_ATR": open_proxy_cfg.orb_selectivity_min_persistence_in_low_atr,
                    "ALLOC_OPENPROXY_HIGH_IMPULSE_THRESHOLD": open_proxy_cfg.orb_selectivity_high_impulse_threshold,
                    "ALLOC_OPENPROXY_MIN_PERSISTENCE_WHEN_HIGH_IMPULSE": open_proxy_cfg.orb_selectivity_min_persistence_when_high_impulse,
                    "ALLOC_OPENPROXY_MEDIUM_IMPULSE_WEAK_PERSISTENCE_FILTER_ENABLED": open_proxy_cfg.enable_medium_impulse_weak_persistence_filter,
                    "ORB_ENABLED": orb_enabled,
                    "ORB_TRIGGER_MODE": getattr(args, "orb_trigger_mode", config.ORB_TRIGGER_MODE),
                    "ORB_PULLBACK_V3_MAX_BARS": orb_pb3_max_bars,
                    "ORB_PULLBACK_V3_TOLERANCE_PTS": orb_pb3_tolerance_pts,
                    "ORB_PULLBACK_V3_ENTRY_MODE": orb_pb3_entry_mode,
                }
            )
        )
        report.set_rejection_counters(signal_engine.rejection_counters)
        report.set_gate_funnel(signal_engine.gate_funnel_report)
        report.set_orb_funnel(
            {
                "or_constructed_days": orb_or_constructed,
                "breaks": orb_break_count,
                "confirmations": orb_confirmation_count,
                "signals": orb_signals_emitted,
                "trigger_mode": orb_trigger_mode,
                "pullback_confirm_bars": orb_pullback_confirm_bars,
                "pullback_v3_max_bars": orb_pb3_max_bars,
                "pullback_v3_tolerance_pts": orb_pb3_tolerance_pts,
                "pullback_v3_entry_mode": orb_pb3_entry_mode,
                "pullback_v3_diagnostics": orb_pb3_diagnostics,
                "engine_mode": engine_mode,
                "allocator_policy": allocator_policy,
                "allocator_decision": allocator_decision,
                "allocator_reason": allocator_reason,
                "open_proxy_diagnostics": {
                    "or_high": open_proxy_result.or_high,
                    "or_low": open_proxy_result.or_low,
                    "opening_range_width_pts": open_proxy_result.opening_range_width_pts,
                    "opening_range_width_atr": open_proxy_result.opening_range_width_atr,
                    "first_3bar_directional_impulse": open_proxy_result.first_3bar_directional_impulse,
                    "signed_imbalance": open_proxy_result.signed_imbalance,
                    "breakout_persistence": open_proxy_result.breakout_persistence,
                    "breakout_direction": open_proxy_result.breakout_direction,
                    "persist_bars_observed": open_proxy_result.persist_bars_observed,
                    "trigger_width": open_proxy_result.trigger_width,
                    "trigger_impulse": open_proxy_result.trigger_impulse,
                    "trigger_persist": open_proxy_result.trigger_persist,
                    "atr_at_decision": open_proxy_result.atr_at_decision,
                    "pre_selectivity_decision": open_proxy_result.pre_selectivity_decision,
                    "pre_selectivity_reason": open_proxy_result.pre_selectivity_reason,
                    "selectivity_refinement_enabled": open_proxy_result.selectivity_refinement_enabled,
                    "selectivity_low_atr_caution": open_proxy_result.selectivity_low_atr_caution,
                    "selectivity_high_impulse_caution": open_proxy_result.selectivity_high_impulse_caution,
                    "selectivity_orb_blocked": open_proxy_result.selectivity_orb_blocked,
                    "selectivity_block_reason": open_proxy_result.selectivity_block_reason,
                    "selectivity_medium_impulse_weak_persistence_caution": open_proxy_result.selectivity_medium_impulse_weak_persistence_caution,
                    "selectivity_v3_orb_blocked": open_proxy_result.selectivity_v3_orb_blocked,
                    "selectivity_v3_block_reason": open_proxy_result.selectivity_v3_block_reason,
                    "pre_v3_selectivity_decision": open_proxy_result.pre_v3_selectivity_decision,
                    "post_v3_selectivity_decision": open_proxy_result.post_v3_selectivity_decision,
                } if open_proxy_result is not None else None,
            }
        )
        artifact_path = report.export()
        print(f"  Report exported → {artifact_path}")

        # ── Run Exit Simulator ──────────────────────────────────────────────
        try:
            from simulation.mr_exit_simulator import MRExitSimulator
            sim = MRExitSimulator()
            run_session = getattr(sim, "run_session", None)
            if callable(run_session):
                diag = run_session(artifact_path)
            else:
                run = getattr(sim, "run", None)
                if not callable(run):
                    raise AttributeError("MRExitSimulator has neither 'run_session' nor 'run'")
                diag = run(artifact_path)
            trades_emitted = diag.get("trades_emitted", 0) if isinstance(diag, dict) else 0
            print(f"  Exit sim ran  → {trades_emitted} trades generated")
        except Exception as exc:
            print(f"  [exit_sim] ERROR: {type(exc).__name__}: {exc}")

    # ── Save / show chart ───────────────────────────────────────────────
    if dashboard_enabled and args.save_path:
        fig.savefig(args.save_path, dpi=140)
        print(f"  Chart saved → {args.save_path}")

    if dashboard_enabled and not args.no_show:
        plt.ioff()
        plt.show()

    return 0


# ═══════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()
    return run_debug_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
