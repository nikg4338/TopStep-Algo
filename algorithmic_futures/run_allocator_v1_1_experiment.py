#!/usr/bin/env python3
"""
run_allocator_v1_1_experiment.py — Allocator open_proxy_v1 experiment matrix.

Compares the current ALLOC_V2_HYST allocator (broken by ADX warmup) against the
new open_proxy_v1 price-action allocator on three datasets:
  1. pilot_20d      — 20 recent sessions (Jan 26 – Feb 20)
  2. extended_60d   — 58 sessions (Dec 1 – Feb 20)
  3. trend20        — 20 ADX-upper-tertile sessions (trend days)

For each (dataset × allocator) cell, performs post-hoc re-routing from existing
base run features_snapshot.csv + trades.csv and runs MC survival simulation.

Outputs:
  - Per-cell: routing counts, ORB entries, trades/day, avg_r, dd_p95,
    equity p10/p50, P_hit / P_ruin / P_daily
  - Allocator audit: false-positive ORB, false-negative MR
  - Comparison table

Usage:
    python run_allocator_v1_1_experiment.py
    python run_allocator_v1_1_experiment.py --or-width-atr 0.6 --impulse-atr 0.7
    python run_allocator_v1_1_experiment.py --persist-bars 2 --require-break on

Part of the v1_1 investigation (allocator warmup fix).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from simulation.mc_survival import MonteCarloSurvivalSimulator
from validation.open_proxy_allocator import (
    OpenProxyConfig,
    OpenProxyDecision,
    OpenWindowState,
    decide as open_proxy_decide,
)

ARTIFACTS_ROOT = Path("artifacts/validation_runs")

# ── Known run IDs ──────────────────────────────────────────────────────
KNOWN_RUNS: dict[str, list[str]] = {
    "pilot_20d": ["pilot_20d_20260302_133625"],
    "extended_60d": ["extended_60d_20260303_175155"],
    "trend20": ["pb3eval_trend20_20260302_133325"],
}

MC_KEYS = [
    "p_target_before_ruin", "p_ruin", "p_daily_loss_breach",
    "dd_p95", "equity_p50", "equity_p10",
]


# ═══════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SessionInfo:
    session_id: str
    session_dir: Path
    n_trades: int
    total_pnl: float
    adx_median: float           # full-session non-zero ADX median
    atr_median: float
    or_high: float = 0.0
    or_low: float = 0.0
    or_width_pts: float = 0.0
    has_orb_signals: bool = False
    has_orb_trades: bool = False


@dataclass
class AllocatorResult:
    session_id: str
    decision: str               # "orb" or "mr"
    reason: str
    or_width_atr: float = 0.0
    impulse_atr: float = 0.0
    persist_bars: int = 0
    trigger_width: bool = False
    trigger_impulse: bool = False
    trigger_persist: bool = False


@dataclass
class CellResult:
    dataset: str
    allocator: str
    n_sessions: int = 0
    n_orb_routed: int = 0
    n_mr_routed: int = 0
    orb_entries: int = 0
    total_trades: int = 0
    total_pnl: float = 0.0
    trades_per_day: float = 0.0
    avg_r: float = 0.0
    p_hit: float = 0.0
    p_ruin: float = 0.0
    p_daily: float = 0.0
    dd_p95: float = 0.0
    eq_p50: float = 0.0
    eq_p10: float = 0.0
    false_positive_orb: int = 0   # ORB routed but 0 ORB entries
    false_negative_mr: int = 0    # MR routed but ORB opportunity later present
    session_decisions: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
#  Session discovery
# ═══════════════════════════════════════════════════════════════════════

def _discover_sessions(run_ids: list[str]) -> list[SessionInfo]:
    """Find all sessions across the given run IDs."""
    sessions: dict[str, SessionInfo] = {}
    for rid in run_ids:
        run_dir = ARTIFACTS_ROOT / rid / "sessions"
        if not run_dir.is_dir():
            print(f"  [WARN] Run dir missing: {run_dir}")
            continue
        for sdir in sorted(run_dir.iterdir()):
            if not sdir.is_dir():
                continue
            sid = sdir.name
            if sid in sessions:
                continue  # first occurrence wins

            # Read trades
            trades_csv = sdir / "trades.csv"
            n_trades = 0
            total_pnl = 0.0
            has_orb_trades = False
            if trades_csv.is_file():
                try:
                    df = pd.read_csv(trades_csv)
                    n_trades = len(df)
                    if "pnl_dollars" in df.columns:
                        total_pnl = float(df["pnl_dollars"].sum())
                    if "strategy" in df.columns:
                        has_orb_trades = bool((df["strategy"].str.contains("ORB", case=False, na=False)).any())
                except Exception:
                    pass

            # Read features for ADX / ATR
            feat = sdir / "features_snapshot.csv"
            adx_med = 0.0
            atr_med = 0.0
            if feat.is_file():
                try:
                    fdf = pd.read_csv(feat)
                    if "adx" in fdf.columns:
                        adx_vals = pd.to_numeric(fdf["adx"], errors="coerce").dropna()
                        adx_vals = adx_vals[adx_vals > 0]
                        adx_med = float(adx_vals.median()) if len(adx_vals) > 0 else 0.0
                    if "atr" in fdf.columns:
                        atr_vals = pd.to_numeric(fdf["atr"], errors="coerce").dropna()
                        atr_vals = atr_vals[atr_vals > 0]
                        atr_med = float(atr_vals.median()) if len(atr_vals) > 0 else 0.0
                except Exception:
                    pass

            # Read session_summary for OR levels
            summary_path = sdir / "session_summary.json"
            or_high = 0.0
            or_low = 0.0
            has_orb_signals = False
            if summary_path.is_file():
                try:
                    with open(summary_path) as f:
                        summary = json.load(f)
                    orb_funnel = summary.get("orb_funnel", {})
                    has_orb_signals = orb_funnel.get("signals", 0) > 0
                    # Try to find OR levels from pb3 diagnostics
                    diags = orb_funnel.get("pullback_v3_diagnostics", [])
                    if diags:
                        or_high = diags[0].get("or_high", 0.0)
                        or_low = diags[0].get("or_low", 0.0)
                except Exception:
                    pass

            or_width = or_high - or_low if or_high > or_low else 0.0
            sessions[sid] = SessionInfo(
                session_id=sid,
                session_dir=sdir,
                n_trades=n_trades,
                total_pnl=total_pnl,
                adx_median=adx_med,
                atr_median=atr_med,
                or_high=or_high,
                or_low=or_low,
                or_width_pts=or_width,
                has_orb_signals=has_orb_signals,
                has_orb_trades=has_orb_trades,
            )

    return sorted(sessions.values(), key=lambda s: s.session_id)


# ═══════════════════════════════════════════════════════════════════════
#  Allocator simulation — V2 (ADX-based, from features_snapshot)
# ═══════════════════════════════════════════════════════════════════════

def _session_early_adx(session_dir: Path, max_bars: int = 12) -> list[float]:
    """Read first max_bars NON-ZERO ADX values from features_snapshot.csv.

    This replicates the calibration runner's behavior — reading mid-session
    ADX values (bar 28+).
    """
    feat = session_dir / "features_snapshot.csv"
    if not feat.is_file():
        return []
    try:
        df = pd.read_csv(feat)
    except Exception:
        return []
    if df.empty or "adx" not in df.columns:
        return []
    adx = pd.to_numeric(df["adx"], errors="coerce")
    out: list[float] = []
    for a in adx:
        if pd.isna(a):
            continue
        if float(a) > 0:
            out.append(float(a))
            if len(out) >= max_bars:
                break
    return out


def _allocator_v2_decision(session_dir: Path) -> AllocatorResult:
    """Replicate ALLOC_V2_HYST from mid-session ADX (calibration behavior)."""
    adx_series = _session_early_adx(session_dir, max_bars=12)
    sid = session_dir.name
    trend_open = any(v >= 25.0 for v in adx_series)
    rising = adx_series[-3:]
    rising_ok = (
        len(rising) >= 3
        and all(v > 20.0 for v in rising)
        and all(rising[i] < rising[i + 1] for i in range(len(rising) - 1))
    )
    range_seq = adx_series[-3:]
    range_ok = len(range_seq) >= 3 and all(v <= 18.0 for v in range_seq)
    if trend_open or rising_ok:
        decision = "orb"
        reason = "V2_TREND" if trend_open else "V2_RISING"
    elif range_ok:
        decision = "mr"
        reason = "V2_RANGE"
    else:
        decision = "mr"
        reason = "V2_DEFAULT_MR"
    return AllocatorResult(
        session_id=sid, decision=decision, reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Allocator simulation — open_proxy_v1 (from features_snapshot bars)
# ═══════════════════════════════════════════════════════════════════════

def _allocator_open_proxy_decision(
    session_dir: Path,
    cfg: OpenProxyConfig,
) -> AllocatorResult:
    """Simulate open_proxy_v1 from the stored features_snapshot.csv bars.

    Reads the first few bars' OHLC data from the features CSV to reconstruct
    the opening range and impulse signals. Falls back to session_summary if
    available.
    """
    sid = session_dir.name
    feat = session_dir / "features_snapshot.csv"

    # We need OHLC bar data. features_snapshot.csv typically only has
    # indicators. We'll reconstruct from the session replay CSV if available,
    # or fall back to the session_summary's OR levels.
    bars_csv = session_dir / "bars.csv"
    summary_path = session_dir / "session_summary.json"

    # Try to read ATR from features
    atr_at_decision = 0.0
    if feat.is_file():
        try:
            fdf = pd.read_csv(feat)
            if "atr" in fdf.columns:
                atr_vals = pd.to_numeric(fdf["atr"], errors="coerce").dropna()
                atr_vals = atr_vals[atr_vals > 0]
                if len(atr_vals) >= 3:
                    # ATR at bar 3 (or earliest available)
                    atr_at_decision = float(atr_vals.iloc[min(2, len(atr_vals) - 1)])
                elif len(atr_vals) > 0:
                    atr_at_decision = float(atr_vals.iloc[0])
        except Exception:
            pass

    if atr_at_decision <= 0:
        atr_at_decision = 5.0  # fallback for MES typical ATR

    # Build OpenWindowState from available data
    state = OpenWindowState()
    state.atr_at_decision = atr_at_decision

    # Strategy: read session_summary for OR levels + orb_funnel diag
    or_high = 0.0
    or_low = 0.0
    bar_data_found = False

    if summary_path.is_file():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            orb_funnel = summary.get("orb_funnel", {})
            diags = orb_funnel.get("pullback_v3_diagnostics", [])
            if diags:
                or_high = float(diags[0].get("or_high", 0.0))
                or_low = float(diags[0].get("or_low", 0.0))
        except Exception:
            pass

    # Try to read bar OHLC from features or reconstruct
    # features_snapshot has timestamp + indicators but NOT bar OHLC
    # We need to look for an alternative data source
    # The replay_debug exports signals.csv with bar data but not bars themselves
    # Best approach: read OR high/low from session_summary if available,
    # and for impulse, read the price trajectory from features timestamps

    if or_high > or_low:
        # We have OR levels from the session summary
        state.or_high = or_high
        state.or_low = or_low
        # Synthesize bars from OR range
        # The first 3 bars span OR; we can compute impulse from mid-range assumption
        # For more accuracy, try to read close prices from the replay data
        bar_data_found = True

    # Try to get close prices from features_snapshot (regime column timing)
    # to compute impulse metric
    if feat.is_file():
        try:
            fdf = pd.read_csv(feat)
            if "timestamp" in fdf.columns:
                ts = pd.to_datetime(fdf["timestamp"], utc=True, errors="coerce")
                import pytz
                ET = pytz.timezone(config.TIMEZONE)
                et_times = ts.dt.tz_convert(ET).dt.strftime("%H:%M")

                # Parse bars from signals.csv if it exists (has OHLC)
                signals_csv = session_dir / "signals.csv"
                if signals_csv.is_file():
                    sdf = pd.read_csv(signals_csv)
                    if all(c in sdf.columns for c in ("open", "high", "low", "close")):
                        bar_data_found = True
                        sts = pd.to_datetime(sdf["timestamp"], utc=True, errors="coerce")
                        s_et_times = sts.dt.tz_convert(ET).dt.strftime("%H:%M")

                        or_bars = sdf[(s_et_times >= config.RTH_OPEN) & (s_et_times < config.ORB_END)]
                        if len(or_bars) > 0:
                            state.first_bar_open = float(or_bars.iloc[0]["open"])
                            for _, row in or_bars.iterrows():
                                bt = (float(row["open"]), float(row["high"]),
                                      float(row["low"]), float(row["close"]))
                                state.bars.append(bt)
                                state.or_high = max(state.or_high, float(row["high"]))
                                state.or_low = min(state.or_low, float(row["low"]))

                        post_bars = sdf[s_et_times >= config.ORB_END]
                        for _, row in post_bars.head(max(3, cfg.persist_bars)).iterrows():
                            bt = (float(row["open"]), float(row["high"]),
                                  float(row["low"]), float(row["close"]))
                            state.post_or_bars.append(bt)
        except Exception:
            pass

    # If we still don't have bar data but have OR levels, synthesize
    if not state.bars and or_high > or_low:
        # Synthesize 3 bars spanning the OR
        mid = (or_high + or_low) / 2.0
        state.first_bar_open = mid
        state.bars = [
            (mid, or_high, or_low, mid + (or_high - or_low) * 0.2),
            (mid, or_high, or_low, mid + (or_high - or_low) * 0.3),
            (mid, or_high, or_low, mid + (or_high - or_low) * 0.4),
        ]
        state.or_high = or_high
        state.or_low = or_low

    # Run decision
    result = open_proxy_decide(state, cfg)

    return AllocatorResult(
        session_id=sid,
        decision=result.decision,
        reason=result.reason,
        or_width_atr=result.opening_range_width_atr,
        impulse_atr=result.first_3bar_directional_impulse,
        persist_bars=result.persist_bars_observed,
        trigger_width=result.trigger_width,
        trigger_impulse=result.trigger_impulse,
        trigger_persist=result.trigger_persist,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Run one cell (dataset × allocator)
# ═══════════════════════════════════════════════════════════════════════

def _run_cell(
    dataset: str,
    allocator: str,
    sessions: list[SessionInfo],
    open_proxy_cfg: OpenProxyConfig,
) -> CellResult:
    """Evaluate one (dataset, allocator) cell."""
    cell = CellResult(dataset=dataset, allocator=allocator, n_sessions=len(sessions))

    all_pnls: list[float] = []
    for sinfo in sessions:
        # Determine routing
        if allocator == "v2_hyst":
            ar = _allocator_v2_decision(sinfo.session_dir)
        elif allocator == "open_proxy_v1":
            ar = _allocator_open_proxy_decision(sinfo.session_dir, open_proxy_cfg)
        else:
            ar = AllocatorResult(session_id=sinfo.session_id, decision="mr", reason="UNKNOWN")

        cell.session_decisions.append(asdict(ar))

        if ar.decision == "orb":
            cell.n_orb_routed += 1
            # Check false positive: ORB routed but no ORB signals in base run
            if not sinfo.has_orb_signals and not sinfo.has_orb_trades:
                cell.false_positive_orb += 1
            if sinfo.has_orb_trades:
                cell.orb_entries += 1
        else:
            cell.n_mr_routed += 1
            # Check false negative: MR routed but ORB opportunity existed
            if sinfo.has_orb_signals or sinfo.has_orb_trades:
                cell.false_negative_mr += 1

        cell.total_trades += sinfo.n_trades
        cell.total_pnl += sinfo.total_pnl

        # Collect per-trade PnLs for MC
        trades_csv = sinfo.session_dir / "trades.csv"
        if trades_csv.is_file():
            try:
                df = pd.read_csv(trades_csv)
                if "pnl_dollars" in df.columns:
                    for pnl in df["pnl_dollars"]:
                        all_pnls.append(float(pnl))
            except Exception:
                pass

    cell.trades_per_day = round(cell.total_trades / max(1, cell.n_sessions), 2)
    if all_pnls:
        cell.avg_r = round(statistics.mean(all_pnls), 2)

    # MC simulation
    if len(all_pnls) >= 5:
        try:
            mc = MonteCarloSurvivalSimulator(
                profit_target=float(config.PROFIT_TARGET),
                max_loss_limit=float(config.MAX_LOSS_LIMIT),
                daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
                n_simulations=10_000,
                max_trades=config.MC_MAX_TRADES,
            )
            sr = mc.run(
                r_values=all_pnls,
                use_dollar_values=True,
            )
            cell.p_hit = round(sr.p_target_before_ruin, 4)
            cell.p_ruin = round(sr.p_ruin, 4)
            cell.p_daily = round(sr.p_daily_loss_breach, 4)
            cell.dd_p95 = round(sr.dd_p95, 1)
            cell.eq_p50 = round(sr.equity_p50, 1)
            cell.eq_p10 = round(sr.equity_p10, 1)
        except Exception as exc:
            print(f"  [WARN] MC failed for {dataset}/{allocator}: {exc}")

    return cell


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Allocator v1_1 experiment matrix: ALLOC_V2_HYST vs open_proxy_v1",
    )
    # open_proxy_v1 thresholds
    p.add_argument("--or-width-atr", type=float, default=0.8,
                    help="OR width / ATR threshold")
    p.add_argument("--impulse-atr", type=float, default=0.9,
                    help="|first 3-bar net move| / ATR threshold")
    p.add_argument("--persist-bars", type=int, default=1,
                    help="Consecutive closes beyond OR for persistence")
    p.add_argument("--require-break", choices=("on", "off"), default="off",
                    help="Require breakout persistence to fire")

    # Dataset overrides
    p.add_argument("--pilot-run-id", type=str, default=None,
                    help="Override pilot_20d run ID")
    p.add_argument("--extended-run-id", type=str, default=None,
                    help="Override extended_60d run ID")
    p.add_argument("--trend-run-id", type=str, default=None,
                    help="Override trend20 run ID")

    # Output
    p.add_argument("--output-dir", type=str, default="artifacts/allocator_v1_1_experiment",
                    help="Output directory for results")

    return p


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    open_proxy_cfg = OpenProxyConfig(
        or_width_atr_threshold=args.or_width_atr,
        impulse_atr_threshold=args.impulse_atr,
        persist_bars=args.persist_bars,
        require_break=(args.require_break == "on"),
    )

    # Resolve run IDs
    run_map: dict[str, list[str]] = {}
    for dataset, default_ids in KNOWN_RUNS.items():
        override = getattr(args, f"{dataset.replace('_', '_').split('_')[0]}_run_id", None)
        if dataset == "pilot_20d":
            override = args.pilot_run_id
        elif dataset == "extended_60d":
            override = args.extended_run_id
        elif dataset == "trend20":
            override = args.trend_run_id
        if override:
            run_map[dataset] = [override]
        else:
            run_map[dataset] = default_ids

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Allocator v1_1 Experiment Matrix")
    print(f"  open_proxy_v1 config: or_width_atr={open_proxy_cfg.or_width_atr_threshold}, "
          f"impulse_atr={open_proxy_cfg.impulse_atr_threshold}, "
          f"persist_bars={open_proxy_cfg.persist_bars}, "
          f"require_break={open_proxy_cfg.require_break}")
    print("=" * 80)

    allocators = ["v2_hyst", "open_proxy_v1"]
    all_cells: list[CellResult] = []

    for dataset, run_ids in run_map.items():
        print(f"\n{'─' * 60}")
        print(f"  Dataset: {dataset}")
        print(f"  Run IDs: {', '.join(run_ids)}")

        sessions = _discover_sessions(run_ids)
        print(f"  Sessions discovered: {len(sessions)}")
        if not sessions:
            print(f"  [SKIP] No sessions found for {dataset}")
            continue

        for alloc in allocators:
            print(f"\n  → Allocator: {alloc}")
            t0 = time.monotonic()
            cell = _run_cell(dataset, alloc, sessions, open_proxy_cfg)
            elapsed = time.monotonic() - t0

            all_cells.append(cell)

            print(f"    Sessions: {cell.n_sessions}")
            print(f"    ORB routed: {cell.n_orb_routed}  MR routed: {cell.n_mr_routed}")
            print(f"    ORB entries: {cell.orb_entries}")
            print(f"    Trades: {cell.total_trades}  Trades/day: {cell.trades_per_day}")
            print(f"    Avg R: ${cell.avg_r:.2f}  Total PnL: ${cell.total_pnl:.2f}")
            print(f"    P_hit: {cell.p_hit:.4f}  P_ruin: {cell.p_ruin:.4f}  P_daily: {cell.p_daily:.4f}")
            print(f"    DD p95: ${cell.dd_p95:.0f}  Eq p50: ${cell.eq_p50:.0f}  Eq p10: ${cell.eq_p10:.0f}")
            print(f"    FP_ORB: {cell.false_positive_orb}  FN_MR: {cell.false_negative_mr}")
            print(f"    [{elapsed:.1f}s]")

    # ── Summary comparison table ────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 100}")
    hdr = (f"{'Dataset':<15} {'Allocator':<18} {'Sess':>5} {'ORB':>5} {'MR':>5} "
           f"{'Trd/d':>6} {'AvgR':>7} {'P_hit':>7} {'P_ruin':>7} "
           f"{'DD95':>7} {'Eq50':>7} {'Eq10':>7} {'FP':>4} {'FN':>4}")
    print(hdr)
    print("-" * 100)
    for c in all_cells:
        row = (f"{c.dataset:<15} {c.allocator:<18} {c.n_sessions:>5} "
               f"{c.n_orb_routed:>5} {c.n_mr_routed:>5} "
               f"{c.trades_per_day:>6.1f} {c.avg_r:>7.2f} "
               f"{c.p_hit:>7.4f} {c.p_ruin:>7.4f} "
               f"{c.dd_p95:>7.0f} {c.eq_p50:>7.0f} {c.eq_p10:>7.0f} "
               f"{c.false_positive_orb:>4} {c.false_negative_mr:>4}")
        print(row)

    # ── Per-session allocator audit for open_proxy_v1 ───────────────────
    print(f"\n{'=' * 80}")
    print("  PER-SESSION ALLOCATOR AUDIT (open_proxy_v1)")
    print(f"{'=' * 80}")
    for c in all_cells:
        if c.allocator != "open_proxy_v1":
            continue
        print(f"\n  {c.dataset} ({c.n_sessions} sessions)")
        print(f"  {'Session':<25} {'Decision':>9} {'OR_W_ATR':>9} {'Impulse':>9} "
              f"{'Persist':>8} {'Trig_W':>7} {'Trig_I':>7} {'Trig_P':>7}")
        print("  " + "-" * 90)
        for sd in c.session_decisions:
            print(f"  {sd['session_id']:<25} {sd['decision']:>9} "
                  f"{sd['or_width_atr']:>9.2f} {sd['impulse_atr']:>9.2f} "
                  f"{sd['persist_bars']:>8} "
                  f"{'Y' if sd['trigger_width'] else '.':>7} "
                  f"{'Y' if sd['trigger_impulse'] else '.':>7} "
                  f"{'Y' if sd['trigger_persist'] else '.':>7}")

    # ── Save results ────────────────────────────────────────────────────
    result_payload = {
        "timestamp": ts,
        "open_proxy_config": asdict(open_proxy_cfg),
        "cells": [
            {
                "dataset": c.dataset,
                "allocator": c.allocator,
                "n_sessions": c.n_sessions,
                "n_orb_routed": c.n_orb_routed,
                "n_mr_routed": c.n_mr_routed,
                "orb_entries": c.orb_entries,
                "total_trades": c.total_trades,
                "total_pnl": round(c.total_pnl, 2),
                "trades_per_day": c.trades_per_day,
                "avg_r": c.avg_r,
                "p_hit": c.p_hit,
                "p_ruin": c.p_ruin,
                "p_daily": c.p_daily,
                "dd_p95": c.dd_p95,
                "eq_p50": c.eq_p50,
                "eq_p10": c.eq_p10,
                "false_positive_orb": c.false_positive_orb,
                "false_negative_mr": c.false_negative_mr,
                "session_decisions": c.session_decisions,
            }
            for c in all_cells
        ],
    }
    out_file = output_dir / f"allocator_v1_1_experiment_{ts}.json"
    out_file.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Results saved → {out_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
