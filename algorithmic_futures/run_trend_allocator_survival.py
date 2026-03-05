#!/usr/bin/env python3
"""
run_trend_allocator_survival.py — Two analyses on the trend-only session pack:

  Part 1: Allocator survival comparison
    Evaluates MR_ONLY, ORB_ONLY, ALLOC_V1_ADX25, ALLOC_V2_HYST on the
    trend20 base run.  Runs MC survival for each.  Reports P_hit, P_ruin,
    dd_p95, equity_p10/p50.

  Part 2: Pullback empirical debug
    For every session, reconstructs the opening range from features_snapshot.csv,
    scans post-OR bars for breakouts, and logs whether price revisited the
    breakout level — how far, how many bars later, and whether continuation
    followed.

Usage:
    python run_trend_allocator_survival.py
    python run_trend_allocator_survival.py --base-run trend20_orb_viability_20260302_011241
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Project imports (must be run from algorithmic_futures/) ──────────
import config
from simulation.mc_survival import MonteCarloSurvivalSimulator


BASE_RUN_ID = "trend20_orb_viability_20260302_011241"
ARTIFACTS_ROOT = Path("artifacts/validation_runs")

TREND_SESSION_IDS = [
    "session_20251208", "session_20251212", "session_20251216", "session_20251217",
    "session_20251223", "session_20251224", "session_20251231", "session_20260105",
    "session_20260106", "session_20260112", "session_20260113", "session_20260120",
    "session_20260121", "session_20260128", "session_20260202", "session_20260203",
    "session_20260204", "session_20260209", "session_20260212", "session_20260218",
]


# ═══════════════════════════════════════════════════════════════════════
#  Part 1 — Allocator Survival Comparison
# ═══════════════════════════════════════════════════════════════════════


def _build_trade_frame(run_dir: Path) -> pd.DataFrame:
    """Build a trade dataframe with signal_type from trades + signals CSVs."""
    agg_csv = run_dir / "aggregate_trades.csv"
    if not agg_csv.is_file():
        return pd.DataFrame()

    trades = pd.read_csv(agg_csv)
    if trades.empty:
        return trades

    trades["session_id"] = trades["session_id"].astype(str)
    trades["signal_ts"] = pd.to_datetime(
        trades["signal_timestamp"].astype(str).str.replace(r"\+00:00$", "", regex=True),
        utc=True, errors="coerce",
    )
    trades["side"] = trades["side"].astype(str).str.upper()

    sig_rows: list[dict[str, Any]] = []
    for sig_csv in sorted((run_dir / "sessions").glob("*/signals.csv")):
        session_id = sig_csv.parent.name
        with sig_csv.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                if str(row.get("approved", "")).strip().lower() != "true":
                    continue
                sig_type = str(row.get("signal_type", "")).strip().upper()
                if sig_type not in {"MR", "ORB"}:
                    continue
                raw_ts = str(row.get("timestamp", "")).replace("+00:00", "")
                sig_ts = pd.Timestamp(raw_ts) if raw_ts else pd.NaT
                if pd.notna(sig_ts):
                    sig_ts = sig_ts.tz_localize("UTC") if sig_ts.tzinfo is None else sig_ts.tz_convert("UTC")
                sig_rows.append({
                    "session_id": session_id,
                    "signal_ts": sig_ts,
                    "side": str(row.get("side", "")).strip().upper(),
                    "signal_type": sig_type,
                })

    sig_df = pd.DataFrame(sig_rows)
    if sig_df.empty:
        trades["signal_type"] = "UNKNOWN"
        return trades

    merged = trades.merge(sig_df, on=["session_id", "signal_ts", "side"], how="left")
    merged["signal_type"] = merged["signal_type"].fillna("UNKNOWN")
    return merged


def _session_early_adx(session_dir: Path, max_bars: int = 12) -> list[float]:
    """First N non-zero ADX values during RTH."""
    feat = session_dir / "features_snapshot.csv"
    if not feat.is_file():
        return []
    try:
        df = pd.read_csv(feat)
    except Exception:
        return []
    if df.empty or "timestamp" not in df.columns or "adx" not in df.columns:
        return []

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    adx = pd.to_numeric(df["adx"], errors="coerce")
    out: list[float] = []
    for t, a in zip(ts, adx):
        if pd.isna(t) or pd.isna(a):
            continue
        et = t.tz_convert(config.TIMEZONE)
        if et.strftime("%H:%M") >= config.RTH_OPEN and float(a) > 0:
            out.append(float(a))
            if len(out) >= max_bars:
                break
    return out


def _allocator_decision(session_dir: Path, kind: str, adx_threshold: float = 25.0) -> str:
    """Decide engine for a session based on allocator policy."""
    if kind == "mr":
        return "mr"
    if kind == "orb":
        return "orb"

    adx_series = _session_early_adx(session_dir, max_bars=12)

    if kind == "v1":
        open_adx = adx_series[0] if adx_series else 0.0
        return "orb" if open_adx >= adx_threshold else "mr"

    if kind == "v2":
        trend_open = any(v >= 25.0 for v in adx_series)
        rising = adx_series[-3:]
        rising_ok = (len(rising) >= 3 and all(v > 20.0 for v in rising)
                     and all(rising[i] < rising[i+1] for i in range(len(rising)-1)))
        range_seq = adx_series[-3:]
        range_ok = len(range_seq) >= 3 and all(v <= 18.0 for v in range_seq)
        if trend_open or rising_ok:
            return "orb"
        if range_ok:
            return "mr"
        return "mr"

    return "mr"


def _filter_trades(trades: pd.DataFrame, run_dir: Path, session_ids: list[str],
                   kind: str, adx_threshold: float = 25.0) -> tuple[pd.DataFrame, dict[str, str]]:
    """Filter trades by allocator policy."""
    day_engine = {}
    for sid in session_ids:
        day_engine[sid] = _allocator_decision(
            run_dir / "sessions" / sid, kind, adx_threshold)

    if trades.empty:
        return trades, day_engine

    if kind == "mr":
        return trades[trades["signal_type"] == "MR"].copy(), day_engine
    if kind == "orb":
        return trades[trades["signal_type"] == "ORB"].copy(), day_engine

    # Allocator modes: keep MR on mr-days, ORB on orb-days
    mask = trades.apply(
        lambda r: (
            (day_engine.get(str(r["session_id"]), "mr") == "mr" and str(r["signal_type"]).upper() == "MR")
            or (day_engine.get(str(r["session_id"]), "mr") == "orb" and str(r["signal_type"]).upper() == "ORB")
        ), axis=1,
    )
    return trades[mask].copy(), day_engine


def _mc_survival(r_values: list[float], session_ids: list[str]) -> dict[str, Any]:
    """Run MC survival and return key metrics."""
    if not r_values:
        return {
            "p_hit": 0.0, "p_ruin": 0.0, "dd_p95": 0.0,
            "equity_p10": 0.0, "equity_p50": 0.0,
            "stress_severe_p_ruin": 0.0,
        }

    sim = MonteCarloSurvivalSimulator()
    results = sim.run_all_scenarios(r_values, seed=42, session_ids=session_ids)
    base = results["base"]
    severe = results.get("severe")

    return {
        "p_hit": float(base.p_target_before_ruin),
        "p_ruin": float(base.p_ruin),
        "dd_p95": float(base.dd_p95),
        "equity_p10": float(base.equity_p10),
        "equity_p50": float(base.equity_p50),
        "stress_severe_p_ruin": float(severe.p_ruin) if severe else 0.0,
    }


def run_allocator_comparison(run_dir: Path) -> dict[str, Any]:
    """Compare allocator modes on the trend-only base run."""
    trades = _build_trade_frame(run_dir)
    session_ids = TREND_SESSION_IDS

    modes = [
        ("MR_ONLY", "mr", 25.0),
        ("ORB_ONLY", "orb", 25.0),
        ("ALLOC_V1_ADX20", "v1", 20.0),
        ("ALLOC_V1_ADX25", "v1", 25.0),
        ("ALLOC_V1_ADX30", "v1", 30.0),
        ("ALLOC_V2_HYST", "v2", 25.0),
        # Also: "no MR on trend" = ORB trades only + zero for MR days
        ("NO_MR_TREND", "orb", 25.0),  # Same as ORB_ONLY for trend pack
    ]

    all_results = {}
    for label, kind, adx_thresh in modes:
        filtered, day_engine = _filter_trades(trades, run_dir, session_ids, kind, adx_thresh)

        r_vals = filtered["pnl_r"].dropna().astype(float).tolist() if not filtered.empty else []
        sids = filtered["session_id"].astype(str).tolist() if not filtered.empty else []
        pnl = filtered["pnl_dollars"].sum() if not filtered.empty and "pnl_dollars" in filtered.columns else 0.0

        n_mr = len(filtered[filtered["signal_type"] == "MR"]) if not filtered.empty else 0
        n_orb = len(filtered[filtered["signal_type"] == "ORB"]) if not filtered.empty else 0
        wins = (filtered["pnl_r"] > 0).sum() if not filtered.empty else 0
        wr = round(100.0 * wins / len(filtered), 1) if not filtered.empty and len(filtered) > 0 else 0.0
        avg_r = round(float(np.mean(r_vals)), 4) if r_vals else 0.0

        # Days allocated to each engine
        orb_days = sum(1 for v in day_engine.values() if v == "orb")
        mr_days = sum(1 for v in day_engine.values() if v == "mr")

        mc = _mc_survival(r_vals, sids)

        all_results[label] = {
            "trades": len(filtered) if not filtered.empty else 0,
            "mr_trades": n_mr,
            "orb_trades": n_orb,
            "win_rate": wr,
            "avg_r": avg_r,
            "total_pnl": round(float(pnl), 2),
            "orb_days": orb_days,
            "mr_days": mr_days,
            **mc,
        }

    return all_results


# ═══════════════════════════════════════════════════════════════════════
#  Part 2 — Pullback Empirical Debug
# ═══════════════════════════════════════════════════════════════════════


def _load_bars_from_ticks(session_id: str) -> pd.DataFrame:
    """Build 5-min OHLCV bars from Databento tick cache for a session date."""
    # Extract date from session_id: session_20260203 → 20260203
    date_str = session_id.replace("session_", "")
    parquet_path = Path(f"data/cache/MES.c.0/trades/{date_str}_143000__{date_str}_210000.parquet")
    if not parquet_path.is_file():
        return pd.DataFrame()

    ticks = pd.read_parquet(parquet_path)
    if ticks.empty:
        return pd.DataFrame()

    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True, errors="coerce")
    ticks = ticks.dropna(subset=["timestamp", "price"])
    ticks = ticks.set_index("timestamp").sort_index()

    # Resample to 5-min bars
    bars = ticks["price"].resample("5min").agg(
        open="first", high="max", low="min", close="last"
    ).dropna()
    bars["volume"] = ticks["size"].resample("5min").sum()
    bars = bars.reset_index()
    bars["et"] = bars["timestamp"].dt.tz_convert(config.TIMEZONE)
    bars["hhmm"] = bars["et"].dt.strftime("%H:%M")
    return bars


def debug_pullbacks_for_session(session_dir: Path, session_id: str) -> dict[str, Any]:
    """Analyze OR breakout and pullback behavior for a single session."""
    bars = _load_bars_from_ticks(session_id)
    if bars.empty:
        return {"session_id": session_id, "error": "no tick data / bar data"}

    # RTH bars only
    rth = bars[bars["hhmm"] >= config.RTH_OPEN].copy()
    if rth.empty:
        return {"session_id": session_id, "error": "no RTH bars"}

    # Build opening range (first ORB_MINUTES / 5 bars)
    orb_bars_count = max(1, config.ORB_MINUTES // 5)
    or_bars = rth.head(orb_bars_count)
    or_high = float(or_bars["high"].max())
    or_low = float(or_bars["low"].min())
    or_range = or_high - or_low

    post_or = rth.iloc[orb_bars_count:].copy()
    if post_or.empty:
        return {
            "session_id": session_id,
            "or_high": or_high,
            "or_low": or_low,
            "or_range": round(or_range, 2),
            "breakouts": [],
            "summary": "no post-OR bars",
        }

    # Get session ADX from features_snapshot
    feat = session_dir / "features_snapshot.csv"
    session_adx = 0.0
    if feat.is_file():
        try:
            fdf = pd.read_csv(feat)
            if "adx" in fdf.columns:
                adx_vals = pd.to_numeric(fdf["adx"], errors="coerce").dropna()
                adx_vals = adx_vals[adx_vals > 0]
                session_adx = float(adx_vals.median()) if len(adx_vals) > 0 else 0.0
        except Exception:
            pass

    # Detect breakouts and track pullback behavior
    breakouts: list[dict] = []
    stale_cutoff = config.ORB_STALE_CUTOFF  # "11:00" ET

    broke_up = False
    broke_down = False

    for break_idx, (idx, bar) in enumerate(post_or.iterrows()):
        hhmm = str(bar["hhmm"])
        if hhmm > stale_cutoff:
            break

        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        bar_ts = bar["timestamp"]

        # Detect upside break
        if not broke_up and close > or_high:
            broke_up = True
            breakout = {
                "direction": "UP",
                "break_bar_ts": str(bar_ts),
                "break_bar_hhmm": hhmm,
                "break_close": round(close, 2),
                "or_high": round(or_high, 2),
                "or_low": round(or_low, 2),
                "overshoot": round(close - or_high, 2),
            }
            # Track pullback: scan subsequent bars
            _track_pullback(post_or, break_idx, breakout, or_high, or_low, "UP")
            breakouts.append(breakout)

        # Detect downside break
        if not broke_down and close < or_low:
            broke_down = True
            breakout = {
                "direction": "DOWN",
                "break_bar_ts": str(bar_ts),
                "break_bar_hhmm": hhmm,
                "break_close": round(close, 2),
                "or_high": round(or_high, 2),
                "or_low": round(or_low, 2),
                "overshoot": round(or_low - close, 2),
            }
            _track_pullback(post_or, break_idx, breakout, or_high, or_low, "DOWN")
            breakouts.append(breakout)

    return {
        "session_id": session_id,
        "or_high": round(or_high, 2),
        "or_low": round(or_low, 2),
        "or_range": round(or_range, 2),
        "session_adx_median": round(session_adx, 2),
        "n_breakouts": len(breakouts),
        "breakouts": breakouts,
    }


def _track_pullback(bars: pd.DataFrame, break_idx: int, breakout: dict,
                    or_high: float, or_low: float, direction: str) -> None:
    """After a break, track what happens: does price revisit the OR level?"""
    subsequent = bars.loc[bars.index > break_idx]
    if subsequent.empty:
        breakout["pullback_occurred"] = False
        breakout["max_adverse_excursion"] = 0.0
        breakout["bars_to_pullback"] = None
        breakout["pullback_depth"] = 0.0
        breakout["continuation_after_pullback"] = None
        breakout["max_continuation"] = 0.0
        breakout["bars_after_break"] = 0
        return

    break_level = or_high if direction == "UP" else or_low

    max_adverse = 0.0
    pullback_bar = None
    pullback_depth = 0.0
    max_continuation = 0.0
    bars_count = 0

    for bar_idx, (_, bar) in enumerate(subsequent.iterrows()):
        bars_count += 1
        close = float(bar["close"])
        low = float(bar["low"])
        high = float(bar["high"])

        if direction == "UP":
            # Adverse = how far price pulls back below break level
            adverse = max(0, break_level - low)
            max_adverse = max(max_adverse, adverse)
            # Did price touch or cross back below break level?
            if low <= break_level and pullback_bar is None:
                pullback_bar = bar_idx + 1  # 1-indexed
                pullback_depth = break_level - low
            # Continuation = how far above break level
            continuation = max(0, high - break_level)
            max_continuation = max(max_continuation, continuation)
        else:  # DOWN
            adverse = max(0, high - break_level)
            max_adverse = max(max_adverse, adverse)
            if high >= break_level and pullback_bar is None:
                pullback_bar = bar_idx + 1
                pullback_depth = high - break_level
            continuation = max(0, break_level - low)
            max_continuation = max(max_continuation, continuation)

    breakout["pullback_occurred"] = pullback_bar is not None
    breakout["max_adverse_excursion"] = round(max_adverse, 2)
    breakout["bars_to_pullback"] = pullback_bar
    breakout["pullback_depth"] = round(pullback_depth, 2)
    breakout["continuation_after_pullback"] = round(max_continuation, 2) if pullback_bar else None
    breakout["max_continuation"] = round(max_continuation, 2)
    breakout["bars_after_break"] = bars_count


def run_pullback_debug(run_dir: Path) -> list[dict]:
    """Run pullback analysis across all trend sessions."""
    results = []
    for sid in TREND_SESSION_IDS:
        session_dir = run_dir / "sessions" / sid
        if not session_dir.is_dir():
            results.append({"session_id": sid, "error": "dir not found"})
            continue
        r = debug_pullbacks_for_session(session_dir, sid)
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="Allocator survival comparison + pullback debug on trend sessions")
    parser.add_argument("--base-run", default=BASE_RUN_ID)
    parser.add_argument("--artifacts-root", default="artifacts/validation_runs")
    parser.add_argument("--skip-survival", action="store_true", help="Skip MC survival (faster, pullback only)")
    parser.add_argument("--skip-pullback", action="store_true", help="Skip pullback debug")
    args = parser.parse_args()

    run_dir = Path(args.artifacts_root) / args.base_run
    if not run_dir.is_dir():
        print(f"ERROR: base run not found: {run_dir}")
        return 1

    output: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "base_run": args.base_run,
    }

    # ── Part 1: Allocator Survival ─────────────────────────────────────
    if not args.skip_survival:
        print("\n" + "═" * 70)
        print("  Part 1: Allocator Survival Comparison (Trend-Only Sessions)")
        print("═" * 70)

        alloc_results = run_allocator_comparison(run_dir)
        output["allocator_comparison"] = alloc_results

        # Print table
        header = f"  {'Mode':<20} {'Trades':>6} {'MR':>4} {'ORB':>4} {'WR%':>6} {'avg_r':>7} {'PnL$':>8} {'P_hit':>6} {'P_ruin':>6} {'dd_p95':>7} {'eq_p10':>8} {'eq_p50':>8} {'ORBd':>4} {'MRd':>4}"
        print(header)
        print("  " + "─" * len(header.strip()))

        for label, m in alloc_results.items():
            print(f"  {label:<20} {m['trades']:>6} {m['mr_trades']:>4} {m['orb_trades']:>4} "
                  f"{m['win_rate']:>5.1f}% {m['avg_r']:>+7.4f} {m['total_pnl']:>+8.1f} "
                  f"{m['p_hit']:>6.3f} {m['p_ruin']:>6.3f} {m['dd_p95']:>7.0f} "
                  f"{m['equity_p10']:>8.0f} {m['equity_p50']:>8.0f} "
                  f"{m['orb_days']:>4} {m['mr_days']:>4}")

        # Key comparison
        mr_only = alloc_results.get("MR_ONLY", {})
        orb_only = alloc_results.get("ORB_ONLY", {})
        v2 = alloc_results.get("ALLOC_V2_HYST", {})

        print(f"\n  ── Key Comparisons ──")
        print(f"  MR→ORB P_hit delta : {orb_only.get('p_hit', 0) - mr_only.get('p_hit', 0):+.4f}")
        print(f"  MR→V2  P_hit delta : {v2.get('p_hit', 0) - mr_only.get('p_hit', 0):+.4f}")
        print(f"  MR→ORB PnL delta   : ${orb_only.get('total_pnl', 0) - mr_only.get('total_pnl', 0):+.1f}")
        print(f"  MR→V2  PnL delta   : ${v2.get('total_pnl', 0) - mr_only.get('total_pnl', 0):+.1f}")
        print(f"  MR P_ruin          : {mr_only.get('p_ruin', 0):.3f}")
        print(f"  ORB P_ruin         : {orb_only.get('p_ruin', 0):.3f}")
        print(f"  V2 P_ruin          : {v2.get('p_ruin', 0):.3f}")

    # ── Part 2: Pullback Debug ─────────────────────────────────────────
    if not args.skip_pullback:
        print(f"\n{'═' * 70}")
        print("  Part 2: Pullback Empirical Debug (Trend-Only Sessions)")
        print("═" * 70)

        pullback_results = run_pullback_debug(run_dir)
        output["pullback_debug"] = pullback_results

        total_breaks = 0
        total_pullbacks = 0
        pullback_depths: list[float] = []
        pullback_bars: list[int] = []
        continuation_after: list[float] = []
        mae_values: list[float] = []
        sessions_with_breaks = 0

        for r in pullback_results:
            if "error" in r:
                print(f"  {r['session_id']}: {r.get('error', 'unknown error')}")
                continue

            n_breaks = r.get("n_breakouts", 0)
            if n_breaks > 0:
                sessions_with_breaks += 1

            for bo in r.get("breakouts", []):
                total_breaks += 1
                mae_values.append(bo.get("max_adverse_excursion", 0))
                if bo.get("pullback_occurred"):
                    total_pullbacks += 1
                    pullback_depths.append(bo.get("pullback_depth", 0))
                    if bo.get("bars_to_pullback") is not None:
                        pullback_bars.append(bo["bars_to_pullback"])
                    if bo.get("continuation_after_pullback") is not None:
                        continuation_after.append(bo["continuation_after_pullback"])

            # Per-session detail
            if n_breaks > 0:
                print(f"\n  {r['session_id']}  (ADX median={r.get('session_adx_median', 0):.1f}, OR: {r['or_low']:.2f}–{r['or_high']:.2f}, range={r['or_range']:.2f})")
                for bo in r["breakouts"]:
                    pb_str = ""
                    if bo.get("pullback_occurred"):
                        pb_str = f"  PULLBACK! depth={bo['pullback_depth']:.2f} after {bo['bars_to_pullback']} bars, continuation={bo.get('continuation_after_pullback', 0):.2f}"
                    else:
                        pb_str = f"  NO pullback (max_adverse={bo['max_adverse_excursion']:.2f})"
                    print(f"    {bo['direction']} break @ {bo['break_bar_hhmm']} close={bo['break_close']:.2f} overshoot={bo['overshoot']:.2f}"
                          f"  MAE={bo['max_adverse_excursion']:.2f}  max_cont={bo['max_continuation']:.2f} bars_after={bo['bars_after_break']}")
                    print(f"      {pb_str}")
            else:
                print(f"  {r['session_id']}  (ADX median={r.get('session_adx_median', 0):.1f}, OR: {r.get('or_low', 0):.2f}–{r.get('or_high', 0):.2f}) — NO breakout before {config.ORB_STALE_CUTOFF}")

        # Aggregate stats
        print(f"\n{'─' * 60}")
        print(f"  PULLBACK SUMMARY")
        print(f"{'─' * 60}")
        print(f"  Sessions with OR breaks : {sessions_with_breaks} / {len(TREND_SESSION_IDS)}")
        print(f"  Total breakouts         : {total_breaks}")
        print(f"  Breakouts with pullback : {total_pullbacks} ({100*total_pullbacks/total_breaks:.0f}%)" if total_breaks else "  Total breakouts: 0")

        if pullback_depths:
            print(f"\n  Pullback depth (pts):")
            print(f"    mean   = {np.mean(pullback_depths):.2f}")
            print(f"    median = {np.median(pullback_depths):.2f}")
            print(f"    max    = {np.max(pullback_depths):.2f}")

        if pullback_bars:
            print(f"  Bars to pullback:")
            print(f"    mean   = {np.mean(pullback_bars):.1f}")
            print(f"    median = {np.median(pullback_bars):.1f}")
            print(f"    max    = {np.max(pullback_bars)}")

        if mae_values:
            print(f"  Max adverse excursion after break (pts):")
            print(f"    mean   = {np.mean(mae_values):.2f}")
            print(f"    median = {np.median(mae_values):.2f}")
            print(f"    max    = {np.max(mae_values):.2f}")

        if continuation_after:
            print(f"  Continuation after pullback (pts):")
            print(f"    mean   = {np.mean(continuation_after):.2f}")
            print(f"    median = {np.median(continuation_after):.2f}")
            print(f"    max    = {np.max(continuation_after):.2f}")

        # Answer the key question
        if total_breaks > 0:
            pb_rate = total_pullbacks / total_breaks
            print(f"\n  ── DIAGNOSIS ──")
            if pb_rate > 0.5:
                print(f"  Pullbacks occur {pb_rate*100:.0f}% of the time after OR breaks.")
                print(f"  Pullback trigger IS viable — needs looser tolerance to capture them.")
                if pullback_depths:
                    p75_depth = np.percentile(pullback_depths, 75)
                    print(f"  Suggested pullback tolerance: within {p75_depth:.1f} pts of OR level (p75)")
            else:
                print(f"  Pullbacks only occur {pb_rate*100:.0f}% of the time after OR breaks.")
                print(f"  Most breaks are clean extensions — pullback trigger will remain low-frequency.")

    # ── Save ───────────────────────────────────────────────────────────
    out_path = Path(args.artifacts_root) / f"trend_allocator_survival_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n  Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
