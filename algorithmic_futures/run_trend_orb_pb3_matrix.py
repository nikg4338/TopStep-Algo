#!/usr/bin/env python3
"""
run_trend_orb_pb3_matrix.py — ORB pullback_v3 parameter sweep on trend sessions.

Runs a grid of (tolerance, max_bars, entry_mode) on the 20 ADX upper-tertile
sessions and reports a compact comparison table.

Usage:
    python run_trend_orb_pb3_matrix.py
    python run_trend_orb_pb3_matrix.py --entry-mode touch_only
    python run_trend_orb_pb3_matrix.py --dry-run
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


# ── Trend session IDs (ADX upper tertile from stratified analysis) ─────
TREND_SESSION_IDS: list[str] = [
    "session_20251208",
    "session_20251212",
    "session_20251216",
    "session_20251217",
    "session_20251223",
    "session_20251224",
    "session_20251231",
    "session_20260105",
    "session_20260106",
    "session_20260112",
    "session_20260113",
    "session_20260120",
    "session_20260121",
    "session_20260128",
    "session_20260202",
    "session_20260203",
    "session_20260204",
    "session_20260209",
    "session_20260212",
    "session_20260218",
]


def _build_trend_pack():
    """Build ValidationPack from trend session IDs."""
    from validation.validation_pack import SessionEntry, ValidationPack, load_pack

    extended = load_pack("extended_60d")
    by_id = {s.session_id: s for s in extended.sessions}

    entries = []
    for sid in TREND_SESSION_IDS:
        s = by_id.get(sid)
        if s is None:
            print(f"  [WARN] session {sid} not in extended_60d, skipping")
            continue
        entries.append(
            SessionEntry(
                session_id=s.session_id,
                start=s.start,
                end=s.end,
                category=s.category,
                symbol=s.symbol,
                tags=list(s.tags),
                notes=s.notes,
            )
        )

    return ValidationPack(
        pack_id="trend20_pb3_matrix",
        description="Trend-tertile sessions for ORB pullback_v3 parameter sweep",
        sessions=entries,
    )


def _extract_metrics(run_dir: Path, n_sessions: int) -> dict[str, Any]:
    """Extract ORB metrics from a completed validation run."""
    # Collect trades + signals across sessions
    sessions_dir = run_dir / "sessions"
    if not sessions_dir.is_dir():
        return {"trades": 0, "error": "no sessions dir"}

    all_trades = []
    orb_diagnostics = []

    for sess_dir in sorted(sessions_dir.iterdir()):
        if not sess_dir.is_dir():
            continue

        # Load trades
        trades_csv = sess_dir / "trades.csv"
        if trades_csv.is_file():
            try:
                df = pd.read_csv(trades_csv)
                if not df.empty:
                    df["session_id"] = sess_dir.name
                    all_trades.append(df)
            except Exception:
                pass

        # Load signals to tag signal_type
        signals_csv = sess_dir / "signals.csv"
        sig_map: dict[tuple, str] = {}
        if signals_csv.is_file():
            try:
                with signals_csv.open(newline="", encoding="utf-8") as fh:
                    rd = csv.DictReader(fh)
                    for row in rd:
                        if str(row.get("approved", "")).strip().lower() != "true":
                            continue
                        sig_type = str(row.get("signal_type", "")).strip().upper()
                        raw_ts = str(row.get("timestamp", ""))
                        side = str(row.get("side", "")).strip().upper()
                        # Normalize timestamp
                        ts_clean = raw_ts.replace("+00:00", "").replace("Z", "")
                        sig_map[(ts_clean, side)] = sig_type
            except Exception:
                pass

        # Load session summary for pb3 diagnostics
        summary_json = sess_dir / "session_summary.json"
        if summary_json.is_file():
            try:
                summary = json.loads(summary_json.read_text(encoding="utf-8"))
                orb_funnel = summary.get("orb_funnel", {})
                pb3_diags = orb_funnel.get("pullback_v3_diagnostics", [])
                for d in pb3_diags:
                    d["session_id"] = sess_dir.name
                    orb_diagnostics.append(d)
            except Exception:
                pass

    if not all_trades:
        return {
            "trades": 0,
            "trades_per_session": 0.0,
            "orb_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "p10_r": 0.0,
            "p90_r": 0.0,
            "pnl": 0.0,
            "breakout_count": 0,
            "pullback_detected_count": 0,
            "entry_count": 0,
            "pb3_diagnostics": orb_diagnostics,
        }

    trades = pd.concat(all_trades, ignore_index=True)

    # Tag signal_type via join
    def _match_sig_type(row):
        ts_raw = str(row.get("signal_timestamp", ""))
        ts_clean = ts_raw.replace("+00:00", "").replace("Z", "")
        side = str(row.get("side", "")).strip().upper()
        return sig_map.get((ts_clean, side), "UNKNOWN")

    # Rebuild sig_map across all sessions
    all_sig_map: dict[tuple, str] = {}
    for sess_dir in sorted(sessions_dir.iterdir()):
        if not sess_dir.is_dir():
            continue
        signals_csv = sess_dir / "signals.csv"
        session_id = sess_dir.name
        if signals_csv.is_file():
            try:
                with signals_csv.open(newline="", encoding="utf-8") as fh:
                    rd = csv.DictReader(fh)
                    for row in rd:
                        if str(row.get("approved", "")).strip().lower() != "true":
                            continue
                        sig_type = str(row.get("signal_type", "")).strip().upper()
                        raw_ts = str(row.get("timestamp", "")).replace("+00:00", "").replace("Z", "")
                        side = str(row.get("side", "")).strip().upper()
                        all_sig_map[(session_id, raw_ts, side)] = sig_type
            except Exception:
                pass

    def _tag_type(row):
        sess = str(row.get("session_id", ""))
        ts = str(row.get("signal_timestamp", "")).replace("+00:00", "").replace("Z", "")
        side = str(row.get("side", "")).strip().upper()
        return all_sig_map.get((sess, ts, side), "UNKNOWN")

    trades["signal_type"] = trades.apply(_tag_type, axis=1)
    orb = trades[trades["signal_type"] == "ORB"]
    mr = trades[trades["signal_type"] == "MR"]
    all_t = trades

    # ORB metrics
    orb_count = len(orb)
    orb_r = orb["pnl_r"].dropna().astype(float) if "pnl_r" in orb.columns else pd.Series(dtype=float)

    # Aggregate diagnostics
    breakout_count = sum(1 for d in orb_diagnostics if not d.get("expired", True) or d.get("pullback_detected", False) or True)
    pullback_detected_count = sum(1 for d in orb_diagnostics if d.get("pullback_detected", False))
    entry_count = sum(1 for d in orb_diagnostics if d.get("entry_triggered", False))

    return {
        "trades": orb_count,
        "trades_per_session": round(orb_count / max(n_sessions, 1), 2),
        "orb_trades": orb_count,
        "mr_trades": len(mr),
        "all_trades": len(all_t),
        "win_rate": round(float((orb_r > 0).mean()) * 100, 1) if len(orb_r) > 0 else 0.0,
        "avg_r": round(float(orb_r.mean()), 4) if len(orb_r) > 0 else 0.0,
        "median_r": round(float(orb_r.median()), 4) if len(orb_r) > 0 else 0.0,
        "p10_r": round(float(np.percentile(orb_r, 10)), 4) if len(orb_r) > 0 else 0.0,
        "p90_r": round(float(np.percentile(orb_r, 90)), 4) if len(orb_r) > 0 else 0.0,
        "pnl": round(float(orb["pnl_dollars"].sum()), 2) if "pnl_dollars" in orb.columns and len(orb) > 0 else 0.0,
        "breakout_count": len(orb_diagnostics),
        "pullback_detected_count": pullback_detected_count,
        "entry_count": entry_count,
        "pb3_diagnostics": orb_diagnostics,
    }


def _run_cell(
    pack,
    *,
    tolerance: float,
    max_bars: int,
    entry_mode: str,
    artifacts_root: str,
    mr_reclaim_mode: str,
    mr_regime_enabled: bool,
) -> dict[str, Any]:
    """Run one cell of the parameter grid."""
    from validation.validation_pack import ValidationPackRunner

    runner = ValidationPackRunner(
        pack,
        artifacts_root=artifacts_root,
        continue_on_error=True,
        mr_reclaim_mode=mr_reclaim_mode,
        mr_regime_enabled=mr_regime_enabled,
        engine_mode="both",
        allocator_policy="none",
        orb_enabled=True,
        orb_trigger_mode="pullback_v3",
        orb_pullback_max_bars=max_bars,
        orb_pullback_tolerance_pts=tolerance,
        orb_pullback_entry_mode=entry_mode,
    )
    manifest = runner.run()
    run_dir = Path(artifacts_root) / manifest.run_id
    metrics = _extract_metrics(run_dir, len(pack.sessions))
    return {
        "run_id": manifest.run_id,
        "tolerance": tolerance,
        "max_bars": max_bars,
        "entry_mode": entry_mode,
        **metrics,
    }


def _run_baseline(
    pack,
    *,
    trigger_mode: str,
    artifacts_root: str,
    mr_reclaim_mode: str,
    mr_regime_enabled: bool,
) -> dict[str, Any]:
    """Run a baseline comparison cell (break or either)."""
    from validation.validation_pack import ValidationPackRunner

    runner = ValidationPackRunner(
        pack,
        artifacts_root=artifacts_root,
        continue_on_error=True,
        mr_reclaim_mode=mr_reclaim_mode,
        mr_regime_enabled=mr_regime_enabled,
        engine_mode="both",
        allocator_policy="none",
        orb_enabled=True,
        orb_trigger_mode=trigger_mode,
    )
    manifest = runner.run()
    run_dir = Path(artifacts_root) / manifest.run_id
    metrics = _extract_metrics(run_dir, len(pack.sessions))
    return {
        "run_id": manifest.run_id,
        "tolerance": 0.0,
        "max_bars": 0,
        "entry_mode": trigger_mode,
        **metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ORB pullback_v3 parameter sweep on trend sessions")
    parser.add_argument("--dry-run", action="store_true", help="Print grid and exit")
    parser.add_argument(
        "--entry-mode",
        choices=("touch_only", "touch_recovery", "both"),
        default="touch_only",
        help="Entry mode(s) to test",
    )
    parser.add_argument(
        "--tolerances",
        type=str,
        default="3.0,5.0,5.4,6.5",
        help="Comma-separated tolerance values",
    )
    parser.add_argument(
        "--max-bars-list",
        type=str,
        default="3,5,7",
        help="Comma-separated max-bars values",
    )
    parser.add_argument("--mr-reclaim-mode", default="off", help="MR reclaim mode")
    parser.add_argument("--mr-regime-enabled", choices=("on", "off"), default="on")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip break/either baselines")
    args = parser.parse_args()

    tolerances = [float(x.strip()) for x in args.tolerances.split(",")]
    max_bars_list = [int(x.strip()) for x in args.max_bars_list.split(",")]
    entry_modes = ["touch_only", "touch_recovery"] if args.entry_mode == "both" else [args.entry_mode]
    mr_regime_enabled = args.mr_regime_enabled == "on"

    # Build grid
    grid: list[dict] = []
    for tol in tolerances:
        for mb in max_bars_list:
            for em in entry_modes:
                grid.append({"tolerance": tol, "max_bars": mb, "entry_mode": em})

    n_cells = len(grid)
    baselines = [] if args.skip_baselines else ["break"]

    print(f"\n{'='*70}")
    print(f"  ORB PULLBACK_V3 PARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"  Tolerances : {tolerances}")
    print(f"  Max bars   : {max_bars_list}")
    print(f"  Entry modes: {entry_modes}")
    print(f"  Grid cells : {n_cells}")
    print(f"  Baselines  : {baselines}")
    print(f"  Sessions   : {len(TREND_SESSION_IDS)}")
    print(f"{'='*70}\n")

    if args.dry_run:
        for i, cell in enumerate(grid, 1):
            print(f"  [{i:2d}] tol={cell['tolerance']} max_bars={cell['max_bars']} entry={cell['entry_mode']}")
        return 0

    # Build pack
    pack = _build_trend_pack()
    artifacts_root = "artifacts/validation_runs"
    results: list[dict[str, Any]] = []

    # Run baselines
    for bl in baselines:
        print(f"\n{'─'*50}")
        print(f"  BASELINE: trigger_mode={bl}")
        print(f"{'─'*50}")
        res = _run_baseline(
            pack,
            trigger_mode=bl,
            artifacts_root=artifacts_root,
            mr_reclaim_mode=args.mr_reclaim_mode,
            mr_regime_enabled=mr_regime_enabled,
        )
        res["trigger_mode"] = bl
        results.append(res)

    # Run grid
    for i, cell in enumerate(grid, 1):
        print(f"\n{'─'*50}")
        print(f"  CELL [{i}/{n_cells}] tol={cell['tolerance']} max_bars={cell['max_bars']} entry={cell['entry_mode']}")
        print(f"{'─'*50}")
        res = _run_cell(
            pack,
            tolerance=cell["tolerance"],
            max_bars=cell["max_bars"],
            entry_mode=cell["entry_mode"],
            artifacts_root=artifacts_root,
            mr_reclaim_mode=args.mr_reclaim_mode,
            mr_regime_enabled=mr_regime_enabled,
        )
        res["trigger_mode"] = "pullback_v3"
        results.append(res)

    # ── Print compact comparison table ──────────────────────────────────
    print(f"\n\n{'='*110}")
    print(f"  ORB PULLBACK_V3 MATRIX RESULTS")
    print(f"{'='*110}")
    header = f"  {'trigger_mode':<14} {'tol':>5} {'bars':>4} {'entry':<16} {'trades':>6} {'t/sess':>6} {'WR%':>6} {'avg_r':>7} {'p10_r':>7} {'p90_r':>7} {'PnL$':>8}"
    print(header)
    print(f"  {'─'*106}")

    for r in results:
        trig = r.get("trigger_mode", "?")
        tol = r.get("tolerance", 0.0)
        mb = r.get("max_bars", 0)
        em = r.get("entry_mode", "?")
        trades = r.get("trades", 0)
        tps = r.get("trades_per_session", 0.0)
        wr = r.get("win_rate", 0.0)
        avg_r = r.get("avg_r", 0.0)
        p10 = r.get("p10_r", 0.0)
        p90 = r.get("p90_r", 0.0)
        pnl = r.get("pnl", 0.0)
        print(
            f"  {trig:<14} {tol:>5.1f} {mb:>4} {em:<16} {trades:>6} {tps:>6.2f} {wr:>5.1f}% {avg_r:>7.4f} {p10:>7.4f} {p90:>7.4f} {pnl:>8.0f}"
        )

    print(f"  {'─'*106}")

    # Print pb3 diagnostics summary
    pb3_results = [r for r in results if r.get("trigger_mode") == "pullback_v3"]
    if pb3_results:
        print(f"\n  PULLBACK_V3 DIAGNOSTICS SUMMARY")
        print(f"  {'tol':>5} {'bars':>4} {'breakouts':>9} {'pb_detect':>9} {'entries':>7}")
        for r in pb3_results:
            tol = r.get("tolerance", 0.0)
            mb = r.get("max_bars", 0)
            bo = r.get("breakout_count", 0)
            pd_ = r.get("pullback_detected_count", 0)
            ec = r.get("entry_count", 0)
            print(f"  {tol:>5.1f} {mb:>4} {bo:>9} {pd_:>9} {ec:>7}")

    # Save results
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(artifacts_root) / f"orb_pb3_matrix_{ts}.json"
    # Strip diagnostics for JSON (too verbose)
    json_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if k != "pb3_diagnostics"}
        json_results.append(r_copy)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "grid": {
            "tolerances": tolerances,
            "max_bars_list": max_bars_list,
            "entry_modes": entry_modes,
        },
        "results": json_results,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Results saved → {out_path}")

    return 0


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    raise SystemExit(main())
