#!/usr/bin/env python3
"""
run_trend_orb_viability.py — ORB viability test on trend-only sessions.

Tests whether ORB signals have convex positive expectancy on the ADX upper-
tertile sessions (the 20 sessions classified as "trend" by the stratified
robustness analysis).

Reports:
  - trades / session
  - win rate
  - avg_r, median_r
  - p10_r, p90_r  (tail distribution)
  - total PnL ($)

Usage:
    python run_trend_orb_viability.py
    python run_trend_orb_viability.py --orb-trigger-mode break
    python run_trend_orb_viability.py --orb-trigger-mode either
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


# ── Trend session IDs ──────────────────────────────────────────────────
# These 20 sessions are the ADX upper-tertile from the stratified robustness
# analysis (sourced from trend20_adx_20260226_232222 manifest).
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
    """Build a ValidationPack containing only trend-tertile sessions."""
    from validation.validation_pack import SessionEntry, ValidationPack, load_pack

    extended = load_pack("extended_60d")
    by_id = {s.session_id: s for s in extended.sessions}

    entries = []
    missing = []
    for sid in TREND_SESSION_IDS:
        s = by_id.get(sid)
        if s is None:
            missing.append(sid)
            continue
        entries.append(
            SessionEntry(
                session_id=s.session_id,
                start=s.start,
                end=s.end,
                category="trend",
                symbol=s.symbol,
                tags=list(s.tags),
                notes=s.notes,
            )
        )

    if missing:
        print(f"  WARNING: {len(missing)} trend sessions not found in extended_60d: {missing}")

    return ValidationPack(
        pack_id="trend20_orb_viability",
        description=f"Trend-tertile ORB viability test ({len(entries)} sessions)",
        sessions=sorted(entries, key=lambda s: s.start),
    )


def _extract_orb_metrics(run_dir: Path) -> dict[str, Any]:
    """Extract ORB-specific trade metrics from a completed run."""
    agg_csv = run_dir / "aggregate_trades.csv"
    if not agg_csv.is_file():
        return {"error": "no aggregate_trades.csv"}

    trades = pd.read_csv(agg_csv)
    if trades.empty:
        return {"error": "empty trades"}

    trades["session_id"] = trades["session_id"].astype(str)
    trades["signal_ts"] = pd.to_datetime(
        trades["signal_timestamp"].astype(str).str.replace(r"\+00:00$", "", regex=True),
        utc=True,
        errors="coerce",
    )
    trades["side"] = trades["side"].astype(str).str.upper()

    # Join with signals.csv to get signal_type
    sig_rows: list[dict[str, Any]] = []
    sessions_dir = run_dir / "sessions"
    for sig_csv in sorted(sessions_dir.glob("*/signals.csv")):
        session_id = sig_csv.parent.name
        with sig_csv.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                approved = str(row.get("approved", "")).strip().lower() == "true"
                if not approved:
                    continue
                sig_type = str(row.get("signal_type", "")).strip().upper()
                if sig_type not in {"MR", "ORB"}:
                    continue
                raw_ts = str(row.get("timestamp", "")).replace("+00:00", "")
                sig_ts = pd.Timestamp(raw_ts) if raw_ts else pd.NaT
                if pd.notna(sig_ts):
                    if sig_ts.tzinfo is None:
                        sig_ts = sig_ts.tz_localize("UTC")
                    else:
                        sig_ts = sig_ts.tz_convert("UTC")
                sig_rows.append({
                    "session_id": session_id,
                    "signal_ts": sig_ts,
                    "side": str(row.get("side", "")).strip().upper(),
                    "signal_type": sig_type,
                })

    sig_df = pd.DataFrame(sig_rows)
    if sig_df.empty:
        trades["signal_type"] = "UNKNOWN"
    else:
        merged = trades.merge(sig_df, on=["session_id", "signal_ts", "side"], how="left")
        merged["signal_type"] = merged["signal_type"].fillna("UNKNOWN")
        trades = merged

    # Filter to ORB trades only
    orb = trades[trades["signal_type"] == "ORB"].copy()
    mr = trades[trades["signal_type"] == "MR"].copy()
    total = trades.copy()

    # Count sessions
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        total_sessions = len(manifest.get("sessions", []))
    else:
        total_sessions = trades["session_id"].nunique()

    def _stats(df: pd.DataFrame, label: str) -> dict[str, Any]:
        if df.empty:
            return {
                "label": label,
                "trade_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate": 0.0,
                "avg_r": 0.0,
                "median_r": 0.0,
                "p10_r": 0.0,
                "p90_r": 0.0,
                "avg_pnl_dollars": 0.0,
                "total_pnl_dollars": 0.0,
                "trades_per_session": 0.0,
                "sessions_with_trades": 0,
            }
        r_vals = df["pnl_r"].dropna().astype(float)
        pnl_vals = df["pnl_dollars"].dropna().astype(float) if "pnl_dollars" in df.columns else pd.Series(dtype=float)
        wins = (r_vals > 0).sum()
        losses = (r_vals <= 0).sum()
        return {
            "label": label,
            "trade_count": int(len(df)),
            "win_count": int(wins),
            "loss_count": int(losses),
            "win_rate": round(100.0 * wins / len(r_vals), 2) if len(r_vals) > 0 else 0.0,
            "avg_r": round(float(r_vals.mean()), 4) if len(r_vals) > 0 else 0.0,
            "median_r": round(float(r_vals.median()), 4) if len(r_vals) > 0 else 0.0,
            "p10_r": round(float(np.percentile(r_vals, 10)), 4) if len(r_vals) > 0 else 0.0,
            "p90_r": round(float(np.percentile(r_vals, 90)), 4) if len(r_vals) > 0 else 0.0,
            "avg_pnl_dollars": round(float(pnl_vals.mean()), 2) if len(pnl_vals) > 0 else 0.0,
            "total_pnl_dollars": round(float(pnl_vals.sum()), 2) if len(pnl_vals) > 0 else 0.0,
            "trades_per_session": round(len(df) / total_sessions, 2) if total_sessions > 0 else 0.0,
            "sessions_with_trades": int(df["session_id"].nunique()),
        }

    return {
        "total_sessions": total_sessions,
        "orb": _stats(orb, "ORB"),
        "mr": _stats(mr, "MR"),
        "all": _stats(total, "ALL"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ORB viability test on trend-only sessions")
    parser.add_argument("--orb-trigger-mode", choices=("break", "pullback", "either"), default="break")
    parser.add_argument("--mr-reclaim-mode", default="off")
    parser.add_argument("--mr-regime-enabled", choices=("on", "off"), default="on")
    parser.add_argument("--artifacts-root", default="artifacts/validation_runs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("\n" + "═" * 70)
    print("  ORB Viability Test — Trend-Only Sessions (ADX upper tertile)")
    print("═" * 70)
    print(f"  Sessions       : {len(TREND_SESSION_IDS)}")
    print(f"  ORB trigger    : {args.orb_trigger_mode}")
    print(f"  MR reclaim     : {args.mr_reclaim_mode}")
    print(f"  MR regime gate : {args.mr_regime_enabled}")
    print()

    pack = _build_trend_pack()
    print(f"  Pack built: {pack.pack_id} ({len(pack.sessions)} sessions)")

    if args.dry_run:
        for i, s in enumerate(pack.sessions, 1):
            print(f"    [{i:>3}] {s.session_id:<30} {s.start}")
        return 0

    from validation.validation_pack import ValidationPackRunner

    runner = ValidationPackRunner(
        pack,
        artifacts_root=args.artifacts_root,
        continue_on_error=True,
        mr_reclaim_mode=args.mr_reclaim_mode,
        mr_regime_enabled=(args.mr_regime_enabled == "on"),
        engine_mode="both",
        allocator_policy="none",
        orb_enabled=True,
        orb_trigger_mode=args.orb_trigger_mode,
    )

    print(f"\n  Running {len(pack.sessions)} sessions (engine=both, ORB={args.orb_trigger_mode})...")
    manifest = runner.run()

    run_dir = Path(args.artifacts_root) / manifest.run_id
    print(f"  Run complete: {manifest.run_id}")
    print(f"  Runtime: {manifest.total_runtime_seconds:.1f}s")

    passed = sum(1 for s in manifest.sessions if s.success)
    failed = len(manifest.sessions) - passed
    print(f"  Sessions: {passed} passed, {failed} failed")

    if failed:
        for s in manifest.sessions:
            if not s.success:
                print(f"    FAIL: {s.session_id}: {s.error_message}")

    # ── Extract metrics ────────────────────────────────────────────────
    metrics = _extract_orb_metrics(run_dir)

    if "error" in metrics:
        print(f"\n  ERROR extracting metrics: {metrics['error']}")
        return 1

    # ── Report ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  RESULTS — Trend-Only Sessions")
    print(f"{'═' * 70}")
    print(f"  Total sessions: {metrics['total_sessions']}")

    for key in ("orb", "mr", "all"):
        s = metrics[key]
        print(f"\n  ── {s['label']} ──────────────────────────────────────")
        print(f"    Trades         : {s['trade_count']}  ({s['win_count']}W / {s['loss_count']}L)")
        print(f"    Win rate       : {s['win_rate']}%")
        print(f"    avg_r          : {s['avg_r']:+.4f}")
        print(f"    median_r       : {s['median_r']:+.4f}")
        print(f"    p10_r          : {s['p10_r']:+.4f}   (left tail)")
        print(f"    p90_r          : {s['p90_r']:+.4f}   (right tail)")
        print(f"    Total PnL      : ${s['total_pnl_dollars']:+.2f}")
        print(f"    Avg PnL/trade  : ${s['avg_pnl_dollars']:+.2f}")
        print(f"    Trades/session : {s['trades_per_session']}")
        print(f"    Sessions w/trades: {s['sessions_with_trades']}/{metrics['total_sessions']}")

    # ── Convexity check ────────────────────────────────────────────────
    orb = metrics["orb"]
    print(f"\n{'═' * 70}")
    print("  CONVEXITY ASSESSMENT")
    print(f"{'═' * 70}")

    if orb["trade_count"] == 0:
        print("  ❌ Zero ORB trades — no signal at all")
    else:
        convex = orb["p90_r"] > abs(orb["p10_r"])
        positive_expectancy = orb["avg_r"] > 0
        adequate_wr = orb["win_rate"] >= 35.0

        status = "✅" if (convex and positive_expectancy) else "❌"
        print(f"  Positive avg_r   : {'YES' if positive_expectancy else 'NO'} ({orb['avg_r']:+.4f})")
        print(f"  Convex tails     : {'YES' if convex else 'NO'} (p90={orb['p90_r']:+.4f} vs |p10|={abs(orb['p10_r']):.4f})")
        print(f"  Win rate ≥ 35%   : {'YES' if adequate_wr else 'NO'} ({orb['win_rate']}%)")
        print(f"\n  {status} ORB {'VIABLE' if (convex and positive_expectancy) else 'NOT VIABLE'} on trend sessions")

    # ── Save results ───────────────────────────────────────────────────
    results = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": {
            "orb_trigger_mode": args.orb_trigger_mode,
            "mr_reclaim_mode": args.mr_reclaim_mode,
            "mr_regime_enabled": args.mr_regime_enabled,
            "session_count": len(TREND_SESSION_IDS),
            "session_ids": TREND_SESSION_IDS,
        },
        "run_id": manifest.run_id,
        "metrics": metrics,
    }
    out_path = Path(args.artifacts_root) / f"trend_orb_viability_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Wrote: {out_path}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
