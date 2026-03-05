#!/usr/bin/env python3
"""
visualize_forward_dashboard.py — Post-run forward-validation dashboard.

Reads the artifacts from a validation run directory and produces a
concise monitoring report covering the exact metrics needed for
controlled deployment of the mainline_combine_v1 preset.

Usage:
    python visualize_forward_dashboard.py <run_dir>
    python visualize_forward_dashboard.py artifacts/validation_runs/<run_id>

Dashboard metrics:
    1. Pass/fail by day
    2. Realized vs simulated slippage
    3. ORB pullback entry quality
    4. 2c activation frequency
    5. Trail headroom usage
    6. Daily loss headroom usage
    7. Allocator engine distribution
    8. Sizing trigger breakdown
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


# ── Helpers ─────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── Dashboard sections ──────────────────────────────────────────────────

def section_sizing_decisions(sizing_log: list[dict]) -> None:
    """Section 1 & 4 & 5 & 6: Per-day sizing, headroom, and activation."""
    n = len(sizing_log)
    if not n:
        print("  (no sizing data)")
        return

    print(f"\n  {'Day':>3}  {'Session':<24} {'Engine':<6} {'C_start':>7} {'C_final':>7} "
          f"{'Equity':>8} {'Trail_HR':>8} {'Day_HR':>8} {'Trigger':<16} {'Downshift':<14}")
    print(f"  {'─'*3}  {'─'*24} {'─'*6} {'─'*7} {'─'*7} {'─'*8} {'─'*8} {'─'*8} {'─'*16} {'─'*14}")

    for r in sizing_log:
        engine = r.get("allocator_engine", r.get("active_engine", ""))[:6]
        trigger = r.get("v3_upsize_trigger", "")
        ds = r.get("downshift_reason", "")
        equity = _safe_float(r.get("equity_after"))
        trail_hr = _safe_float(r.get("trail_headroom"))
        day_hr = _safe_float(r.get("day_headroom"))
        print(
            f"  {r.get('day_index', 0):>3}  {r.get('session_id', ''):<24} "
            f"{engine:<6} {r.get('contracts_start', 1):>7} {r.get('contracts_final', 1):>7} "
            f"${equity:>7,.0f} ${trail_hr:>7,.0f} ${day_hr:>7,.0f} "
            f"{trigger or '—':<16} {ds or '—':<14}"
        )


def section_activation_summary(sizing_log: list[dict]) -> None:
    """Section 4: 2c activation frequency and trigger breakdown."""
    n = len(sizing_log)
    if not n:
        return

    days_started_2c = sum(1 for r in sizing_log if r.get("contracts_start", 1) == 2)
    days_ever_2c = sum(
        1 for r in sizing_log
        if r.get("contracts_start", 1) == 2 or r.get("v3_upsize_trigger", "") != ""
    )
    orb_sessions = sum(1 for r in sizing_log if r.get("v3_orb_day", False))
    downshifts = sum(1 for r in sizing_log if r.get("downshift_reason", ""))
    profit_locks = sum(1 for r in sizing_log if r.get("profit_lock_triggered", False))

    traction = sum(1 for r in sizing_log if r.get("v3_upsize_trigger") == "traction")
    first_win = sum(1 for r in sizing_log if r.get("v3_upsize_trigger") == "first_trade_win")
    orb_up = sum(1 for r in sizing_log if r.get("v3_upsize_trigger") == "orb_day")

    print(f"\n  ┌─ 2C ACTIVATION ({'dynamic_v3' if any(r.get('v3_upsize_trigger') is not None for r in sizing_log) else 'unknown'}) ─┐")
    print(f"  │ Days started at 2c : {days_started_2c}/{n} ({100*days_started_2c/n:.0f}%)")
    print(f"  │ Days ever at 2c    : {days_ever_2c}/{n} ({100*days_ever_2c/n:.0f}%)")
    print(f"  │ ORB sessions       : {orb_sessions}/{n} ({100*orb_sessions/n:.0f}%)")
    print(f"  │ Downshifts         : {downshifts}")
    print(f"  │ Profit locks       : {profit_locks}")
    print(f"  │ Triggers           : traction={traction}  first_win={first_win}  orb={orb_up}")
    print(f"  └{'─'*44}┘")


def section_headroom_usage(sizing_log: list[dict]) -> None:
    """Section 5 & 6: Trail and daily headroom min/max/mean."""
    n = len(sizing_log)
    if not n:
        return

    trail = [_safe_float(r.get("trail_headroom")) for r in sizing_log]
    day = [_safe_float(r.get("day_headroom")) for r in sizing_log]

    def _stats(vals):
        return min(vals), max(vals), sum(vals) / len(vals)

    t_min, t_max, t_mean = _stats(trail)
    d_min, d_max, d_mean = _stats(day)

    print(f"\n  ┌─ HEADROOM USAGE ─┐")
    print(f"  │ Trail headroom  : min=${t_min:,.0f}  max=${t_max:,.0f}  mean=${t_mean:,.0f}")
    print(f"  │ Daily headroom  : min=${d_min:,.0f}  max=${d_max:,.0f}  mean=${d_mean:,.0f}")

    # Flag danger zones
    if t_min < 400:
        print(f"  │ ⚠ Trail headroom dipped below $400 (MLL proximity)")
    if d_min < 200:
        print(f"  │ ⚠ Daily headroom dipped below $200")
    print(f"  └{'─'*44}┘")


def section_allocator_distribution(sizing_log: list[dict], alloc_decisions: list[dict] | None) -> None:
    """Section 7: Allocator engine distribution."""
    # Try allocator decisions file first, fall back to sizing log
    engines: list[str] = []
    if alloc_decisions:
        engines = [d.get("decision", d.get("engine", "unknown")) for d in alloc_decisions]
    else:
        engines = [r.get("allocator_engine", r.get("active_engine", "unknown")) for r in sizing_log]

    if not engines:
        return

    from collections import Counter
    dist = Counter(engines)
    total = len(engines)

    print(f"\n  ┌─ ALLOCATOR DISTRIBUTION ─┐")
    for eng, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  │ {eng:<12} : {cnt:>3}/{total} ({100*cnt/total:.0f}%)")
    print(f"  └{'─'*30}┘")


def section_orb_quality(run_dir: Path) -> None:
    """Section 3: ORB pullback entry quality from session summaries."""
    sessions_dir = run_dir / "sessions"
    if not sessions_dir.is_dir():
        return

    orb_entries = []
    for sdir in sorted(sessions_dir.iterdir()):
        summary_path = sdir / "session_summary.json"
        if not summary_path.is_file():
            continue
        summary = _load_json(summary_path)
        funnel = summary.get("orb_funnel", {})
        if funnel:
            orb_entries.append({
                "session": sdir.name,
                "breakouts": funnel.get("breakouts_detected", 0),
                "pullbacks": funnel.get("pullbacks_detected", 0),
                "entries": funnel.get("entries_triggered", 0),
                "stale_rejects": funnel.get("stale_rejects", 0),
            })

    if not orb_entries:
        return

    total_breakouts = sum(e["breakouts"] for e in orb_entries)
    total_pullbacks = sum(e["pullbacks"] for e in orb_entries)
    total_entries = sum(e["entries"] for e in orb_entries)
    sessions_with_entry = sum(1 for e in orb_entries if e["entries"] > 0)

    print(f"\n  ┌─ ORB PULLBACK QUALITY ─┐")
    print(f"  │ Sessions with ORB funnel : {len(orb_entries)}")
    print(f"  │ Total breakouts          : {total_breakouts}")
    print(f"  │ Total pullbacks          : {total_pullbacks}")
    print(f"  │ Total entries             : {total_entries}")
    print(f"  │ Sessions with entry       : {sessions_with_entry}/{len(orb_entries)}")
    if total_breakouts:
        print(f"  │ Pullback rate             : {100*total_pullbacks/total_breakouts:.0f}%")
        print(f"  │ Entry conversion          : {100*total_entries/total_breakouts:.0f}%")
    print(f"  └{'─'*35}┘")


def section_pnl_summary(sizing_log: list[dict]) -> None:
    """Section 1: Pass/fail by day (PnL positive/negative)."""
    n = len(sizing_log)
    if not n:
        return

    days_pos = 0
    days_neg = 0
    days_flat = 0
    pnls = []

    for r in sizing_log:
        eb = _safe_float(r.get("equity_before"))
        ea = _safe_float(r.get("equity_after"))
        day_pnl = ea - eb
        pnls.append(day_pnl)
        if day_pnl > 0:
            days_pos += 1
        elif day_pnl < 0:
            days_neg += 1
        else:
            days_flat += 1

    total_pnl = sum(pnls)
    max_day = max(pnls) if pnls else 0
    min_day = min(pnls) if pnls else 0
    mean_day = total_pnl / n if n else 0

    print(f"\n  ┌─ DAILY PnL ─┐")
    print(f"  │ Win days  : {days_pos}/{n} ({100*days_pos/n:.0f}%)")
    print(f"  │ Loss days : {days_neg}/{n}")
    print(f"  │ Flat days : {days_flat}/{n}")
    print(f"  │ Total PnL : ${total_pnl:,.2f}")
    print(f"  │ Mean/day  : ${mean_day:,.2f}")
    print(f"  │ Best day  : ${max_day:,.2f}")
    print(f"  │ Worst day : ${min_day:,.2f}")
    print(f"  └{'─'*20}┘")


def section_preset_check(run_dir: Path) -> None:
    """Verify the sizing config matches the mainline preset."""
    config_path = run_dir / "sizing_config.json"
    if not config_path.is_file():
        print("\n  ⚠ No sizing_config.json found — cannot verify preset")
        return

    cfg = _load_json(config_path)
    expected = {
        "policy": "dynamic_v3",
        "v3_earned_traction": 75.0,
        "v3_giveback_floor": 25.0,
        "v3_orb_upsize_allowed": True,
        "v3_day_headroom_up": 800.0,
        "v3_trail_headroom_up": 1400.0,
    }

    mismatches = []
    for k, v in expected.items():
        actual = cfg.get(k)
        if actual != v:
            mismatches.append(f"{k}: expected={v}, actual={actual}")

    if mismatches:
        print(f"\n  ⚠ PRESET MISMATCH (expected mainline_combine_v1):")
        for m in mismatches:
            print(f"    • {m}")
    else:
        print(f"\n  ✓ Sizing config matches mainline_combine_v1")


# ── Main ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Forward-validation dashboard for mainline_combine_v1"
    )
    parser.add_argument(
        "run_dir",
        help="Path to the validation run directory",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory", file=sys.stderr)
        return 1

    print(f"\n{'═'*70}")
    print(f"  FORWARD-VALIDATION DASHBOARD")
    print(f"  Run: {run_dir.name}")
    print(f"{'═'*70}")

    # Load artifacts
    sizing_log: list[dict] = []
    sizing_log_path = run_dir / "sizing_decisions.json"
    if sizing_log_path.is_file():
        sizing_log = _load_json(sizing_log_path)

    alloc_decisions: list[dict] | None = None
    alloc_path = run_dir / "allocator_decisions.json"
    if alloc_path.is_file():
        alloc_decisions = _load_json(alloc_path)

    # Run all dashboard sections
    section_preset_check(run_dir)
    section_pnl_summary(sizing_log)
    section_sizing_decisions(sizing_log)
    section_activation_summary(sizing_log)
    section_headroom_usage(sizing_log)
    section_allocator_distribution(sizing_log, alloc_decisions)
    section_orb_quality(run_dir)

    print(f"\n{'═'*70}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
