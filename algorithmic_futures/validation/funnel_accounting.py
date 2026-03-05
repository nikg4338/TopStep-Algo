"""Per-session funnel accounting report for validation runs.

Usage:
    python -m validation.funnel_accounting --run-dir artifacts/validation_runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _int(d: dict[str, Any], key: str) -> int:
    try:
        return int(d.get(key, 0) or 0)
    except Exception:
        return 0


def _session_row(summary: dict[str, Any]) -> dict[str, Any]:
    gate = (summary.get("gate_funnel") or {})
    drop = gate.get("drop_ledger") or {}
    z_values_tod = gate.get("z_values_time_of_day") or {}

    return {
        "session_id": summary.get("session_id", "unknown"),
        "day": str(summary.get("session_id", "unknown")).split("_")[0],
        "candidate_mode": gate.get("candidate_mode", "unknown"),
        "bars_evaluated": _int(drop, "bars_evaluated") or _int(gate, "bars_evaluated"),
        "eligible_session_bars": _int(drop, "eligible_session_bars") or _int(gate, "eligible_session_bars"),
        "z_cross_events": _int(drop, "z_cross_events") or _int(gate, "z_cross_events") or _int(gate, "z_cross_inside_to_outside"),
        "dedupe_rejects": _int(drop, "dedupe_rejects"),
        "attempt_limit_rejects": _int(drop, "attempt_limit_rejects"),
        "cooldown_rejects": _int(drop, "cooldown_rejects"),
        "in_position_rejects": _int(drop, "in_position_rejects"),
        "regime_rejects": _int(drop, "regime_rejects"),
        "spread_liquidity_rejects": _int(drop, "spread_liquidity_rejects"),
        "candidates_formed": _int(drop, "candidates_formed") or _int(gate, "candidates_total"),
        "orders_submitted": _int(drop, "orders_submitted") or _int(gate, "approved_trades"),
        "fills": _int(drop, "fills") or _int(gate, "approved_trades"),
        "trades": _int(drop, "trades") or _int(gate, "approved_trades"),
        "session_z_min": float((gate.get("session_z_stats") or {}).get("min", 0.0) or 0.0),
        "session_z_p50": float((gate.get("session_z_stats") or {}).get("p50", 0.0) or 0.0),
        "session_z_max": float((gate.get("session_z_stats") or {}).get("max", 0.0) or 0.0),
        "first_eligible_bar_outside": _int(gate, "first_eligible_bar_outside"),
        "z_cross_time_of_day": dict(gate.get("z_cross_time_of_day") or {}),
        "z_values_time_of_day": z_values_tod,
        "cross_body_impulse_abs_values": list(gate.get("cross_body_impulse_abs_values") or []),
        "cross_range_impulse_values": list(gate.get("cross_range_impulse_values") or []),
    }


def _print(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No session_summary.json files found.")
        return

    ledger_keys = [
        "bars_evaluated",
        "eligible_session_bars",
        "z_cross_events",
        "dedupe_rejects",
        "attempt_limit_rejects",
        "cooldown_rejects",
        "in_position_rejects",
        "regime_rejects",
        "spread_liquidity_rejects",
        "candidates_formed",
        "orders_submitted",
        "fills",
        "trades",
    ]

    print("\nThroughput Audit (per session/day)")
    print("=" * 170)
    print(
        "session_id                  mode    bars  elig    z_x dedupe attempt cool in_pos regime spread  cands orders fills trades"
    )
    print("-" * 170)

    totals = {k: 0 for k in ledger_keys}
    body_impulses: list[float] = []
    range_impulses: list[float] = []
    z_cross_hist_total: dict[str, int] = {}
    z_bucket_minmedmax: dict[str, dict[str, float]] = {}
    daily_totals: dict[str, dict[str, int]] = {}
    outside_first_count = 0

    for row in rows:
        day = row["day"]
        if day not in daily_totals:
            daily_totals[day] = {k: 0 for k in ledger_keys}

        for key in ledger_keys:
            val = int(row.get(key, 0) or 0)
            totals[key] += val
            daily_totals[day][key] += val

        outside_first_count += int(row.get("first_eligible_bar_outside", 0) or 0)

        print(
            f"{row['session_id']:<26} {row['candidate_mode']:<6} "
            f"{row['bars_evaluated']:>6} {row['eligible_session_bars']:>5} {row['z_cross_events']:>6} "
            f"{row['dedupe_rejects']:>6} {row['attempt_limit_rejects']:>7} {row['cooldown_rejects']:>4} "
            f"{row['in_position_rejects']:>6} {row['regime_rejects']:>6} {row['spread_liquidity_rejects']:>6} "
            f"{row['candidates_formed']:>6} {row['orders_submitted']:>6} {row['fills']:>5} {row['trades']:>6}"
        )

        body_impulses.extend(float(v) for v in row.get("cross_body_impulse_abs_values", []) if isinstance(v, (int, float)))
        range_impulses.extend(float(v) for v in row.get("cross_range_impulse_values", []) if isinstance(v, (int, float)))

        for bucket, count in (row.get("z_cross_time_of_day") or {}).items():
            z_cross_hist_total[bucket] = z_cross_hist_total.get(bucket, 0) + int(count or 0)
        for bucket, stats in (row.get("z_values_time_of_day") or {}).items():
            z_bucket_minmedmax[bucket] = {
                "min": float(stats.get("min", 0.0) or 0.0),
                "p50": float(stats.get("p50", 0.0) or 0.0),
                "max": float(stats.get("max", 0.0) or 0.0),
            }

    print("-" * 170)
    print(
        f"{'TOTAL':<26} {'-':<6} {totals['bars_evaluated']:>6} {totals['eligible_session_bars']:>5} {totals['z_cross_events']:>6} "
        f"{totals['dedupe_rejects']:>6} {totals['attempt_limit_rejects']:>7} {totals['cooldown_rejects']:>4} "
        f"{totals['in_position_rejects']:>6} {totals['regime_rejects']:>6} {totals['spread_liquidity_rejects']:>6} "
        f"{totals['candidates_formed']:>6} {totals['orders_submitted']:>6} {totals['fills']:>5} {totals['trades']:>6}"
    )
    print("=" * 170)

    print("\nDaily totals")
    print("=" * 170)
    print("day                      bars  elig    z_x dedupe attempt cool in_pos regime spread  cands orders fills trades")
    print("-" * 170)
    for day, day_row in sorted(daily_totals.items()):
        print(
            f"{day:<24} {day_row['bars_evaluated']:>6} {day_row['eligible_session_bars']:>5} {day_row['z_cross_events']:>6} "
            f"{day_row['dedupe_rejects']:>6} {day_row['attempt_limit_rejects']:>7} {day_row['cooldown_rejects']:>4} "
            f"{day_row['in_position_rejects']:>6} {day_row['regime_rejects']:>6} {day_row['spread_liquidity_rejects']:>6} "
            f"{day_row['candidates_formed']:>6} {day_row['orders_submitted']:>6} {day_row['fills']:>5} {day_row['trades']:>6}"
        )
    print("=" * 170)

    eligible_total = max(1, totals["eligible_session_bars"])
    print(
        "first eligible bar already outside rate: "
        f"{outside_first_count}/{len(rows)} ({outside_first_count/len(rows):.2%})"
    )
    print(f"z_cross / eligible_session_bars: {totals['z_cross_events']}/{eligible_total} ({totals['z_cross_events']/eligible_total:.2%})")

    if z_cross_hist_total:
        print("\nTime-of-day z-cross histogram")
        print("=" * 60)
        for bucket, count in sorted(z_cross_hist_total.items()):
            print(f"  {bucket}: {count}")

    if z_bucket_minmedmax:
        print("\nTime-of-day z-value min/p50/max")
        print("=" * 80)
        for bucket, stats in sorted(z_bucket_minmedmax.items()):
            print(
                f"  {bucket}: {stats['min']:.4f} / {stats['p50']:.4f} / {stats['max']:.4f}"
            )
    if body_impulses and range_impulses:
        body_sorted = sorted(body_impulses)
        range_sorted = sorted(range_impulses)

        def _pct(vals: list[float], q: float) -> float:
            if not vals:
                return 0.0
            idx = (q / 100.0) * (len(vals) - 1)
            lo = int(idx)
            hi = min(lo + 1, len(vals) - 1)
            w = idx - lo
            return vals[lo] * (1 - w) + vals[hi] * w

        print("Cross-bar impulse sanity (z-cross bars):")
        print(
            "  body_impulse_abs p50/p75/p90/p95/max = "
            f"{_pct(body_sorted, 50):.4f}/{_pct(body_sorted, 75):.4f}/{_pct(body_sorted, 90):.4f}/{_pct(body_sorted, 95):.4f}/{max(body_sorted):.4f}"
        )
        print(
            "  range_impulse    p50/p75/p90/p95/max = "
            f"{_pct(range_sorted, 50):.4f}/{_pct(range_sorted, 75):.4f}/{_pct(range_sorted, 90):.4f}/{_pct(range_sorted, 95):.4f}/{max(range_sorted):.4f}"
        )
    print("Notes: spread/liquidity and in-position rejects are scaffolded counters and may be zero until those gates are active.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print per-session funnel accounting for a validation run")
    parser.add_argument("--run-dir", required=True, help="Path to validation run dir")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    session_paths = sorted((run_dir / "sessions").glob("*/session_summary.json"))
    rows = [_session_row(_load_json(p)) for p in session_paths]
    _print(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
