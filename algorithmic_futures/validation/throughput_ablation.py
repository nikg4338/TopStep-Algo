"""Run D0-D5 throughput ablation grid for MR upstream flow audit.

Usage:
    python -m validation.throughput_ablation --pack pilot_20d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validation.validation_pack import ValidationPackRunner, load_pack


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _sum_drop_ledger(run_dir: Path) -> dict[str, int]:
    totals: dict[str, int] = {
        "bars_evaluated": 0,
        "eligible_session_bars": 0,
        "z_cross_events": 0,
        "dedupe_rejects": 0,
        "attempt_limit_rejects": 0,
        "cooldown_rejects": 0,
        "in_position_rejects": 0,
        "regime_rejects": 0,
        "spread_liquidity_rejects": 0,
        "candidates_formed": 0,
        "orders_submitted": 0,
        "fills": 0,
        "trades": 0,
    }
    for p in sorted((run_dir / "sessions").glob("*/session_summary.json")):
        summary = _load_json(p)
        gate = summary.get("gate_funnel") or {}
        drop = gate.get("drop_ledger") or {}
        for key in totals:
            totals[key] += int(drop.get(key, gate.get(key, 0)) or 0)
    return totals


def _sankey_line(drop: dict[str, int]) -> str:
    return (
        f"bars={drop['bars_evaluated']} -> eligible={drop['eligible_session_bars']} -> "
        f"z_cross={drop['z_cross_events']} -> cands={drop['candidates_formed']} -> "
        f"orders={drop['orders_submitted']} -> fills={drop['fills']} -> trades={drop['trades']}"
    )


def run_throughput_ablation(
    *,
    pack_name: str,
    artifacts_root: str,
    continue_on_error: bool,
    mr_sigma_entry: float,
    mr_soft_range_impulse_k: float,
) -> list[tuple[str, str]]:
    pack = load_pack(pack_name)

    grid = [
        ("D0", True, True, 1),
        ("D1", False, True, 1),
        ("D2", True, False, 1),
        ("D3", True, True, 0),
        ("D4", False, True, 0),
        ("D5", False, False, 1),
    ]

    run_dirs: list[tuple[str, str]] = []
    comparison: dict[str, Any] = {}

    for label, dedupe_enabled, attempt_enabled, cooldown_bars in grid:
        runner = ValidationPackRunner(
            pack,
            artifacts_root=artifacts_root,
            continue_on_error=continue_on_error,
            mr_reclaim_mode="off",
            mr_sigma_entry=mr_sigma_entry,
            mr_soft_impulse_k=mr_soft_range_impulse_k,
            mr_dedupe_enabled=dedupe_enabled,
            mr_attempt_cap_enabled=attempt_enabled,
            mr_cooldown_bars=cooldown_bars,
        )
        manifest = runner.run()
        run_dir = Path(artifacts_root) / manifest.run_id
        run_dirs.append((label, str(run_dir)))

        agg = _load_json(run_dir / "aggregate_metrics.json")
        drop = _sum_drop_ledger(run_dir)
        comparison[label] = {
            "run_id": manifest.run_id,
            "run_dir": str(run_dir),
            "controls": {
                "mr_sigma_entry": mr_sigma_entry,
                "mr_reclaim_mode": "off",
                "dedupe_enabled": dedupe_enabled,
                "attempt_cap_enabled": attempt_enabled,
                "cooldown_bars": cooldown_bars,
            },
            "aggregate_metrics": {
                "trade_count_total": int(agg.get("trade_count_total", 0) or 0),
                "trades_per_session_mean": float(agg.get("trades_per_session_mean", 0.0) or 0.0),
                "win_rate": float(agg.get("win_rate", 0.0) or 0.0),
                "avg_r": float(agg.get("avg_r", 0.0) or 0.0),
                "mae_p95": float(agg.get("mae_p95", 0.0) or 0.0),
            },
            "drop_ledger_total": drop,
            "sankey": _sankey_line(drop),
        }

        print(f"[{label}] {manifest.run_id}")
        print(f"  {_sankey_line(drop)}")

    if run_dirs:
        out_dir = Path(run_dirs[-1][1]).parent
        out_path = out_dir / f"{pack_name}_throughput_ablation_summary.json"
        out_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote comparison summary -> {out_path}")

    return run_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D0-D5 throughput ablation grid")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--artifacts-root", default="artifacts/validation_runs")
    parser.add_argument("--no-continue-on-error", action="store_true")
    parser.add_argument("--mr-sigma-entry", type=float, default=1.4)
    parser.add_argument("--mr-soft-range-impulse-k", type=float, default=1.2)
    args = parser.parse_args()

    run_throughput_ablation(
        pack_name=args.pack,
        artifacts_root=args.artifacts_root,
        continue_on_error=not args.no_continue_on_error,
        mr_sigma_entry=float(args.mr_sigma_entry),
        mr_soft_range_impulse_k=float(args.mr_soft_range_impulse_k),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
