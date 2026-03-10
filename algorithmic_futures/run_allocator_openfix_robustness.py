"""Robustness runner for allocator-openfix candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from validation.candidate_openfix import (
    ensure_report_dir,
    load_run_dir,
    pass_fail,
    run_pack_for_preset,
    run_robustness_from_run,
    write_csv,
    write_markdown,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run robustness checks for allocator-openfix candidate.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preset", default="mainline_combine_v1_1_allocator_openfix")
    parser.add_argument("--pack", default="extended_60d")
    parser.add_argument("--artifacts-root", default="artifacts/validation_runs")
    parser.add_argument("--output-root", default="artifacts/candidate_reports")
    parser.add_argument("--seeds", default="11,29,42")
    parser.add_argument("--slippage-ticks", default="0,1,2")
    args = parser.parse_args()

    run_dir = load_run_dir(args.run_id) if args.run_id else run_pack_for_preset(args.pack, args.preset, args.artifacts_root)
    seeds = _parse_int_list(args.seeds)
    slippage_ticks = _parse_int_list(args.slippage_ticks)
    rows = run_robustness_from_run(run_dir, seeds, slippage_ticks)
    report_dir = ensure_report_dir("allocator_openfix_robustness", Path(args.output_root))
    write_csv(report_dir / "robustness.csv", rows)

    pass_rows = [
        r for r in rows
        if r["p_ruin"] < config.MC_RUIN_THRESHOLD and r["dd_p95"] < config.MC_DRAWDOWN_P95_MAX
    ]
    overall_pass = len(pass_rows) == len(rows)
    worst_target = min((r["p_target_before_ruin"] for r in rows), default=0.0)
    worst_ruin = max((r["p_ruin"] for r in rows), default=1.0)
    worst_dd = max((r["dd_p95"] for r in rows), default=0.0)

    write_markdown(
        report_dir / "summary.md",
        [
            "# Allocator Openfix Robustness",
            "",
            f"- Run: {Path(run_dir).name}",
            f"- Overall verdict: {pass_fail(overall_pass)}",
            f"- Worst P(target): {worst_target:.4f}",
            f"- Worst P(ruin): {worst_ruin:.4f}",
            f"- Worst DD p95: ${worst_dd:,.2f}",
            "",
            "## Gates",
            "",
            f"- P(ruin) < {config.MC_RUIN_THRESHOLD:.2f}: {pass_fail(worst_ruin < config.MC_RUIN_THRESHOLD)}",
            f"- DD p95 < ${config.MC_DRAWDOWN_P95_MAX:,.0f}: {pass_fail(worst_dd < config.MC_DRAWDOWN_P95_MAX)}",
            "",
            "## Notes",
            "",
            "- Scenarios use seed variation plus slippage sensitivity on the realized scaled trade stream.",
            "- Missed-fill and delayed-fill stress are out of scope unless the simulator gains native execution-latency hooks.",
        ],
    )
    print(f"Robustness outputs written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
