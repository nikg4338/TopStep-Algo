"""Audit live-sim style integrity for allocator-openfix validation runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from validation.candidate_openfix import (
    ensure_report_dir,
    evaluate_live_integrity,
    load_run_dir,
    pass_fail,
    run_pack_for_preset,
    write_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate live-sim integrity for allocator-openfix candidate.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preset", default="mainline_combine_v1_1_allocator_openfix")
    parser.add_argument("--pack", default="extended_60d")
    parser.add_argument("--artifacts-root", default="artifacts/validation_runs")
    parser.add_argument("--output-root", default="artifacts/candidate_reports")
    args = parser.parse_args()

    run_dir = load_run_dir(args.run_id) if args.run_id else run_pack_for_preset(args.pack, args.preset, args.artifacts_root)
    live_checks = evaluate_live_integrity(run_dir)
    checks = list(live_checks.items())
    overall = all(flag for _, flag in checks)
    report_dir = ensure_report_dir("allocator_openfix_live_sim", Path(args.output_root))
    write_markdown(
        report_dir / "summary.md",
        [
            "# Allocator Openfix Live-Sim Validation",
            "",
            f"- Run: {Path(run_dir).name}",
            f"- Timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"- Overall: {pass_fail(overall)}",
            "",
            "## Checks",
            "",
            *[f"- {name}: {pass_fail(flag)}" for name, flag in checks],
            "",
            "## Notes",
            "",
            "- No-lookahead audit uses allocator_debug/session summary fields plus a guard that entries do not predate the OR decision cutover.",
            "- EOD flatten audit requires every trade to have an exit timestamp no later than 16:05 ET / 21:05 UTC in current winter-session data.",
        ],
    )
    print(f"Live-sim audit written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
