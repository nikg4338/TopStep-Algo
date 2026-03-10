"""Forward-shadow tracker for unseen sessions using the frozen allocator-openfix candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validation.candidate_openfix import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    build_forward_shadow_rows,
    build_runner_kwargs_from_preset,
    ensure_report_dir,
    load_run_dir,
    summarize_forward_shadow,
    summarize_run,
    write_csv,
    write_markdown,
)
from validation.session_generator import generate_sessions_for_range
from validation.validation_pack import SessionEntry, ValidationPack, ValidationPackRunner


def _load_seen_session_ids(run_ids: list[str], artifacts_root: Path) -> set[str]:
    seen: set[str] = set()
    for run_id in run_ids:
        run_dir = load_run_dir(run_id, artifacts_root)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seen.update(str(s.get("session_id")) for s in manifest.get("sessions", []) if s.get("session_id"))
    return seen


def _build_pack_from_range(start_date: str, end_date: str, *, exclude_session_ids: set[str]) -> ValidationPack:
    raw_sessions = generate_sessions_for_range(start_date, end_date, category="unlabeled")
    entries = [
        SessionEntry(
            session_id=s["session_id"],
            start=s["start"],
            end=s["end"],
            category=s["category"],
            symbol=s["symbol"],
            notes="forward_shadow",
        )
        for s in raw_sessions
        if s["session_id"] not in exclude_session_ids
    ]
    return ValidationPack(
        pack_id=f"forward_shadow_{start_date.replace('-', '')}_{end_date.replace('-', '')}",
        description="Forward-shadow unseen-session tracker",
        sessions=entries,
    )


def _run_custom_pack(pack: ValidationPack, preset: str, artifacts_root: str) -> Path:
    kwargs = build_runner_kwargs_from_preset(preset, artifacts_root)
    runner = ValidationPackRunner(pack, **kwargs)
    manifest = runner.run()
    return Path(artifacts_root) / manifest.run_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Run forward-shadow tracking for unseen sessions.")
    parser.add_argument("--preset", default="mainline_combine_v1_1_allocator_openfix")
    parser.add_argument("--baseline-preset", default=None, help="Optional baseline preset for false-positive inference")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--exclude-run-id", action="append", default=[], help="Prior runs whose session_ids should be excluded as already seen")
    parser.add_argument("--candidate-run-id", default=None, help="Use an existing candidate run instead of replaying")
    parser.add_argument("--baseline-run-id", default=None, help="Use an existing baseline run for false-positive inference")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_REPORT_ROOT))
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    seen_ids = _load_seen_session_ids(args.exclude_run_id, artifacts_root)
    pack = _build_pack_from_range(args.start_date, args.end_date, exclude_session_ids=seen_ids)
    if not pack.sessions:
        raise SystemExit("No unseen sessions remain in the requested date range.")

    candidate_run_dir = load_run_dir(args.candidate_run_id, artifacts_root) if args.candidate_run_id else _run_custom_pack(pack, args.preset, args.artifacts_root)
    baseline_run_dir = None
    if args.baseline_run_id:
        baseline_run_dir = load_run_dir(args.baseline_run_id, artifacts_root)
    elif args.baseline_preset:
        baseline_run_dir = _run_custom_pack(pack, args.baseline_preset, args.artifacts_root)

    candidate_summary = summarize_run(candidate_run_dir)
    candidate_summary["preset_name"] = args.preset
    baseline_summary = summarize_run(baseline_run_dir) if baseline_run_dir else None
    rows = build_forward_shadow_rows(candidate_summary, baseline_summary=baseline_summary)
    summary = summarize_forward_shadow(rows)

    report_dir = ensure_report_dir("allocator_openfix_forward_shadow", Path(args.output_root))
    write_csv(report_dir / "tracker.csv", rows)
    write_markdown(
        report_dir / "summary.md",
        [
            "# Allocator Openfix Forward Shadow",
            "",
            f"- Preset: {args.preset}",
            f"- Candidate run: {Path(candidate_run_dir).name}",
            f"- Baseline run: {Path(baseline_run_dir).name if baseline_run_dir else 'N/A'}",
            f"- Date range: {args.start_date} -> {args.end_date}",
            "",
            "## Summary",
            f"- Sessions processed: {summary['sessions_processed']}",
            f"- ORB-routed sessions: {summary['orb_routed_sessions']}",
            f"- ORB route rate: {summary['orb_route_rate']:.2%}",
            f"- ORB win rate: {summary['orb_win_rate']:.2%}",
            f"- False-positive ORB count: {summary['false_positive_orb_count']}",
            f"- False-positive ORB rate: {summary['false_positive_orb_rate']:.2%}",
            f"- Cumulative equity: ${summary['cumulative_equity']:,.2f}",
            f"- Current drawdown: ${summary['current_drawdown']:,.2f}",
            f"- Progress to target: {summary['progress_to_target']:.2%}",
            f"- Rule breach count: {summary['rule_breach_count']}",
            f"- Tracker verdict: {summary['status']}",
            "",
            "## Interpretation",
            "- stable: compliance clean and route quality acceptable",
            "- watch: no hard violation, but route quality or equity drift needs review",
            "- degrading: rule issues or meaningful deterioration in drawdown / route quality",
        ],
    )
    print(
        f"Forward shadow: sessions={summary['sessions_processed']} | "
        f"ORB={summary['orb_routed_sessions']} ({summary['orb_route_rate']:.1%}) | "
        f"equity=${summary['cumulative_equity']:,.0f} | verdict={summary['status']}"
    )
    print(f"Forward-shadow outputs written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
