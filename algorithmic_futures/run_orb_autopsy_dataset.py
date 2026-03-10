"""Build an ORB autopsy dataset from existing validation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.candidate_openfix import DEFAULT_ARTIFACT_ROOT, DEFAULT_REPORT_ROOT
from validation.orb_autopsy import (
    build_dataset_for_windows,
    parse_window_specs,
    summarize_orb_dataset,
    write_orb_autopsy_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ORB autopsy dataset from existing run artifacts.")
    parser.add_argument("--candidate-window", action="append", required=True, help="WINDOW_LABEL=RUN_ID")
    parser.add_argument("--baseline-window", action="append", default=[], help="WINDOW_LABEL=RUN_ID")
    parser.add_argument("--candidate-preset", default="mainline_combine_v1_1_allocator_openfix")
    parser.add_argument("--baseline-preset", default="mainline_combine_v1")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_REPORT_ROOT))
    args = parser.parse_args()

    specs = parse_window_specs(
        args.candidate_window,
        args.baseline_window,
        preset_name=args.candidate_preset,
        baseline_preset_name=args.baseline_preset,
    )
    rows = build_dataset_for_windows(specs, artifacts_root=Path(args.artifacts_root))
    report_dir = write_orb_autopsy_artifacts(rows, label="orb_autopsy_dataset", output_root=Path(args.output_root))
    summary = summarize_orb_dataset(rows)
    print(
        f"ORB autopsy dataset: rows={summary['rows']} | "
        f"false_positive={summary['false_positive_count']} ({summary['false_positive_rate']:.1%}) | "
        f"labels={summary['label_counts']}"
    )
    print(f"ORB autopsy dataset outputs written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())