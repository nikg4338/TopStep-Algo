"""Run pairwise conditional expectancy analysis from an existing validation run."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.candidate_openfix import DEFAULT_REPORT_ROOT
from validation.pairwise_edge_analysis import (
    build_trade_level_dataset,
    analyze_pairwise_slices,
    rank_pairwise_rows,
    write_pairwise_edge_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pairwise conditional edge analysis on an existing validation run.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", help="Validation run id under artifacts/validation_runs")
    source.add_argument("--run-dir", help="Absolute or relative path to a validation run directory")
    parser.add_argument("--reporting-min-sample", type=int, default=10)
    parser.add_argument("--candidate-min-sample", type=int, default=20)
    parser.add_argument("--output-root", default=str(DEFAULT_REPORT_ROOT))
    args = parser.parse_args()

    run_source = args.run_dir or args.run_id
    dataset = build_trade_level_dataset(run_source)
    rows, bucket_definitions, mr_proxy = analyze_pairwise_slices(
        dataset,
        reporting_min_sample=args.reporting_min_sample,
        candidate_min_sample=args.candidate_min_sample,
    )
    report_dir = write_pairwise_edge_artifacts(
        dataset,
        rows,
        bucket_definitions,
        run_id=str(run_source),
        reporting_min_sample=args.reporting_min_sample,
        candidate_min_sample=args.candidate_min_sample,
        mr_proxy_feature=mr_proxy,
        output_root=Path(args.output_root),
    )

    orb_top = rank_pairwise_rows(rows, "ORB", reporting_min_sample=args.reporting_min_sample)
    mr_top = rank_pairwise_rows(rows, "MR", reporting_min_sample=args.reporting_min_sample)
    print(
        f"Pairwise edge analysis: rows={len(rows)} | "
        f"ORB_top={(orb_top[0]['feature_1'] + '×' + orb_top[0]['feature_2']) if orb_top else 'n/a'} | "
        f"MR_top={(mr_top[0]['feature_1'] + '×' + mr_top[0]['feature_2']) if mr_top else 'n/a'}"
    )
    print(f"Pairwise edge outputs written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
