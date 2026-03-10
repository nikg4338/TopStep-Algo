"""Generate descriptive good-vs-bad ORB autopsy report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from validation.candidate_openfix import DEFAULT_REPORT_ROOT, ensure_report_dir, write_csv, write_markdown
from validation.orb_autopsy import (
    candidate_failure_hypotheses,
    grouped_categorical_summary,
    grouped_numeric_summary,
    strongest_numeric_discriminators,
    summarize_orb_dataset,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ORB autopsy comparison report from dataset CSV.")
    parser.add_argument("--dataset", required=True, help="Path to orb_autopsy_dataset.csv")
    parser.add_argument("--output-root", default=str(DEFAULT_REPORT_ROOT))
    args = parser.parse_args()

    rows = _read_csv(Path(args.dataset))
    summary = summarize_orb_dataset(rows)
    numeric_summary = grouped_numeric_summary(rows)
    categorical_summary = grouped_categorical_summary(rows)
    discriminators = strongest_numeric_discriminators(rows)
    hypotheses = candidate_failure_hypotheses(rows)
    report_dir = ensure_report_dir("orb_autopsy_report", Path(args.output_root))
    write_csv(report_dir / "numeric_summary.csv", numeric_summary)
    write_csv(report_dir / "categorical_summary.csv", categorical_summary)
    write_csv(report_dir / "discriminators.csv", discriminators)
    lines = [
        "# ORB Autopsy Report",
        "",
        "## Dataset composition",
        f"- ORB rows: {summary['rows']}",
        f"- Labels: {summary['label_counts']}",
        f"- Windows: {summary['window_counts']}",
        f"- False-positive ORB rows: {summary['false_positive_count']} ({summary['false_positive_rate']:.2%})",
        "",
        "## Good vs bad ORB feature summaries",
    ]
    for row in discriminators[:5]:
        lines.append(
            f"- {row['feature']}: good median={row['good_median']}, bad median={row['bad_median']}, gap={row['median_gap']}"
        )
    lines.extend([
        "",
        "## False-positive ORB summaries",
        f"- Baseline-relative labels available for {summary['baseline_available_count']} row(s)",
        f"- False-positive ORB count: {summary['false_positive_count']}",
        "",
        "## Candidate failure hypotheses",
    ])
    lines.extend([f"- {item}" for item in hypotheses])
    lines.extend([
        "",
        "## Recommended next research filter ideas",
        "- Test one targeted selectivity rule in a new candidate branch rather than broad threshold retuning.",
        "- Prioritize a skip/filter that rejects early expansion without follow-through in older rotational or gap-fade regimes.",
        "- Compare any future filter only against the frozen allocator-openfix baseline and the frozen mainline baseline.",
    ])
    write_markdown(report_dir / "summary.md", lines)
    print(
        f"ORB autopsy report: rows={summary['rows']} | false_positive={summary['false_positive_count']} | "
        f"top_feature={(discriminators[0]['feature'] if discriminators else 'n/a')}"
    )
    print(f"ORB autopsy report written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())