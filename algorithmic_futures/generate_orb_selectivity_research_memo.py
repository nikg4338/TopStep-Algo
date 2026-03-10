"""Generate a research handoff memo for the next ORB selectivity experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from validation.candidate_openfix import DEFAULT_REPORT_ROOT, ensure_report_dir, write_markdown
from validation.orb_autopsy import candidate_failure_hypotheses, strongest_numeric_discriminators, summarize_orb_dataset


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ORB selectivity research memo from autopsy dataset.")
    parser.add_argument("--dataset", required=True, help="Path to orb_autopsy_dataset.csv")
    parser.add_argument("--output-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--next-experiment-name", default="mainline_combine_v1_2_orb_selectivity")
    args = parser.parse_args()

    rows = _read_csv(Path(args.dataset))
    summary = summarize_orb_dataset(rows)
    discriminators = strongest_numeric_discriminators(rows)
    hypotheses = candidate_failure_hypotheses(rows)
    report_dir = ensure_report_dir("orb_selectivity_research_memo", Path(args.output_root))
    top_features = ", ".join(row["feature"] for row in discriminators[:3]) if discriminators else "insufficient data"
    lines = [
        "# ORB Selectivity Research Memo",
        "",
        "This memo is a research handoff only. It does not change frozen strategy behavior.",
        "",
        "## Current read",
        f"- ORB autopsy rows: {summary['rows']}",
        f"- Labels: {summary['label_counts']}",
        f"- False-positive ORB rows: {summary['false_positive_count']} ({summary['false_positive_rate']:.2%})",
        "",
        "## Top recurring characteristics of bad ORB sessions",
    ]
    lines.extend([f"- {item}" for item in hypotheses])
    lines.extend([
        "",
        "## Strongest candidate discriminators",
        f"- Top numeric separators: {top_features}",
        "",
        "## Confidence / caveats",
        "- Confidence is moderate at best until more labeled ORB sessions accumulate across additional windows.",
        "- Findings are descriptive and evidence-driven; no machine learning or retuning was applied here.",
        "- Baseline-relative labels are strongest where baseline counterfactual runs exist; otherwise labels fall back to realized ORB PnL.",
        "",
        "## Suggested next experiment",
        f"- Create `{args.next_experiment_name}` as a new candidate branch that adds one targeted ORB selectivity filter.",
        "- Compare that future branch against both the frozen allocator-openfix baseline and the frozen mainline baseline.",
        "- Do not alter `mainline_combine_v1_1_allocator_openfix` in place.",
    ])
    write_markdown(report_dir / "summary.md", lines)
    print(f"ORB selectivity research memo written to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())