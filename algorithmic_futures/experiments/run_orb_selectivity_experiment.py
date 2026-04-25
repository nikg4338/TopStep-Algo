"""
experiments/run_orb_selectivity_experiment.py — ORB selectivity research.

Scores simple filters over the ORB diagnostics dataset to find conditions where
ORB adds positive convexity without adding too much drawdown. This is a
research-only report and does not modify live ORB rules.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DIAGNOSTICS_ROOT = PROJECT_ROOT / "artifacts" / "orb_diagnostics"
DEFAULT_MIN_TRADES = 30
DEFAULT_DOLLARS_PER_R = 100.0
DEFAULT_COMBINE_TARGET_DOLLARS = 3000.0
MAX_PAIRWISE_FILTERS = 80


@dataclass(frozen=True)
class FilterSpec:
    """A named boolean filter over the ORB diagnostics dataframe."""

    name: str
    description: str
    fn: Callable[[pd.DataFrame], pd.Series]
    complexity: int = 1


@dataclass(frozen=True)
class FilterResult:
    """Metrics for one ORB selectivity filter."""

    name: str
    description: str
    complexity: int
    allowed: int
    rejected: int
    win_rate: float
    avg_r: float
    avg_win_r: float
    avg_loss_r: float
    avg_mfe_r: float
    avg_mae_r: float
    convexity_ratio: float
    dd_contribution_r: float
    dd_contribution_dollars: float
    target_contribution_pct: float
    total_r: float
    low_confidence: bool


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def find_latest_diagnostics_csv(root: Path = DEFAULT_DIAGNOSTICS_ROOT) -> Path:
    """Return the newest diagnostics CSV under artifacts/orb_diagnostics."""
    candidates = sorted(root.glob("*/orb_diagnostics.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No orb_diagnostics.csv found under {root}")
    return candidates[-1]


def load_diagnostics(path: Path) -> pd.DataFrame:
    """Load and normalize the ORB diagnostics CSV."""
    df = pd.read_csv(path)
    required = {"final_r", "max_favorable_excursion", "max_adverse_excursion", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Diagnostics CSV missing required columns: {sorted(missing)}")

    for col in (
        "opening_range_width",
        "atr",
        "opening_impulse",
        "one_sidedness_score",
        "pullback_depth",
        "max_favorable_excursion",
        "max_adverse_excursion",
        "final_r",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("atr_regime", "vwap_relationship", "breakout_direction", "label"):
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    return df


def max_drawdown(values: list[float]) -> float:
    """Return max peak-to-trough drawdown for a sequence of R values."""
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return round(max_dd, 4)


def evaluate_filter(
    df: pd.DataFrame,
    spec: FilterSpec,
    *,
    min_trades: int,
    dollars_per_r: float,
    combine_target_dollars: float,
) -> FilterResult:
    """Apply one filter and compute selectivity metrics."""
    mask = spec.fn(df).fillna(False)
    allowed_df = df[mask & df["final_r"].notna()].copy()
    rejected = int(len(df[df["final_r"].notna()]) - len(allowed_df))
    final_r = allowed_df["final_r"].astype(float)
    wins = final_r[final_r > 0]
    losses = final_r[final_r < 0]
    allowed = int(len(allowed_df))
    total_r = float(final_r.sum()) if allowed else 0.0
    dd_r = max_drawdown(final_r.tolist()) if allowed else 0.0
    avg_mfe = safe_float(allowed_df["max_favorable_excursion"].mean()) if allowed else 0.0
    avg_mae = safe_float(allowed_df["max_adverse_excursion"].mean()) if allowed else 0.0
    target_contribution = (
        (total_r * dollars_per_r) / combine_target_dollars
        if combine_target_dollars
        else 0.0
    )

    return FilterResult(
        name=spec.name,
        description=spec.description,
        complexity=spec.complexity,
        allowed=allowed,
        rejected=rejected,
        win_rate=float((final_r > 0).mean()) if allowed else 0.0,
        avg_r=safe_float(final_r.mean()) if allowed else 0.0,
        avg_win_r=safe_float(wins.mean()) if not wins.empty else 0.0,
        avg_loss_r=safe_float(losses.mean()) if not losses.empty else 0.0,
        avg_mfe_r=avg_mfe,
        avg_mae_r=avg_mae,
        convexity_ratio=round(avg_mfe / avg_mae, 4) if avg_mae > 0 else 0.0,
        dd_contribution_r=dd_r,
        dd_contribution_dollars=round(dd_r * dollars_per_r, 2),
        target_contribution_pct=round(target_contribution, 4),
        total_r=round(total_r, 4),
        low_confidence=allowed < min_trades,
    )


def _numeric_quantiles(df: pd.DataFrame, col: str) -> dict[str, float]:
    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "q25": float(values.quantile(0.25)),
        "q50": float(values.quantile(0.50)),
        "q75": float(values.quantile(0.75)),
    }


def build_filter_specs(df: pd.DataFrame) -> list[FilterSpec]:
    """Create simple ORB selectivity filters from available diagnostics columns."""
    specs: list[FilterSpec] = [
        FilterSpec("all_orb_trades", "No selectivity filter; baseline ORB diagnostics", lambda d: d["final_r"].notna()),
    ]

    if "opening_range_width" in df.columns:
        q = _numeric_quantiles(df, "opening_range_width")
        if q:
            specs.extend(
                [
                    FilterSpec(
                        "or_width_le_median",
                        f"Opening range width <= median ({q['q50']:.2f})",
                        lambda d, x=q["q50"]: d["opening_range_width"] <= x,
                    ),
                    FilterSpec(
                        "or_width_between_q25_q75",
                        f"Opening range width between Q25/Q75 ({q['q25']:.2f}-{q['q75']:.2f})",
                        lambda d, lo=q["q25"], hi=q["q75"]: d["opening_range_width"].between(lo, hi),
                    ),
                    FilterSpec(
                        "or_width_ge_median",
                        f"Opening range width >= median ({q['q50']:.2f})",
                        lambda d, x=q["q50"]: d["opening_range_width"] >= x,
                    ),
                ]
            )

    if "atr_regime" in df.columns:
        for regime in sorted(v for v in df["atr_regime"].dropna().unique() if v != "unknown"):
            specs.append(
                FilterSpec(
                    f"atr_regime_{regime}",
                    f"ATR regime is {regime}",
                    lambda d, r=regime: d["atr_regime"] == r,
                )
            )

    if "opening_impulse" in df.columns:
        q = _numeric_quantiles(df.assign(opening_impulse_abs=df["opening_impulse"].abs()), "opening_impulse_abs")
        if q:
            specs.extend(
                [
                    FilterSpec(
                        "abs_impulse_ge_median",
                        f"Absolute opening impulse >= median ({q['q50']:.2f})",
                        lambda d, x=q["q50"]: d["opening_impulse"].abs() >= x,
                    ),
                    FilterSpec(
                        "abs_impulse_ge_q75",
                        f"Absolute opening impulse >= Q75 ({q['q75']:.2f})",
                        lambda d, x=q["q75"]: d["opening_impulse"].abs() >= x,
                    ),
                    FilterSpec(
                        "positive_opening_impulse",
                        "Opening impulse is positive",
                        lambda d: d["opening_impulse"] > 0,
                    ),
                    FilterSpec(
                        "negative_opening_impulse",
                        "Opening impulse is negative",
                        lambda d: d["opening_impulse"] < 0,
                    ),
                ]
            )

    if "one_sidedness_score" in df.columns:
        q = _numeric_quantiles(df, "one_sidedness_score")
        if q:
            specs.extend(
                [
                    FilterSpec(
                        "one_sidedness_ge_median",
                        f"One-sidedness >= median ({q['q50']:.2f})",
                        lambda d, x=q["q50"]: d["one_sidedness_score"] >= x,
                    ),
                    FilterSpec(
                        "one_sidedness_ge_q75",
                        f"One-sidedness >= Q75 ({q['q75']:.2f})",
                        lambda d, x=q["q75"]: d["one_sidedness_score"] >= x,
                    ),
                ]
            )

    if "vwap_relationship" in df.columns:
        for rel in sorted(v for v in df["vwap_relationship"].dropna().unique() if v != "unknown"):
            specs.append(
                FilterSpec(
                    f"vwap_{rel}",
                    f"Entry is {rel.replace('_', ' ')}",
                    lambda d, value=rel: d["vwap_relationship"] == value,
                )
            )
        specs.append(
            FilterSpec(
                "vwap_extension_side",
                "BUY above VWAP or SELL below VWAP",
                lambda d: (
                    ((d["breakout_direction"].str.upper() == "BUY") & (d["vwap_relationship"] == "above_vwap"))
                    | ((d["breakout_direction"].str.upper() == "SELL") & (d["vwap_relationship"] == "below_vwap"))
                ),
            )
        )
        specs.append(
            FilterSpec(
                "vwap_pullback_side",
                "BUY below VWAP or SELL above VWAP",
                lambda d: (
                    ((d["breakout_direction"].str.upper() == "BUY") & (d["vwap_relationship"] == "below_vwap"))
                    | ((d["breakout_direction"].str.upper() == "SELL") & (d["vwap_relationship"] == "above_vwap"))
                ),
            )
        )

    if "pullback_depth" in df.columns:
        q = _numeric_quantiles(df, "pullback_depth")
        if q:
            specs.extend(
                [
                    FilterSpec(
                        "pullback_touch_or_none",
                        "Pullback depth <= 0.25 points",
                        lambda d: d["pullback_depth"].fillna(0.0) <= 0.25,
                    ),
                    FilterSpec(
                        "pullback_depth_le_median",
                        f"Pullback depth <= median ({q['q50']:.2f})",
                        lambda d, x=q["q50"]: d["pullback_depth"].fillna(0.0) <= x,
                    ),
                    FilterSpec(
                        "pullback_depth_le_q75",
                        f"Pullback depth <= Q75 ({q['q75']:.2f})",
                        lambda d, x=q["q75"]: d["pullback_depth"].fillna(0.0) <= x,
                    ),
                ]
            )

    return specs


def build_pairwise_specs(specs: list[FilterSpec]) -> list[FilterSpec]:
    """Build simple two-condition combinations, capped to keep the report readable."""
    simple_specs = [s for s in specs if s.name != "all_orb_trades" and s.complexity == 1]
    pairwise: list[FilterSpec] = []
    for left, right in combinations(simple_specs, 2):
        if len(pairwise) >= MAX_PAIRWISE_FILTERS:
            break
        pairwise.append(
            FilterSpec(
                name=f"{left.name}__and__{right.name}",
                description=f"{left.description}; AND {right.description}",
                fn=lambda d, a=left, b=right: a.fn(d).fillna(False) & b.fn(d).fillna(False),
                complexity=2,
            )
        )
    return pairwise


def rank_results(results: list[FilterResult]) -> list[FilterResult]:
    """Rank filters with a simple preference for expectancy, count, convexity, and simplicity."""
    return sorted(
        results,
        key=lambda r: (
            not r.low_confidence,
            r.avg_r > 0,
            r.total_r,
            r.avg_r,
            r.convexity_ratio,
            -r.dd_contribution_r,
            -r.complexity,
        ),
        reverse=True,
    )


def result_to_table_row(result: FilterResult) -> str:
    confidence = "LOW" if result.low_confidence else "OK"
    return (
        f"| `{result.name}` | {result.allowed} | {result.rejected} | "
        f"{result.win_rate:.1%} | {result.avg_r:.4f} | {result.avg_win_r:.4f} | "
        f"{result.avg_loss_r:.4f} | {result.avg_mfe_r:.4f} | {result.avg_mae_r:.4f} | "
        f"{result.dd_contribution_r:.2f} | ${result.dd_contribution_dollars:.0f} | "
        f"{result.target_contribution_pct:.1%} | {confidence} |"
    )


def render_report(
    *,
    source_csv: Path,
    df: pd.DataFrame,
    results: list[FilterResult],
    min_trades: int,
    dollars_per_r: float,
    combine_target_dollars: float,
) -> str:
    ranked = rank_results(results)
    baseline = next((r for r in results if r.name == "all_orb_trades"), None)
    top3 = ranked[:3]
    low_conf_count = sum(1 for r in results if r.low_confidence)

    lines = [
        "# ORB Selectivity Experiment",
        "",
        f"- Source diagnostics: `{source_csv}`",
        f"- Rows loaded: {len(df)}",
        f"- ORB trades with final R: {int(df['final_r'].notna().sum())}",
        f"- Confidence floor: {min_trades} ORB trades",
        f"- Combine contribution estimate: total R * ${dollars_per_r:.2f}/R / ${combine_target_dollars:.0f} target",
        "",
    ]
    if baseline:
        lines.extend(
            [
                "## Baseline",
                "",
                "| Filter | Allowed | Rejected | Win Rate | Avg R | Avg Win R | Avg Loss R | MFE R | MAE R | DD R | DD $ | Target Contribution | Confidence |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                result_to_table_row(baseline),
                "",
            ]
        )

    lines.extend(
        [
            "## Top 3 Candidate Filters",
            "",
            "| Filter | Allowed | Rejected | Win Rate | Avg R | Avg Win R | Avg Loss R | MFE R | MAE R | DD R | DD $ | Target Contribution | Confidence |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(result_to_table_row(row) for row in top3)
    lines.append("")

    lines.extend(
        [
            "## Full Ranking",
            "",
            "| Filter | Allowed | Rejected | Win Rate | Avg R | Avg Win R | Avg Loss R | MFE R | MAE R | DD R | DD $ | Target Contribution | Confidence |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(result_to_table_row(row) for row in ranked)
    lines.append("")

    lines.extend(
        [
            "## Overfitting Warning",
            "",
            f"- {low_conf_count} of {len(results)} tested filters are below the {min_trades}-trade confidence floor.",
            "- Do not promote any LOW-confidence filter directly into live ORB rules.",
            "- Prefer one-condition filters unless a two-condition combination survives a larger out-of-sample pack.",
            "- This experiment estimates drawdown and Combine target contribution from diagnostic R values, not a full account-level Monte Carlo simulation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(results: list[FilterResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FilterResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)


def run_experiment(args: argparse.Namespace) -> tuple[Path, Path, list[FilterResult]]:
    source_csv = args.diagnostics_csv or find_latest_diagnostics_csv(args.diagnostics_root)
    df = load_diagnostics(source_csv)
    specs = build_filter_specs(df)
    if not args.simple_only:
        specs.extend(build_pairwise_specs(specs))

    results = [
        evaluate_filter(
            df,
            spec,
            min_trades=args.min_trades,
            dollars_per_r=args.dollars_per_r,
            combine_target_dollars=args.combine_target_dollars,
        )
        for spec in specs
    ]
    ranked = rank_results(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output_root / f"orb_selectivity_experiment_{timestamp}.md"
    csv_path = args.output_root / f"orb_selectivity_experiment_{timestamp}.csv"
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            source_csv=source_csv,
            df=df,
            results=results,
            min_trades=args.min_trades,
            dollars_per_r=args.dollars_per_r,
            combine_target_dollars=args.combine_target_dollars,
        ),
        encoding="utf-8",
    )
    write_csv(ranked, csv_path)
    return report_path, csv_path, ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ORB selectivity filters from diagnostics")
    parser.add_argument("--diagnostics-root", type=Path, default=DEFAULT_DIAGNOSTICS_ROOT)
    parser.add_argument("--diagnostics-csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_DIAGNOSTICS_ROOT)
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--dollars-per-r", type=float, default=DEFAULT_DOLLARS_PER_R)
    parser.add_argument("--combine-target-dollars", type=float, default=DEFAULT_COMBINE_TARGET_DOLLARS)
    parser.add_argument("--simple-only", action="store_true", help="Disable two-condition combinations")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path, csv_path, ranked = run_experiment(args)
    print(f"Report: {report_path}")
    print(f"CSV: {csv_path}")
    print("Top 3:")
    for row in ranked[:3]:
        confidence = "LOW" if row.low_confidence else "OK"
        print(
            f"  {row.name}: allowed={row.allowed}, avg_r={row.avg_r:.4f}, "
            f"dd_r={row.dd_contribution_r:.2f}, target={row.target_contribution_pct:.1%}, confidence={confidence}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
