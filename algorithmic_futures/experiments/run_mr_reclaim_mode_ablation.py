"""
experiments/run_mr_reclaim_mode_ablation.py — MR reclaim-mode ablation runner.

Runs the same validation pack four times with only ``mr_reclaim_mode`` changed:
on, off, soft, touch. Outputs per-mode run artifacts plus JSON/CSV/Markdown
summary files under ``artifacts/candidate_reports``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RECLAIM_MODES: tuple[str, ...] = ("on", "off", "soft", "touch")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "candidate_reports"


@dataclass(frozen=True)
class ReclaimModeMetrics:
    """Summary metrics for one reclaim-mode validation run."""

    reclaim_mode: str
    run_dir: str
    candidate_pool_size: int
    approved_trades: int
    sessions_total: int
    sessions_with_trades: int
    trades_per_session: float
    win_rate: float
    avg_r: float
    avg_win_r: float
    avg_loss_r: float
    avg_trade_pnl: float
    p_target: float
    p_ruin: float
    dd_p95: float
    losing_streak_p95: float
    stress_mild_target_probability: float
    stress_severe_target_probability: float
    promotion_gate_result: str


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, returning an empty dict for missing/invalid files."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sum_candidate_pool(run_dir: Path) -> int:
    """Sum MR candidate counts from per-session replay summaries."""
    total = 0
    sessions_dir = run_dir / "sessions"
    if not sessions_dir.is_dir():
        return total

    for summary_path in sessions_dir.glob("*/session_summary.json"):
        summary = load_json(summary_path)
        gate_funnel = summary.get("gate_funnel", {}) or {}
        total += safe_int(gate_funnel.get("candidates_total"))
    return total


def average_trade_pnl(run_dir: Path) -> float:
    """Compute average trade P&L from aggregate_trades.csv."""
    aggregate_csv = run_dir / "aggregate_trades.csv"
    if not aggregate_csv.is_file():
        return 0.0

    values: list[float] = []
    with aggregate_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if "pnl_dollars" in row and row.get("pnl_dollars") not in (None, ""):
                values.append(safe_float(row.get("pnl_dollars")))
            elif "pnl_r" in row and row.get("pnl_r") not in (None, ""):
                values.append(safe_float(row.get("pnl_r")))

    return round(sum(values) / len(values), 4) if values else 0.0


def extract_mode_metrics(reclaim_mode: str, run_dir: Path) -> ReclaimModeMetrics:
    """Extract the ablation metric set from a completed validation run."""
    aggregate = load_json(run_dir / "aggregate_metrics.json")
    mc_base = load_json(run_dir / "mc_results.json")
    mc_mild = load_json(run_dir / "mc_results_stress_mild.json")
    mc_severe = load_json(run_dir / "mc_results_stress_severe.json")
    gate = load_json(run_dir / "gate_result.json")

    approved_trades = safe_int(aggregate.get("trade_count_total"))
    sessions_total = safe_int(aggregate.get("sessions_total"))
    trades_per_session = approved_trades / sessions_total if sessions_total else 0.0
    gate_result = "PASS" if gate.get("overall_pass") is True else "FAIL"
    if not gate:
        gate_result = "N/A"

    return ReclaimModeMetrics(
        reclaim_mode=reclaim_mode,
        run_dir=str(run_dir),
        candidate_pool_size=sum_candidate_pool(run_dir),
        approved_trades=approved_trades,
        sessions_total=sessions_total,
        sessions_with_trades=safe_int(aggregate.get("sessions_with_trades")),
        trades_per_session=round(trades_per_session, 4),
        win_rate=safe_float(aggregate.get("win_rate")),
        avg_r=safe_float(aggregate.get("avg_r")),
        avg_win_r=safe_float(aggregate.get("avg_win_r")),
        avg_loss_r=safe_float(aggregate.get("avg_loss_r")),
        avg_trade_pnl=average_trade_pnl(run_dir),
        p_target=safe_float(mc_base.get("p_target_before_ruin")),
        p_ruin=safe_float(mc_base.get("p_ruin")),
        dd_p95=safe_float(mc_base.get("dd_p95")),
        losing_streak_p95=safe_float(mc_base.get("losing_streak_p95")),
        stress_mild_target_probability=safe_float(mc_mild.get("p_target_before_ruin")),
        stress_severe_target_probability=safe_float(mc_severe.get("p_target_before_ruin")),
        promotion_gate_result=gate_result,
    )


def metrics_to_markdown(metrics: list[ReclaimModeMetrics], title: str) -> str:
    """Render a compact Markdown comparison table."""
    lines = [
        f"# {title}",
        "",
        "| Mode | Candidates | Trades | Sessions W/ Trades | Trades/Session | Win Rate | Avg R | Avg Win R | Avg Loss R | Avg P&L | P(Target) | P(Ruin) | DD p95 | Loss Streak p95 | Mild P(Target) | Severe P(Target) | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics:
        lines.append(
            "| "
            f"{row.reclaim_mode} | "
            f"{row.candidate_pool_size} | "
            f"{row.approved_trades} | "
            f"{row.sessions_with_trades} | "
            f"{row.trades_per_session:.2f} | "
            f"{row.win_rate:.2%} | "
            f"{row.avg_r:.4f} | "
            f"{row.avg_win_r:.4f} | "
            f"{row.avg_loss_r:.4f} | "
            f"{row.avg_trade_pnl:.2f} | "
            f"{row.p_target:.2%} | "
            f"{row.p_ruin:.2%} | "
            f"{row.dd_p95:.0f} | "
            f"{row.losing_streak_p95:.1f} | "
            f"{row.stress_mild_target_probability:.2%} | "
            f"{row.stress_severe_target_probability:.2%} | "
            f"{row.promotion_gate_result} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_summary_artifacts(metrics: list[ReclaimModeMetrics], output_dir: Path) -> dict[str, Path]:
    """Write JSON, CSV, and Markdown summary artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(row) for row in metrics]

    json_path = output_dir / "mr_reclaim_mode_ablation_summary.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir / "mr_reclaim_mode_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    md_path = output_dir / "mr_reclaim_mode_ablation_summary.md"
    md_path.write_text(
        metrics_to_markdown(metrics, "MR Reclaim Mode Ablation"),
        encoding="utf-8",
    )

    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def run_mode(args: argparse.Namespace, output_dir: Path, mode: str) -> ReclaimModeMetrics:
    """Run one validation pack with a specific reclaim mode."""
    from validation.validation_pack import ValidationPackRunner, load_pack

    pack = load_pack(args.pack)
    mode_root = output_dir / f"mode_{mode}"
    runner = ValidationPackRunner(
        pack,
        artifacts_root=str(mode_root),
        continue_on_error=not args.fail_fast,
        mr_reclaim_mode=mode,
        mr_soft_impulse_k=args.mr_soft_impulse_k,
        mr_dedupe_enabled=args.mr_dedupe_enabled,
        mr_attempt_cap_enabled=not args.disable_attempt_cap,
        mr_cooldown_bars=args.mr_cooldown_bars,
        mr_first_outside_enabled=args.mr_first_outside_enabled,
        mr_touch_latch_reset_buffer=args.mr_touch_latch_reset_buffer,
        mr_dedupe_window_bars=args.mr_dedupe_window_bars,
        mr_dedupe_min_delta_z=args.mr_dedupe_min_delta_z,
        mr_regime_enabled=not args.disable_regime_filter,
        engine_mode=args.engine_mode,
        allocator_policy=args.allocator_policy,
        orb_enabled=args.orb_enabled,
        orb_trigger_mode=args.orb_trigger_mode,
        sizing_policy=args.sizing_policy,
        fixed_contracts=args.fixed_contracts,
    )
    manifest = runner.run()
    run_dir = mode_root / manifest.run_id
    return extract_mode_metrics(mode, run_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MR reclaim-mode ablation with fixed validation-pack config"
    )
    parser.add_argument("--pack", default="pilot_20d", help="Validation pack name")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for candidate report outputs",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional output folder name under output-root",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed session")
    parser.add_argument(
        "--engine-mode",
        choices=("mr", "orb", "both"),
        default="mr",
        help="Default isolates MR candidate formation",
    )
    parser.add_argument(
        "--allocator-policy",
        choices=("none", "v1", "v2", "open_proxy_v1"),
        default="none",
    )
    parser.add_argument("--orb-enabled", action="store_true", help="Enable ORB during the pack")
    parser.add_argument(
        "--orb-trigger-mode",
        choices=("break", "pullback", "either", "pullback_v3"),
        default="pullback_v3",
    )
    parser.add_argument(
        "--sizing-policy",
        choices=("fixed", "dynamic_v1", "dynamic_v2", "dynamic_v3"),
        default="fixed",
    )
    parser.add_argument("--fixed-contracts", type=int, default=2)
    parser.add_argument("--mr-soft-impulse-k", type=float, default=0.25)
    parser.add_argument("--mr-cooldown-bars", type=int, default=1)
    parser.add_argument("--mr-dedupe-enabled", action="store_true")
    parser.add_argument("--disable-attempt-cap", action="store_true")
    parser.add_argument("--mr-first-outside-enabled", action="store_true")
    parser.add_argument("--mr-touch-latch-reset-buffer", type=float, default=0.2)
    parser.add_argument("--mr-dedupe-window-bars", type=int, default=1)
    parser.add_argument("--mr-dedupe-min-delta-z", type=float, default=0.35)
    parser.add_argument("--disable-regime-filter", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    ablation_id = args.run_id or f"mr_reclaim_mode_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_root / ablation_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nMR reclaim-mode ablation")
    print(f"Pack        : {args.pack}")
    print(f"Output dir  : {output_dir}")
    print(f"Engine mode : {args.engine_mode}")

    metrics: list[ReclaimModeMetrics] = []
    for mode in RECLAIM_MODES:
        print(f"\n{'=' * 72}")
        print(f"Running reclaim_mode={mode}")
        print(f"{'=' * 72}")
        metrics.append(run_mode(args, output_dir, mode))

    paths = write_summary_artifacts(metrics, output_dir)
    print("\nSummary artifacts:")
    for label, path in paths.items():
        print(f"  {label:8s}: {path}")
    print("")
    print(metrics_to_markdown(metrics, "MR Reclaim Mode Ablation"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
