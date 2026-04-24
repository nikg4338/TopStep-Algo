"""Run a narrow MR edge iteration around first-outside and entry quality."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.mr_edge_iteration import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    run_mr_edge_iteration,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a narrow MR edge iteration.")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--no-continue-on-error", action="store_true")
    parser.add_argument("--mr-sigma-entries", nargs="+", type=float, default=[1.25, 1.3, 1.35])
    parser.add_argument("--mr-soft-range-impulse-ks", nargs="+", type=float, default=[1.0, 1.1, 1.2])
    parser.add_argument("--mr-cooldown-bars", nargs="+", type=int, default=[1])
    parser.add_argument("--mr-first-outside-modes", nargs="+", choices=("on", "off"), default=["on"])
    parser.add_argument("--mr-dedupe-modes", nargs="+", choices=("on", "off"), default=["on"])
    parser.add_argument("--mr-reclaim-mode", default="off")
    parser.add_argument("--mr-attempt-cap-mode", choices=("on", "off"), default="on")
    parser.add_argument("--mr-regime-mode", choices=("on", "off"), default="on")
    parser.add_argument("--engine-mode", choices=("mr", "both"), default="both")
    parser.add_argument("--allocator-policy", choices=("none", "v1", "v2", "open_proxy_v1"), default="open_proxy_v1")
    parser.add_argument("--reference-label", default="")
    args = parser.parse_args()

    summary = run_mr_edge_iteration(
        pack_name=args.pack,
        artifacts_root=Path(args.artifacts_root),
        output_root=Path(args.output_root),
        continue_on_error=not args.no_continue_on_error,
        sigma_entries=args.mr_sigma_entries,
        soft_range_impulse_ks=args.mr_soft_range_impulse_ks,
        cooldown_bars=args.mr_cooldown_bars,
        first_outside_modes=args.mr_first_outside_modes,
        dedupe_modes=args.mr_dedupe_modes,
        reclaim_mode=args.mr_reclaim_mode,
        attempt_cap_mode=args.mr_attempt_cap_mode,
        regime_mode=args.mr_regime_mode,
        engine_mode=args.engine_mode,
        allocator_policy=args.allocator_policy,
        reference_label=args.reference_label or None,
    )
    best = summary.get("best_candidate") or {}
    print(
        f"MR edge iteration outputs written to {summary['output_dir']}\n"
        f"Best edge candidate: {best.get('label')} | class={best.get('edge_classification')} | "
        f"avg_r={best.get('avg_r', 0):.3f} | p_target={best.get('p_target_before_ruin', 0):.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())