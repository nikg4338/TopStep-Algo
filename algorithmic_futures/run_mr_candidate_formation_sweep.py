"""Run a focused MR candidate-formation sweep around first-outside, dedupe, and cooldown."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.mr_candidate_formation import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    run_mr_candidate_formation_sweep,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a focused MR candidate-formation sweep.")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--no-continue-on-error", action="store_true")
    parser.add_argument("--mr-sigma-entries", nargs="+", type=float, default=[1.3, 1.4])
    parser.add_argument("--mr-reclaim-modes", nargs="+", default=["off"])
    parser.add_argument("--mr-cooldown-bars", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--mr-first-outside-modes", nargs="+", choices=("on", "off"), default=["off", "on"])
    parser.add_argument("--mr-dedupe-modes", nargs="+", choices=("on", "off"), default=["off", "on"])
    parser.add_argument("--mr-attempt-cap-modes", nargs="+", choices=("on", "off"), default=["on"])
    parser.add_argument("--mr-regime-modes", nargs="+", choices=("on", "off"), default=["on"])
    parser.add_argument("--mr-soft-range-impulse-k", type=float, default=1.2)
    parser.add_argument("--engine-mode", choices=("mr", "both"), default="both")
    parser.add_argument("--allocator-policy", choices=("none", "v1", "v2", "open_proxy_v1"), default="open_proxy_v1")
    args = parser.parse_args()

    summary = run_mr_candidate_formation_sweep(
        pack_name=args.pack,
        artifacts_root=Path(args.artifacts_root),
        output_root=Path(args.output_root),
        continue_on_error=not args.no_continue_on_error,
        sigma_entries=args.mr_sigma_entries,
        reclaim_modes=args.mr_reclaim_modes,
        cooldown_bars=args.mr_cooldown_bars,
        first_outside_modes=args.mr_first_outside_modes,
        dedupe_modes=args.mr_dedupe_modes,
        attempt_cap_modes=args.mr_attempt_cap_modes,
        regime_modes=args.mr_regime_modes,
        soft_range_impulse_k=float(args.mr_soft_range_impulse_k),
        engine_mode=args.engine_mode,
        allocator_policy=args.allocator_policy,
    )
    best = (summary.get("formation_ranking") or [{}])[0]
    print(
        f"MR candidate-formation sweep outputs written to {summary['output_dir']}\n"
        f"Best formation candidate: {best.get('label')} | class={best.get('formation_classification')} | "
        f"avg_r={best.get('avg_r', 0):.3f} | p_target={best.get('p_target_before_ruin', 0):.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())