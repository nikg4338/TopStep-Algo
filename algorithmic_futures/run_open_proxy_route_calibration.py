"""Run a focused open_proxy_v1 route-quality calibration sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from validation.open_proxy_route_calibration import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    run_open_proxy_route_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a focused open_proxy_v1 route-quality calibration sweep.")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--reference-preset", default="mainline_combine_v1_4_execution_bridge")
    parser.add_argument("--artifacts-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--no-continue-on-error", action="store_true")
    parser.add_argument("--persist-bars", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--low-atr-persistences", nargs="+", type=int, default=[2])
    parser.add_argument("--high-impulse-persistences", nargs="+", type=int, default=[1])
    parser.add_argument("--medium-impulse-min-atrs", nargs="+", type=float, default=[8.0])
    parser.add_argument("--medium-impulse-max-atrs", nargs="+", type=float, default=[15.0])
    parser.add_argument("--medium-impulse-mins", nargs="+", type=float, default=[0.9, 1.0])
    parser.add_argument("--medium-impulse-maxs", nargs="+", type=float, default=[1.8, 2.0])
    parser.add_argument("--medium-impulse-min-persistences", nargs="+", type=int, default=[2, 3])
    args = parser.parse_args()

    summary = run_open_proxy_route_calibration(
        pack_name=args.pack,
        reference_preset=args.reference_preset,
        artifacts_root=Path(args.artifacts_root),
        output_root=Path(args.output_root),
        continue_on_error=not args.no_continue_on_error,
        persist_bars=args.persist_bars,
        low_atr_persistences=args.low_atr_persistences,
        high_impulse_persistences=args.high_impulse_persistences,
        medium_impulse_min_atrs=args.medium_impulse_min_atrs,
        medium_impulse_max_atrs=args.medium_impulse_max_atrs,
        medium_impulse_mins=args.medium_impulse_mins,
        medium_impulse_maxs=args.medium_impulse_maxs,
        medium_impulse_min_persistences=args.medium_impulse_min_persistences,
    )
    best = summary.get("best_candidate") or {}
    print(
        f"Open proxy route calibration outputs written to {summary['output_dir']}\n"
        f"Best route candidate: {best.get('label')} | class={best.get('classification')} | "
        f"p_target={best.get('target_probability', 0):.2%} | fp_orb={best.get('false_positive_orb_rate', 0):.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())