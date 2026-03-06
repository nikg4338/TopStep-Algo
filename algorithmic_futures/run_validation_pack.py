"""CLI entry point for running a validation pack.

Usage:
    python run_validation_pack.py --pack baseline_v1
    python run_validation_pack.py --pack baseline_v1 --artifacts-root /tmp/val_runs
    python run_validation_pack.py --pack baseline_v1 --no-continue-on-error
    python run_validation_pack.py --preset mainline_combine_v1 --pack baseline_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
import config


# ── Preset loader ───────────────────────────────────────────────────────
PRESETS_DIR = Path(__file__).resolve().parent / "presets"

_PRESET_TO_CLI: dict[str, dict] = {}   # populated lazily


def _load_preset(name: str) -> dict:
    """Load a preset JSON and return a flat dict of CLI-compatible overrides."""
    if name in _PRESET_TO_CLI:
        return _PRESET_TO_CLI[name]

    path = PRESETS_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Preset file not found: {path}")

    with open(path) as f:
        preset = json.load(f)

    s = preset.get("sizing", {})
    t = preset.get("trend_engine", {})
    a = preset.get("allocator", {})
    overrides = {
        # Sizing
        "sizing_policy": s.get("policy", "dynamic_v3"),
        "dyn_v3_earned_traction": s.get("v3_earned_traction", 75.0),
        "dyn_v3_giveback_floor": s.get("v3_giveback_floor", 25.0),
        "dyn_v3_orb_upsize_allowed": "on" if s.get("v3_orb_upsize_allowed", True) else "off",
        "dyn_v3_day_headroom_up": s.get("v3_day_headroom_up", 800.0),
        "dyn_v3_day_headroom_down": s.get("v3_day_headroom_down", 600.0),
        "dyn_v3_trail_headroom_up": s.get("v3_trail_headroom_up", 1400.0),
        "dyn_v3_trail_headroom_down": s.get("v3_trail_headroom_down", 1200.0),
        # ORB pullback v3
        "orb_trigger_mode": t.get("trigger_mode", "pullback_v3"),
        "orb_pullback_confirm_bars": t.get("pullback_v3_max_bars", 3),
        # Engine / allocator
        "engine_mode": "both",
        "allocator_policy": a.get("policy", "v2"),
        "orb_enabled": "on",
        "mr_reclaim_mode": "off",
        "mr_regime_enabled": "on",
    }
    # open_proxy_v1 thresholds from preset
    if a.get("policy") == "open_proxy_v1":
        overrides["alloc_openproxy_or_width_atr"] = a.get("open_proxy_or_width_atr", 2.2)
        overrides["alloc_openproxy_impulse_atr"] = a.get("open_proxy_impulse_atr", 0.9)
        overrides["alloc_openproxy_persist_bars"] = a.get("open_proxy_persist_bars", 1)
        overrides["alloc_openproxy_require_break"] = "on" if a.get("open_proxy_require_break", False) else "off"
    _PRESET_TO_CLI[name] = overrides
    return overrides


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run a validation pack (batch replay regression test)"
    )
    parser.add_argument(
        "--pack",
        required=True,
        help="Pack name to run (e.g. 'baseline_v1'). See BUILTIN_PACKS.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Load a preset config (e.g. 'mainline_combine_v1'). Overrides CLI defaults for sizing, allocator, ORB params. Explicit flags still take precedence.",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/validation_runs",
        help="Root directory for validation run outputs (default: artifacts/validation_runs)",
    )
    parser.add_argument(
        "--no-continue-on-error",
        action="store_true",
        help="Abort the pack run on the first session failure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print session list without executing (requires a generated pack)",
    )
    parser.add_argument(
        "--mr-reclaim-mode",
        choices=("on", "off", "soft", "touch"),
        default="on",
        help="MR candidate mode: 'on' requires reclaim, 'off' threshold-cross, 'soft' threshold-cross + light momentum confirm",
    )
    parser.add_argument(
        "--mr-soft-range-impulse-k",
        type=float,
        default=1.2,
        help="Soft-v3 range impulse threshold k_range in ATR units",
    )
    parser.add_argument(
        "--mr-soft-impulse-k",
        type=float,
        default=None,
        help="Deprecated alias for --mr-soft-range-impulse-k",
    )
    parser.add_argument(
        "--mr-dedupe-enabled",
        choices=("on", "off"),
        default=("on" if config.MR_EXCURSION_DEDUPE_ENABLED else "off"),
        help="Enable/disable MR excursion dedupe gate",
    )
    parser.add_argument(
        "--mr-attempt-cap-enabled",
        choices=("on", "off"),
        default="on",
        help="Enable/disable MR attempt-cap gate",
    )
    parser.add_argument(
        "--mr-cooldown-bars",
        type=int,
        default=config.MR_COOLDOWN_BARS,
        help="MR cooldown bars",
    )
    parser.add_argument(
        "--mr-first-outside-enabled",
        choices=("on", "off"),
        default=("on" if config.MR_FIRST_OUTSIDE_ENABLED else "off"),
        help="Enable first-eligible outside candidate salvage rule",
    )
    parser.add_argument(
        "--mr-touch-latch-reset-buffer",
        type=float,
        default=config.MR_TOUCH_LATCH_RESET_BUFFER,
        help="Touch mode latch reset buffer in z units",
    )
    parser.add_argument(
        "--mr-dedupe-window-bars",
        type=int,
        default=config.MR_DEDUPE_WINDOW_BARS,
        help="Smarter dedupe window in bars",
    )
    parser.add_argument(
        "--mr-dedupe-min-delta-z",
        type=float,
        default=config.MR_DEDUPE_MIN_DELTA_Z,
        help="Smarter dedupe required |z| progression",
    )
    parser.add_argument(
        "--mr-regime-enabled",
        choices=("on", "off"),
        default="on",
        help="Enable/disable MR regime gate",
    )
    parser.add_argument(
        "--throughput-ablation",
        action="store_true",
        help="Run D0-D5 throughput ablation grid (forces mr-reclaim-mode=off)",
    )
    parser.add_argument(
        "--engine-matrix",
        action="store_true",
        help="Run engine mode matrix (MR-only, ORB-only, allocator variants) across trend + mixed packs",
    )
    parser.add_argument(
        "--matrix-trend-source-run-id",
        default="trend20_adx_20260226_232222",
        help="Validation run_id whose sessions define the trend-selected 20-session pack",
    )
    parser.add_argument(
        "--matrix-random-draws",
        type=int,
        default=3,
        help="Number of random 20-session draws from extended_60d",
    )
    parser.add_argument(
        "--matrix-random-seed",
        type=int,
        default=42,
        help="Seed for random 20-session draws",
    )
    parser.add_argument(
        "--matrix-random-draw-size",
        type=int,
        default=20,
        help="Sessions per random draw",
    )
    parser.add_argument(
        "--engine-mode",
        choices=("mr", "orb", "both"),
        default="both",
        help="Strategy engine mode: mr-only, orb-only, or both",
    )
    parser.add_argument(
        "--allocator-policy",
        choices=("none", "v1", "v2", "open_proxy_v1"),
        default="none",
        help="Day-level engine allocator policy (applies with --engine-mode both)",
    )
    parser.add_argument(
        "--allocator-v1-adx-threshold",
        type=float,
        default=25.0,
        help="Allocator v1 trend threshold: ADX >= threshold -> ORB-only day",
    )
    parser.add_argument(
        "--allocator-v2-trend-open-threshold",
        type=float,
        default=25.0,
        help="Allocator v2 trend condition at open window",
    )
    parser.add_argument(
        "--allocator-v2-rising-threshold",
        type=float,
        default=20.0,
        help="Allocator v2 rising-ADX floor",
    )
    parser.add_argument(
        "--allocator-v2-rising-bars",
        type=int,
        default=3,
        help="Allocator v2 rising-ADX consecutive bars requirement",
    )
    parser.add_argument(
        "--allocator-v2-range-threshold",
        type=float,
        default=18.0,
        help="Allocator v2 range-ADX ceiling",
    )
    parser.add_argument(
        "--allocator-v2-range-bars",
        type=int,
        default=3,
        help="Allocator v2 range-ADX consecutive bars requirement",
    )
    # ── open_proxy_v1 allocator flags (calibrated to ~53% ORB routing) ─
    parser.add_argument(
        "--alloc-openproxy-or-width-atr",
        type=float,
        default=2.2,
        help="open_proxy_v1: OR width / ATR threshold for trend signal (calibrated: 2.2)",
    )
    parser.add_argument(
        "--alloc-openproxy-impulse-atr",
        type=float,
        default=0.9,
        help="open_proxy_v1: |first 3-bar net move| / ATR threshold",
    )
    parser.add_argument(
        "--alloc-openproxy-persist-bars",
        type=int,
        default=1,
        help="open_proxy_v1: consecutive closes beyond OR for persistence",
    )
    parser.add_argument(
        "--alloc-openproxy-require-break",
        choices=("on", "off"),
        default="off",
        help="open_proxy_v1: require breakout persistence (not just width/impulse)",
    )
    parser.add_argument(
        "--orb-enabled",
        choices=("on", "off"),
        default="off",
        help="Enable ORB Engine 2 scaffold in replay/validation runs",
    )
    parser.add_argument(
        "--orb-trigger-mode",
        choices=("break", "pullback", "either"),
        default=config.ORB_TRIGGER_MODE,
        help="ORB trigger mode",
    )
    parser.add_argument(
        "--orb-pullback-confirm-bars",
        type=int,
        default=config.ORB_PULLBACK_CONFIRM_BARS,
        help="Max bars to wait for ORB pullback confirmation",
    )
    parser.add_argument(
        "--orb-allocator-enabled",
        choices=("on", "off"),
        default="off",
        help="Enable day-level ORB allocator gates",
    )
    parser.add_argument(
        "--orb-day-pnl-floor",
        type=float,
        default=-120.0,
        help="Allocator ORB enable floor on cumulative day PnL",
    )
    parser.add_argument(
        "--orb-max-loss-cluster",
        type=int,
        default=2,
        help="Allocator ORB disable threshold for loss-cluster size",
    )

    # ── Sizing policy ───────────────────────────────────────────────────
    parser.add_argument(
        "--sizing-policy",
        choices=("fixed", "dynamic_v1", "dynamic_v2", "dynamic_v3"),
        default="fixed",
        help="Contract sizing policy: 'fixed', 'dynamic_v1', 'dynamic_v2', or 'dynamic_v3' (multi-trigger earned upsize)",
    )
    parser.add_argument(
        "--fixed-contracts",
        type=int,
        default=2,
        help="Fixed contract count (sizing-policy=fixed)",
    )
    parser.add_argument(
        "--dyn-up-trail-headroom",
        type=float,
        default=1400.0,
        help="Dynamic v1: trail headroom >= X to upsize to 2c",
    )
    parser.add_argument(
        "--dyn-up-day-headroom",
        type=float,
        default=700.0,
        help="Dynamic v1: day headroom >= X to upsize to 2c",
    )
    parser.add_argument(
        "--dyn-down-trail-headroom",
        type=float,
        default=1200.0,
        help="Dynamic v1: trail headroom < X => force downshift to 1c",
    )
    parser.add_argument(
        "--dyn-down-day-headroom",
        type=float,
        default=600.0,
        help="Dynamic v1: day headroom < X => force downshift to 1c",
    )
    parser.add_argument(
        "--dyn-loss-streak-up-max",
        type=int,
        default=1,
        help="Dynamic v1: loss streak <= X to upsize",
    )
    parser.add_argument(
        "--dyn-loss-streak-down-min",
        type=int,
        default=2,
        help="Dynamic v1: loss streak >= X => force downshift",
    )
    parser.add_argument(
        "--dyn-profit-lock",
        type=float,
        default=2000.0,
        help="Dynamic v1: equity >= X => lock 1c for remainder of run",
    )
    parser.add_argument(
        "--dyn-shock-loss-frac",
        type=float,
        default=0.6,
        help="Dynamic v1: single loss >= frac * daily_loss_limit => force downshift",
    )
    # ── Dynamic v2 (earned upsize + vol throttle) ────────────────────────
    parser.add_argument(
        "--dyn-vol-atr-cap",
        type=float,
        default=14.0,
        help="Dynamic v2: session median ATR >= X => cap at 1c (vol throttle)",
    )
    parser.add_argument(
        "--dyn-earned-traction",
        type=float,
        default=150.0,
        help="Dynamic v2: day PnL >= X => unlock 2c (earned upsize)",
    )
    parser.add_argument(
        "--dyn-earned-giveback",
        type=float,
        default=50.0,
        help="Dynamic v2: day PnL drops below (traction - giveback) => revert to 1c",
    )
    # ── Dynamic v3 (multi-trigger earned upsize) ─────────────────────────
    parser.add_argument(
        "--dyn-v3-earned-traction",
        type=float,
        default=75.0,
        help="Dynamic v3: day PnL >= X => unlock 2c via traction trigger",
    )
    parser.add_argument(
        "--dyn-v3-giveback-floor",
        type=float,
        default=25.0,
        help="Dynamic v3: day PnL drops below X => revert to 1c",
    )
    parser.add_argument(
        "--dyn-v3-orb-upsize-allowed",
        choices=("on", "off"),
        default="off",
        help="Dynamic v3: auto-upsize to 2c on ORB days at session start",
    )
    parser.add_argument(
        "--dyn-v3-day-headroom-up",
        type=float,
        default=800.0,
        help="Dynamic v3: day headroom >= X required to upsize",
    )
    parser.add_argument(
        "--dyn-v3-day-headroom-down",
        type=float,
        default=600.0,
        help="Dynamic v3: day headroom < X => force downshift to 1c",
    )
    parser.add_argument(
        "--dyn-v3-trail-headroom-up",
        type=float,
        default=1400.0,
        help="Dynamic v3: trail headroom >= X required to upsize",
    )
    parser.add_argument(
        "--dyn-v3-trail-headroom-down",
        type=float,
        default=1200.0,
        help="Dynamic v3: trail headroom < X => force downshift to 1c",
    )
    args = parser.parse_args()

    # ── Apply preset defaults (explicit CLI flags take precedence) ──────
    if args.preset:
        try:
            preset_overrides = _load_preset(args.preset)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        # Only override args that the user did NOT explicitly set on the CLI
        explicit = {act.dest for act in parser._actions if act.dest in sys.argv}
        for key, val in preset_overrides.items():
            if f"--{key.replace('_', '-')}" not in " ".join(sys.argv):
                setattr(args, key, val)
        print(f"  [preset] Loaded '{args.preset}' from {PRESETS_DIR / f'{args.preset}.json'}")

    from validation.validation_pack import ValidationPackRunner, load_pack

    try:
        pack = load_pack(args.pack)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"\n  Pack: {pack.pack_id}")
        print(f"  Description: {pack.description}")
        print(f"  Sessions: {len(pack.sessions)}")
        print()
        for i, s in enumerate(pack.sessions, 1):
            print(f"  [{i:>3}] {s.session_id:<30} {s.start} → {s.end}  ({s.category})")
        print()
        return 0

    if args.throughput_ablation:
        from validation.throughput_ablation import run_throughput_ablation
        run_dirs = run_throughput_ablation(
            pack_name=args.pack,
            artifacts_root=args.artifacts_root,
            continue_on_error=not args.no_continue_on_error,
            mr_soft_range_impulse_k=args.mr_soft_impulse_k if args.mr_soft_impulse_k is not None else args.mr_soft_range_impulse_k,
        )
        print("\nThroughput ablation complete. Run dirs:")
        for label, run_dir in run_dirs:
            print(f"  {label}: {run_dir}")
        return 0

    if args.engine_matrix:
        from validation.engine_mode_matrix import run_engine_mode_matrix
        summary = run_engine_mode_matrix(
            artifacts_root=args.artifacts_root,
            trend_source_run_id=args.matrix_trend_source_run_id,
            random_draws=max(0, int(args.matrix_random_draws)),
            random_seed=int(args.matrix_random_seed),
            random_draw_size=max(1, int(args.matrix_random_draw_size)),
            mr_reclaim_mode=args.mr_reclaim_mode,
            mr_regime_enabled=(args.mr_regime_enabled == "on"),
        )
        print(f"\nEngine mode matrix complete: {summary.get('output_path')}")
        return 0

    soft_range_k = args.mr_soft_range_impulse_k
    if args.mr_soft_impulse_k is not None:
        soft_range_k = args.mr_soft_impulse_k

    runner = ValidationPackRunner(
        pack,
        artifacts_root=args.artifacts_root,
        continue_on_error=not args.no_continue_on_error,
        mr_reclaim_mode=args.mr_reclaim_mode,
        mr_soft_impulse_k=soft_range_k,
        mr_dedupe_enabled=(args.mr_dedupe_enabled == "on"),
        mr_attempt_cap_enabled=(args.mr_attempt_cap_enabled == "on"),
        mr_cooldown_bars=max(0, int(args.mr_cooldown_bars)),
        mr_first_outside_enabled=(args.mr_first_outside_enabled == "on"),
        mr_touch_latch_reset_buffer=float(args.mr_touch_latch_reset_buffer),
        mr_dedupe_window_bars=max(0, int(args.mr_dedupe_window_bars)),
        mr_dedupe_min_delta_z=float(args.mr_dedupe_min_delta_z),
        mr_regime_enabled=(args.mr_regime_enabled == "on"),
        engine_mode=args.engine_mode,
        allocator_policy=args.allocator_policy,
        allocator_v1_adx_threshold=float(args.allocator_v1_adx_threshold),
        allocator_v2_trend_open_threshold=float(args.allocator_v2_trend_open_threshold),
        allocator_v2_rising_threshold=float(args.allocator_v2_rising_threshold),
        allocator_v2_rising_bars=max(1, int(args.allocator_v2_rising_bars)),
        allocator_v2_range_threshold=float(args.allocator_v2_range_threshold),
        allocator_v2_range_bars=max(1, int(args.allocator_v2_range_bars)),
        alloc_openproxy_or_width_atr=float(getattr(args, "alloc_openproxy_or_width_atr", 2.2)),
        alloc_openproxy_impulse_atr=float(getattr(args, "alloc_openproxy_impulse_atr", 0.9)),
        alloc_openproxy_persist_bars=max(0, int(getattr(args, "alloc_openproxy_persist_bars", 1))),
        alloc_openproxy_require_break=(getattr(args, "alloc_openproxy_require_break", "off") == "on"),
        orb_enabled=(args.orb_enabled == "on"),
        orb_trigger_mode=args.orb_trigger_mode,
        orb_pullback_confirm_bars=max(1, int(args.orb_pullback_confirm_bars)),
        orb_allocator_enabled=(args.orb_allocator_enabled == "on"),
        orb_day_pnl_floor=float(args.orb_day_pnl_floor),
        orb_max_loss_cluster=max(0, int(args.orb_max_loss_cluster)),
        sizing_policy=args.sizing_policy,
        fixed_contracts=max(1, int(args.fixed_contracts)),
        dyn_up_trail_headroom=float(args.dyn_up_trail_headroom),
        dyn_up_day_headroom=float(args.dyn_up_day_headroom),
        dyn_down_trail_headroom=float(args.dyn_down_trail_headroom),
        dyn_down_day_headroom=float(args.dyn_down_day_headroom),
        dyn_loss_streak_up_max=max(0, int(args.dyn_loss_streak_up_max)),
        dyn_loss_streak_down_min=max(1, int(args.dyn_loss_streak_down_min)),
        dyn_profit_lock=float(args.dyn_profit_lock),
        dyn_shock_loss_frac=float(args.dyn_shock_loss_frac),
        dyn_vol_atr_cap=float(args.dyn_vol_atr_cap),
        dyn_earned_traction=float(args.dyn_earned_traction),
        dyn_earned_giveback=float(args.dyn_earned_giveback),
        dyn_v3_earned_traction=float(args.dyn_v3_earned_traction),
        dyn_v3_giveback_floor=float(args.dyn_v3_giveback_floor),
        dyn_v3_orb_upsize_allowed=(args.dyn_v3_orb_upsize_allowed == "on"),
        dyn_v3_day_headroom_up=float(args.dyn_v3_day_headroom_up),
        dyn_v3_day_headroom_down=float(args.dyn_v3_day_headroom_down),
        dyn_v3_trail_headroom_up=float(args.dyn_v3_trail_headroom_up),
        dyn_v3_trail_headroom_down=float(args.dyn_v3_trail_headroom_down),
    )
    manifest = runner.run()

    # ── Summary ─────────────────────────────────────────────────────────
    passed = sum(1 for s in manifest.sessions if s.success)
    failed = len(manifest.sessions) - passed

    print(f"\n{'═'*70}")
    print(f"  Validation Pack Complete: {manifest.pack_id}")
    print(f"  Run ID   : {manifest.run_id}")
    print(f"  Sessions : {passed} passed, {failed} failed, {len(manifest.sessions)} total")
    print(f"  Runtime  : {manifest.total_runtime_seconds:.1f}s")
    print(f"  Hash     : {manifest.config_hash}")
    print(f"{'═'*70}")

    if failed:
        print("\n  Failed sessions:")
        for s in manifest.sessions:
            if not s.success:
                print(f"    • {s.session_id}: {s.error_message}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
