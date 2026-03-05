"""
validation/sizing_comparison.py — Compare sizing policies on the same packs.

Runs a validation pack under multiple sizing policies (fixed-1c, fixed-2c,
dynamic_v1) and compares MC survival metrics side-by-side.

Usage:
    python -m validation.sizing_comparison

This script assumes at least one base run already exists (to reuse
session replays).  If no base run is available, it runs from scratch.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import config
from simulation.mc_survival import MonteCarloSurvivalSimulator
from validation.sizing_policy import SizingConfig, SizingPolicy, apply_sizing_to_trades
from validation.validation_pack import ValidationPackRunner, load_pack


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SizingSpec:
    """One arm of the comparison."""

    label: str
    policy: str
    fixed_contracts: int = 2
    # dynamic_v1 thresholds default to SizingConfig defaults


# Default comparison arms
DEFAULT_ARMS: list[SizingSpec] = [
    SizingSpec(label="fixed_1c", policy="fixed", fixed_contracts=1),
    SizingSpec(label="fixed_2c", policy="fixed", fixed_contracts=2),
    SizingSpec(label="dynamic_v1", policy="dynamic_v1"),
    SizingSpec(label="dynamic_v2", policy="dynamic_v2"),
]


# ═══════════════════════════════════════════════════════════════════════
#  Core logic — rescore existing run with alternate sizing
# ═══════════════════════════════════════════════════════════════════════


def _load_session_trades(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load per-session trades DataFrames from an existing run.

    Returns dict[session_id → DataFrame].
    Reads the *original* (1-contract) trades to allow re-scaling.
    """
    sessions_dir = run_dir / "sessions"
    if not sessions_dir.is_dir():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")

    trades_by_session: dict[str, pd.DataFrame] = {}
    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        trades_csv = session_dir / "trades.csv"
        if not trades_csv.is_file():
            continue
        df = pd.read_csv(trades_csv)
        if df.empty:
            continue
        # Undo any previous scaling — normalize to 1-contract PnL
        if "contracts" in df.columns:
            contracts = df["contracts"].fillna(1).astype(float).replace(0, 1)
            for col in ("pnl_dollars", "pnl_points", "mae_points", "mfe_points"):
                if col in df.columns:
                    df[col] = df[col] / contracts
            df.drop(columns=["contracts"], inplace=True, errors="ignore")
        trades_by_session[session_dir.name] = df

    return trades_by_session


def _get_session_atr_median(session_dir: Path, max_bars: int = 12) -> float:
    """Compute median ATR of first `max_bars` bars from features_snapshot.csv."""
    features_path = session_dir / "features_snapshot.csv"
    if not features_path.is_file():
        return 0.0
    try:
        df = pd.read_csv(features_path, nrows=max_bars)
        if "atr" in df.columns:
            vals = df["atr"].dropna()
            if len(vals) > 0:
                return float(vals.median())
    except Exception:
        pass
    return 0.0


def _get_session_regime_engine(session_dir: Path, default_engine: str = "both") -> tuple[str, str]:
    """Extract regime and active_engine from session artifacts."""
    summary_path = session_dir / "session_summary.json"
    regime = "unknown"
    active_engine = default_engine

    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            orb_funnel = summary.get("orb_funnel", {})
            alloc_decision = orb_funnel.get("allocator_decision")
            if alloc_decision and alloc_decision in {"mr", "orb", "both"}:
                active_engine = alloc_decision
            regime_dist = summary.get("regime_distribution", {})
            if regime_dist:
                top = max(regime_dist, key=regime_dist.get)  # type: ignore[arg-type]
                lower = top.lower()
                if "range" in lower:
                    regime = "range"
                elif "trend" in lower:
                    regime = "trend"
                elif "chop" in lower:
                    regime = "chop"
        except Exception:
            pass

    return regime, active_engine


def rescore_with_sizing(
    run_dir: Path,
    spec: SizingSpec,
    session_order: list[str],
) -> dict[str, Any]:
    """Apply a sizing policy to an existing run's trades and compute MC metrics.

    Parameters
    ----------
    run_dir : Path
        Existing validation run directory.
    spec : SizingSpec
        The sizing arm to apply.
    session_order : list[str]
        Ordered session_ids (the combine day sequence).

    Returns
    -------
    dict
        MC metrics + sizing summary.
    """
    sessions_dir = run_dir / "sessions"

    sizing_cfg = SizingConfig(
        policy=spec.policy,
        fixed_contracts=spec.fixed_contracts,
        daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
        trail_dd_limit=float(config.MAX_LOSS_LIMIT),
    )
    policy = SizingPolicy(sizing_cfg)

    all_scaled_pnls: list[float] = []
    daily_pnls: list[float] = []

    for idx, sid in enumerate(session_order, 1):
        session_dir = sessions_dir / sid
        trades_csv = session_dir / "trades.csv"
        regime, engine = _get_session_regime_engine(session_dir)
        atr_median = _get_session_atr_median(session_dir)

        if not trades_csv.is_file():
            policy.decide_day_start(sid, regime, engine, idx,
                                    session_atr_median=atr_median)
            policy.end_of_day()
            daily_pnls.append(0.0)
            continue

        df = pd.read_csv(trades_csv)
        if df.empty:
            policy.decide_day_start(sid, regime, engine, idx,
                                    session_atr_median=atr_median)
            policy.end_of_day()
            daily_pnls.append(0.0)
            continue

        # Normalize to 1-contract if previously scaled
        if "contracts" in df.columns:
            contracts_col = df["contracts"].fillna(1).astype(float).replace(0, 1)
            for col in ("pnl_dollars", "pnl_points", "mae_points", "mfe_points"):
                if col in df.columns:
                    df[col] = df[col] / contracts_col
            df.drop(columns=["contracts"], inplace=True, errors="ignore")

        # Apply sizing
        policy.decide_day_start(sid, regime, engine, idx,
                                session_atr_median=atr_median)
        day_total = 0.0

        for _, row in df.iterrows():
            c = policy.contracts
            base_pnl = float(row.get("pnl_dollars", 0.0))
            scaled_pnl = base_pnl * c
            all_scaled_pnls.append(scaled_pnl)
            day_total += scaled_pnl
            policy.on_trade(scaled_pnl)

        policy.end_of_day()
        daily_pnls.append(day_total)

    # ── Run MC survival ─────────────────────────────────────────────────
    mc_result: dict[str, Any] = {}
    if all_scaled_pnls:
        try:
            mc = MonteCarloSurvivalSimulator(
                profit_target=float(config.PROFIT_TARGET),
                max_loss_limit=float(config.MAX_LOSS_LIMIT),
                daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
                n_simulations=10_000,
            )
            result_obj = mc.run(
                all_scaled_pnls,
                use_dollar_values=True,
            )
            mc_result = {
                k: getattr(result_obj, k)
                for k in (
                    "p_target_before_ruin", "p_ruin", "p_daily_loss_breach",
                    "dd_p95", "equity_p50", "equity_p10",
                    "losing_streak_p95", "median_trades_to_target",
                )
                if hasattr(result_obj, k)
            }
        except Exception as exc:
            mc_result = {"error": str(exc)}

    # ── Sizing summary ──────────────────────────────────────────────────
    days_at_2c = sum(1 for r in policy.daily_log if r.contracts_start == 2)
    downshifts = sum(1 for r in policy.daily_log if r.downshift_reason)
    vol_throttled_days = sum(1 for r in policy.daily_log if getattr(r, "vol_throttled", False))
    earned_upsize_days = sum(1 for r in policy.daily_log if getattr(r, "earned_upsize_triggered", False))

    return {
        "label": spec.label,
        "policy": spec.policy,
        "fixed_contracts": spec.fixed_contracts,
        "total_trades": len(all_scaled_pnls),
        "total_days": len(session_order),
        "final_equity": round(policy.equity, 2),
        "peak_equity": round(policy.peak_equity, 2),
        "trailing_dd_used": round(policy.trailing_dd_used, 2),
        "days_at_2c": days_at_2c,
        "intraday_downshifts": downshifts,
        "vol_throttled_days": vol_throttled_days,
        "earned_upsize_days": earned_upsize_days,
        "profit_lock_triggered": policy.profit_lock_triggered,
        "daily_pnls": [round(d, 2) for d in daily_pnls],
        "mc": mc_result,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Comparison runner
# ═══════════════════════════════════════════════════════════════════════


def run_comparison(
    run_dir: str | Path,
    arms: list[SizingSpec] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compare multiple sizing policies on an existing validation run.

    Parameters
    ----------
    run_dir : str or Path
        Path to a completed validation run (with sessions/ subdirectory).
    arms : list[SizingSpec], optional
        Sizing arms to compare. Defaults to fixed_1c, fixed_2c, dynamic_v1.
    output_path : str or Path, optional
        Where to write the comparison JSON. Defaults to run_dir/sizing_comparison.json.

    Returns
    -------
    dict
        Full comparison results.
    """
    run_dir = Path(run_dir)
    arms = arms or DEFAULT_ARMS
    output_path = Path(output_path) if output_path else run_dir / "sizing_comparison.json"

    # Get session order from manifest
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session_order = [
        s["session_id"]
        for s in manifest.get("sessions", [])
        if s.get("success", True)
    ]

    if not session_order:
        raise ValueError("No successful sessions found in manifest")

    print(f"\n{'═'*70}")
    print(f"  SIZING COMPARISON — {len(arms)} arms × {len(session_order)} sessions")
    print(f"  Source run: {run_dir.name}")
    print(f"{'═'*70}")

    results: list[dict[str, Any]] = []
    for arm in arms:
        t0 = time.monotonic()
        print(f"\n  ▶ {arm.label} (policy={arm.policy}, contracts={arm.fixed_contracts})")
        result = rescore_with_sizing(run_dir, arm, session_order)
        elapsed = time.monotonic() - t0

        mc = result.get("mc", {})
        p_hit = mc.get("p_target_before_ruin", "N/A")
        p_ruin = mc.get("p_ruin", "N/A")
        p_breach = mc.get("p_daily_loss_breach", "N/A")

        print(f"    equity={result['final_equity']:.0f}  peak={result['peak_equity']:.0f}  "
              f"trail_dd={result['trailing_dd_used']:.0f}")
        print(f"    P_hit={p_hit}  P_ruin={p_ruin}  P_daily_breach={p_breach}")
        if result["policy"] in ("dynamic_v1", "dynamic_v2"):
            print(f"    days_at_2c={result['days_at_2c']}/{result['total_days']}  "
                  f"downshifts={result['intraday_downshifts']}  lock={result['profit_lock_triggered']}")
        if result["policy"] == "dynamic_v2":
            print(f"    vol_throttled={result.get('vol_throttled_days', 0)}  "
                  f"earned_upsize={result.get('earned_upsize_days', 0)}")
        print(f"    ({elapsed:.1f}s)")

        results.append(result)

    # ── Write output ────────────────────────────────────────────────────
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "source_run": run_dir.name,
        "sessions": len(session_order),
        "arms": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"\n  Comparison written → {output_path}")

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  {'Arm':<15} {'Equity':>8} {'P_hit':>8} {'P_ruin':>8} {'P_breach':>9} {'Trail DD':>9}")
    print(f"{'─'*70}")
    for r in results:
        mc = r.get("mc", {})
        print(f"  {r['label']:<15} "
              f"{r['final_equity']:>8.0f} "
              f"{mc.get('p_target_before_ruin', 'N/A'):>8} "
              f"{mc.get('p_ruin', 'N/A'):>8} "
              f"{mc.get('p_daily_loss_breach', 'N/A'):>9} "
              f"{r['trailing_dd_used']:>9.0f}")
    print(f"{'─'*70}\n")

    return comparison


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare sizing policies on an existing validation run"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the validation run directory (e.g. artifacts/validation_runs/pilot_20d_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for comparison JSON (default: <run_dir>/sizing_comparison.json)",
    )
    args = parser.parse_args()

    run_comparison(args.run_dir, output_path=args.output)


if __name__ == "__main__":
    main()
