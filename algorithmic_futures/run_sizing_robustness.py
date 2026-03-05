"""
run_sizing_robustness.py — 100-draw robustness comparison across sizing policies.

Pools sessions from multiple base validation runs, draws N random subsets,
rescores each draw under fixed_1c / fixed_2c / dynamic_v1 / dynamic_v2,
runs MC survival on each, and aggregates results.

Usage:
    python run_sizing_robustness.py \
        --base-runs pilot_20d_20260227_005250 random20_01_20260227_005957 \
                    random20_02_20260227_010608 random20_03_20260227_011226 \
                    trend20_adx_20260226_232222 \
        --n-draws 100 --draw-size 20 --seed 42

Output: artifacts/validation_runs/sizing_robustness_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from simulation.mc_survival import MonteCarloSurvivalSimulator
from validation.sizing_policy import SizingConfig, SizingPolicy


# ═══════════════════════════════════════════════════════════════════════
#  Data types
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SessionInfo:
    """Metadata for one pooled session."""
    session_id: str          # e.g. "session_20260120"
    session_dir: Path        # absolute path to session directory
    regime: str              # majority regime
    active_engine: str       # allocator decision
    atr_median: float        # median first-hour ATR (for v2 vol throttle)
    source_run: str          # which base run it came from


SIZING_ARMS = ["fixed_1c", "fixed_2c", "dynamic_v1", "dynamic_v2"]

MC_METRICS = [
    "p_target_before_ruin", "p_ruin", "p_daily_loss_breach",
    "dd_p95", "equity_p50", "equity_p10",
]


# ═══════════════════════════════════════════════════════════════════════
#  Session pooling
# ═══════════════════════════════════════════════════════════════════════


def _get_regime_engine(session_dir: Path) -> tuple[str, str]:
    """Extract regime and engine from session_summary.json."""
    summary_path = session_dir / "session_summary.json"
    regime = "unknown"
    active_engine = "both"

    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            orb_funnel = summary.get("orb_funnel", {})
            alloc_decision = orb_funnel.get("allocator_decision")
            if alloc_decision and alloc_decision in {"mr", "orb", "both"}:
                active_engine = alloc_decision
            regime_dist = summary.get("regime_distribution", {})
            if regime_dist:
                top = max(regime_dist, key=regime_dist.get)
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


def pool_sessions(
    base_run_ids: list[str],
    artifacts_root: Path,
) -> list[SessionInfo]:
    """Collect all sessions from the given base runs into a flat pool."""
    pool: list[SessionInfo] = []

    for run_id in base_run_ids:
        run_dir = artifacts_root / run_id
        sessions_dir = run_dir / "sessions"
        if not sessions_dir.is_dir():
            print(f"  ⚠ Skipping {run_id}: no sessions/ directory")
            continue

        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            print(f"  ⚠ Skipping {run_id}: no manifest.json")
            continue

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        session_entries = manifest.get("sessions", [])
        successful_sids = {
            s["session_id"] for s in session_entries if s.get("success", True)
        }

        for session_dir in sorted(sessions_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            sid = session_dir.name
            if sid not in successful_sids:
                continue
            trades_csv = session_dir / "trades.csv"
            if not trades_csv.is_file():
                continue

            regime, engine = _get_regime_engine(session_dir)
            atr_median = _get_session_atr_median(session_dir)

            pool.append(SessionInfo(
                session_id=sid,
                session_dir=session_dir,
                regime=regime,
                active_engine=engine,
                atr_median=atr_median,
                source_run=run_id,
            ))

    return pool


# ═══════════════════════════════════════════════════════════════════════
#  Rescoring — apply a sizing policy to a sequence of sessions
# ═══════════════════════════════════════════════════════════════════════


def _make_sizing_config(arm: str) -> SizingConfig:
    """Create SizingConfig for a named arm."""
    if arm == "fixed_1c":
        return SizingConfig(
            policy="fixed", fixed_contracts=1,
            daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
            trail_dd_limit=float(config.MAX_LOSS_LIMIT),
        )
    elif arm == "fixed_2c":
        return SizingConfig(
            policy="fixed", fixed_contracts=2,
            daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
            trail_dd_limit=float(config.MAX_LOSS_LIMIT),
        )
    elif arm == "dynamic_v1":
        return SizingConfig(
            policy="dynamic_v1",
            daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
            trail_dd_limit=float(config.MAX_LOSS_LIMIT),
        )
    elif arm == "dynamic_v2":
        return SizingConfig(
            policy="dynamic_v2",
            daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
            trail_dd_limit=float(config.MAX_LOSS_LIMIT),
            vol_atr_cap=14.0,
            earned_traction=150.0,
            earned_giveback=50.0,
        )
    else:
        raise ValueError(f"Unknown arm: {arm}")


def rescore_draw(
    sessions: list[SessionInfo],
    arm: str,
) -> dict[str, Any]:
    """Apply a sizing policy to a draw of sessions and return MC metrics.

    Parameters
    ----------
    sessions : list[SessionInfo]
        Ordered sequence of sessions (the "combine days").
    arm : str
        Sizing arm label (fixed_1c, fixed_2c, dynamic_v1, dynamic_v2).

    Returns
    -------
    dict with MC metrics + sizing summary.
    """
    sizing_cfg = _make_sizing_config(arm)
    policy = SizingPolicy(sizing_cfg)

    all_scaled_pnls: list[float] = []
    session_ids_for_mc: list[str] = []

    for idx, sinfo in enumerate(sessions, 1):
        trades_csv = sinfo.session_dir / "trades.csv"

        if not trades_csv.is_file():
            policy.decide_day_start(
                sinfo.session_id, sinfo.regime, sinfo.active_engine, idx,
                session_atr_median=sinfo.atr_median,
            )
            policy.end_of_day()
            continue

        df = pd.read_csv(trades_csv)
        if df.empty:
            policy.decide_day_start(
                sinfo.session_id, sinfo.regime, sinfo.active_engine, idx,
                session_atr_median=sinfo.atr_median,
            )
            policy.end_of_day()
            continue

        # Normalize to 1-contract base
        if "contracts" in df.columns:
            contracts_col = df["contracts"].fillna(1).astype(float).replace(0, 1)
            for col in ("pnl_dollars", "pnl_points", "mae_points", "mfe_points"):
                if col in df.columns:
                    df[col] = df[col] / contracts_col
            df.drop(columns=["contracts"], inplace=True, errors="ignore")

        # Day start
        policy.decide_day_start(
            sinfo.session_id, sinfo.regime, sinfo.active_engine, idx,
            session_atr_median=sinfo.atr_median,
        )

        for _, row in df.iterrows():
            c = policy.contracts
            base_pnl = float(row.get("pnl_dollars", 0.0))
            scaled_pnl = base_pnl * c
            all_scaled_pnls.append(scaled_pnl)
            session_ids_for_mc.append(sinfo.session_id)
            policy.on_trade(scaled_pnl)

        policy.end_of_day()

    # ── Run MC ──────────────────────────────────────────────────────────
    result: dict[str, Any] = {"arm": arm, "trade_count": len(all_scaled_pnls)}

    if all_scaled_pnls:
        try:
            mc = MonteCarloSurvivalSimulator(
                profit_target=float(config.PROFIT_TARGET),
                max_loss_limit=float(config.MAX_LOSS_LIMIT),
                daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
                n_simulations=10_000,
            )
            r = mc.run(
                all_scaled_pnls,
                use_dollar_values=True,
                session_ids=session_ids_for_mc,
            )
            for k in MC_METRICS:
                result[k] = getattr(r, k, None)
        except Exception as exc:
            result["error"] = str(exc)
    else:
        for k in MC_METRICS:
            result[k] = None

    # Sizing summary
    result["final_equity"] = round(policy.equity, 2)
    result["days_at_2c_start"] = sum(1 for r in policy.daily_log if r.contracts_start == 2)
    result["intraday_downshifts"] = sum(1 for r in policy.daily_log if r.downshift_reason)
    result["vol_throttled_days"] = sum(1 for r in policy.daily_log if getattr(r, "vol_throttled", False))
    result["earned_upsize_days"] = sum(1 for r in policy.daily_log if getattr(r, "earned_upsize_triggered", False))

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Aggregation
# ═══════════════════════════════════════════════════════════════════════


def _percentile_stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values."""
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p25": 0.0,
                "median": 0.0, "p75": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "median": round(float(np.median(arr)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def aggregate_draws(draws: list[dict[str, Any]]) -> dict[str, dict[str, dict]]:
    """Aggregate per-draw results by arm.

    Returns
    -------
    dict[arm_label → dict[metric_name → percentile_stats]]
    """
    # Group by arm
    by_arm: dict[str, list[dict]] = {}
    for d in draws:
        arm = d["arm"]
        by_arm.setdefault(arm, []).append(d)

    agg: dict[str, dict[str, dict]] = {}
    for arm, arm_draws in by_arm.items():
        arm_stats: dict[str, dict] = {}
        for metric in MC_METRICS + ["trade_count", "final_equity",
                                     "days_at_2c_start", "intraday_downshifts",
                                     "vol_throttled_days", "earned_upsize_days"]:
            values = [d.get(metric) for d in arm_draws if d.get(metric) is not None]
            arm_stats[metric] = _percentile_stats([float(v) for v in values if v is not None])
        agg[arm] = arm_stats

    return agg


# ═══════════════════════════════════════════════════════════════════════
#  Main driver
# ═══════════════════════════════════════════════════════════════════════


def run_robustness(
    base_run_ids: list[str],
    artifacts_root: str | Path = "artifacts/validation_runs",
    n_draws: int = 100,
    draw_size: int = 20,
    seed: int = 42,
    arms: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the full robustness comparison.

    Parameters
    ----------
    base_run_ids : list[str]
        Run IDs whose sessions form the pool.
    artifacts_root : Path
        Root of validation run artifacts.
    n_draws : int
        Number of random session draws.
    draw_size : int
        Sessions per draw.
    seed : int
        RNG seed.
    arms : list[str]
        Sizing arms to evaluate. Defaults to all four.

    Returns
    -------
    dict
        Full results including config, per-draw details, and aggregate stats.
    """
    artifacts_root = Path(artifacts_root)
    arms = arms or SIZING_ARMS
    rng = np.random.default_rng(seed)

    print(f"\n{'═'*70}")
    print(f"  SIZING ROBUSTNESS — {n_draws} draws × {draw_size} sessions × {len(arms)} arms")
    print(f"  Base runs: {base_run_ids}")
    print(f"{'═'*70}")

    # ── Pool sessions ───────────────────────────────────────────────────
    t0 = time.monotonic()
    pool = pool_sessions(base_run_ids, artifacts_root)
    print(f"\n  Pool: {len(pool)} sessions from {len(base_run_ids)} runs")

    if len(pool) < draw_size:
        raise ValueError(
            f"Pool has {len(pool)} sessions but draw_size={draw_size}. "
            f"Need at least {draw_size} sessions."
        )

    # ── Generate draws ──────────────────────────────────────────────────
    draws_indices: list[np.ndarray] = []
    for _ in range(n_draws):
        idx = rng.choice(len(pool), size=draw_size, replace=False)
        draws_indices.append(idx)

    # ── Execute draws × arms ────────────────────────────────────────────
    all_draw_results: list[dict[str, Any]] = []
    total_combos = n_draws * len(arms)
    done = 0

    for draw_idx, indices in enumerate(draws_indices):
        draw_sessions = [pool[i] for i in indices]
        draw_sids = [s.session_id for s in draw_sessions]

        for arm in arms:
            result = rescore_draw(draw_sessions, arm)
            result["draw_idx"] = draw_idx
            result["session_ids"] = draw_sids
            all_draw_results.append(result)

            done += 1
            if done % 50 == 0 or done == total_combos:
                elapsed = time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_combos - done) / rate if rate > 0 else 0
                print(f"    [{done}/{total_combos}] {elapsed:.0f}s elapsed, "
                      f"~{eta:.0f}s remaining")

    # ── Aggregate ───────────────────────────────────────────────────────
    aggregate = aggregate_draws(all_draw_results)

    # ── Write output ────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = artifacts_root / f"sizing_robustness_{ts}.json"

    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_draws": n_draws,
            "draw_size": draw_size,
            "seed": seed,
            "arms": arms,
            "pool_size": len(pool),
            "base_runs": base_run_ids,
        },
        "aggregate": aggregate,
        "draws": all_draw_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")

    total_elapsed = time.monotonic() - t0
    print(f"\n  Output → {output_path}")
    print(f"  Total time: {total_elapsed:.1f}s")

    # ── Print summary table ─────────────────────────────────────────────
    _print_summary(aggregate, n_draws)

    return output


def _print_summary(aggregate: dict, n_draws: int) -> None:
    """Print a concise summary table to stdout."""
    print(f"\n{'═'*90}")
    print(f"  ROBUSTNESS SUMMARY — {n_draws} draws")
    print(f"{'═'*90}")
    print(f"  {'Arm':<14} {'P_hit':>18} {'P_ruin':>18} "
          f"{'dd_p95':>14} {'Equity_p50':>14}")
    print(f"  {'':>14} {'mean/med/p10':>18} {'mean/p90/max':>18} "
          f"{'mean':>14} {'mean':>14}")
    print(f"{'─'*90}")

    for arm in SIZING_ARMS:
        stats = aggregate.get(arm)
        if not stats:
            continue
        ph = stats.get("p_target_before_ruin", {})
        pr = stats.get("p_ruin", {})
        dd = stats.get("dd_p95", {})
        eq = stats.get("equity_p50", {})

        ph_str = f"{ph.get('mean', 0):.3f}/{ph.get('median', 0):.3f}/{ph.get('p10', 0):.3f}"
        pr_str = f"{pr.get('mean', 0):.4f}/{pr.get('p90', 0):.4f}/{pr.get('max', 0):.4f}"
        dd_str = f"${dd.get('mean', 0):,.0f}"
        eq_str = f"${eq.get('mean', 0):,.0f}"

        print(f"  {arm:<14} {ph_str:>18} {pr_str:>18} {dd_str:>14} {eq_str:>14}")

    print(f"{'─'*90}")

    # Extra v2-specific stats
    for arm in ("dynamic_v1", "dynamic_v2"):
        stats = aggregate.get(arm)
        if not stats:
            continue
        d2c = stats.get("days_at_2c_start", {})
        ds = stats.get("intraday_downshifts", {})
        extras = [f"  {arm}: days_at_2c={d2c.get('mean', 0):.1f}  downshifts={ds.get('mean', 0):.1f}"]
        if arm == "dynamic_v2":
            vt = stats.get("vol_throttled_days", {})
            eu = stats.get("earned_upsize_days", {})
            extras.append(f"    vol_throttled={vt.get('mean', 0):.1f}  earned_upsize={eu.get('mean', 0):.1f}")
        for line in extras:
            print(line)

    print(f"{'═'*90}\n")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="100-draw sizing robustness comparison across fixed_1c/2c, dynamic_v1/v2"
    )
    parser.add_argument(
        "--base-runs",
        nargs="+",
        default=[
            "pilot_20d_20260227_005250",
            "random20_01_20260227_005957",
            "random20_02_20260227_010608",
            "random20_03_20260227_011226",
            "trend20_adx_20260226_232222",
        ],
        help="Run IDs whose sessions form the pool",
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/validation_runs",
        help="Root of validation run artifacts",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=100,
        help="Number of random draws (default: 100)",
    )
    parser.add_argument(
        "--draw-size",
        type=int,
        default=20,
        help="Sessions per draw (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="Sizing arms to test (default: all four)",
    )
    args = parser.parse_args()

    try:
        run_robustness(
            base_run_ids=args.base_runs,
            artifacts_root=args.artifacts_root,
            n_draws=args.n_draws,
            draw_size=args.draw_size,
            seed=args.seed,
            arms=args.arms,
        )
        return 0
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
