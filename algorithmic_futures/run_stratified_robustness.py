"""
run_stratified_robustness.py — Stratified 100-draw robustness comparison.

Unlike run_sizing_robustness.py which used convenience sampling from a flat
pool, this version:
  1. Classifies each session by **ADX tertile** (range / mixed / trend)
     using the non-zero median ADX from features_snapshot.csv
  2. Deduplicates sessions by date (keeps run with most trades)
  3. Draws stratified samples: equal sessions per stratum per draw
  4. Rescores each draw under fixed_1c / fixed_2c / dynamic_v1 / dynamic_v2

Usage:
    python run_stratified_robustness.py \
        --n-draws 100 --draw-size 21 --seed 42

Output: artifacts/validation_runs/stratified_robustness_<timestamp>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root on sys.path
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
class StratifiedSession:
    """One unique session with ADX classification and trade data."""
    session_id: str          # e.g. "session_20260120"
    session_dir: Path        # absolute path to best session directory
    adx_median: float        # median of non-zero ADX values
    atr_median: float        # median of non-zero ATR values
    regime_label: str        # "range", "mixed", or "trend" (from ADX)
    n_trades: int            # number of trades
    source_run: str          # which validation run provided trades
    adx_run: str             # which validation run provided ADX data


SIZING_ARMS = ["fixed_1c", "fixed_2c", "dynamic_v1", "dynamic_v2"]

MC_METRICS = [
    "p_target_before_ruin", "p_ruin", "p_daily_loss_breach",
    "dd_p95", "equity_p50", "equity_p10",
]


# ═══════════════════════════════════════════════════════════════════════
#  ADX Classification
# ═══════════════════════════════════════════════════════════════════════


def _compute_session_adx(features_path: Path) -> tuple[float, float, int]:
    """Compute median of non-zero ADX and ATR from features_snapshot.csv.

    Returns (adx_median, atr_median, n_adx_bars).
    """
    if not features_path.is_file():
        return 0.0, 0.0, 0
    adx_vals: list[float] = []
    atr_vals: list[float] = []
    try:
        with open(features_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = float(row.get("adx", 0))
                if a > 0:
                    adx_vals.append(a)
                t = float(row.get("atr", 0))
                if t > 0:
                    atr_vals.append(t)
    except Exception:
        return 0.0, 0.0, 0

    adx_med = statistics.median(adx_vals) if adx_vals else 0.0
    atr_med = statistics.median(atr_vals) if atr_vals else 0.0
    return adx_med, atr_med, len(adx_vals)


def classify_sessions_by_adx(
    sessions: list[dict],
) -> tuple[float, float, list[dict]]:
    """Assign regime labels via ADX tertiles.

    Parameters
    ----------
    sessions : list[dict]
        Each dict must have 'adx_median'.

    Returns
    -------
    (thresh_low, thresh_high, updated_sessions)
        thresh_low: ADX below → "range"
        thresh_high: ADX above → "trend"
        Between → "mixed"
    """
    adx_values = sorted(s["adx_median"] for s in sessions if s["adx_median"] > 0)
    if len(adx_values) < 3:
        raise ValueError(f"Need ≥3 sessions with ADX > 0, got {len(adx_values)}")

    n = len(adx_values)
    thresh_low = adx_values[n // 3]
    thresh_high = adx_values[2 * n // 3]

    for s in sessions:
        if s["adx_median"] < thresh_low:
            s["regime_label"] = "range"
        elif s["adx_median"] > thresh_high:
            s["regime_label"] = "trend"
        else:
            s["regime_label"] = "mixed"

    return thresh_low, thresh_high, sessions


# ═══════════════════════════════════════════════════════════════════════
#  Session Pooling (deduplicated by date)
# ═══════════════════════════════════════════════════════════════════════


def pool_all_sessions(
    artifacts_root: Path,
    min_feature_rows: int = 60,
) -> list[StratifiedSession]:
    """Scan ALL validation runs, deduplicate sessions by ID, classify by ADX.

    For each session date, keeps the run with the most trades (for trade data)
    and the run with the most ADX bars (for classification).
    """
    # Collect raw data per session across all runs
    raw: dict[str, dict] = {}  # session_id → info

    for run_id in sorted(os.listdir(artifacts_root)):
        sess_dir = artifacts_root / run_id / "sessions"
        if not sess_dir.is_dir():
            continue
        for sess_name in sorted(os.listdir(sess_dir)):
            if not sess_name.startswith("session_"):
                continue
            sp = sess_dir / sess_name
            feat = sp / "features_snapshot.csv"
            trades_file = sp / "trades.csv"

            if not feat.is_file():
                continue

            # Check feature row count (skip short replay sessions)
            with open(feat) as f:
                total_rows = sum(1 for _ in f) - 1
            if total_rows < min_feature_rows:
                continue

            # ADX / ATR
            adx_med, atr_med, n_adx_bars = _compute_session_adx(feat)

            # Trades
            n_trades = 0
            if trades_file.is_file():
                with open(trades_file) as f:
                    n_trades = max(0, sum(1 for _ in f) - 1)

            if sess_name not in raw:
                raw[sess_name] = {
                    "adx_median": 0.0,
                    "atr_median": 0.0,
                    "n_adx_bars": 0,
                    "adx_run": "",
                    "n_trades": 0,
                    "trade_dir": None,
                    "trade_run": "",
                }

            # Keep best ADX source
            if n_adx_bars > raw[sess_name]["n_adx_bars"]:
                raw[sess_name]["adx_median"] = adx_med
                raw[sess_name]["atr_median"] = atr_med
                raw[sess_name]["n_adx_bars"] = n_adx_bars
                raw[sess_name]["adx_run"] = run_id

            # Keep best trade source
            if n_trades > raw[sess_name]["n_trades"]:
                raw[sess_name]["n_trades"] = n_trades
                raw[sess_name]["trade_dir"] = sp
                raw[sess_name]["trade_run"] = run_id

    # Filter: need both ADX and trades
    valid = {k: v for k, v in raw.items() if v["adx_median"] > 0 and v["n_trades"] > 0}

    if not valid:
        raise ValueError("No sessions found with both ADX data and trades")

    # Classify by ADX tertile
    sessions_list = [
        {"session_id": k, **v} for k, v in valid.items()
    ]
    thresh_low, thresh_high, sessions_list = classify_sessions_by_adx(sessions_list)

    # Build StratifiedSession objects
    pool: list[StratifiedSession] = []
    for s in sessions_list:
        pool.append(StratifiedSession(
            session_id=s["session_id"],
            session_dir=s["trade_dir"],
            adx_median=s["adx_median"],
            atr_median=s["atr_median"],
            regime_label=s["regime_label"],
            n_trades=s["n_trades"],
            source_run=s["trade_run"],
            adx_run=s["adx_run"],
        ))

    pool.sort(key=lambda s: s.adx_median)
    return pool, thresh_low, thresh_high


# ═══════════════════════════════════════════════════════════════════════
#  Rescoring — apply a sizing policy to sessions
# ═══════════════════════════════════════════════════════════════════════


def _make_sizing_config(arm: str) -> SizingConfig:
    """Create SizingConfig for a named arm."""
    base = dict(
        daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
        trail_dd_limit=float(config.MAX_LOSS_LIMIT),
    )
    if arm == "fixed_1c":
        return SizingConfig(policy="fixed", fixed_contracts=1, **base)
    elif arm == "fixed_2c":
        return SizingConfig(policy="fixed", fixed_contracts=2, **base)
    elif arm == "dynamic_v1":
        return SizingConfig(policy="dynamic_v1", **base)
    elif arm == "dynamic_v2":
        return SizingConfig(
            policy="dynamic_v2",
            vol_atr_cap=14.0,
            earned_traction=150.0,
            earned_giveback=50.0,
            **base,
        )
    else:
        raise ValueError(f"Unknown arm: {arm}")


def rescore_draw(
    sessions: list[StratifiedSession],
    arm: str,
) -> dict[str, Any]:
    """Apply sizing policy to a draw of sessions and return MC metrics."""
    sizing_cfg = _make_sizing_config(arm)
    policy = SizingPolicy(sizing_cfg)

    all_scaled_pnls: list[float] = []
    session_ids_for_mc: list[str] = []

    for idx, sinfo in enumerate(sessions, 1):
        trades_csv = sinfo.session_dir / "trades.csv"

        if not trades_csv.is_file():
            policy.decide_day_start(
                sinfo.session_id, sinfo.regime_label, "both", idx,
                session_atr_median=sinfo.atr_median,
            )
            policy.end_of_day()
            continue

        df = pd.read_csv(trades_csv)
        if df.empty:
            policy.decide_day_start(
                sinfo.session_id, sinfo.regime_label, "both", idx,
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

        # Day start — pass ADX-based regime
        policy.decide_day_start(
            sinfo.session_id, sinfo.regime_label, "both", idx,
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

    # ── Run MC ──
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
    result["vol_throttled_days"] = sum(
        1 for r in policy.daily_log if getattr(r, "vol_throttled", False)
    )
    result["earned_upsize_days"] = sum(
        1 for r in policy.daily_log if getattr(r, "earned_upsize_triggered", False)
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Aggregation
# ═══════════════════════════════════════════════════════════════════════


def _percentile_stats(values: list[float]) -> dict[str, float]:
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return {"mean": 0, "std": 0, "p10": 0, "p25": 0,
                "median": 0, "p75": 0, "p90": 0, "min": 0, "max": 0}
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
    by_arm: dict[str, list[dict]] = {}
    for d in draws:
        by_arm.setdefault(d["arm"], []).append(d)

    agg: dict[str, dict[str, dict]] = {}
    for arm, arm_draws in by_arm.items():
        arm_stats: dict[str, dict] = {}
        for metric in MC_METRICS + [
            "trade_count", "final_equity",
            "days_at_2c_start", "intraday_downshifts",
            "vol_throttled_days", "earned_upsize_days",
        ]:
            values = [d.get(metric) for d in arm_draws if d.get(metric) is not None]
            arm_stats[metric] = _percentile_stats([float(v) for v in values if v is not None])
        agg[arm] = arm_stats

    return agg


# ═══════════════════════════════════════════════════════════════════════
#  Stratified Drawing
# ═══════════════════════════════════════════════════════════════════════


def stratified_draw(
    pool: list[StratifiedSession],
    draw_size: int,
    rng: np.random.Generator,
) -> list[StratifiedSession]:
    """Draw `draw_size` sessions with equal representation per stratum.

    Splits draw_size into 3 equal parts (rounding residual to mixed).
    Samples WITH REPLACEMENT within each stratum.
    Final list is shuffled to randomize day ordering.
    """
    by_stratum: dict[str, list[StratifiedSession]] = {"range": [], "mixed": [], "trend": []}
    for s in pool:
        by_stratum.setdefault(s.regime_label, []).append(s)

    # Equal split: base = draw_size // 3, remainder to mixed
    base_per_stratum = draw_size // 3
    remainder = draw_size - base_per_stratum * 3
    counts = {
        "range": base_per_stratum,
        "trend": base_per_stratum,
        "mixed": base_per_stratum + remainder,
    }

    drawn: list[StratifiedSession] = []
    for stratum, n in counts.items():
        available = by_stratum.get(stratum, [])
        if not available:
            raise ValueError(f"No sessions in stratum '{stratum}'")
        indices = rng.choice(len(available), size=n, replace=True)
        drawn.extend(available[i] for i in indices)

    # Shuffle to randomize day ordering within the combine
    rng.shuffle(drawn)
    return drawn


# ═══════════════════════════════════════════════════════════════════════
#  Main Driver
# ═══════════════════════════════════════════════════════════════════════


def run_stratified_robustness(
    artifacts_root: str | Path = "artifacts/validation_runs",
    n_draws: int = 100,
    draw_size: int = 21,
    seed: int = 42,
    arms: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the stratified robustness comparison.

    Parameters
    ----------
    artifacts_root :
        Root of validation run artifacts.
    n_draws :
        Number of stratified draws.
    draw_size :
        Sessions per draw (should be divisible by 3 for equal strata).
    seed :
        RNG seed.
    arms :
        Sizing arms to evaluate.
    """
    artifacts_root = Path(artifacts_root)
    arms = arms or SIZING_ARMS
    rng = np.random.default_rng(seed)

    print(f"\n{'═'*70}")
    print(f"  STRATIFIED ROBUSTNESS — {n_draws} draws × {draw_size} sessions × {len(arms)} arms")
    print(f"{'═'*70}")

    # ── Pool & classify ─────────────────────────────────────────────────
    t0 = time.monotonic()
    pool, thresh_low, thresh_high = pool_all_sessions(artifacts_root)

    by_stratum = {"range": [], "mixed": [], "trend": []}
    for s in pool:
        by_stratum[s.regime_label].append(s)

    print(f"\n  Pool: {len(pool)} unique sessions (ADX-classified)")
    print(f"  ADX tertile thresholds: <{thresh_low:.1f} (range), "
          f"{thresh_low:.1f}–{thresh_high:.1f} (mixed), >{thresh_high:.1f} (trend)")
    for label in ("range", "mixed", "trend"):
        group = by_stratum[label]
        trades = sum(s.n_trades for s in group)
        adx_vals = [s.adx_median for s in group]
        adx_range = f"ADX {min(adx_vals):.0f}–{max(adx_vals):.0f}" if adx_vals else "—"
        print(f"    {label:6s}: {len(group):3d} sessions, {trades:3d} trades  ({adx_range})")

    total_trades = sum(s.n_trades for s in pool)
    print(f"  Total trades in pool: {total_trades}")

    # ── Generate & score draws ──────────────────────────────────────────
    all_draw_results: list[dict[str, Any]] = []
    total_combos = n_draws * len(arms)
    done = 0

    for draw_idx in range(n_draws):
        draw_sessions = stratified_draw(pool, draw_size, rng)
        draw_sids = [s.session_id for s in draw_sessions]
        draw_strata = {label: sum(1 for s in draw_sessions if s.regime_label == label)
                       for label in ("range", "mixed", "trend")}

        for arm in arms:
            result = rescore_draw(draw_sessions, arm)
            result["draw_idx"] = draw_idx
            result["session_ids"] = draw_sids
            result["strata_counts"] = draw_strata
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
    output_path = artifacts_root / f"stratified_robustness_{ts}.json"

    output = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "method": "stratified_adx_tertile",
        "config": {
            "n_draws": n_draws,
            "draw_size": draw_size,
            "seed": seed,
            "arms": arms,
            "pool_size": len(pool),
            "adx_thresh_low": round(thresh_low, 2),
            "adx_thresh_high": round(thresh_high, 2),
            "strata": {
                label: {
                    "n_sessions": len(by_stratum[label]),
                    "n_trades": sum(s.n_trades for s in by_stratum[label]),
                    "adx_range": [
                        round(min(s.adx_median for s in by_stratum[label]), 1),
                        round(max(s.adx_median for s in by_stratum[label]), 1),
                    ] if by_stratum[label] else [0, 0],
                }
                for label in ("range", "mixed", "trend")
            },
        },
        "aggregate": aggregate,
        "draws": all_draw_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")

    total_elapsed = time.monotonic() - t0
    print(f"\n  Output → {output_path}")
    print(f"  Total time: {total_elapsed:.1f}s")

    _print_summary(aggregate, n_draws, draw_size, pool, by_stratum, thresh_low, thresh_high)

    return output


def _print_summary(
    aggregate: dict,
    n_draws: int,
    draw_size: int,
    pool: list,
    by_stratum: dict,
    thresh_low: float,
    thresh_high: float,
) -> None:
    """Print summary table."""
    print(f"\n{'═'*95}")
    print(f"  STRATIFIED ROBUSTNESS SUMMARY — {n_draws} draws × {draw_size} sessions")
    print(f"  Pool: {len(pool)} sessions | ADX thresholds: <{thresh_low:.1f} range, "
          f"{thresh_low:.1f}–{thresh_high:.1f} mixed, >{thresh_high:.1f} trend")
    for label in ("range", "mixed", "trend"):
        g = by_stratum[label]
        print(f"    {label}: {len(g)} sessions, {sum(s.n_trades for s in g)} trades")
    print(f"{'═'*95}")
    print(f"  {'Arm':<14} {'P_hit':>18} {'P_ruin':>18} "
          f"{'dd_p95':>14} {'Equity_p50':>14}")
    print(f"  {'':>14} {'mean/med/p10':>18} {'mean/p90/max':>18} "
          f"{'mean':>14} {'mean':>14}")
    print(f"{'─'*95}")

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

    print(f"{'─'*95}")

    # Extra dynamic stats
    for arm in ("dynamic_v1", "dynamic_v2"):
        stats = aggregate.get(arm)
        if not stats:
            continue
        d2c = stats.get("days_at_2c_start", {})
        ds = stats.get("intraday_downshifts", {})
        extras = [f"  {arm}: days_at_2c={d2c.get('mean', 0):.1f}/{draw_size}  "
                  f"downshifts={ds.get('mean', 0):.1f}"]
        if arm == "dynamic_v2":
            vt = stats.get("vol_throttled_days", {})
            eu = stats.get("earned_upsize_days", {})
            extras.append(
                f"    vol_throttled={vt.get('mean', 0):.1f}  "
                f"earned_upsize={eu.get('mean', 0):.1f}  "
                f"days_at_2c={d2c.get('mean', 0):.1f}"
            )
        for line in extras:
            print(line)

    print(f"{'═'*95}\n")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stratified 100-draw sizing robustness (ADX tertile classification)"
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts/validation_runs",
        help="Root of validation run artifacts",
    )
    parser.add_argument(
        "--n-draws", type=int, default=100,
        help="Number of stratified draws (default: 100)",
    )
    parser.add_argument(
        "--draw-size", type=int, default=21,
        help="Sessions per draw — use multiple of 3 for equal strata (default: 21)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--arms", nargs="+", default=None,
        help="Sizing arms to test (default: all four)",
    )
    args = parser.parse_args()

    try:
        run_stratified_robustness(
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
