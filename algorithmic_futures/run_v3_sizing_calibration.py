#!/usr/bin/env python3
"""
run_v3_sizing_calibration.py — Dynamic v3 sizing calibration experiment.

6-arm comparison via 100-draw stratified robustness:
  1. fixed_1c             — baseline
  2. fixed_2c             — ceiling
  3. dynamic_v3_50_25     — traction=$50, giveback=$25
  4. dynamic_v3_75_25     — traction=$75, giveback=$25
  5. dynamic_v3_100_25    — traction=$100, giveback=$25
  6. dynamic_v3_orb_start — traction=$75, giveback=$25, orb_upsize=True

Reuses the base-run session pool from the pb3 evaluation step.

Usage:
    python run_v3_sizing_calibration.py
    python run_v3_sizing_calibration.py --base-run-ids <id1> <id2> ...
    python run_v3_sizing_calibration.py --n-draws 50 --draw-size 21
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from simulation.mc_survival import MonteCarloSurvivalSimulator

# ── Known base run IDs from pb3 evaluation ─────────────────────────────
DEFAULT_BASE_RUN_IDS = [
    "pb3eval_trend20_20260302_133325",
    "pilot_20d_20260302_133625",
    "pb3eval_random01_20260302_134030",
    "pb3eval_random02_20260302_134339",
    "pb3eval_random03_20260302_134658",
]

ARTIFACTS_ROOT = Path("artifacts/validation_runs")

MC_KEYS = [
    "p_target_before_ruin", "p_ruin", "p_daily_loss_breach",
    "dd_p95", "equity_p50", "equity_p10",
]

# ── Sizing arms ────────────────────────────────────────────────────────
SIZING_ARMS = [
    "fixed_1c",
    "fixed_2c",
    "dynamic_v3_50_25",
    "dynamic_v3_75_25",
    "dynamic_v3_100_25",
    "dynamic_v3_orb_start",
]


# ═══════════════════════════════════════════════════════════════════════
#  Session pool (reused from run_pb3_evaluation)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class StratSession:
    session_id: str
    session_dir: Path
    adx_median: float
    atr_median: float
    regime_label: str  # range / mixed / trend
    n_trades: int
    source_run: str
    active_engine: str = "both"  # allocator v2 decision: mr / orb / both


def _session_early_adx(session_dir: Path, max_bars: int = 12) -> list[float]:
    """Read first max_bars RTH ADX values from features_snapshot.csv."""
    feat = session_dir / "features_snapshot.csv"
    if not feat.is_file():
        return []
    try:
        df = pd.read_csv(feat)
    except Exception:
        return []
    if df.empty or "timestamp" not in df.columns or "adx" not in df.columns:
        return []
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    adx = pd.to_numeric(df["adx"], errors="coerce")
    out: list[float] = []
    for t, a in zip(ts, adx):
        if pd.isna(t) or pd.isna(a):
            continue
        et = t.tz_convert(config.TIMEZONE)
        if et.strftime("%H:%M") >= config.RTH_OPEN and float(a) > 0:
            out.append(float(a))
            if len(out) >= max_bars:
                break
    return out


def _allocator_v2_decision(session_dir: Path) -> str:
    """Replicate allocator V2 hysteresis engine routing from early ADX."""
    adx_series = _session_early_adx(session_dir, max_bars=12)
    trend_open = any(v >= 25.0 for v in adx_series)
    rising = adx_series[-3:]
    rising_ok = (len(rising) >= 3 and all(v > 20.0 for v in rising)
                 and all(rising[i] < rising[i + 1] for i in range(len(rising) - 1)))
    range_seq = adx_series[-3:]
    range_ok = len(range_seq) >= 3 and all(v <= 18.0 for v in range_seq)
    if trend_open or rising_ok:
        return "orb"
    if range_ok:
        return "mr"
    return "mr"


def _compute_session_adx(features_path: Path) -> tuple[float, float, int]:
    if not features_path.is_file():
        return 0.0, 0.0, 0
    adx_vals: list[float] = []
    atr_vals: list[float] = []
    try:
        with open(features_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = float(row.get("adx", row.get("adx_14", 0)))
                if v > 0:
                    adx_vals.append(v)
                t = float(row.get("atr", row.get("atr_14", 0)))
                if t > 0:
                    atr_vals.append(t)
    except Exception:
        return 0.0, 0.0, 0
    import statistics
    adx_med = statistics.median(adx_vals) if adx_vals else 0.0
    atr_med = statistics.median(atr_vals) if atr_vals else 0.0
    return adx_med, atr_med, len(adx_vals)


def pool_all_sessions(
    run_ids: list[str],
    min_feature_rows: int = 60,
) -> tuple[list[StratSession], float, float]:
    """Pool sessions from given runs, classify by ADX tertile."""
    import os
    raw: dict[str, dict] = {}

    for run_id in run_ids:
        sess_dir = ARTIFACTS_ROOT / run_id / "sessions"
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
            with open(feat) as f:
                total_rows = sum(1 for _ in f) - 1
            if total_rows < min_feature_rows:
                continue
            adx_med, atr_med, n_adx_bars = _compute_session_adx(feat)
            n_trades = 0
            if trades_file.is_file():
                with open(trades_file) as f:
                    n_trades = max(0, sum(1 for _ in f) - 1)

            if sess_name not in raw:
                raw[sess_name] = {
                    "adx_median": 0.0, "atr_median": 0.0, "n_adx_bars": 0,
                    "n_trades": 0, "trade_dir": None, "trade_run": "",
                }
            if n_adx_bars > raw[sess_name]["n_adx_bars"]:
                raw[sess_name]["adx_median"] = adx_med
                raw[sess_name]["atr_median"] = atr_med
                raw[sess_name]["n_adx_bars"] = n_adx_bars
            if n_trades > raw[sess_name]["n_trades"]:
                raw[sess_name]["n_trades"] = n_trades
                raw[sess_name]["trade_dir"] = sp
                raw[sess_name]["trade_run"] = run_id

    valid = {k: v for k, v in raw.items() if v["adx_median"] > 0 and v["n_trades"] > 0}
    if len(valid) < 3:
        raise ValueError(f"Need ≥3 sessions, got {len(valid)}")

    adx_values = sorted(v["adx_median"] for v in valid.values())
    n = len(adx_values)
    thresh_low = adx_values[n // 3]
    thresh_high = adx_values[2 * n // 3]

    pool: list[StratSession] = []
    for k, v in valid.items():
        if v["adx_median"] < thresh_low:
            label = "range"
        elif v["adx_median"] > thresh_high:
            label = "trend"
        else:
            label = "mixed"
        engine = _allocator_v2_decision(v["trade_dir"])
        pool.append(StratSession(
            session_id=k,
            session_dir=v["trade_dir"],
            adx_median=v["adx_median"],
            atr_median=v["atr_median"],
            regime_label=label,
            n_trades=v["n_trades"],
            source_run=v["trade_run"],
            active_engine=engine,
        ))
    pool.sort(key=lambda s: s.adx_median)
    return pool, thresh_low, thresh_high


def stratified_draw(
    pool: list[StratSession], draw_size: int, rng: np.random.Generator,
) -> list[StratSession]:
    by_stratum: dict[str, list[StratSession]] = {"range": [], "mixed": [], "trend": []}
    for s in pool:
        by_stratum.setdefault(s.regime_label, []).append(s)
    base = draw_size // 3
    remainder = draw_size - base * 3
    counts = {"range": base, "trend": base, "mixed": base + remainder}
    drawn: list[StratSession] = []
    for stratum, n in counts.items():
        available = by_stratum.get(stratum, [])
        if not available:
            raise ValueError(f"No sessions in stratum '{stratum}'")
        indices = rng.choice(len(available), size=n, replace=True)
        drawn.extend(available[i] for i in indices)
    rng.shuffle(drawn)
    return drawn


# ═══════════════════════════════════════════════════════════════════════
#  Sizing config factory
# ═══════════════════════════════════════════════════════════════════════


def _make_sizing_config(arm: str):
    from validation.sizing_policy import SizingConfig
    base = dict(
        daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
        trail_dd_limit=float(config.MAX_LOSS_LIMIT),
    )
    if arm == "fixed_1c":
        return SizingConfig(policy="fixed", fixed_contracts=1, **base)
    elif arm == "fixed_2c":
        return SizingConfig(policy="fixed", fixed_contracts=2, **base)
    elif arm.startswith("dynamic_v3"):
        # Parse arm name: dynamic_v3_<traction>_<giveback> or dynamic_v3_orb_start
        if arm == "dynamic_v3_orb_start":
            return SizingConfig(
                policy="dynamic_v3",
                v3_earned_traction=75.0,
                v3_giveback_floor=25.0,
                v3_orb_upsize_allowed=True,
                **base,
            )
        else:
            # e.g. dynamic_v3_50_25
            parts = arm.split("_")
            traction = float(parts[2])
            giveback = float(parts[3])
            return SizingConfig(
                policy="dynamic_v3",
                v3_earned_traction=traction,
                v3_giveback_floor=giveback,
                v3_orb_upsize_allowed=False,
                **base,
            )
    else:
        raise ValueError(f"Unknown arm: {arm}")


# ═══════════════════════════════════════════════════════════════════════
#  Rescore a draw with a given arm
# ═══════════════════════════════════════════════════════════════════════


def rescore_draw(sessions: list[StratSession], arm: str) -> dict[str, Any]:
    from validation.sizing_policy import SizingPolicy
    sizing_cfg = _make_sizing_config(arm)
    policy = SizingPolicy(sizing_cfg)
    all_scaled_pnls: list[float] = []
    session_ids_for_mc: list[str] = []
    trades_at_2c: int = 0

    for idx, sinfo in enumerate(sessions, 1):
        trades_csv = sinfo.session_dir / "trades.csv"
        if not trades_csv.is_file():
            policy.decide_day_start(
                sinfo.session_id, sinfo.regime_label, sinfo.active_engine, idx,
                session_atr_median=sinfo.atr_median,
            )
            policy.end_of_day()
            continue
        df = pd.read_csv(trades_csv)
        if df.empty:
            policy.decide_day_start(
                sinfo.session_id, sinfo.regime_label, sinfo.active_engine, idx,
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

        policy.decide_day_start(
            sinfo.session_id, sinfo.regime_label, sinfo.active_engine, idx,
            session_atr_median=sinfo.atr_median,
        )
        for _, row in df.iterrows():
            c = policy.contracts
            if c >= 2:
                trades_at_2c += 1
            base_pnl = float(row.get("pnl_dollars", 0.0))
            scaled_pnl = base_pnl * c
            all_scaled_pnls.append(scaled_pnl)
            session_ids_for_mc.append(sinfo.session_id)
            policy.on_trade(scaled_pnl)
        policy.end_of_day()

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
                all_scaled_pnls, use_dollar_values=True,
                session_ids=session_ids_for_mc,
            )
            for k in MC_KEYS:
                result[k] = getattr(r, k, None)
        except Exception as exc:
            result["error"] = str(exc)
    else:
        for k in MC_KEYS:
            result[k] = None

    result["final_equity"] = round(policy.equity, 2)
    result["days_started_2c"] = sum(1 for r in policy.daily_log if r.contracts_start == 2)
    result["days_ever_2c"] = sum(
        1 for r in policy.daily_log
        if r.contracts_start == 2 or getattr(r, "earned_upsize_triggered", False)
    )
    result["trades_at_2c"] = trades_at_2c
    result["intraday_downshifts"] = sum(1 for r in policy.daily_log if r.downshift_reason)
    result["earned_upsize_days"] = sum(
        1 for r in policy.daily_log if getattr(r, "earned_upsize_triggered", False)
    )

    # v3-specific diagnostics
    result["v3_traction_days"] = sum(
        1 for r in policy.daily_log if r.v3_upsize_trigger == "traction"
    )
    result["v3_first_win_days"] = sum(
        1 for r in policy.daily_log if r.v3_upsize_trigger == "first_trade_win"
    )
    result["v3_orb_upsize_days"] = sum(
        1 for r in policy.daily_log if r.v3_upsize_trigger == "orb_day"
    )
    result["v3_any_upsize_days"] = sum(
        1 for r in policy.daily_log if r.v3_upsize_trigger != ""
    )
    result["v3_orb_sessions"] = sum(
        1 for r in policy.daily_log if r.v3_orb_day
    )

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Aggregation
# ═══════════════════════════════════════════════════════════════════════


def _percentile_stats(values: list[float]) -> dict[str, float]:
    arr = np.array([v for v in values if v is not None])
    if len(arr) == 0:
        return {"mean": 0, "std": 0, "p10": 0, "median": 0, "p90": 0, "max": 0}
    return {
        "mean": round(float(np.mean(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "p10": round(float(np.percentile(arr, 10)), 6),
        "median": round(float(np.median(arr)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "max": round(float(np.max(arr)), 6),
    }


DIAG_METRICS = [
    "trade_count", "final_equity",
    "days_started_2c", "days_ever_2c", "trades_at_2c",
    "intraday_downshifts", "earned_upsize_days",
    "v3_traction_days", "v3_first_win_days", "v3_orb_upsize_days",
    "v3_any_upsize_days", "v3_orb_sessions",
]


def aggregate_draw_results(draws: list[dict]) -> dict[str, dict[str, dict]]:
    by_arm: dict[str, list[dict]] = {}
    for d in draws:
        by_arm.setdefault(d["arm"], []).append(d)
    agg: dict[str, dict[str, dict]] = {}
    for arm, arm_draws in by_arm.items():
        stats: dict[str, dict] = {}
        for metric in MC_KEYS + DIAG_METRICS:
            values = [d.get(metric) for d in arm_draws if d.get(metric) is not None]
            stats[metric] = _percentile_stats([float(v) for v in values if v is not None])
        agg[arm] = stats
    return agg


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dynamic v3 sizing calibration — 6-arm stratified robustness"
    )
    parser.add_argument("--base-run-ids", nargs="+", default=DEFAULT_BASE_RUN_IDS,
                        help="Base run IDs to pool sessions from")
    parser.add_argument("--n-draws", type=int, default=100,
                        help="Number of stratified draws")
    parser.add_argument("--draw-size", type=int, default=21,
                        help="Sessions per draw (divisible by 3 ideal)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Override arm list (default: all 6)")
    args = parser.parse_args()

    arms = args.arms or SIZING_ARMS
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_global = time.monotonic()

    print(f"\n{'═'*95}")
    print(f"  DYNAMIC V3 SIZING CALIBRATION")
    print(f"  Arms: {arms}")
    print(f"  Draws: {args.n_draws} × {args.draw_size} sessions")
    print(f"  Seed: {args.seed}")
    print(f"{'═'*95}")

    # ── Pool sessions ───────────────────────────────────────────────────
    print("\n  Pooling sessions from base runs...")
    pool, thresh_low, thresh_high = pool_all_sessions(args.base_run_ids)

    by_stratum: dict[str, list[StratSession]] = {"range": [], "mixed": [], "trend": []}
    for s in pool:
        by_stratum[s.regime_label].append(s)

    print(f"  Pool: {len(pool)} sessions (ADX classified)")
    print(f"  Thresholds: <{thresh_low:.1f} range, "
          f"{thresh_low:.1f}–{thresh_high:.1f} mixed, >{thresh_high:.1f} trend")
    for label in ("range", "mixed", "trend"):
        g = by_stratum[label]
        n_orb = sum(1 for s in g if s.active_engine == "orb")
        print(f"    {label}: {len(g)} sessions, {sum(s.n_trades for s in g)} trades, "
              f"{n_orb} orb-routed")

    # ── Run draws ───────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    all_draw_results: list[dict] = []
    total_combos = args.n_draws * len(arms)
    done = 0
    t0 = time.monotonic()

    for draw_idx in range(args.n_draws):
        draw_sessions = stratified_draw(pool, args.draw_size, rng)
        draw_sids = [s.session_id for s in draw_sessions]
        draw_strata = {
            label: sum(1 for s in draw_sessions if s.regime_label == label)
            for label in ("range", "mixed", "trend")
        }
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
                print(f"    [{done}/{total_combos}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    aggregate = aggregate_draw_results(all_draw_results)

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n{'═'*110}")
    print(f"  V3 SIZING CALIBRATION — {args.n_draws} draws × {args.draw_size} sessions")
    print(f"{'═'*110}")
    print(f"  {'Arm':<22} {'P_hit':>18} {'P_ruin':>18} "
          f"{'P_daily':>8} {'dd_p95':>9} {'Eq_p50':>9} {'Eq_p10':>9}")
    print(f"  {'':>22} {'mean/med/p10':>18} {'mean/p90/max':>18}")
    print(f"{'─'*110}")

    for arm in arms:
        s = aggregate.get(arm)
        if not s:
            continue
        ph = s.get("p_target_before_ruin", {})
        pr = s.get("p_ruin", {})
        pd_ = s.get("p_daily_loss_breach", {})
        dd = s.get("dd_p95", {})
        eq50 = s.get("equity_p50", {})
        eq10 = s.get("equity_p10", {})

        ph_str = f"{ph.get('mean', 0):.3f}/{ph.get('median', 0):.3f}/{ph.get('p10', 0):.3f}"
        pr_str = f"{pr.get('mean', 0):.4f}/{pr.get('p90', 0):.4f}/{pr.get('max', 0):.4f}"

        print(
            f"  {arm:<22} {ph_str:>18} {pr_str:>18} "
            f"{pd_.get('mean', 0):>8.4f} "
            f"${dd.get('mean', 0):>8,.0f} "
            f"${eq50.get('mean', 0):>8,.0f} "
            f"${eq10.get('mean', 0):>8,.0f}"
        )

    # ── v3 activation diagnostics ───────────────────────────────────────
    print(f"\n{'─'*130}")
    print(f"  V3 ACTIVATION DIAGNOSTICS (mean per {args.draw_size}-day draw)")
    print(f"{'─'*130}")
    print(f"  {'Arm':<22} {'start@2c':>9} {'ever@2c':>8} {'t@2c':>6} "
          f"{'upsize':>7} {'traction':>9} {'1st_win':>8} {'orb_up':>7} "
          f"{'orb_sess':>9} {'downshift':>10}")
    print(f"{'─'*130}")

    for arm in arms:
        s = aggregate.get(arm)
        if not s:
            continue
        d2c_s = s.get("days_started_2c", {}).get("mean", 0)
        d2c_e = s.get("days_ever_2c", {}).get("mean", 0)
        t2c = s.get("trades_at_2c", {}).get("mean", 0)
        any_up = s.get("v3_any_upsize_days", {}).get("mean", 0)
        trac = s.get("v3_traction_days", {}).get("mean", 0)
        fw = s.get("v3_first_win_days", {}).get("mean", 0)
        orb = s.get("v3_orb_upsize_days", {}).get("mean", 0)
        orb_s = s.get("v3_orb_sessions", {}).get("mean", 0)
        ds = s.get("intraday_downshifts", {}).get("mean", 0)
        print(
            f"  {arm:<22} {d2c_s:>9.1f} {d2c_e:>8.1f} {t2c:>6.1f} "
            f"{any_up:>7.1f} {trac:>9.1f} {fw:>8.1f} {orb:>7.1f} "
            f"{orb_s:>9.1f} {ds:>10.1f}"
        )

    print(f"{'═'*110}")

    # ── Write artifacts ─────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t_global
    output: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "experiment": "v3_sizing_calibration",
        "arms": arms,
        "base_run_ids": args.base_run_ids,
        "n_draws": args.n_draws,
        "draw_size": args.draw_size,
        "seed": args.seed,
        "pool_size": len(pool),
        "thresholds": {"low": thresh_low, "high": thresh_high},
        "strata": {
            label: {
                "n_sessions": len(by_stratum[label]),
                "n_trades": sum(s.n_trades for s in by_stratum[label]),
            }
            for label in ("range", "mixed", "trend")
        },
        "aggregate": aggregate,
        "total_runtime_seconds": round(total_elapsed, 1),
    }

    json_path = ARTIFACTS_ROOT / f"v3_sizing_calibration_{ts}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n  JSON → {json_path}")

    # CSV summary
    csv_path = ARTIFACTS_ROOT / f"v3_sizing_calibration_{ts}.csv"
    csv_rows: list[dict] = []
    for arm_name, astats in aggregate.items():
        csv_rows.append({
            "arm": arm_name,
            "p_hit_mean": round(astats.get("p_target_before_ruin", {}).get("mean", 0), 4),
            "p_hit_median": round(astats.get("p_target_before_ruin", {}).get("median", 0), 4),
            "p_hit_p10": round(astats.get("p_target_before_ruin", {}).get("p10", 0), 4),
            "p_ruin_mean": round(astats.get("p_ruin", {}).get("mean", 0), 4),
            "p_ruin_p90": round(astats.get("p_ruin", {}).get("p90", 0), 4),
            "p_daily_mean": round(astats.get("p_daily_loss_breach", {}).get("mean", 0), 4),
            "dd_p95_mean": round(astats.get("dd_p95", {}).get("mean", 0), 0),
            "eq_p50_mean": round(astats.get("equity_p50", {}).get("mean", 0), 0),
            "eq_p10_mean": round(astats.get("equity_p10", {}).get("mean", 0), 0),
            "days_started_2c": round(astats.get("days_started_2c", {}).get("mean", 0), 1),
            "days_ever_2c": round(astats.get("days_ever_2c", {}).get("mean", 0), 1),
            "trades_at_2c": round(astats.get("trades_at_2c", {}).get("mean", 0), 1),
            "v3_any_upsize": round(astats.get("v3_any_upsize_days", {}).get("mean", 0), 1),
            "v3_traction": round(astats.get("v3_traction_days", {}).get("mean", 0), 1),
            "v3_first_win": round(astats.get("v3_first_win_days", {}).get("mean", 0), 1),
            "v3_orb_upsize": round(astats.get("v3_orb_upsize_days", {}).get("mean", 0), 1),
            "v3_orb_sessions": round(astats.get("v3_orb_sessions", {}).get("mean", 0), 1),
            "downshifts": round(astats.get("intraday_downshifts", {}).get("mean", 0), 1),
        })

    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  CSV → {csv_path}")

    print(f"\n  Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
