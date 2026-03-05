#!/usr/bin/env python3
"""
run_pb3_evaluation.py — End-to-end evaluation of ORB Pullback v3 engine.

Two experiments:
  1. Allocator comparison on trend20 / pilot_20d / stratified pool
     Arms: MR_ONLY, ORB_ONLY, ALLOC_V2_HYST
  2. 100-draw stratified robustness
     Arms: fixed_1c, fixed_2c, dynamic_v2

Generates base runs with pullback_v3 (tol=5.0, bars=3, touch_only),
then post-hoc scores each allocator/sizing arm via MC survival.

Usage:
    python run_pb3_evaluation.py
    python run_pb3_evaluation.py --skip-base-runs   # reuse existing
    python run_pb3_evaluation.py --skip-robustness   # allocator only
    python run_pb3_evaluation.py --skip-allocator    # robustness only
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
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
from validation.validation_pack import (
    SessionEntry,
    ValidationPack,
    ValidationPackRunner,
    load_pack,
)


# ── ORB Pullback v3 best config ────────────────────────────────────────
PB3_TOLERANCE = 5.0
PB3_MAX_BARS = 3
PB3_ENTRY_MODE = "touch_only"

# ── Trend session IDs (ADX upper tertile) ──────────────────────────────
TREND_SESSION_IDS = [
    "session_20251208", "session_20251212", "session_20251216", "session_20251217",
    "session_20251223", "session_20251224", "session_20251231", "session_20260105",
    "session_20260106", "session_20260112", "session_20260113", "session_20260120",
    "session_20260121", "session_20260128", "session_20260202", "session_20260203",
    "session_20260204", "session_20260209", "session_20260212", "session_20260218",
]

ARTIFACTS_ROOT = Path("artifacts/validation_runs")


# ═══════════════════════════════════════════════════════════════════════
#  Pack builders
# ═══════════════════════════════════════════════════════════════════════


def _build_trend_pack() -> ValidationPack:
    extended = load_pack("extended_60d")
    by_id = {s.session_id: s for s in extended.sessions}
    entries = [
        SessionEntry(
            session_id=s.session_id, start=s.start, end=s.end,
            category=s.category, symbol=s.symbol,
        )
        for sid in TREND_SESSION_IDS
        if (s := by_id.get(sid)) is not None
    ]
    return ValidationPack(
        pack_id="pb3eval_trend20",
        description="Trend-tertile sessions for pb3 evaluation",
        sessions=entries,
    )


def _build_random_packs(
    n_draws: int = 3,
    draw_size: int = 20,
    seed: int = 42,
) -> list[ValidationPack]:
    rng = random.Random(seed)
    extended = load_pack("extended_60d")
    all_sessions = list(extended.sessions)
    packs = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(packs) < n_draws and attempts < n_draws * 30:
        attempts += 1
        sample = rng.sample(all_sessions, draw_size)
        key = tuple(sorted(s.session_id for s in sample))
        if key in seen:
            continue
        seen.add(key)
        sample_sorted = sorted(sample, key=lambda s: s.start)
        packs.append(
            ValidationPack(
                pack_id=f"pb3eval_random{len(packs)+1:02d}",
                description=f"Random {draw_size}-session draw #{len(packs)+1} for pb3 eval",
                sessions=[
                    SessionEntry(
                        session_id=s.session_id, start=s.start, end=s.end,
                        category=s.category, symbol=s.symbol,
                    )
                    for s in sample_sorted
                ],
            )
        )
    return packs


# ═══════════════════════════════════════════════════════════════════════
#  Base run generation — runs replay with pullback_v3 ORB
# ═══════════════════════════════════════════════════════════════════════


def generate_base_run(pack: ValidationPack) -> str:
    """Run a validation pack with engine_mode=both and ORB pullback_v3."""
    runner = ValidationPackRunner(
        pack,
        artifacts_root=str(ARTIFACTS_ROOT),
        continue_on_error=True,
        mr_reclaim_mode="off",
        mr_regime_enabled=True,
        engine_mode="both",
        allocator_policy="none",
        orb_enabled=True,
        orb_trigger_mode="pullback_v3",
        orb_pullback_max_bars=PB3_MAX_BARS,
        orb_pullback_tolerance_pts=PB3_TOLERANCE,
        orb_pullback_entry_mode=PB3_ENTRY_MODE,
    )
    manifest = runner.run()
    return manifest.run_id


# ═══════════════════════════════════════════════════════════════════════
#  Trade frame builder (shared across experiments)
# ═══════════════════════════════════════════════════════════════════════


def _build_trade_frame(run_dir: Path) -> pd.DataFrame:
    """Build a trade dataframe with signal_type from trades + signals CSVs."""
    agg_csv = run_dir / "aggregate_trades.csv"
    if not agg_csv.is_file():
        return pd.DataFrame()

    trades = pd.read_csv(agg_csv)
    if trades.empty:
        return trades

    trades["session_id"] = trades["session_id"].astype(str)
    trades["signal_ts"] = pd.to_datetime(
        trades["signal_timestamp"].astype(str).str.replace(r"\+00:00$", "", regex=True),
        utc=True, errors="coerce",
    )
    trades["side"] = trades["side"].astype(str).str.upper()

    sig_rows: list[dict[str, Any]] = []
    for sig_csv in sorted((run_dir / "sessions").glob("*/signals.csv")):
        session_id = sig_csv.parent.name
        with sig_csv.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                if str(row.get("approved", "")).strip().lower() != "true":
                    continue
                sig_type = str(row.get("signal_type", "")).strip().upper()
                if sig_type not in {"MR", "ORB"}:
                    continue
                raw_ts = str(row.get("timestamp", "")).replace("+00:00", "")
                sig_ts = pd.Timestamp(raw_ts) if raw_ts else pd.NaT
                if pd.notna(sig_ts):
                    sig_ts = sig_ts.tz_localize("UTC") if sig_ts.tzinfo is None else sig_ts.tz_convert("UTC")
                sig_rows.append({
                    "session_id": session_id,
                    "signal_ts": sig_ts,
                    "side": str(row.get("side", "")).strip().upper(),
                    "signal_type": sig_type,
                })

    sig_df = pd.DataFrame(sig_rows)
    if sig_df.empty:
        trades["signal_type"] = "UNKNOWN"
        return trades

    merged = trades.merge(sig_df, on=["session_id", "signal_ts", "side"], how="left")
    merged["signal_type"] = merged["signal_type"].fillna("UNKNOWN")
    return merged


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 1 — Allocator comparison
# ═══════════════════════════════════════════════════════════════════════


def _session_early_adx(session_dir: Path, max_bars: int = 12) -> list[float]:
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


def _allocator_decision(session_dir: Path, kind: str) -> str:
    if kind == "mr":
        return "mr"
    if kind == "orb":
        return "orb"
    # v2 hysteresis
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


def _filter_trades(
    trades: pd.DataFrame, run_dir: Path, session_ids: list[str], kind: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    day_engine = {}
    for sid in session_ids:
        day_engine[sid] = _allocator_decision(run_dir / "sessions" / sid, kind)
    if trades.empty:
        return trades, day_engine
    if kind == "mr":
        return trades[trades["signal_type"] == "MR"].copy(), day_engine
    if kind == "orb":
        return trades[trades["signal_type"] == "ORB"].copy(), day_engine
    # Allocator: keep MR on mr-days, ORB on orb-days
    mask = trades.apply(
        lambda r: (
            (day_engine.get(str(r["session_id"]), "mr") == "mr" and str(r["signal_type"]).upper() == "MR")
            or (day_engine.get(str(r["session_id"]), "mr") == "orb" and str(r["signal_type"]).upper() == "ORB")
        ), axis=1,
    )
    return trades[mask].copy(), day_engine


def _mc_survival(r_values: list[float], session_ids: list[str]) -> dict[str, Any]:
    if not r_values:
        return {k: 0.0 for k in [
            "p_hit", "p_ruin", "p_daily_breach", "dd_p95",
            "equity_p10", "equity_p50",
        ]}
    sim = MonteCarloSurvivalSimulator()
    results = sim.run_all_scenarios(r_values, seed=42, session_ids=session_ids)
    base = results["base"]
    return {
        "p_hit": float(base.p_target_before_ruin),
        "p_ruin": float(base.p_ruin),
        "p_daily_breach": float(base.p_daily_loss_breach),
        "dd_p95": float(base.dd_p95),
        "equity_p10": float(base.equity_p10),
        "equity_p50": float(base.equity_p50),
    }


ALLOCATOR_MODES = [
    ("MR_ONLY", "mr"),
    ("ORB_ONLY", "orb"),
    ("ALLOC_V2_HYST", "v2"),
]


def run_allocator_comparison(
    run_dir: Path, session_ids: list[str], pack_key: str,
) -> list[dict[str, Any]]:
    """Compare allocator modes on a single base run."""
    trades = _build_trade_frame(run_dir)
    n_sessions = len(session_ids)
    results = []

    for label, kind in ALLOCATOR_MODES:
        filtered, day_engine = _filter_trades(trades, run_dir, session_ids, kind)
        r_vals = filtered["pnl_r"].dropna().astype(float).tolist() if not filtered.empty else []
        sids = filtered["session_id"].astype(str).tolist() if not filtered.empty else []
        wins = (filtered["pnl_r"] > 0).sum() if not filtered.empty else 0

        mc = _mc_survival(r_vals, sids)

        results.append({
            "pack": pack_key,
            "arm": label,
            "trades": len(r_vals),
            "trades_per_day": round(len(r_vals) / n_sessions, 2) if n_sessions else 0,
            "avg_r": round(float(np.mean(r_vals)), 4) if r_vals else 0.0,
            "wr": round(100.0 * wins / len(filtered), 1) if not filtered.empty and len(filtered) > 0 else 0.0,
            "orb_days": sum(1 for v in day_engine.values() if v == "orb"),
            "mr_days": sum(1 for v in day_engine.values() if v == "mr"),
            **mc,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 2 — 100-draw stratified robustness
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StratSession:
    session_id: str
    session_dir: Path
    adx_median: float
    atr_median: float
    regime_label: str  # range / mixed / trend
    n_trades: int
    source_run: str


SIZING_ARMS = ["fixed_1c", "fixed_2c", "dynamic_v2"]

MC_KEYS = [
    "p_target_before_ruin", "p_ruin", "p_daily_loss_breach",
    "dd_p95", "equity_p50", "equity_p10",
]


def _compute_session_adx(features_path: Path) -> tuple[float, float, int]:
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
        pool.append(StratSession(
            session_id=k,
            session_dir=v["trade_dir"],
            adx_median=v["adx_median"],
            atr_median=v["atr_median"],
            regime_label=label,
            n_trades=v["n_trades"],
            source_run=v["trade_run"],
        ))
    pool.sort(key=lambda s: s.adx_median)
    return pool, thresh_low, thresh_high


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


def rescore_draw(sessions: list[StratSession], arm: str) -> dict[str, Any]:
    from validation.sizing_policy import SizingPolicy
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
    result["days_at_2c_start"] = sum(1 for r in policy.daily_log if r.contracts_start == 2)
    result["intraday_downshifts"] = sum(1 for r in policy.daily_log if r.downshift_reason)
    result["vol_throttled_days"] = sum(
        1 for r in policy.daily_log if getattr(r, "vol_throttled", False)
    )
    result["earned_upsize_days"] = sum(
        1 for r in policy.daily_log if getattr(r, "earned_upsize_triggered", False)
    )
    return result


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


def aggregate_draw_results(draws: list[dict]) -> dict[str, dict[str, dict]]:
    by_arm: dict[str, list[dict]] = {}
    for d in draws:
        by_arm.setdefault(d["arm"], []).append(d)
    agg: dict[str, dict[str, dict]] = {}
    for arm, arm_draws in by_arm.items():
        stats: dict[str, dict] = {}
        for metric in MC_KEYS + [
            "trade_count", "final_equity",
            "days_at_2c_start", "intraday_downshifts",
            "vol_throttled_days", "earned_upsize_days",
        ]:
            values = [d.get(metric) for d in arm_draws if d.get(metric) is not None]
            stats[metric] = _percentile_stats([float(v) for v in values if v is not None])
        agg[arm] = stats
    return agg


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation of ORB Pullback v3 engine"
    )
    parser.add_argument("--skip-base-runs", action="store_true",
                        help="Reuse existing base runs (pass --base-run-ids)")
    parser.add_argument("--base-run-ids", nargs="+", default=None,
                        help="Existing base run IDs to reuse (requires --skip-base-runs)")
    parser.add_argument("--skip-allocator", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--n-draws", type=int, default=100)
    parser.add_argument("--draw-size", type=int, default=21)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_global = time.monotonic()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    output: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "orb_config": {
            "trigger_mode": "pullback_v3",
            "tolerance_pts": PB3_TOLERANCE,
            "max_bars": PB3_MAX_BARS,
            "entry_mode": PB3_ENTRY_MODE,
        },
    }

    # ════════════════════════════════════════════════════════════════════
    #  Step 1: Generate base runs with pullback_v3
    # ════════════════════════════════════════════════════════════════════
    base_run_ids: dict[str, str] = {}

    if args.skip_base_runs and args.base_run_ids:
        # Map from pack keys to provided IDs
        keys = ["trend20", "pilot_20d", "random01", "random02", "random03"]
        for i, rid in enumerate(args.base_run_ids):
            if i < len(keys):
                base_run_ids[keys[i]] = rid
    else:
        print(f"\n{'═'*70}")
        print("  STEP 1: Generating base runs with ORB pullback_v3")
        print(f"  tol={PB3_TOLERANCE} bars={PB3_MAX_BARS} entry={PB3_ENTRY_MODE}")
        print(f"{'═'*70}")

        # Trend pack
        print("\n  ▶ Pack: trend20")
        trend_pack = _build_trend_pack()
        base_run_ids["trend20"] = generate_base_run(trend_pack)
        print(f"    run_id = {base_run_ids['trend20']}")

        # Pilot pack
        print("\n  ▶ Pack: pilot_20d")
        pilot_pack = load_pack("pilot_20d")
        base_run_ids["pilot_20d"] = generate_base_run(pilot_pack)
        print(f"    run_id = {base_run_ids['pilot_20d']}")

        # Random packs
        random_packs = _build_random_packs(n_draws=3, draw_size=20, seed=42)
        for i, rp in enumerate(random_packs, 1):
            key = f"random{i:02d}"
            print(f"\n  ▶ Pack: {key}")
            base_run_ids[key] = generate_base_run(rp)
            print(f"    run_id = {base_run_ids[key]}")

    output["base_run_ids"] = base_run_ids
    print(f"\n  Base runs: {json.dumps(base_run_ids, indent=4)}")

    # ════════════════════════════════════════════════════════════════════
    #  Step 2: Allocator comparison
    # ════════════════════════════════════════════════════════════════════
    alloc_results: list[dict[str, Any]] = []

    if not args.skip_allocator:
        print(f"\n{'═'*70}")
        print("  STEP 2: Allocator Comparison (MR_ONLY / ORB_ONLY / ALLOC_V2_HYST)")
        print(f"{'═'*70}")

        for pack_key, run_id in base_run_ids.items():
            run_dir = ARTIFACTS_ROOT / run_id
            if not run_dir.is_dir():
                print(f"  ⚠ Skipping {pack_key}: {run_dir} not found")
                continue

            # Get session list from manifest
            manifest_path = run_dir / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                session_ids = [
                    s["session_id"] for s in manifest.get("sessions", [])
                    if s.get("success", True)
                ]
            else:
                session_ids = [
                    d.name for d in sorted((run_dir / "sessions").iterdir())
                    if d.is_dir() and d.name.startswith("session_")
                ]

            print(f"\n  ▶ Pack={pack_key} (run={run_id}, {len(session_ids)} sessions)")
            pack_results = run_allocator_comparison(run_dir, session_ids, pack_key)
            alloc_results.extend(pack_results)

            for r in pack_results:
                print(
                    f"    {r['arm']:<15} trades={r['trades']:>3} "
                    f"t/d={r['trades_per_day']:.2f} WR={r['wr']:.1f}% "
                    f"avg_r={r['avg_r']:+.4f} P_hit={r['p_hit']:.3f} "
                    f"P_ruin={r['p_ruin']:.4f} dd_p95=${r['dd_p95']:,.0f} "
                    f"eq_p50=${r['equity_p50']:,.0f}"
                )

        output["allocator_comparison"] = alloc_results

    # ════════════════════════════════════════════════════════════════════
    #  Step 3: 100-draw stratified robustness
    # ════════════════════════════════════════════════════════════════════
    robustness_output: dict[str, Any] = {}

    if not args.skip_robustness:
        print(f"\n{'═'*70}")
        print(f"  STEP 3: {args.n_draws}-draw Stratified Robustness")
        print(f"  Arms: {SIZING_ARMS}")
        print(f"{'═'*70}")

        rng = np.random.default_rng(args.seed)
        pool, thresh_low, thresh_high = pool_all_sessions(list(base_run_ids.values()))

        by_stratum = {"range": [], "mixed": [], "trend": []}
        for s in pool:
            by_stratum[s.regime_label].append(s)

        print(f"\n  Pool: {len(pool)} sessions (ADX classified)")
        print(f"  Thresholds: <{thresh_low:.1f} range, "
              f"{thresh_low:.1f}–{thresh_high:.1f} mixed, >{thresh_high:.1f} trend")
        for label in ("range", "mixed", "trend"):
            g = by_stratum[label]
            print(f"    {label}: {len(g)} sessions, {sum(s.n_trades for s in g)} trades")

        all_draw_results: list[dict] = []
        total_combos = args.n_draws * len(SIZING_ARMS)
        done = 0
        t0 = time.monotonic()

        for draw_idx in range(args.n_draws):
            draw_sessions = stratified_draw(pool, args.draw_size, rng)
            draw_sids = [s.session_id for s in draw_sessions]
            draw_strata = {
                label: sum(1 for s in draw_sessions if s.regime_label == label)
                for label in ("range", "mixed", "trend")
            }
            for arm in SIZING_ARMS:
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

        robustness_output = {
            "method": "stratified_adx_tertile",
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
        }
        output["robustness"] = robustness_output

        # Print summary
        print(f"\n{'═'*95}")
        print(f"  STRATIFIED ROBUSTNESS SUMMARY — {args.n_draws} draws × {args.draw_size} sessions")
        print(f"{'═'*95}")
        print(f"  {'Arm':<14} {'P_hit':>18} {'P_ruin':>18} "
              f"{'P_daily':>10} {'dd_p95':>10} {'Eq_p50':>10} {'Eq_p10':>10}")
        print(f"  {'':>14} {'mean/med/p10':>18} {'mean/p90/max':>18}")
        print(f"{'─'*95}")

        for arm in SIZING_ARMS:
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
                f"  {arm:<14} {ph_str:>18} {pr_str:>18} "
                f"{pd_.get('mean', 0):>10.4f} "
                f"${dd.get('mean', 0):>9,.0f} "
                f"${eq50.get('mean', 0):>9,.0f} "
                f"${eq10.get('mean', 0):>9,.0f}"
            )

        print(f"{'─'*95}")
        # Dynamic v2 diagnostics
        dv2 = aggregate.get("dynamic_v2")
        if dv2:
            vt = dv2.get("vol_throttled_days", {})
            eu = dv2.get("earned_upsize_days", {})
            d2c = dv2.get("days_at_2c_start", {})
            ds = dv2.get("intraday_downshifts", {})
            print(f"  dynamic_v2 diagnostics:")
            print(f"    vol_throttled_days  mean={vt.get('mean', 0):.1f}")
            print(f"    earned_upsize_days  mean={eu.get('mean', 0):.1f}")
            print(f"    days_at_2c         mean={d2c.get('mean', 0):.1f}/{args.draw_size}")
            print(f"    intraday_downshifts mean={ds.get('mean', 0):.1f}")
        print(f"{'═'*95}")

    # ════════════════════════════════════════════════════════════════════
    #  Step 4: Write summary artifacts
    # ════════════════════════════════════════════════════════════════════
    total_elapsed = time.monotonic() - t_global
    output["total_runtime_seconds"] = round(total_elapsed, 1)

    # Write JSON
    json_path = ARTIFACTS_ROOT / f"pb3_evaluation_{ts}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\n  JSON → {json_path}")

    # Write CSV summary
    csv_path = ARTIFACTS_ROOT / f"pb3_evaluation_{ts}.csv"
    csv_rows: list[dict] = []

    # Allocator rows
    for r in alloc_results:
        csv_rows.append({
            "experiment": "allocator",
            "pack": r["pack"],
            "arm": r["arm"],
            "trades": r["trades"],
            "trades_per_day": r["trades_per_day"],
            "avg_r": r["avg_r"],
            "wr": r["wr"],
            "p_hit": round(r["p_hit"], 4),
            "p_ruin": round(r["p_ruin"], 4),
            "p_daily_breach": round(r["p_daily_breach"], 4),
            "dd_p95": round(r["dd_p95"], 0),
            "equity_p50": round(r["equity_p50"], 0),
            "equity_p10": round(r["equity_p10"], 0),
        })

    # Robustness rows (aggregate means)
    if robustness_output:
        agg = robustness_output.get("aggregate", {})
        for arm_name, astats in agg.items():
            csv_rows.append({
                "experiment": "robustness_100draw",
                "pack": "stratified_pool",
                "arm": arm_name,
                "trades": round(astats.get("trade_count", {}).get("mean", 0)),
                "trades_per_day": round(
                    astats.get("trade_count", {}).get("mean", 0) / args.draw_size, 2
                ) if args.draw_size else 0,
                "avg_r": "",
                "wr": "",
                "p_hit": round(astats.get("p_target_before_ruin", {}).get("mean", 0), 4),
                "p_ruin": round(astats.get("p_ruin", {}).get("mean", 0), 4),
                "p_daily_breach": round(astats.get("p_daily_loss_breach", {}).get("mean", 0), 4),
                "dd_p95": round(astats.get("dd_p95", {}).get("mean", 0), 0),
                "equity_p50": round(astats.get("equity_p50", {}).get("mean", 0), 0),
                "equity_p10": round(astats.get("equity_p10", {}).get("mean", 0), 0),
            })

    if csv_rows:
        fieldnames = [
            "experiment", "pack", "arm", "trades", "trades_per_day",
            "avg_r", "wr", "p_hit", "p_ruin", "p_daily_breach",
            "dd_p95", "equity_p50", "equity_p10",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"  CSV → {csv_path}")

    print(f"\n  Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
