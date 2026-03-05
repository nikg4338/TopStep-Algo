from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import config
from simulation.mc_survival import MonteCarloSurvivalSimulator
from validation.validation_pack import SessionEntry, ValidationPack, ValidationPackRunner, load_pack


@dataclass(frozen=True)
class ModeSpec:
    label: str
    kind: str
    allocator_v1_adx_threshold: float = 25.0


@dataclass(frozen=True)
class PackSpec:
    key: str
    pack: ValidationPack
    base_run_id: str | None = None


def _pack_from_session_ids(pack_id: str, description: str, source_pack: ValidationPack, session_ids: list[str]) -> ValidationPack:
    by_id = {s.session_id: s for s in source_pack.sessions}
    sessions: list[SessionEntry] = []
    for sid in session_ids:
        s = by_id.get(sid)
        if s is None:
            continue
        sessions.append(
            SessionEntry(
                session_id=s.session_id,
                start=s.start,
                end=s.end,
                category=s.category,
                symbol=s.symbol,
                tags=list(s.tags),
                notes=s.notes,
            )
        )
    return ValidationPack(pack_id=pack_id, description=description, sessions=sessions)


def _load_trend_pack_from_run(source_run_id: str) -> ValidationPack:
    run_dir = Path("artifacts/validation_runs") / source_run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"trend source manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session_ids = [str(s.get("session_id")) for s in manifest.get("sessions", []) if s.get("session_id")]
    extended = load_pack("extended_60d")
    return _pack_from_session_ids(
        pack_id="trend20_adx_source",
        description=f"Trend-selected sessions sourced from {source_run_id}",
        source_pack=extended,
        session_ids=session_ids,
    )


def _make_random_packs(extended: ValidationPack, *, n_draws: int, draw_size: int, seed: int) -> list[ValidationPack]:
    rng = random.Random(seed)
    all_sessions = list(extended.sessions)
    if draw_size > len(all_sessions):
        raise ValueError(f"draw_size={draw_size} exceeds available sessions={len(all_sessions)}")

    packs: list[ValidationPack] = []
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
                pack_id=f"random20_{len(packs)+1:02d}",
                description=f"Random {draw_size}-session draw #{len(packs)+1} from extended_60d (seed={seed})",
                sessions=[
                    SessionEntry(
                        session_id=s.session_id,
                        start=s.start,
                        end=s.end,
                        category=s.category,
                        symbol=s.symbol,
                        tags=list(s.tags),
                        notes=s.notes,
                    )
                    for s in sample_sorted
                ],
            )
        )
    return packs


def _build_trade_frame(run_dir: Path) -> pd.DataFrame:
    agg_csv = run_dir / "aggregate_trades.csv"
    if not agg_csv.is_file():
        return pd.DataFrame()

    trades = pd.read_csv(agg_csv)
    if trades.empty:
        return trades

    trades["session_id"] = trades["session_id"].astype(str)
    trades["signal_ts"] = pd.to_datetime(trades["signal_timestamp"], utc=True, errors="coerce")
    trades["side"] = trades["side"].astype(str).str.upper()

    sig_rows: list[dict[str, Any]] = []
    for sig_csv in sorted((run_dir / "sessions").glob("*/signals.csv")):
        session_id = sig_csv.parent.name
        with sig_csv.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                approved = str(row.get("approved", "")).strip().lower() == "true"
                if not approved:
                    continue
                sig_type = str(row.get("signal_type", "")).strip().upper()
                if sig_type not in {"MR", "ORB"}:
                    continue
                raw_ts = row.get("timestamp", "")
                sig_ts = pd.Timestamp(str(raw_ts)) if raw_ts else pd.NaT
                if pd.notna(sig_ts):
                    if sig_ts.tzinfo is None:
                        sig_ts = sig_ts.tz_localize("UTC")
                    else:
                        sig_ts = sig_ts.tz_convert("UTC")
                sig_rows.append(
                    {
                        "session_id": session_id,
                        "signal_ts": sig_ts,
                        "side": str(row.get("side", "")).strip().upper(),
                        "signal_type": sig_type,
                    }
                )

    sig_df = pd.DataFrame(sig_rows)
    if sig_df.empty:
        trades["signal_type"] = "UNKNOWN"
        return trades

    merged = trades.merge(sig_df, on=["session_id", "signal_ts", "side"], how="left")
    merged["signal_type"] = merged["signal_type"].fillna("UNKNOWN")
    return merged


def _session_early_adx(session_dir: Path, max_bars: int = 12) -> list[float]:
    features_path = session_dir / "features_snapshot.csv"
    if not features_path.is_file():
        return []

    try:
        df = pd.read_csv(features_path)
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
        hhmm = et.strftime("%H:%M")
        if config.RTH_OPEN <= hhmm and float(a) > 0:
            out.append(float(a))
            if len(out) >= max_bars:
                break
    return out


def _allocator_day_engine(run_dir: Path, session_id: str, mode: ModeSpec) -> str:
    if mode.kind == "mr":
        return "mr"
    if mode.kind == "orb":
        return "orb"

    adx_series = _session_early_adx(run_dir / "sessions" / session_id, max_bars=12)
    if mode.kind == "v1":
        open_adx = adx_series[0] if adx_series else 0.0
        return "orb" if open_adx >= mode.allocator_v1_adx_threshold else "mr"

    if mode.kind == "v2":
        trend_open = any(v >= 25.0 for v in adx_series)
        rising_seq = adx_series[-3:]
        rising_ok = len(rising_seq) >= 3 and all(v > 20.0 for v in rising_seq) and all(rising_seq[i] < rising_seq[i + 1] for i in range(len(rising_seq) - 1))
        range_seq = adx_series[-3:]
        range_ok = len(range_seq) >= 3 and all(v <= 18.0 for v in range_seq)
        if trend_open or rising_ok:
            return "orb"
        if range_ok:
            return "mr"
        return "mr"

    return "mr"


def _filter_mode_trades(run_dir: Path, session_ids: list[str], mode: ModeSpec) -> tuple[pd.DataFrame, dict[str, str]]:
    trades = _build_trade_frame(run_dir)
    day_engine = {sid: _allocator_day_engine(run_dir, sid, mode) for sid in session_ids}

    if trades.empty:
        return trades, day_engine

    if mode.kind == "mr":
        filtered = trades[trades["signal_type"] == "MR"].copy()
    elif mode.kind == "orb":
        filtered = trades[trades["signal_type"] == "ORB"].copy()
    else:
        mask = trades.apply(
            lambda r: (
                (day_engine.get(str(r["session_id"]), "mr") == "mr" and str(r["signal_type"]).upper() == "MR")
                or (day_engine.get(str(r["session_id"]), "mr") == "orb" and str(r["signal_type"]).upper() == "ORB")
            ),
            axis=1,
        )
        filtered = trades[mask].copy()

    return filtered, day_engine


def _mode_metrics(filtered: pd.DataFrame) -> dict[str, Any]:
    if filtered.empty:
        return {
            "decision_metrics": {
                "p_target_before_ruin": 0.0,
                "p_ruin": 0.0,
                "p_daily_loss_breach": 0.0,
                "dd_p95": 0.0,
                "final_pnl_median": 0.0,
                "final_pnl_p10": 0.0,
            },
            "diagnostics": {
                "trade_count_total": 0,
                "losing_streak_max": 0,
                "losing_streak_p95": 0.0,
                "per_trade_r_distribution": [],
                "per_day_trade_count_distribution": [],
                "per_day_pnl_distribution": [],
            },
            "stress": {},
        }

    r_values = filtered["pnl_r"].dropna().astype(float).tolist()
    session_ids = filtered["session_id"].astype(str).tolist()

    sim = MonteCarloSurvivalSimulator()
    all_results = sim.run_all_scenarios(r_values, seed=42, session_ids=session_ids)
    base = all_results["base"]

    per_day_trade_count = filtered.groupby("session_id").size().astype(int).tolist()
    per_day_pnl = filtered.groupby("session_id")["pnl_dollars"].sum().astype(float).tolist()

    loss_streak_max = 0
    curr = 0
    for val in filtered["pnl_r"].astype(float).tolist():
        if val <= 0:
            curr += 1
            loss_streak_max = max(loss_streak_max, curr)
        else:
            curr = 0

    stress = {}
    for name in ("mild", "severe", "tilt_bad_week"):
        if name not in all_results:
            continue
        s = all_results[name]
        stress[name] = {
            "p_target_before_ruin": float(s.p_target_before_ruin),
            "p_ruin": float(s.p_ruin),
            "dd_p95": float(s.dd_p95),
        }

    return {
        "decision_metrics": {
            "p_target_before_ruin": float(base.p_target_before_ruin),
            "p_ruin": float(base.p_ruin),
            "p_daily_loss_breach": float(base.p_daily_loss_breach),
            "dd_p95": float(base.dd_p95),
            "final_pnl_median": float(base.equity_p50),
            "final_pnl_p10": float(base.equity_p10),
        },
        "diagnostics": {
            "trade_count_total": int(len(filtered)),
            "losing_streak_max": int(loss_streak_max),
            "losing_streak_p95": float(base.losing_streak_p95),
            "per_trade_r_distribution": r_values,
            "per_day_trade_count_distribution": per_day_trade_count,
            "per_day_pnl_distribution": per_day_pnl,
        },
        "stress": stress,
    }


def _ensure_base_run(pack: ValidationPack, *, artifacts_root: str, mr_reclaim_mode: str, mr_regime_enabled: bool) -> str:
    runner = ValidationPackRunner(
        pack,
        artifacts_root=artifacts_root,
        continue_on_error=True,
        mr_reclaim_mode=mr_reclaim_mode,
        mr_regime_enabled=mr_regime_enabled,
        engine_mode="both",
        allocator_policy="none",
        orb_enabled=True,
        orb_trigger_mode="break",
    )
    manifest = runner.run()
    return manifest.run_id


def run_engine_mode_matrix(
    *,
    artifacts_root: str = "artifacts/validation_runs",
    trend_source_run_id: str = "trend20_adx_20260226_232222",
    random_draws: int = 3,
    random_seed: int = 42,
    random_draw_size: int = 20,
    mr_reclaim_mode: str = "off",
    mr_regime_enabled: bool = True,
) -> dict[str, Any]:
    trend_pack = _load_trend_pack_from_run(trend_source_run_id)
    pilot_pack = load_pack("pilot_20d")
    extended = load_pack("extended_60d")
    random_packs = _make_random_packs(
        extended,
        n_draws=random_draws,
        draw_size=random_draw_size,
        seed=random_seed,
    )

    pack_specs: list[PackSpec] = [
        PackSpec(key="trend20", pack=trend_pack, base_run_id=trend_source_run_id),
        PackSpec(key="pilot_20d", pack=pilot_pack),
    ]
    pack_specs.extend(PackSpec(key=p.pack_id, pack=p) for p in random_packs)

    mode_specs = [
        ModeSpec(label="MR_ONLY", kind="mr"),
        ModeSpec(label="ORB_ONLY", kind="orb"),
        ModeSpec(label="ALLOC_V1_ADX20", kind="v1", allocator_v1_adx_threshold=20.0),
        ModeSpec(label="ALLOC_V1_ADX25", kind="v1", allocator_v1_adx_threshold=25.0),
        ModeSpec(label="ALLOC_V1_ADX30", kind="v1", allocator_v1_adx_threshold=30.0),
        ModeSpec(label="ALLOC_V2_HYST", kind="v2"),
    ]

    results: list[dict[str, Any]] = []
    for pack_spec in pack_specs:
        base_run_id = pack_spec.base_run_id
        if not base_run_id:
            print(f"\n[MATRIX] generating base run for pack={pack_spec.key}")
            base_run_id = _ensure_base_run(
                pack_spec.pack,
                artifacts_root=artifacts_root,
                mr_reclaim_mode=mr_reclaim_mode,
                mr_regime_enabled=mr_regime_enabled,
            )
        base_run_dir = Path(artifacts_root) / base_run_id

        for mode in mode_specs:
            print(f"[MATRIX] evaluating pack={pack_spec.key} mode={mode.label} from base_run={base_run_id}")
            filtered, day_engine = _filter_mode_trades(
                base_run_dir,
                [s.session_id for s in pack_spec.pack.sessions],
                mode,
            )
            metrics = _mode_metrics(filtered)
            results.append(
                {
                    "pack": {
                        "key": pack_spec.key,
                        "pack_id": pack_spec.pack.pack_id,
                        "description": pack_spec.pack.description,
                        "session_ids": [s.session_id for s in pack_spec.pack.sessions],
                        "base_run_id": base_run_id,
                    },
                    "mode": {
                        "label": mode.label,
                        "kind": mode.kind,
                        "allocator_v1_adx_threshold": mode.allocator_v1_adx_threshold,
                        "day_engine": day_engine,
                    },
                    **metrics,
                }
            )

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "config": {
            "trend_source_run_id": trend_source_run_id,
            "random_draws": random_draws,
            "random_seed": random_seed,
            "random_draw_size": random_draw_size,
            "mr_reclaim_mode": mr_reclaim_mode,
            "mr_regime_enabled": mr_regime_enabled,
        },
        "results": results,
    }

    out_path = Path(artifacts_root) / f"engine_mode_matrix_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[MATRIX] wrote {out_path}")
    summary["output_path"] = str(out_path)
    return summary
