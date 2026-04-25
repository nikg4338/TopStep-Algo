"""
experiments/run_orb_diagnostics.py — ORB diagnostic labeling report.

Builds a deterministic label set for ORB opportunities from an existing
validation run or an existing ORB autopsy CSV. This is a diagnostics tool only;
it does not change ORB trading rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "orb_diagnostics"
DEFAULT_GOOD_R = 0.75
DEFAULT_GOOD_MFE_R = 1.0
DEFAULT_BAD_R = -0.75
DEFAULT_BAD_MAE_R = 1.0
DEFAULT_NEUTRAL_R = 0.25
OUTPUT_COLUMNS = [
    "date",
    "session_id",
    "source_run_id",
    "opportunity_id",
    "opening_range_width",
    "atr",
    "atr_regime",
    "opening_impulse",
    "one_sidedness_score",
    "vwap_relationship",
    "pullback_depth",
    "breakout_direction",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "final_r",
    "label",
]


@dataclass(frozen=True)
class OrbLabelThresholds:
    """Deterministic ORB label thresholds, expressed in R where applicable."""

    good_r: float = DEFAULT_GOOD_R
    good_mfe_r: float = DEFAULT_GOOD_MFE_R
    bad_r: float = DEFAULT_BAD_R
    bad_mae_r: float = DEFAULT_BAD_MAE_R
    neutral_r: float = DEFAULT_NEUTRAL_R


@dataclass(frozen=True)
class OrbDiagnosticRow:
    """Normalized ORB diagnostic row."""

    date: str
    session_id: str
    source_run_id: str
    opportunity_id: str
    opening_range_width: float | None
    atr: float | None
    atr_regime: str
    opening_impulse: float | None
    one_sidedness_score: float | None
    vwap_relationship: str
    pullback_depth: float | None
    breakout_direction: str
    max_favorable_excursion: float | None
    max_adverse_excursion: float | None
    final_r: float | None
    label: str


def safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None for blanks and invalid values."""
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, returning an empty dict for missing/invalid files."""
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def classify_atr_regime(atr: float | None = None, atr_percentile: float | None = None) -> str:
    """Classify ATR context from percentile when available, else coarse ATR."""
    if atr_percentile is not None:
        if atr_percentile <= 33.0:
            return "low"
        if atr_percentile <= 66.0:
            return "medium"
        return "high"

    if atr is None:
        return "unknown"
    if atr < 8.0:
        return "low"
    if atr < 16.0:
        return "medium"
    return "high"


def classify_vwap_relationship(candidate_price: float | None, vwap: float | None, tolerance: float = 0.25) -> str:
    """Return candidate price relationship to VWAP."""
    if candidate_price is None or vwap is None:
        return "unknown"
    diff = candidate_price - vwap
    if abs(diff) <= tolerance:
        return "at_vwap"
    return "above_vwap" if diff > 0 else "below_vwap"


def label_orb_result(
    *,
    has_valid_setup: bool,
    final_r: float | None,
    max_favorable_r: float | None,
    max_adverse_r: float | None,
    thresholds: OrbLabelThresholds | None = None,
) -> str:
    """Classify an ORB opportunity into good/bad/neutral/no-trade."""
    if not has_valid_setup:
        return "no-trade"

    limits = thresholds or OrbLabelThresholds()
    final = final_r if final_r is not None else 0.0
    mfe = max_favorable_r if max_favorable_r is not None else 0.0
    mae = max_adverse_r if max_adverse_r is not None else 0.0

    if mfe >= limits.good_mfe_r or final >= limits.good_r:
        return "good_orb"
    if final <= limits.bad_r or mae >= limits.bad_mae_r:
        return "bad_orb"
    if abs(final) <= limits.neutral_r:
        return "neutral_orb"
    return "neutral_orb"


def _session_date(session_id: str) -> str:
    return session_id.replace("session_", "")


def _source_run_id(run_dir: Path) -> str:
    manifest = load_json(run_dir / "manifest.json")
    return str(manifest.get("run_id") or run_dir.name)


def _load_features(session_dir: Path) -> dict[str, float | None]:
    features = session_dir / "features_snapshot.csv"
    if not features.is_file():
        return {"atr": None, "atr_percentile": None}
    try:
        df = pd.read_csv(features)
    except Exception:
        return {"atr": None, "atr_percentile": None}
    out: dict[str, float | None] = {}
    for col in ("atr", "atr_percentile"):
        if col not in df.columns:
            out[col] = None
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        vals = vals[vals > 0]
        out[col] = float(vals.median()) if not vals.empty else None
    return out


def _load_opening_context(session_id: str, data_root: Path) -> dict[str, float | None]:
    """Compute OR width and opening impulse from cached MES tick bars."""
    date = _session_date(session_id)
    tick_path = data_root / "MES.c.0" / "trades" / f"{date}_143000__{date}_210000.parquet"
    if not tick_path.is_file():
        return {
            "opening_range_width": None,
            "opening_impulse": None,
            "one_sidedness_score": None,
        }
    try:
        ticks = pd.read_parquet(tick_path)
    except Exception:
        return {
            "opening_range_width": None,
            "opening_impulse": None,
            "one_sidedness_score": None,
        }
    if ticks.empty or "timestamp" not in ticks.columns or "price" not in ticks.columns:
        return {
            "opening_range_width": None,
            "opening_impulse": None,
            "one_sidedness_score": None,
        }

    ticks = ticks.copy()
    ticks["timestamp"] = pd.to_datetime(ticks["timestamp"], utc=True, errors="coerce")
    ticks = ticks.dropna(subset=["timestamp", "price"]).set_index("timestamp").sort_index()
    bars = ticks["price"].resample("5min").agg(open="first", high="max", low="min", close="last").dropna()
    if bars.empty:
        return {
            "opening_range_width": None,
            "opening_impulse": None,
            "one_sidedness_score": None,
        }

    opening = bars.head(3)
    or_high = safe_float(opening["high"].max())
    or_low = safe_float(opening["low"].min())
    or_open = safe_float(opening["open"].iloc[0])
    or_close = safe_float(opening["close"].iloc[-1])
    if or_high is None or or_low is None or or_open is None or or_close is None:
        width = impulse = one_sidedness = None
    else:
        width = max(or_high - or_low, 0.0)
        impulse = (or_close - or_open) / width if width else 0.0
        one_sidedness = abs(impulse)
    return {
        "opening_range_width": round(width, 4) if width is not None else None,
        "opening_impulse": round(impulse, 4) if impulse is not None else None,
        "one_sidedness_score": round(one_sidedness, 4) if one_sidedness is not None else None,
    }


def _orb_diagnostics_by_index(session_summary: dict[str, Any]) -> dict[int, dict[str, Any]]:
    orb_funnel = session_summary.get("orb_funnel", {}) or {}
    diagnostics = orb_funnel.get("pullback_v3_diagnostics", []) or []
    return {idx: row for idx, row in enumerate(diagnostics)}


def _read_approved_orb_signals(session_dir: Path) -> list[dict[str, Any]]:
    signals_path = session_dir / "signals.csv"
    if not signals_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with signals_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("signal_type", "")).strip().upper() != "ORB":
                continue
            if str(row.get("approved", "")).strip().lower() != "true":
                continue
            rows.append(row)
    return rows


def _read_trades_by_signal(session_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    trades_path = session_dir / "trades.csv"
    if not trades_path.is_file():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with trades_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (str(row.get("signal_timestamp", "")).replace("+00:00", ""), str(row.get("side", "")).upper())
            out[key] = row
    return out


def rows_from_validation_run(run_dir: Path, data_root: Path, thresholds: OrbLabelThresholds) -> list[OrbDiagnosticRow]:
    """Build ORB diagnostic rows from a completed validation run directory."""
    sessions_dir = run_dir / "sessions"
    if not sessions_dir.is_dir():
        raise FileNotFoundError(f"No sessions directory found under {run_dir}")

    source_id = _source_run_id(run_dir)
    rows: list[OrbDiagnosticRow] = []
    for session_dir in sorted(p for p in sessions_dir.iterdir() if p.is_dir()):
        session_id = session_dir.name
        session_summary = load_json(session_dir / "session_summary.json")
        features = _load_features(session_dir)
        opening = _load_opening_context(session_id, data_root)
        diagnostics = _orb_diagnostics_by_index(session_summary)
        signals = _read_approved_orb_signals(session_dir)
        trades_by_signal = _read_trades_by_signal(session_dir)
        atr = features.get("atr")
        atr_regime = classify_atr_regime(atr, features.get("atr_percentile"))

        if not signals:
            rows.append(
                OrbDiagnosticRow(
                    date=_session_date(session_id),
                    session_id=session_id,
                    source_run_id=source_id,
                    opportunity_id=f"{session_id}:no_trade",
                    opening_range_width=opening["opening_range_width"],
                    atr=atr,
                    atr_regime=atr_regime,
                    opening_impulse=opening["opening_impulse"],
                    one_sidedness_score=opening["one_sidedness_score"],
                    vwap_relationship="unknown",
                    pullback_depth=None,
                    breakout_direction="none",
                    max_favorable_excursion=None,
                    max_adverse_excursion=None,
                    final_r=None,
                    label="no-trade",
                )
            )
            continue

        for idx, signal in enumerate(signals):
            signal_ts = str(signal.get("timestamp", "")).replace("+00:00", "")
            side = str(signal.get("side", "")).upper()
            trade = trades_by_signal.get((signal_ts, side), {})
            final_r = safe_float(trade.get("pnl_r"))
            mfe_points = safe_float(trade.get("mfe_points"))
            mae_points = safe_float(trade.get("mae_points"))
            entry = safe_float(trade.get("entry_price"))
            stop = safe_float(trade.get("stop_price"))
            risk_points = abs(entry - stop) if entry is not None and stop is not None else None
            max_favorable_r = (mfe_points / risk_points) if mfe_points is not None and risk_points else None
            max_adverse_r = (mae_points / risk_points) if mae_points is not None and risk_points else None
            diagnostic = diagnostics.get(idx, {})
            candidate_price = safe_float(signal.get("candidate_price"))
            vwap = safe_float(signal.get("vwap"))

            rows.append(
                OrbDiagnosticRow(
                    date=_session_date(session_id),
                    session_id=session_id,
                    source_run_id=source_id,
                    opportunity_id=f"{session_id}:orb_{idx + 1}",
                    opening_range_width=opening["opening_range_width"],
                    atr=atr,
                    atr_regime=atr_regime,
                    opening_impulse=opening["opening_impulse"],
                    one_sidedness_score=opening["one_sidedness_score"],
                    vwap_relationship=classify_vwap_relationship(candidate_price, vwap),
                    pullback_depth=safe_float(diagnostic.get("pullback_depth_pts")),
                    breakout_direction=side or str(diagnostic.get("breakout_direction", "")).upper(),
                    max_favorable_excursion=round(max_favorable_r, 4) if max_favorable_r is not None else None,
                    max_adverse_excursion=round(max_adverse_r, 4) if max_adverse_r is not None else None,
                    final_r=final_r,
                    label=label_orb_result(
                        has_valid_setup=bool(trade),
                        final_r=final_r,
                        max_favorable_r=max_favorable_r,
                        max_adverse_r=max_adverse_r,
                        thresholds=thresholds,
                    ),
                )
            )
    return rows


def rows_from_autopsy_csv(path: Path, thresholds: OrbLabelThresholds) -> list[OrbDiagnosticRow]:
    """Normalize an existing ORB autopsy dataset CSV into the diagnostics schema."""
    rows: list[OrbDiagnosticRow] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            session_id = str(row.get("session_id", ""))
            label = str(row.get("label", "")).strip()
            if label not in {"good_orb", "bad_orb", "neutral_orb", "no-trade"}:
                session_pnl = safe_float(row.get("session_pnl"))
                label = label_orb_result(
                    has_valid_setup=str(row.get("route", "")).lower() == "orb",
                    final_r=session_pnl,
                    max_favorable_r=None,
                    max_adverse_r=None,
                    thresholds=thresholds,
                )
            atr = safe_float(row.get("atr"))
            rows.append(
                OrbDiagnosticRow(
                    date=str(row.get("date", "")),
                    session_id=session_id,
                    source_run_id=str(row.get("source_run_id", path.parent.name)),
                    opportunity_id=f"{session_id}:autopsy_{idx + 1}",
                    opening_range_width=safe_float(row.get("opening_range_width")),
                    atr=atr,
                    atr_regime=classify_atr_regime(atr),
                    opening_impulse=safe_float(row.get("impulse")),
                    one_sidedness_score=safe_float(row.get("one_sidedness")),
                    vwap_relationship="unknown",
                    pullback_depth=None,
                    breakout_direction=str(row.get("breakout_direction", "") or "unknown").upper(),
                    max_favorable_excursion=None,
                    max_adverse_excursion=None,
                    final_r=None,
                    label=label,
                )
            )
    return rows


def label_counts(rows: list[OrbDiagnosticRow]) -> dict[str, int]:
    counts = {"good_orb": 0, "bad_orb": 0, "neutral_orb": 0, "no-trade": 0}
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1
    return counts


def rows_to_markdown(rows: list[OrbDiagnosticRow], title: str) -> str:
    counts = label_counts(rows)
    lines = [
        f"# {title}",
        "",
        f"- Rows: {len(rows)}",
        f"- Labels: {counts}",
        "",
        "| Label | Count | Avg Final R | Avg MFE R | Avg MAE R |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("good_orb", "bad_orb", "neutral_orb", "no-trade"):
        subset = [r for r in rows if r.label == label]
        final_rs = [r.final_r for r in subset if r.final_r is not None]
        mfes = [r.max_favorable_excursion for r in subset if r.max_favorable_excursion is not None]
        maes = [r.max_adverse_excursion for r in subset if r.max_adverse_excursion is not None]
        avg_final = sum(final_rs) / len(final_rs) if final_rs else 0.0
        avg_mfe = sum(mfes) / len(mfes) if mfes else 0.0
        avg_mae = sum(maes) / len(maes) if maes else 0.0
        lines.append(f"| {label} | {len(subset)} | {avg_final:.4f} | {avg_mfe:.4f} | {avg_mae:.4f} |")
    lines.append("")
    return "\n".join(lines)


def write_outputs(rows: list[OrbDiagnosticRow], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "orb_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    json_path = output_dir / "orb_diagnostics.json"
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8")

    md_path = output_dir / "orb_diagnostics_summary.md"
    md_path.write_text(rows_to_markdown(rows, "ORB Diagnostics"), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def find_latest_autopsy_csv(root: Path) -> Path | None:
    candidates = sorted(root.glob("candidate_reports/orb_autopsy_dataset_*/orb_autopsy_dataset.csv"))
    return candidates[-1] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic ORB diagnostic labels")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-run", type=Path, help="Completed validation run directory")
    source.add_argument("--autopsy-csv", type=Path, help="Existing ORB autopsy dataset CSV")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=f"orb_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "cache")
    parser.add_argument("--good-r", type=float, default=DEFAULT_GOOD_R)
    parser.add_argument("--good-mfe-r", type=float, default=DEFAULT_GOOD_MFE_R)
    parser.add_argument("--bad-r", type=float, default=DEFAULT_BAD_R)
    parser.add_argument("--bad-mae-r", type=float, default=DEFAULT_BAD_MAE_R)
    parser.add_argument("--neutral-r", type=float, default=DEFAULT_NEUTRAL_R)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = OrbLabelThresholds(
        good_r=args.good_r,
        good_mfe_r=args.good_mfe_r,
        bad_r=args.bad_r,
        bad_mae_r=args.bad_mae_r,
        neutral_r=args.neutral_r,
    )

    if args.source_run:
        rows = rows_from_validation_run(args.source_run, args.data_root, thresholds)
        source_label = str(args.source_run)
    else:
        autopsy_csv = args.autopsy_csv or find_latest_autopsy_csv(PROJECT_ROOT / "artifacts")
        if autopsy_csv is None:
            raise FileNotFoundError("No --source-run provided and no ORB autopsy dataset was found")
        rows = rows_from_autopsy_csv(autopsy_csv, thresholds)
        source_label = str(autopsy_csv)

    output_dir = args.output_root / args.run_id
    artifacts = write_outputs(rows, output_dir)

    print(f"ORB diagnostics source: {source_label}")
    print(f"Rows: {len(rows)}")
    print(f"Labels: {label_counts(rows)}")
    print("Artifacts:")
    for kind, path in artifacts.items():
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
