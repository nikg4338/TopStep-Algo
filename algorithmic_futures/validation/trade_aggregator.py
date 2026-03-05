"""
validation/trade_aggregator.py — Aggregate per-session trades into run-level outputs.

Discovers ``trades.csv`` files under ``<run_root>/sessions/**/``,
concatenates them, computes aggregate statistics, and writes:
  - ``<run_root>/aggregate_trades.csv``
  - ``<run_root>/aggregate_metrics.json``

Empty session CSVs (header-only) are silently skipped.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────


def _percentile(values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile (0-100) of *values*."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = (pct / 100.0) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    w = idx - lo
    return s[lo] * (1 - w) + s[hi] * w


def _max_losing_streak(pnl_r_values: list[float]) -> int:
    """Compute the longest consecutive-loss streak (pnl_r <= 0)."""
    best = 0
    curr = 0
    for r in pnl_r_values:
        if r <= 0:
            curr += 1
            best = max(best, curr)
        else:
            curr = 0
    return best


# ── Public API ───────────────────────────────────────────────────────────


def aggregate_trades(
    run_root: str | Path,
    *,
    min_trade_count: int = 10,
    early_stop_n_bars: int = 3,
) -> dict[str, Any]:
    """Discover, concatenate, and summarise all per-session trades.

    Parameters
    ----------
    run_root:
        Path to a completed validation run directory (must contain
        ``sessions/`` sub-tree).
    min_trade_count:
        If the total trade count is below this value the output
        ``readiness`` flag is set to ``false``.

    Returns
    -------
    dict
        The aggregate-metrics dictionary (also written to JSON on disk).
    """
    run_root = Path(run_root)
    sessions_dir = run_root / "sessions"

    # ── Discover trades.csv files ────────────────────────────────────
    csv_paths = sorted(sessions_dir.glob("**/trades.csv")) if sessions_dir.is_dir() else []

    print(f"  [aggregator] scanning {sessions_dir}")
    for p in csv_paths:
        print(f"    found: {p}")

    # ── Read & concatenate ──────────────────────────────────────────
    all_rows: list[dict[str, str]] = []
    sessions_total = 0
    sessions_with_trades = 0
    per_session_counts: list[tuple[str, int]] = []

    for csv_path in csv_paths:
        session_id = csv_path.parent.name
        sessions_total += 1
        rows = _read_csv_rows(csv_path)
        count = len(rows)
        per_session_counts.append((session_id, count))

        if count == 0:
            print(f"    {session_id}: 0 rows (skipped)")
            continue

        sessions_with_trades += 1
        for row in rows:
            # Enrich with provenance columns
            row.setdefault("run_id", run_root.name)
            row.setdefault("session_id", session_id)
            row["source_path"] = str(csv_path)
        all_rows.extend(rows)
        print(f"    {session_id}: {count} rows")

    trade_count = len(all_rows)
    print(f"  [aggregator] total: {trade_count} trades across {sessions_total} sessions "
          f"({sessions_with_trades} with trades)")

    # ── Write aggregate_trades.csv ──────────────────────────────────
    agg_csv_path = run_root / "aggregate_trades.csv"
    _write_aggregate_csv(all_rows, agg_csv_path)
    print(f"  [aggregator] wrote {agg_csv_path}")

    # ── Compute metrics ─────────────────────────────────────────────
    metrics = _compute_aggregate_metrics(
        all_rows,
        sessions_total=sessions_total,
        sessions_with_trades=sessions_with_trades,
        min_trade_count=min_trade_count,
        early_stop_n_bars=early_stop_n_bars,
        run_root_name=run_root.name,
        per_session_counts=per_session_counts,
    )

    # ── Write aggregate_metrics.json ────────────────────────────────
    metrics_path = run_root / "aggregate_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"  [aggregator] wrote {metrics_path}")

    # ── Patch manifest.json (if present) ────────────────────────────
    _patch_manifest(run_root, agg_csv_path, metrics_path)

    return metrics


# ── Internal helpers ─────────────────────────────────────────────────────


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of row-dicts.  Returns [] on empty/missing."""
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception as exc:
        logger.warning("Could not read %s: %s", csv_path, exc)
        return []


def _write_aggregate_csv(rows: list[dict[str, str]], dest: Path) -> None:
    """Write all rows to *dest* as CSV.  Creates header-only file if empty."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        # Write a header-only sentinel
        fieldnames = [
            "trade_id", "session_id", "signal_timestamp", "side",
            "entry_timestamp", "entry_price", "stop_price", "target_price",
            "exit_timestamp", "exit_price", "exit_reason",
            "pnl_points", "pnl_dollars", "pnl_r",
            "mae_points", "mfe_points", "hold_minutes", "hold_bars",
            "regime_at_entry", "sigma_band_level",
            "run_id", "source_path",
        ]
        with dest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
        return

    fieldnames = list(rows[0].keys())
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _compute_aggregate_metrics(
    rows: list[dict[str, str]],
    *,
    sessions_total: int,
    sessions_with_trades: int,
    min_trade_count: int,
    early_stop_n_bars: int,
    run_root_name: str,
    per_session_counts: list[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Build the aggregate-metrics dict from raw CSV rows."""
    trade_count = len(rows)

    # Parse numeric vectors
    pnl_r_vals = _floats(rows, "pnl_r")
    pnl_pts_vals = _floats(rows, "pnl_points")
    mae_vals = _floats(rows, "mae_points")

    wins_r = [v for v in pnl_r_vals if v > 0]
    losses_r = [v for v in pnl_r_vals if v <= 0]

    win_rate = len(wins_r) / trade_count if trade_count else 0.0
    avg_r = statistics.mean(pnl_r_vals) if pnl_r_vals else 0.0
    std_r = statistics.stdev(pnl_r_vals) if len(pnl_r_vals) >= 2 else 0.0
    avg_win_r = statistics.mean(wins_r) if wins_r else 0.0
    avg_loss_r = statistics.mean(losses_r) if losses_r else 0.0
    max_loss_r = min(pnl_r_vals) if pnl_r_vals else 0.0
    max_win_r = max(pnl_r_vals) if pnl_r_vals else 0.0

    readiness = trade_count >= min_trade_count
    readiness_reason = "" if readiness else "insufficient_trades"

    # Per-session statistics
    session_counts = [c for _, c in (per_session_counts or []) if c > 0]
    trades_per_session_mean = statistics.mean(session_counts) if session_counts else 0.0
    trades_per_session_std = (
        statistics.stdev(session_counts)
        if len(session_counts) >= 2
        else 0.0
    )

    early_stop_count = 0
    for row in rows:
        exit_reason = str(row.get("exit_reason", "")).strip().lower()
        if exit_reason != "stop":
            continue

        hold_bars_raw = row.get("hold_bars")
        hold_bars_val: float | None = None
        if hold_bars_raw not in (None, ""):
            try:
                hold_bars_val = float(hold_bars_raw)
            except (ValueError, TypeError):
                hold_bars_val = None

        if hold_bars_val is None:
            hold_minutes_raw = row.get("hold_minutes")
            if hold_minutes_raw not in (None, ""):
                try:
                    hold_bars_val = float(hold_minutes_raw) / 5.0
                except (ValueError, TypeError):
                    hold_bars_val = None

        if hold_bars_val is not None and hold_bars_val <= early_stop_n_bars:
            early_stop_count += 1

    early_stop_rate = early_stop_count / trade_count if trade_count else 0.0

    return {
        "run_id": run_root_name,
        "generated": datetime.now(timezone.utc).isoformat(),
        "trade_count_total": trade_count,
        "total_sessions": sessions_total,
        "sessions_total": sessions_total,
        "sessions_with_trades": sessions_with_trades,
        "trades_per_session_mean": round(trades_per_session_mean, 2),
        "trades_per_session_std": round(trades_per_session_std, 2),
        "win_rate": round(win_rate, 4),
        "avg_r": round(avg_r, 4),
        "std_r": round(std_r, 4),
        "avg_win_r": round(avg_win_r, 4),
        "avg_loss_r": round(avg_loss_r, 4),
        "max_loss_r": round(max_loss_r, 4),
        "max_win_r": round(max_win_r, 4),
        "p5_r": round(_percentile(pnl_r_vals, 5), 4),
        "p50_r": round(_percentile(pnl_r_vals, 50), 4),
        "p95_r": round(_percentile(pnl_r_vals, 95), 4),
        "mae_p50": round(_percentile(mae_vals, 50), 4),
        "mae_p90": round(_percentile(mae_vals, 90), 4),
        "mae_p95": round(_percentile(mae_vals, 95), 4),
        "early_stop_n_bars": early_stop_n_bars,
        "early_stop_count": early_stop_count,
        "early_stop_rate": round(early_stop_rate, 4),
        "losing_streak_max": _max_losing_streak(pnl_r_vals),
        "readiness": readiness,
        "readiness_reason": readiness_reason,
    }


def _floats(rows: list[dict[str, str]], key: str) -> list[float]:
    """Extract a numeric column from raw CSV rows, skipping blanks."""
    out: list[float] = []
    for row in rows:
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            out.append(float(raw))
        except (ValueError, TypeError):
            continue
    return out


def _patch_manifest(
    run_root: Path,
    agg_csv_path: Path,
    metrics_path: Path,
) -> None:
    """Add aggregate artifact pointers to manifest.json (if it exists)."""
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        return

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return

    data.setdefault("aggregate_artifacts", {})
    data["aggregate_artifacts"]["aggregate_trades_csv"] = str(
        agg_csv_path.relative_to(run_root)
    )
    data["aggregate_artifacts"]["aggregate_metrics_json"] = str(
        metrics_path.relative_to(run_root)
    )

    manifest_path.write_text(
        json.dumps(data, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("Patched manifest with aggregate artifact pointers")
