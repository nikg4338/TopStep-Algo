"""
validation/scorecard.py — Scorecard aggregation for validation runs.

Reads per-session artifacts (signals.csv, session_summary.json, and
optionally trades.csv) produced by replay sessions, computes aggregate
metrics, and writes structured outputs to ``{run_dir}/scorecard/``.

This module is **self-contained** — it reads flat files from disk and
does not import any other project modules.

Outputs
-------
- ``aggregate_metrics.json``  — full metrics dict
- ``aggregate_metrics.csv``   — flat key/value export
- ``summary.md``              — human-readable markdown report
- ``by_session.csv``          — one row per session with core metrics
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

_FLOAT_FMT = ".4f"


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division with zero-denominator guard."""
    return numerator / denominator if denominator else default


def _pct(value: float) -> str:
    """Format a ratio as a percentage string (e.g. ``'42.86%'``)."""
    return f"{value * 100:.2f}%"


def _merge_counters(*dicts: dict[str, int]) -> dict[str, int]:
    """Merge multiple ``{key: count}`` dicts by summing values."""
    merged: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            merged[k] = merged.get(k, 0) + int(v)
    return merged


def _percentile(values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile (0-100) of a sorted list."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (pct / 100.0) * (len(sorted_v) - 1)
    low = int(idx)
    high = min(low + 1, len(sorted_v) - 1)
    weight = idx - low
    return sorted_v[low] * (1 - weight) + sorted_v[high] * weight


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts. Returns [] on error."""
    try:
        with path.open(newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", path, exc)
        return []


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file into a dict. Returns {} on error."""
    try:
        with path.open() as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s: %s", path, exc)
        return {}


def _float_or(value: Any, default: float | None = 0.0) -> float | None:
    """Coerce to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Session-level metric computation ────────────────────────────────────

def _compute_session_signal_metrics(signals: list[dict[str, str]]) -> dict[str, Any]:
    """Derive signal-level metrics from rows of ``signals.csv``."""
    total = len(signals)
    approved = [r for r in signals if r.get("approved", "").lower() in ("true", "1", "yes")]
    rejected = [r for r in signals if r.get("approved", "").lower() in ("false", "0", "no")]

    rejection_breakdown: dict[str, int] = {}
    for r in rejected:
        reason = r.get("rejection_reason", "") or "unknown"
        rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + 1

    regime_distribution: dict[str, int] = {}
    for r in signals:
        regime = r.get("regime", "") or "unknown"
        regime_distribution[regime] = regime_distribution.get(regime, 0) + 1

    side_counts: dict[str, int] = {}
    for r in signals:
        side = (r.get("side", "") or "UNKNOWN").upper()
        side_counts[side] = side_counts.get(side, 0) + 1

    sigma_values = [
        _float_or(r.get("sigma_points", r.get("sigma_value")), None)
        for r in approved
    ]
    sigma_values = [v for v in sigma_values if v is not None]

    z_values = [_float_or(r.get("z_score"), None) for r in approved]
    z_values = [v for v in z_values if v is not None]

    band_values = [_float_or(r.get("band_level"), None) for r in approved]
    band_values = [v for v in band_values if v is not None]

    return {
        "total_candidates": total,
        "total_approved": len(approved),
        "total_rejected": len(rejected),
        "approval_rate": _safe_div(len(approved), total),
        "rejection_breakdown": rejection_breakdown,
        "regime_distribution": regime_distribution,
        "signal_counts_by_side": side_counts,
        "avg_sigma_points": (statistics.mean(sigma_values) if sigma_values else 0.0),
        "avg_z_score": (statistics.mean(z_values) if z_values else 0.0),
        "avg_sigma_distance": (statistics.mean(sigma_values) if sigma_values else 0.0),
        "avg_band_level": (statistics.mean(band_values) if band_values else 0.0),
    }


def _compute_trade_metrics(trades: list[dict[str, str]]) -> dict[str, Any]:
    """Derive trade-level metrics from rows of ``trades.csv``."""
    if not trades:
        return {}

    pnl_dollars = [_float_or(t.get("pnl_dollars")) for t in trades]
    pnl_r = [_float_or(t.get("pnl_r")) for t in trades]
    mae_points = [_float_or(t.get("mae_points")) for t in trades]
    mfe_points = [_float_or(t.get("mfe_points")) for t in trades]
    hold_minutes = [_float_or(t.get("hold_minutes")) for t in trades]

    wins = [p for p in pnl_dollars if p > 0]
    losses = [p for p in pnl_dollars if p <= 0]

    win_rate = _safe_div(len(wins), len(pnl_dollars))
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = statistics.mean(losses) if losses else 0.0  # will be ≤0
    payoff_ratio = _safe_div(avg_win, abs(avg_loss)) if avg_loss != 0.0 else 0.0
    expectancy_dollars = statistics.mean(pnl_dollars) if pnl_dollars else 0.0
    expectancy_r = statistics.mean(pnl_r) if pnl_r else 0.0

    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_win_dollars": avg_win,
        "avg_loss_dollars": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy_dollars": expectancy_dollars,
        "expectancy_r": expectancy_r,
        "mae_avg": statistics.mean(mae_points) if mae_points else 0.0,
        "mae_p95": _percentile(mae_points, 95),
        "mfe_avg": statistics.mean(mfe_points) if mfe_points else 0.0,
        "mfe_p95": _percentile(mfe_points, 95),
        "hold_time_avg": statistics.mean(hold_minutes) if hold_minutes else 0.0,
        "hold_time_p95": _percentile(hold_minutes, 95),
    }


# ── ScorecardAggregator ────────────────────────────────────────────────

class ScorecardAggregator:
    """Reads all session artifacts from a validation run directory and
    produces aggregate scorecard outputs.

    Parameters
    ----------
    run_dir:
        Path to the validation run root, e.g.
        ``artifacts/validation_runs/{run_id}/``.
    compare_to_dir:
        Optional path to a prior validation run directory used to
        compute deltas for key metrics.
    """

    def __init__(self, run_dir: str, compare_to_dir: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.compare_to_dir = Path(compare_to_dir) if compare_to_dir else None

    # ── Public API ──────────────────────────────────────────────────────

    def generate(self) -> dict[str, Any]:
        """Main entry: aggregate all session artifacts and write outputs.

        Returns
        -------
        dict
            The full aggregate metrics dictionary (also written to disk).
        """
        manifest = self._load_manifest()
        session_entries = manifest.get("sessions", [])
        run_id = manifest.get("run_id", self.run_dir.name)
        pack_id = manifest.get("pack_id", "unknown")

        # Build session_id → category map from manifest
        category_map: dict[str, str] = {}
        for entry in session_entries:
            sid = entry.get("session_id", "")
            category_map[sid] = entry.get("category", "default")

        # Discover session directories
        sessions_root = self.run_dir / "sessions"
        session_dirs: list[Path] = []
        if sessions_root.is_dir():
            session_dirs = sorted(
                [d for d in sessions_root.iterdir() if d.is_dir()]
            )

        # Collect per-session metrics
        session_metrics: list[dict[str, Any]] = []
        succeeded = 0
        failed = 0

        # Build a quick lookup of success status from manifest
        success_map: dict[str, bool] = {}
        for entry in session_entries:
            sid = entry.get("session_id", "")
            success_map[sid] = entry.get("success", True)

        for sdir in session_dirs:
            sid = sdir.name
            if not success_map.get(sid, True):
                failed += 1
                session_metrics.append({
                    "session_id": sid,
                    "category": category_map.get(sid, "default"),
                    "success": False,
                })
                continue

            succeeded += 1
            sm = self._process_session(sdir)
            sm["session_id"] = sid
            sm["category"] = category_map.get(sid, "default")
            sm["success"] = True
            session_metrics.append(sm)

        # Also count manifest entries that have no directory
        for entry in session_entries:
            sid = entry.get("session_id", "")
            if not any(sm["session_id"] == sid for sm in session_metrics):
                is_success = entry.get("success", True)
                if is_success:
                    succeeded += 1
                else:
                    failed += 1
                session_metrics.append({
                    "session_id": sid,
                    "category": category_map.get(sid, "default"),
                    "success": is_success,
                })

        total_sessions = succeeded + failed

        # ── Aggregate ───────────────────────────────────────────────────
        ok_sessions = [s for s in session_metrics if s.get("success")]

        agg_candidates = sum(s.get("total_candidates", 0) for s in ok_sessions)
        agg_approved = sum(s.get("total_approved", 0) for s in ok_sessions)
        agg_rejected = sum(s.get("total_rejected", 0) for s in ok_sessions)

        agg_rejection_bkdn = _merge_counters(
            *(s.get("rejection_breakdown", {}) for s in ok_sessions)
        )
        agg_regime_dist = _merge_counters(
            *(s.get("regime_distribution", {}) for s in ok_sessions)
        )

        cand_per_session = [s.get("total_candidates", 0) for s in ok_sessions]
        avg_cand = _safe_div(sum(cand_per_session), len(cand_per_session))
        med_cand = statistics.median(cand_per_session) if cand_per_session else 0.0

        # Per-category stats
        per_category: dict[str, dict[str, Any]] = {}
        for s in ok_sessions:
            cat = s.get("category", "default")
            if cat not in per_category:
                per_category[cat] = {"total_candidates": 0, "total_approved": 0}
            per_category[cat]["total_candidates"] += s.get("total_candidates", 0)
            per_category[cat]["total_approved"] += s.get("total_approved", 0)
        for cat, stats in per_category.items():
            stats["approval_rate"] = _safe_div(
                stats["total_approved"], stats["total_candidates"]
            )

        # Aggregate trade metrics (if any session has them)
        all_trade_metrics = [s.get("trade_metrics", {}) for s in ok_sessions if s.get("trade_metrics")]
        agg_trade: dict[str, Any] = {}
        if all_trade_metrics:
            total_trades = sum(t.get("total_trades", 0) for t in all_trade_metrics)
            # Weighted averages by trade count
            if total_trades > 0:
                def _wavg(key: str) -> float:
                    vals = [(t.get(key, 0.0), t.get("total_trades", 0)) for t in all_trade_metrics]
                    return sum(v * n for v, n in vals) / total_trades

                agg_trade = {
                    "total_trades": total_trades,
                    "win_rate": _wavg("win_rate"),
                    "avg_win_dollars": _wavg("avg_win_dollars"),
                    "avg_loss_dollars": _wavg("avg_loss_dollars"),
                    "payoff_ratio": _wavg("payoff_ratio"),
                    "expectancy_dollars": _wavg("expectancy_dollars"),
                    "expectancy_r": _wavg("expectancy_r"),
                    "mae_avg": _wavg("mae_avg"),
                    "mfe_avg": _wavg("mfe_avg"),
                    "hold_time_avg": _wavg("hold_time_avg"),
                }

        aggregate: dict[str, Any] = {
            "run_id": run_id,
            "pack_id": pack_id,
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_sessions_run": total_sessions,
            "total_sessions_succeeded": succeeded,
            "total_sessions_failed": failed,
            "total_candidates": agg_candidates,
            "total_approved": agg_approved,
            "total_rejected": agg_rejected,
            "aggregate_approval_rate": _safe_div(agg_approved, agg_candidates),
            "aggregate_rejection_breakdown": agg_rejection_bkdn,
            "aggregate_regime_distribution": agg_regime_dist,
            "avg_candidates_per_session": avg_cand,
            "median_candidates_per_session": med_cand,
            "per_category_stats": per_category,
        }

        if agg_trade:
            aggregate["trade_metrics"] = agg_trade

        # ── Baseline comparison ─────────────────────────────────────────
        baseline: dict[str, Any] | None = None
        deltas: dict[str, Any] = {}
        if self.compare_to_dir:
            baseline = self._load_baseline()
            if baseline:
                deltas = self._compute_deltas(aggregate, baseline)
                aggregate["baseline_comparison"] = deltas

        # ── Write outputs ───────────────────────────────────────────────
        scorecard_dir = self.run_dir / "scorecard"
        scorecard_dir.mkdir(parents=True, exist_ok=True)

        self._write_aggregate_json(scorecard_dir, aggregate)
        self._write_aggregate_csv(scorecard_dir, aggregate)
        self._write_by_session_csv(scorecard_dir, session_metrics)
        self._write_summary_md(
            scorecard_dir,
            aggregate,
            session_metrics,
            agg_rejection_bkdn,
            baseline,
            deltas,
        )

        logger.info("Scorecard written → %s", scorecard_dir.resolve())
        return aggregate

    # ── Internal: loading ───────────────────────────────────────────────

    def _load_manifest(self) -> dict[str, Any]:
        """Load manifest.json from the run directory."""
        path = self.run_dir / "manifest.json"
        return _read_json(path)

    def _load_baseline(self) -> dict[str, Any] | None:
        """Load prior run's aggregate_metrics.json for comparison."""
        if not self.compare_to_dir:
            return None
        path = self.compare_to_dir / "scorecard" / "aggregate_metrics.json"
        data = _read_json(path)
        return data or None

    def _process_session(self, sdir: Path) -> dict[str, Any]:
        """Read artifacts from a single session directory and compute metrics."""
        result: dict[str, Any] = {}

        # Prefer signals.csv over summary for signal-level metrics
        signals_path = sdir / "signals.csv"
        summary_path = sdir / "session_summary.json"
        trades_path = sdir / "trades.csv"

        signals = _read_csv(signals_path)
        summary = _read_json(summary_path)

        if signals:
            result.update(_compute_session_signal_metrics(signals))
        elif summary:
            # Fall back to session_summary.json
            result["total_candidates"] = summary.get("total_candidates", 0)
            result["total_approved"] = summary.get("total_approved", 0)
            result["total_rejected"] = summary.get("total_rejected", 0)
            result["approval_rate"] = _safe_div(
                result["total_approved"], result["total_candidates"]
            )
            result["rejection_breakdown"] = summary.get("rejection_breakdown", {})
            result["regime_distribution"] = summary.get("regime_distribution", {})
            result["signal_counts_by_side"] = {}
            result["avg_sigma_points"] = 0.0
            result["avg_z_score"] = 0.0
            result["avg_sigma_distance"] = 0.0
            result["avg_band_level"] = 0.0
        else:
            logger.warning("No signal data found for session dir %s", sdir)
            result.update({
                "total_candidates": 0,
                "total_approved": 0,
                "total_rejected": 0,
                "approval_rate": 0.0,
                "rejection_breakdown": {},
                "regime_distribution": {},
                "signal_counts_by_side": {},
                "avg_sigma_points": 0.0,
                "avg_z_score": 0.0,
                "avg_sigma_distance": 0.0,
                "avg_band_level": 0.0,
            })

        # Trades (optional)
        if trades_path.exists():
            trades = _read_csv(trades_path)
            trade_metrics = _compute_trade_metrics(trades)
            if trade_metrics:
                result["trade_metrics"] = trade_metrics

        return result

    # ── Internal: baseline comparison ───────────────────────────────────

    @staticmethod
    def _compute_deltas(
        current: dict[str, Any],
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute deltas between current and baseline for key metrics."""
        deltas: dict[str, Any] = {}

        def _delta(key: str, source_cur: dict, source_base: dict, *, nested: str | None = None) -> None:
            cur_src = source_cur.get(nested, {}) if nested else source_cur
            base_src = source_base.get(nested, {}) if nested else source_base
            cur_val = cur_src.get(key)
            base_val = base_src.get(key)
            label = f"{nested}.{key}" if nested else key
            if cur_val is not None and base_val is not None:
                try:
                    deltas[label] = {
                        "current": cur_val,
                        "previous": base_val,
                        "delta": float(cur_val) - float(base_val),
                    }
                except (TypeError, ValueError):
                    deltas[label] = {
                        "current": cur_val,
                        "previous": base_val,
                        "delta": "N/A",
                    }
            else:
                deltas[label] = {
                    "current": cur_val if cur_val is not None else "N/A",
                    "previous": base_val if base_val is not None else "N/A",
                    "delta": "N/A",
                }

        _delta("aggregate_approval_rate", current, baseline)
        _delta("total_candidates", current, baseline)
        _delta("total_approved", current, baseline)
        _delta("total_rejected", current, baseline)
        _delta("avg_candidates_per_session", current, baseline)

        # Trade-level deltas (if available)
        _delta("win_rate", current, baseline, nested="trade_metrics")
        _delta("expectancy_dollars", current, baseline, nested="trade_metrics")
        _delta("expectancy_r", current, baseline, nested="trade_metrics")
        _delta("payoff_ratio", current, baseline, nested="trade_metrics")

        return deltas

    # ── Internal: writers ───────────────────────────────────────────────

    @staticmethod
    def _write_aggregate_json(out_dir: Path, aggregate: dict[str, Any]) -> None:
        path = out_dir / "aggregate_metrics.json"
        with path.open("w") as fh:
            json.dump(aggregate, fh, indent=2, default=str)
        logger.debug("Wrote %s", path)

    @staticmethod
    def _write_aggregate_csv(out_dir: Path, aggregate: dict[str, Any]) -> None:
        """Flat key/value CSV of scalar aggregate metrics."""
        path = out_dir / "aggregate_metrics.csv"
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["metric", "value"])
            for key, value in aggregate.items():
                if isinstance(value, (dict, list)):
                    writer.writerow([key, json.dumps(value, default=str)])
                else:
                    writer.writerow([key, value])
        logger.debug("Wrote %s", path)

    @staticmethod
    def _write_by_session_csv(
        out_dir: Path,
        session_metrics: list[dict[str, Any]],
    ) -> None:
        path = out_dir / "by_session.csv"
        fieldnames = [
            "session_id",
            "category",
            "success",
            "total_candidates",
            "total_approved",
            "total_rejected",
            "approval_rate",
            "avg_sigma_points",
            "avg_z_score",
            "avg_sigma_distance",
            "avg_band_level",
            "has_trades",
            "win_rate",
            "expectancy_dollars",
        ]
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for sm in session_metrics:
                tm = sm.get("trade_metrics", {})
                row = {
                    "session_id": sm.get("session_id", ""),
                    "category": sm.get("category", ""),
                    "success": sm.get("success", ""),
                    "total_candidates": sm.get("total_candidates", 0),
                    "total_approved": sm.get("total_approved", 0),
                    "total_rejected": sm.get("total_rejected", 0),
                    "approval_rate": f"{sm.get('approval_rate', 0):.4f}" if sm.get("success") else "",
                    "avg_sigma_points": f"{sm.get('avg_sigma_points', 0):.4f}" if sm.get("success") else "",
                    "avg_z_score": f"{sm.get('avg_z_score', 0):.4f}" if sm.get("success") else "",
                    "avg_sigma_distance": f"{sm.get('avg_sigma_distance', 0):.4f}" if sm.get("success") else "",
                    "avg_band_level": f"{sm.get('avg_band_level', 0):.4f}" if sm.get("success") else "",
                    "has_trades": bool(tm),
                    "win_rate": f"{tm.get('win_rate', 0):.4f}" if tm else "",
                    "expectancy_dollars": f"{tm.get('expectancy_dollars', 0):.2f}" if tm else "",
                }
                writer.writerow(row)
        logger.debug("Wrote %s", path)

    @staticmethod
    def _write_summary_md(
        out_dir: Path,
        aggregate: dict[str, Any],
        session_metrics: list[dict[str, Any]],
        rejection_breakdown: dict[str, int],
        baseline: dict[str, Any] | None,
        deltas: dict[str, Any],
    ) -> None:
        run_id = aggregate.get("run_id", "unknown")
        pack_id = aggregate.get("pack_id", "unknown")
        generated = aggregate.get("generated", datetime.now(timezone.utc).isoformat())
        n_success = aggregate.get("total_sessions_succeeded", 0)
        n_total = aggregate.get("total_sessions_run", 0)

        lines: list[str] = []

        # Header
        lines.append(f"# Validation Scorecard — {run_id}")
        lines.append("")
        lines.append(f"**Generated:** {generated}")
        lines.append(f"**Pack:** {pack_id}")
        lines.append(f"**Sessions:** {n_success}/{n_total} succeeded")
        lines.append("")

        # Aggregate metrics table
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Candidates | {aggregate.get('total_candidates', 0)} |")
        lines.append(f"| Total Approved | {aggregate.get('total_approved', 0)} |")
        lines.append(f"| Total Rejected | {aggregate.get('total_rejected', 0)} |")
        lines.append(f"| Approval Rate | {_pct(aggregate.get('aggregate_approval_rate', 0))} |")
        lines.append(f"| Avg Candidates/Session | {aggregate.get('avg_candidates_per_session', 0):.2f} |")
        lines.append(f"| Median Candidates/Session | {aggregate.get('median_candidates_per_session', 0):.1f} |")

        tm = aggregate.get("trade_metrics", {})
        if tm:
            lines.append(f"| Total Trades | {tm.get('total_trades', 0)} |")
            lines.append(f"| Win Rate | {_pct(tm.get('win_rate', 0))} |")
            lines.append(f"| Avg Win ($) | {tm.get('avg_win_dollars', 0):.2f} |")
            lines.append(f"| Avg Loss ($) | {tm.get('avg_loss_dollars', 0):.2f} |")
            lines.append(f"| Payoff Ratio | {tm.get('payoff_ratio', 0):.2f} |")
            lines.append(f"| Expectancy ($) | {tm.get('expectancy_dollars', 0):.2f} |")
            lines.append(f"| Expectancy (R) | {tm.get('expectancy_r', 0):.4f} |")
            lines.append(f"| MAE Avg | {tm.get('mae_avg', 0):.2f} |")
            lines.append(f"| MFE Avg | {tm.get('mfe_avg', 0):.2f} |")
            lines.append(f"| Hold Time Avg (min) | {tm.get('hold_time_avg', 0):.1f} |")

        lines.append("")

        # Per-session summary
        lines.append("## Per-Session Summary")
        lines.append("")
        lines.append("| Session | Category | Candidates | Approved | Rate | Regime |")
        lines.append("|---------|----------|------------|----------|------|--------|")
        for sm in session_metrics:
            sid = sm.get("session_id", "")
            cat = sm.get("category", "")
            if not sm.get("success"):
                lines.append(f"| {sid} | {cat} | — | — | FAILED | — |")
                continue
            cand = sm.get("total_candidates", 0)
            appr = sm.get("total_approved", 0)
            rate = _pct(sm.get("approval_rate", 0))
            regime_dist = sm.get("regime_distribution", {})
            top_regime = max(regime_dist, key=regime_dist.get, default="—") if regime_dist else "—"
            lines.append(f"| {sid} | {cat} | {cand} | {appr} | {rate} | {top_regime} |")
        lines.append("")

        # Rejection breakdown
        lines.append("## Rejection Breakdown")
        lines.append("")
        total_rej = sum(rejection_breakdown.values()) if rejection_breakdown else 0
        lines.append("| Reason | Count | % |")
        lines.append("|--------|-------|---|")
        if rejection_breakdown:
            for reason, count in sorted(rejection_breakdown.items(), key=lambda x: -x[1]):
                pct_val = _pct(_safe_div(count, total_rej))
                lines.append(f"| {reason} | {count} | {pct_val} |")
        else:
            lines.append("| (none) | 0 | — |")
        lines.append("")

        # Per-category stats
        per_cat = aggregate.get("per_category_stats", {})
        if per_cat:
            lines.append("## Per-Category Stats")
            lines.append("")
            lines.append("| Category | Candidates | Approved | Rate |")
            lines.append("|----------|------------|----------|------|")
            for cat, stats in sorted(per_cat.items()):
                lines.append(
                    f"| {cat} | {stats['total_candidates']} "
                    f"| {stats['total_approved']} "
                    f"| {_pct(stats['approval_rate'])} |"
                )
            lines.append("")

        # Baseline comparison
        if baseline and deltas:
            lines.append("## Baseline Comparison")
            lines.append("")
            lines.append("| Metric | Current | Previous | Delta |")
            lines.append("|--------|---------|----------|-------|")
            for metric, vals in deltas.items():
                cur = vals.get("current", "N/A")
                prev = vals.get("previous", "N/A")
                delta = vals.get("delta", "N/A")
                if isinstance(cur, float):
                    cur = f"{cur:.4f}"
                if isinstance(prev, float):
                    prev = f"{prev:.4f}"
                if isinstance(delta, float):
                    delta_sign = "+" if delta > 0 else ""
                    delta = f"{delta_sign}{delta:.4f}"
                lines.append(f"| {metric} | {cur} | {prev} | {delta} |")
            lines.append("")

        path = out_dir / "summary.md"
        with path.open("w") as fh:
            fh.write("\n".join(lines))
        logger.debug("Wrote %s", path)


# ── Standalone entry-point ──────────────────────────────────────────────

def run_scorecard(run_dir: str, compare_to: str | None = None) -> dict[str, Any]:
    """Create a :class:`ScorecardAggregator` and generate the scorecard.

    Parameters
    ----------
    run_dir:
        Path to the validation run root directory.
    compare_to:
        Optional path to a prior run directory for baseline comparison.

    Returns
    -------
    dict
        The aggregate metrics dictionary.
    """
    aggregator = ScorecardAggregator(run_dir, compare_to_dir=compare_to)
    return aggregator.generate()


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate validation scorecard")
    parser.add_argument("run_dir", help="Path to validation run directory")
    parser.add_argument(
        "--compare-to",
        default=None,
        help="Path to prior run directory for baseline comparison",
    )
    args = parser.parse_args()

    result = run_scorecard(args.run_dir, compare_to=args.compare_to)
    print(json.dumps(result, indent=2, default=str))
