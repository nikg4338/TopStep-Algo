"""
reporting/replay_report.py — Replay Session Report export module.

Collects MRSignal candidates, RegimeFeatures snapshots, and session
metadata from a replay run and writes structured artifacts to disk
for post-session analysis.

Artifacts produced per session:
  • signals.csv          — one row per MR candidate signal
  • session_summary.json — aggregate stats + config snapshot
  • features_snapshot.csv — (optional) regime feature time-series
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import config as _config
from strategies.mr_signal_engine import MRSignal
from regime.regime_v1 import RegimeFeatures

logger = logging.getLogger(__name__)

_REPORT_VERSION = "1.0.0"

# ── Column definitions ──────────────────────────────────────────────────

_SIGNAL_COLUMNS: list[str] = [
    "timestamp",
    "regime",
    "signal_type",
    "side",
    "candidate_price",
    "stop_reference",
    "target_reference",
    "approved",
    "rejection_reason",
    "band_level",
    "vwap",
    "sigma_value",
    "sigma_points",
    "z_score",
    "session_id",
]

_FEATURE_COLUMNS: list[str] = [
    "timestamp",
    "adx",
    "atr",
    "atr_percentile",
    "realized_vol",
    "vol_percentile",
    "regime",
]


# ── ReplaySessionReport ────────────────────────────────────────────────

class ReplaySessionReport:
    """Accumulates replay-session data and exports structured artifacts.

    Typical usage::

        report = ReplaySessionReport("run_001", "MES", "2026-02-18T14:30:00Z", "2026-02-18T16:00:00Z")
        report.set_tick_stats(ticks_processed=12000, bars_closed=180, ...)
        report.add_signals(engine.candidates)
        report.add_features(classifier.feature_history)
        report.set_config_snapshot({...})
        report.export()
    """

    def __init__(
        self,
        session_id: str,
        symbol: str,
        replay_start: str,
        replay_end: str,
    ) -> None:
        self.session_id: str = session_id
        self.symbol: str = symbol
        self.replay_start: str = replay_start
        self.replay_end: str = replay_end

        # Tick / bar statistics (set via set_tick_stats)
        self.ticks_processed: int = 0
        self.bars_closed: int = 0
        self.bars_partial_flushed: int = 0
        self.unique_buckets: int = 0

        # Signal & feature buffers
        self._signals: list[MRSignal] = []
        self._features: list[RegimeFeatures] = []

        # Optional config mirror
        self._config_snapshot: dict[str, Any] = {}

        # Optional rejection counters from signal engine
        self._rejection_counters: dict[str, int] = {}
        self._gate_funnel: dict[str, Any] = {}
        self._orb_funnel: dict[str, Any] = {}

    # ── Setters ─────────────────────────────────────────────────────────

    def set_tick_stats(
        self,
        ticks_processed: int,
        bars_closed: int,
        bars_partial_flushed: int,
        unique_buckets: int,
    ) -> None:
        """Record aggregate tick/bar counts for the session."""
        self.ticks_processed = ticks_processed
        self.bars_closed = bars_closed
        self.bars_partial_flushed = bars_partial_flushed
        self.unique_buckets = unique_buckets

    def add_signals(self, signals: list[MRSignal]) -> None:
        """Append a batch of MRSignal candidates."""
        self._signals.extend(signals)

    def add_features(self, features: list[RegimeFeatures]) -> None:
        """Append a batch of RegimeFeatures snapshots."""
        self._features.extend(features)

    def set_config_snapshot(self, config: dict[str, Any]) -> None:
        """Store a snapshot of the active configuration for provenance."""
        self._config_snapshot = dict(config)

    def set_rejection_counters(self, counters: dict[str, int]) -> None:
        """Store structured rejection counters from signal engine."""
        self._rejection_counters = dict(counters)

    def set_gate_funnel(self, funnel: dict[str, Any]) -> None:
        """Store ordered gate funnel diagnostics from signal engine."""
        self._gate_funnel = dict(funnel)

    def set_orb_funnel(self, funnel: dict[str, Any]) -> None:
        """Store ORB signal formation diagnostics."""
        self._orb_funnel = dict(funnel)

    # ── Export ──────────────────────────────────────────────────────────

    def export(self, base_dir: str | None = None) -> Path:
        """Write all artifacts to disk and return the output directory.

        Parameters
        ----------
        base_dir:
            Override for ``ARTIFACTS_DIR``.  When *None* the value from
            ``config.ARTIFACTS_DIR`` is used.

        Returns
        -------
        pathlib.Path
            Absolute path to the session artifact directory.
        """
        root = Path(base_dir) if base_dir else Path(_config.ARTIFACTS_DIR)
        out_dir = root / self.session_id
        out_dir.mkdir(parents=True, exist_ok=True)

        self._write_signals_csv(out_dir)
        self._write_session_summary(out_dir)

        if _config.EXPORT_FEATURES_SNAPSHOT:
            self._write_features_csv(out_dir)

        abs_path = out_dir.resolve()
        logger.info("Replay report exported → %s", abs_path)
        print(f"Replay report exported → {abs_path}")
        return abs_path

    # ── Internal writers ────────────────────────────────────────────────

    def _write_signals_csv(self, out_dir: Path) -> None:
        """Write signals.csv — one row per MR candidate."""
        path = out_dir / "signals.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_SIGNAL_COLUMNS)
            writer.writeheader()
            for sig in self._signals:
                writer.writerow(
                    {
                        "timestamp": sig.timestamp.isoformat(),
                        "regime": sig.regime_at_signal or "",
                        "signal_type": sig.signal_type,
                        "side": sig.side,
                        "candidate_price": sig.entry_reference_price,
                        "stop_reference": sig.stop_reference,
                        "target_reference": sig.target_reference,
                        "approved": sig.approved,
                        "rejection_reason": sig.rejection_reason,
                        "band_level": sig.band_level_hit,
                        "vwap": sig.vwap_at_signal,
                        "sigma_value": sig.sigma_at_signal,
                        "sigma_points": sig.sigma_at_signal,
                        "z_score": sig.z_at_signal,
                        "session_id": self.session_id,
                    }
                )
        logger.debug("Wrote %d signal rows → %s", len(self._signals), path)

    def _write_session_summary(self, out_dir: Path) -> None:
        """Write session_summary.json — aggregate stats + config."""
        approved = [s for s in self._signals if s.approved]
        rejected = [s for s in self._signals if not s.approved]

        rejection_breakdown: dict[str, int] = dict(
            Counter(s.rejection_reason for s in rejected if s.rejection_reason)
        )
        regime_distribution: dict[str, int] = dict(
            Counter(
                s.regime_at_signal or "unknown" for s in self._signals
            )
        )

        summary: dict[str, Any] = {
            "session_id": self.session_id,
            "symbol": self.symbol,
            "replay_start": self.replay_start,
            "replay_end": self.replay_end,
            "ticks_processed": self.ticks_processed,
            "bars_closed": self.bars_closed,
            "bars_partial_flushed": self.bars_partial_flushed,
            "unique_buckets": self.unique_buckets,
            "total_candidates": len(self._signals),
            "total_approved": len(approved),
            "total_rejected": len(rejected),
            "rejection_breakdown": rejection_breakdown,
            "rejection_counters": self._rejection_counters,
            "gate_funnel": self._gate_funnel,
            "orb_funnel": self._orb_funnel,
            "regime_distribution": regime_distribution,
            "config_snapshot": self._config_snapshot,
            "notes": "",
            "version": _REPORT_VERSION,
        }

        path = out_dir / "session_summary.json"
        with path.open("w") as fh:
            json.dump(summary, fh, indent=2, default=str)
        logger.debug("Wrote session summary → %s", path)

    def _write_features_csv(self, out_dir: Path) -> None:
        """Write features_snapshot.csv — regime feature time-series."""
        path = out_dir / "features_snapshot.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FEATURE_COLUMNS)
            writer.writeheader()
            for feat in self._features:
                writer.writerow(
                    {
                        "timestamp": feat.timestamp.isoformat(),
                        "adx": feat.adx,
                        "atr": feat.atr,
                        "atr_percentile": feat.atr_percentile,
                        "realized_vol": feat.realized_vol,
                        "vol_percentile": feat.vol_percentile,
                        "regime": feat.regime or "",
                    }
                )
        logger.debug("Wrote %d feature rows → %s", len(self._features), path)
