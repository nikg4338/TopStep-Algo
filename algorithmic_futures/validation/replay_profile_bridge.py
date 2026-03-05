"""
validation/replay_profile_bridge.py — Bridge from replay trade CSVs to
Monte Carlo simulation profiles.

Reads completed validation-run artefacts (trades.csv per session), computes
aggregate win-rate / payoff statistics, persists a *ReplayDerivedProfile*
JSON + Markdown interpretation, and optionally feeds the profile straight
into MonteCarloValidator.

Usage (standalone)::

    from validation.replay_profile_bridge import generate_mc_profile
    profile = generate_mc_profile("artifacts/validation_runs/run_abc123")

Usage (programmatic)::

    bridge = ReplayProfileBridge("artifacts/validation_runs/run_abc123")
    profile = bridge.build_profile(category_filter="trend")
    bridge.write_profile(profile)
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from risk.monte_carlo import MonteCarloResult

logger = logging.getLogger(__name__)


def _trades_per_session(trades: list[dict[str, Any]]) -> list[int]:
    """Count trades per unique session_id."""
    counts: dict[str, int] = {}
    for t in trades:
        sid = t.get("session_id", "unknown")
        counts[sid] = counts.get(sid, 0) + 1
    return list(counts.values())


# ── Profile dataclass ───────────────────────────────────────────────────


@dataclass
class ReplayDerivedProfile:
    """Aggregated trade statistics suitable for Monte Carlo simulation."""

    sample_size_trades: int
    sample_size_sessions: int
    win_rate: float
    avg_win_dollars: float
    avg_loss_dollars: float
    avg_win_r: float
    avg_loss_r: float
    payoff_ratio: float
    expectancy_dollars: float
    expectancy_r: float
    pnl_std_dollars: float
    pnl_std_r: float
    trades_per_session_avg: float
    sessions_in_sample: int
    recommended_mc_horizon_trades: int
    notes: str = ""
    source: str = "replay_derived"


# ── Bridge ──────────────────────────────────────────────────────────────


class ReplayProfileBridge:
    """Reads replay artefacts and builds :class:`ReplayDerivedProfile`s."""

    def __init__(self, run_dir: str, min_trade_count: int = 10) -> None:
        self.run_dir = Path(run_dir)
        self.min_trade_count = min_trade_count

        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {self.run_dir}")

        self._manifest: dict[str, Any] | None = None

    # ── aggregate-first loading ──────────────────────────────────────

    def _load_aggregate_trades(self) -> list[dict[str, Any]] | None:
        """Try loading ``aggregate_trades.csv`` from run root.

        Returns *None* if the file doesn't exist or is empty.
        """
        agg_path = self.run_dir / "aggregate_trades.csv"
        if not agg_path.is_file():
            logger.info("aggregate_trades.csv not found at %s — will scan sessions", agg_path)
            return None

        trades = self._load_trades(agg_path)
        if not trades:
            logger.info("aggregate_trades.csv exists but has 0 rows — will scan sessions")
            return None

        logger.info("Loaded %d trades from %s", len(trades), agg_path)
        return trades

    def build_profile_from_aggregate(
        self,
        category_filter: str | None = None,
    ) -> ReplayDerivedProfile:
        """Build profile preferring aggregate_trades.csv, falling back to sessions.

        This method also writes ``mc_profile.json`` at the run root for
        downstream consumers (MC survival simulator, promotion gate).
        """
        # 1. Try aggregate_trades.csv first
        agg_trades = self._load_aggregate_trades()

        if agg_trades is not None:
            # Optional category filtering on 'session_id' → manifest map
            if category_filter is not None:
                cat_map = self._session_category_map()
                cf_lower = category_filter.lower()
                agg_trades = [
                    t for t in agg_trades
                    if cat_map.get(t.get("session_id", ""), "unknown").lower() == cf_lower
                ]

            session_ids = {t.get("session_id", "") for t in agg_trades if t.get("session_id")}
            tps = _trades_per_session(agg_trades)
            print(f"  [bridge] loaded aggregate_trades.csv: {len(agg_trades)} trades, "
                  f"{len(session_ids)} sessions")

            profile = self._compute_profile(agg_trades, len(session_ids), tps)
        else:
            # 2. Fallback: scan sessions/**/trades.csv
            print("  [bridge] aggregate_trades.csv not available — scanning sessions/")
            trade_files = self._discover_trade_files()
            if not trade_files:
                msg = (f"No trades found under {self.run_dir}. "
                       "Paths scanned: sessions/**/trades.csv. "
                       "Recommendation: run the trade aggregator first.")
                logger.warning(msg)
                print(f"  [bridge] WARNING: {msg}")

            all_trades: list[dict[str, Any]] = []
            session_ids_set: set[str] = set()
            tps_list: list[int] = []

            for sid, csv_path in trade_files:
                trades = self._load_trades(csv_path)
                print(f"    {sid}: {len(trades)} rows from {csv_path}")
                if trades:
                    all_trades.extend(trades)
                    session_ids_set.add(sid)
                    tps_list.append(len(trades))

            print(f"  [bridge] total: {len(all_trades)} rows from {len(trade_files)} files "
                  f"({len(session_ids_set)} sessions with trades)")

            if not all_trades:
                msg = (f"Zero trade rows found across {len(trade_files)} files. "
                       "Recommendation: run the trade aggregator or verify exit-sim output.")
                logger.warning(msg)
                print(f"  [bridge] WARNING: {msg}")

            profile = self._compute_profile(all_trades, len(session_ids_set), tps_list)

        # Write mc_profile.json at run root
        self._write_mc_profile_json(profile)

        return profile

    def _write_mc_profile_json(self, profile: ReplayDerivedProfile) -> str:
        """Write ``mc_profile.json`` at run root with profile stats + r_values summary."""
        out_path = self.run_dir / "mc_profile.json"
        payload: dict[str, Any] = {
            "trade_count": profile.sample_size_trades,
            "win_rate": profile.win_rate,
            "avg_r": profile.expectancy_r,
            "std_r": profile.pnl_std_r,
            "avg_win_r": profile.avg_win_r,
            "avg_loss_r": profile.avg_loss_r,
            "payoff_ratio": profile.payoff_ratio,
            "expectancy_dollars": profile.expectancy_dollars,
            "pnl_std_dollars": profile.pnl_std_dollars,
            "quantiles": {
                "p5": round(profile.avg_loss_r * 1.5, 4) if profile.avg_loss_r else 0.0,
                "p50": profile.expectancy_r,
                "p95": round(profile.avg_win_r * 1.5, 4) if profile.avg_win_r else 0.0,
            },
            "sessions_in_sample": profile.sessions_in_sample,
            "recommended_mc_horizon": profile.recommended_mc_horizon_trades,
            "notes": profile.notes,
            "source": profile.source,
        }
        out_path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote mc_profile.json → %s", out_path)
        print(f"  [bridge] wrote {out_path}")
        return str(out_path)

    # ── manifest ────────────────────────────────────────────────────────

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest is not None:
            return self._manifest

        manifest_path = self.run_dir / "manifest.json"
        if not manifest_path.exists():
            logger.warning("manifest.json not found in %s; category filtering disabled", self.run_dir)
            self._manifest = {}
            return self._manifest

        with open(manifest_path, "r", encoding="utf-8") as fh:
            self._manifest = json.load(fh)
        assert self._manifest is not None  # satisfy type-checker after json.load
        return self._manifest

    def _session_category_map(self) -> dict[str, str]:
        """Return {session_id: category} from the manifest."""
        manifest = self._load_manifest()
        sessions = manifest.get("sessions", [])
        return {s["session_id"]: s.get("category", "unknown") for s in sessions if "session_id" in s}

    # ── CSV loading ─────────────────────────────────────────────────────

    def _discover_trade_files(self) -> list[tuple[str, Path]]:
        """Return [(session_id, trades.csv path), ...] for every session dir."""
        sessions_dir = self.run_dir / "sessions"
        if not sessions_dir.is_dir():
            logger.warning("No sessions/ directory found under %s", self.run_dir)
            return []

        results: list[tuple[str, Path]] = []
        for entry in sorted(sessions_dir.iterdir()):
            if entry.is_dir():
                csv_path = entry / "trades.csv"
                if csv_path.is_file():
                    results.append((entry.name, csv_path))
        return results

    @staticmethod
    def _load_trades(csv_path: Path) -> list[dict[str, Any]]:
        """Load a single trades.csv and coerce numeric columns."""
        trades: list[dict[str, Any]] = []
        numeric_cols = {
            "entry_price", "stop_price", "target_price", "exit_price",
            "pnl_points", "pnl_dollars", "pnl_r",
            "mae_points", "mfe_points", "hold_minutes",
        }
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                for col in numeric_cols:
                    raw = row.get(col)
                    if raw is not None and raw != "":
                        try:
                            row[col] = float(raw)
                        except (ValueError, TypeError):
                            row[col] = None
                    else:
                        row[col] = None
                trades.append(row)
        return trades

    # ── profile computation ─────────────────────────────────────────────

    def build_profile(self, category_filter: str | None = None) -> ReplayDerivedProfile:
        """Build an aggregate profile from all (or category-filtered) sessions.

        Parameters
        ----------
        category_filter : str, optional
            If provided, only include sessions whose manifest category matches
            this value (case-insensitive).

        Returns
        -------
        ReplayDerivedProfile
        """
        cat_map = self._session_category_map()
        trade_files = self._discover_trade_files()

        if category_filter is not None:
            category_filter_lower = category_filter.lower()
            trade_files = [
                (sid, path) for sid, path in trade_files
                if cat_map.get(sid, "unknown").lower() == category_filter_lower
            ]

        all_trades: list[dict[str, Any]] = []
        session_ids: set[str] = set()
        trades_per_session: list[int] = []

        for session_id, csv_path in trade_files:
            trades = self._load_trades(csv_path)
            if trades:
                all_trades.extend(trades)
                session_ids.add(session_id)
                trades_per_session.append(len(trades))

        return self._compute_profile(all_trades, len(session_ids), trades_per_session)

    def build_category_profiles(self) -> dict[str, ReplayDerivedProfile]:
        """Build a separate profile for each category in the manifest."""
        cat_map = self._session_category_map()
        categories = sorted(set(cat_map.values()))

        if not categories:
            logger.warning("No categories found in manifest; returning empty dict")
            return {}

        profiles: dict[str, ReplayDerivedProfile] = {}
        for category in categories:
            profile = self.build_profile(category_filter=category)
            profiles[category] = profile

        return profiles

    def _compute_profile(
        self,
        trades: list[dict[str, Any]],
        n_sessions: int,
        trades_per_session: list[int],
    ) -> ReplayDerivedProfile:
        """Compute all profile fields from a flat list of trade dicts."""
        n_trades = len(trades)

        # ── edge case: no trades ────────────────────────────────────────
        if n_trades == 0:
            logger.warning("No trades found — returning zeroed profile")
            return ReplayDerivedProfile(
                sample_size_trades=0,
                sample_size_sessions=0,
                win_rate=0.0,
                avg_win_dollars=0.0,
                avg_loss_dollars=0.0,
                avg_win_r=0.0,
                avg_loss_r=0.0,
                payoff_ratio=0.0,
                expectancy_dollars=0.0,
                expectancy_r=0.0,
                pnl_std_dollars=0.0,
                pnl_std_r=0.0,
                trades_per_session_avg=0.0,
                sessions_in_sample=0,
                recommended_mc_horizon_trades=0,
                notes="WARNING: No trades in sample — profile is empty.",
            )

        # ── extract P&L vectors ─────────────────────────────────────────
        pnl_dollars = [t["pnl_dollars"] for t in trades if t.get("pnl_dollars") is not None]
        pnl_r = [t["pnl_r"] for t in trades if t.get("pnl_r") is not None]

        if not pnl_dollars:
            logger.warning("All pnl_dollars values are null — returning zeroed profile")
            return ReplayDerivedProfile(
                sample_size_trades=n_trades,
                sample_size_sessions=n_sessions,
                win_rate=0.0,
                avg_win_dollars=0.0,
                avg_loss_dollars=0.0,
                avg_win_r=0.0,
                avg_loss_r=0.0,
                payoff_ratio=0.0,
                expectancy_dollars=0.0,
                expectancy_r=0.0,
                pnl_std_dollars=0.0,
                pnl_std_r=0.0,
                trades_per_session_avg=0.0,
                sessions_in_sample=n_sessions,
                recommended_mc_horizon_trades=0,
                notes="WARNING: All PnL values are null — profile is not usable.",
            )

        wins_dollars = [p for p in pnl_dollars if p > 0]
        losses_dollars = [p for p in pnl_dollars if p <= 0]

        wins_r = [p for p in pnl_r if p > 0] if pnl_r else []
        losses_r = [p for p in pnl_r if p <= 0] if pnl_r else []

        n_effective = len(pnl_dollars)
        n_wins = len(wins_dollars)
        n_losses = len(losses_dollars)

        win_rate = n_wins / n_effective if n_effective > 0 else 0.0

        # ── averages (handle all-wins / all-losses) ─────────────────────
        avg_win_dollars = statistics.mean(wins_dollars) if wins_dollars else 0.0
        avg_loss_dollars = statistics.mean(losses_dollars) if losses_dollars else 0.0
        avg_win_r = statistics.mean(wins_r) if wins_r else 0.0
        avg_loss_r = statistics.mean(losses_r) if losses_r else 0.0

        # Payoff ratio = |avg_win / avg_loss|; guard division by zero
        if avg_loss_dollars != 0:
            payoff_ratio = abs(avg_win_dollars / avg_loss_dollars)
        else:
            payoff_ratio = float("inf") if avg_win_dollars > 0 else 0.0

        # Expectancy
        expectancy_dollars = statistics.mean(pnl_dollars)
        expectancy_r = statistics.mean(pnl_r) if pnl_r else 0.0

        # Standard deviation (need ≥ 2 observations)
        pnl_std_dollars = statistics.stdev(pnl_dollars) if len(pnl_dollars) >= 2 else 0.0
        pnl_std_r = statistics.stdev(pnl_r) if len(pnl_r) >= 2 else 0.0

        # Trades per session
        trades_per_session_avg = (
            statistics.mean(trades_per_session) if trades_per_session else 0.0
        )

        # Recommended MC horizon: min(sample * 5, 1000)
        recommended_mc_horizon_trades = min(n_effective * 5, 1000)

        # Notes
        notes_parts: list[str] = []
        if n_effective < self.min_trade_count:
            notes_parts.append(
                f"WARNING: Sample size ({n_effective}) is below minimum "
                f"threshold ({self.min_trade_count}). Profile may be unreliable."
            )
        if n_wins == 0:
            notes_parts.append("WARNING: No winning trades in sample.")
        if n_losses == 0:
            notes_parts.append("WARNING: No losing trades in sample — payoff ratio is infinite.")

        notes = " | ".join(notes_parts) if notes_parts else ""

        return ReplayDerivedProfile(
            sample_size_trades=n_effective,
            sample_size_sessions=n_sessions,
            win_rate=win_rate,
            avg_win_dollars=round(avg_win_dollars, 2),
            avg_loss_dollars=round(avg_loss_dollars, 2),
            avg_win_r=round(avg_win_r, 4),
            avg_loss_r=round(avg_loss_r, 4),
            payoff_ratio=round(payoff_ratio, 4),
            expectancy_dollars=round(expectancy_dollars, 2),
            expectancy_r=round(expectancy_r, 4),
            pnl_std_dollars=round(pnl_std_dollars, 2),
            pnl_std_r=round(pnl_std_r, 4),
            trades_per_session_avg=round(trades_per_session_avg, 2),
            sessions_in_sample=n_sessions,
            recommended_mc_horizon_trades=recommended_mc_horizon_trades,
            notes=notes,
        )

    # ── serialisation ───────────────────────────────────────────────────

    def write_profile(
        self,
        profile: ReplayDerivedProfile,
        output_dir: str | None = None,
    ) -> str:
        """Persist profile as JSON + Markdown and return the JSON path.

        Parameters
        ----------
        profile : ReplayDerivedProfile
        output_dir : str, optional
            Defaults to ``{run_dir}/mc_profile/``.

        Returns
        -------
        str  Absolute path to the written JSON file.
        """
        out = Path(output_dir) if output_dir else self.run_dir / "mc_profile"
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "replay_derived_profile.json"
        md_path = out / "replay_derived_profile.md"

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(asdict(profile), fh, indent=2, default=str)

        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(_render_markdown(profile, self.min_trade_count))

        logger.info("Wrote replay-derived profile to %s", json_path)
        return str(json_path)

    def write_category_profiles(
        self,
        profiles: dict[str, ReplayDerivedProfile],
        output_dir: str | None = None,
    ) -> str:
        """Persist per-category profiles as a single JSON and return its path.

        Parameters
        ----------
        profiles : dict[str, ReplayDerivedProfile]
        output_dir : str, optional
            Defaults to ``{run_dir}/mc_profile/``.

        Returns
        -------
        str  Absolute path to the written JSON file.
        """
        out = Path(output_dir) if output_dir else self.run_dir / "mc_profile"
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / "by_category_profiles.json"
        payload = {cat: asdict(profile) for cat, profile in profiles.items()}

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

        logger.info("Wrote category profiles to %s", json_path)
        return str(json_path)


# ── Standalone helpers ──────────────────────────────────────────────────


def load_replay_profile(profile_path: str) -> ReplayDerivedProfile:
    """Deserialise a :class:`ReplayDerivedProfile` from a JSON file.

    Parameters
    ----------
    profile_path : str
        Path to ``replay_derived_profile.json``.

    Returns
    -------
    ReplayDerivedProfile
    """
    path = Path(profile_path)
    if not path.is_file():
        raise FileNotFoundError(f"Profile JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    return ReplayDerivedProfile(**data)


def run_mc_with_replay_profile(
    profile: ReplayDerivedProfile,
    **mc_kwargs: Any,
) -> MonteCarloResult:
    """Run a Monte Carlo simulation using a replay-derived profile.

    Parameters
    ----------
    profile : ReplayDerivedProfile
        Profile supplying win_rate, avg_win_dollars, avg_loss_dollars.
    **mc_kwargs
        Forwarded to :class:`MonteCarloValidator.__init__`
        (e.g. ``n_simulations``, ``max_trades``, ``starting_capital``).

    Returns
    -------
    MonteCarloResult
    """
    from risk.monte_carlo import MonteCarloValidator  # deferred import

    # Ensure avg_loss is negative as required by the validator interface.
    avg_loss = profile.avg_loss_dollars
    if avg_loss > 0:
        avg_loss = -avg_loss

    # Guard against degenerate profiles that would fail validation.
    if profile.avg_win_dollars <= 0:
        raise ValueError(
            "Cannot run MC with avg_win_dollars <= 0 "
            f"(got {profile.avg_win_dollars}). Profile may be empty or all-loss."
        )
    if avg_loss >= 0:
        raise ValueError(
            "Cannot run MC with avg_loss_dollars >= 0 after negation "
            f"(got {avg_loss}). Profile may be empty or all-win."
        )

    # Override max_trades with recommended horizon if not explicitly set.
    if "max_trades" not in mc_kwargs and profile.recommended_mc_horizon_trades > 0:
        mc_kwargs["max_trades"] = profile.recommended_mc_horizon_trades

    validator = MonteCarloValidator(**mc_kwargs)
    result = validator.run(
        win_rate=profile.win_rate,
        avg_win=profile.avg_win_dollars,
        avg_loss=avg_loss,
    )

    logger.info(
        "MC simulation complete — ruin=%.2f%% target=%.2f%% accepted=%s",
        result.ruin_probability * 100,
        result.target_probability * 100,
        result.accepted,
    )
    return result


def generate_mc_profile(
    run_dir: str,
    min_trade_count: int = 10,
) -> ReplayDerivedProfile:
    """One-shot convenience: build, write, and return a replay-derived profile.

    Prefers ``aggregate_trades.csv`` when available, else scans sessions.
    Also writes ``mc_profile.json`` at the run root.

    Parameters
    ----------
    run_dir : str
        Path to a validation run directory.
    min_trade_count : int
        Minimum trades required before the profile is considered adequate.

    Returns
    -------
    ReplayDerivedProfile
    """
    bridge = ReplayProfileBridge(run_dir, min_trade_count=min_trade_count)
    profile = bridge.build_profile_from_aggregate()
    bridge.write_profile(profile)
    logger.info("Profile generated for run: %s", run_dir)
    return profile


# ── Markdown renderer (private) ────────────────────────────────────────


def _render_markdown(profile: ReplayDerivedProfile, min_trade_count: int) -> str:
    """Render a human-readable Markdown interpretation of the profile."""
    adequacy = (
        "ADEQUATE"
        if profile.sample_size_trades >= min_trade_count
        else "LOW — increase sample"
    )

    notes_section = profile.notes if profile.notes else "None"

    return (
        "# Replay-Derived Monte Carlo Profile\n"
        "\n"
        f"**Source:** {profile.source}\n"
        f"**Sample:** {profile.sample_size_trades} trades across "
        f"{profile.sessions_in_sample} sessions\n"
        "\n"
        "## Key Parameters\n"
        "| Parameter | Value |\n"
        "|-----------|-------|\n"
        f"| Win Rate | {profile.win_rate * 100:.1f}% |\n"
        f"| Avg Win | ${profile.avg_win_dollars:,.2f} |\n"
        f"| Avg Loss | ${profile.avg_loss_dollars:,.2f} |\n"
        f"| Payoff Ratio | {profile.payoff_ratio:.4f} |\n"
        f"| Expectancy | ${profile.expectancy_dollars:,.2f}/trade |\n"
        f"| PnL Std Dev | ${profile.pnl_std_dollars:,.2f} |\n"
        "\n"
        "## MC Readiness\n"
        f"- Recommended horizon: {profile.recommended_mc_horizon_trades} trades\n"
        f"- Sample adequacy: {adequacy}\n"
        "\n"
        "## Notes\n"
        f"{notes_section}\n"
    )
