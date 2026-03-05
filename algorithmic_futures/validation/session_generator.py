"""
validation/session_generator.py — Generate SessionEntry objects from a date range.

Uses ``pandas_market_calendars`` to enumerate valid CME Equity trading days,
then produces one full-RTH SessionEntry per day.  Skips early-close days.

Usage:
    from validation.session_generator import generate_sessions_for_range

    sessions = generate_sessions_for_range("2025-12-22", "2026-02-20")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────

DEFAULT_EXCHANGE = "CME_Equity"

# Full RTH session in UTC (09:30–16:00 ET = 14:30–21:00 UTC)
DEFAULT_SESSION_START_UTC = "14:30:00"
DEFAULT_SESSION_END_UTC = "21:00:00"

# Full RTH session duration (6 h 30 min) — used to detect early-close days
FULL_RTH_MINUTES = 390

# Default bar interval for expected-bars calculation
DEFAULT_BAR_INTERVAL_MIN = 5


@dataclass
class SessionTemplate:
    """Configurable session window template (all times in UTC)."""

    start_time_utc: str = DEFAULT_SESSION_START_UTC  # e.g. "14:30:00"
    end_time_utc: str = DEFAULT_SESSION_END_UTC      # e.g. "21:00:00"
    symbol: str = "MES.c.0"
    bar_interval_min: int = DEFAULT_BAR_INTERVAL_MIN
    skip_early_close: bool = True  # v1: skip shortened days


def generate_sessions_for_range(
    start_date: str,
    end_date: str,
    *,
    template: SessionTemplate | None = None,
    exchange: str = DEFAULT_EXCHANGE,
    category: str = "unlabeled",
) -> list[dict]:
    """Generate one SessionEntry-compatible dict per valid trading day.

    Parameters
    ----------
    start_date:
        First calendar date (inclusive), ``YYYY-MM-DD``.
    end_date:
        Last calendar date (inclusive), ``YYYY-MM-DD``.
    template:
        Session window template.  Uses full-RTH defaults when *None*.
    exchange:
        ``pandas_market_calendars`` exchange name.
    category:
        Category label for generated sessions. Use ``"unlabeled"`` for
        auto-generated packs (avoids regime-labelling bias).

    Returns
    -------
    list[dict]
        Each dict has keys: ``session_id``, ``start``, ``end``, ``category``,
        ``symbol``, ``expected_bars``, ``is_early_close``.
    """
    import pandas_market_calendars as mcal  # type: ignore[import-untyped]

    tmpl = template or SessionTemplate()
    cal = mcal.get_calendar(exchange)

    # Get schedule — includes early close info
    schedule = cal.schedule(start_date=start_date, end_date=end_date)

    sessions: list[dict] = []
    skipped_early_close = 0

    for date_idx, row in schedule.iterrows():
        date_str = pd.Timestamp(str(date_idx)).strftime("%Y-%m-%d")
        market_open: pd.Timestamp = row["market_open"]
        market_close: pd.Timestamp = row["market_close"]

        # Detect early close
        rth_minutes = (market_close - market_open).total_seconds() / 60
        is_early_close = rth_minutes < FULL_RTH_MINUTES

        if is_early_close and tmpl.skip_early_close:
            logger.info("Skipping early-close day: %s (%.0f min)", date_str, rth_minutes)
            skipped_early_close += 1
            continue

        # Build session window
        session_start = f"{date_str}T{tmpl.start_time_utc}Z"
        session_end = f"{date_str}T{tmpl.end_time_utc}Z"

        # Calculate expected bar count
        start_dt = datetime.fromisoformat(session_start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(session_end.replace("Z", "+00:00"))
        window_minutes = (end_dt - start_dt).total_seconds() / 60
        expected_bars = int(window_minutes / tmpl.bar_interval_min)

        session_id = f"session_{date_str.replace('-', '')}"

        sessions.append({
            "session_id": session_id,
            "start": session_start,
            "end": session_end,
            "category": category,
            "symbol": tmpl.symbol,
            "expected_bars": expected_bars,
            "is_early_close": is_early_close,
        })

    if skipped_early_close:
        logger.info(
            "Skipped %d early-close days out of %d total",
            skipped_early_close, len(schedule),
        )

    logger.info(
        "Generated %d sessions for %s → %s (%d skipped)",
        len(sessions), start_date, end_date, skipped_early_close,
    )
    return sessions


def print_sessions_dry_run(sessions: Sequence[dict]) -> None:
    """Print a formatted table of generated sessions for --dry-run mode."""
    print(f"\n{'─'*80}")
    print(f"  {'DATE':<12} {'START (UTC)':<22} {'END (UTC)':<22} {'BARS':>5} {'EARLY?':>6}")
    print(f"{'─'*80}")
    for s in sessions:
        date = s["session_id"].replace("session_", "")
        date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        early = "YES" if s.get("is_early_close") else "no"
        print(f"  {date_fmt:<12} {s['start']:<22} {s['end']:<22} {s['expected_bars']:>5} {early:>6}")
    print(f"{'─'*80}")
    print(f"  Total: {len(sessions)} sessions\n")
