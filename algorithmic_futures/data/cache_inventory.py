"""Utilities for inspecting cached Databento session coverage.

This module intentionally stays read-only. It infers cached RTH session
coverage from the existing Parquet filename convention used by
`data.databento_provider`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DATABENTO_SYMBOL


@dataclass(frozen=True)
class CachedSession:
    session_id: str
    date: str
    start: str
    end: str
    file_path: Path


def trades_cache_dir(symbol: str = DATABENTO_SYMBOL, schema: str = "trades") -> Path:
    return Path(__file__).resolve().parent / "cache" / symbol / schema


def _slug_to_iso(slug: str) -> str:
    dt = datetime.strptime(slug, "%Y%m%d_%H%M%S")
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_cached_session_file(path: Path) -> CachedSession:
    stem = path.stem
    if "__" not in stem:
        raise ValueError(f"Unrecognized cache filename: {path.name}")
    start_slug, end_slug = stem.split("__", 1)
    if len(start_slug) < 8:
        raise ValueError(f"Unrecognized cache filename: {path.name}")
    date = start_slug[:8]
    return CachedSession(
        session_id=f"session_{date}",
        date=date,
        start=_slug_to_iso(start_slug),
        end=_slug_to_iso(end_slug),
        file_path=path,
    )


def list_cached_sessions(
    *,
    symbol: str = DATABENTO_SYMBOL,
    schema: str = "trades",
    cache_dir: Path | None = None,
) -> list[CachedSession]:
    root = cache_dir or trades_cache_dir(symbol=symbol, schema=schema)
    if not root.is_dir():
        return []
    sessions = [parse_cached_session_file(path) for path in root.glob("*.parquet")]
    return sorted(sessions, key=lambda session: (session.date, session.start, session.end))


def summarize_cached_sessions(
    *,
    symbol: str = DATABENTO_SYMBOL,
    schema: str = "trades",
    cache_dir: Path | None = None,
    include_gaps: bool = True,
) -> dict[str, Any]:
    sessions = list_cached_sessions(symbol=symbol, schema=schema, cache_dir=cache_dir)
    summary: dict[str, Any] = {
        "symbol": symbol,
        "schema": schema,
        "cache_dir": str(cache_dir or trades_cache_dir(symbol=symbol, schema=schema)),
        "session_count": len(sessions),
        "earliest_session": "",
        "latest_session": "",
        "earliest_date": "",
        "latest_date": "",
        "gap_count": 0,
        "gap_session_ids": [],
    }
    if not sessions:
        return summary

    summary["earliest_session"] = sessions[0].session_id
    summary["latest_session"] = sessions[-1].session_id
    summary["earliest_date"] = f"{sessions[0].date[:4]}-{sessions[0].date[4:6]}-{sessions[0].date[6:]}"
    summary["latest_date"] = f"{sessions[-1].date[:4]}-{sessions[-1].date[4:6]}-{sessions[-1].date[6:]}"

    if include_gaps:
        from validation.session_generator import generate_sessions_for_range

        expected = generate_sessions_for_range(summary["earliest_date"], summary["latest_date"])
        cached_ids = {session.session_id for session in sessions}
        missing_ids = sorted(
            session["session_id"] for session in expected if session["session_id"] not in cached_ids
        )
        summary["gap_count"] = len(missing_ids)
        summary["gap_session_ids"] = missing_ids

    return summary