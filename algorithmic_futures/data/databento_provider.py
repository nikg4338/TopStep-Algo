"""
data/databento_provider.py — Databento replay/historical data adapter.

Provides a simple historical trade fetch + replay loop that can feed
existing IntradayBarAggregator logic without changing strategy code.

Includes a local Parquet disk cache keyed by (symbol, schema, start, end)
to avoid redundant API calls.  Cache dir: ``data/cache/<symbol>/<schema>/``.
An index manifest (``data/cache/manifest.json``) lists all cached entries.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config import DATABENTO_DATASET, DATABENTO_STYPE_IN, DATABENTO_SYMBOL

logger = logging.getLogger(__name__)

# ── Cache configuration ────────────────────────────────────────────────

_CACHE_ROOT = Path(__file__).resolve().parent / "cache"
_CACHE_MANIFEST = _CACHE_ROOT / "manifest.json"


def _cache_key(symbol: str, schema: str, start: str, end: str, dataset: str) -> str:
    """Deterministic string key for a cached fetch."""
    return f"{dataset}|{symbol}|{schema}|{start}|{end}"


def _cache_path(symbol: str, schema: str, start: str, end: str) -> Path:
    """Return ``data/cache/<symbol>/<schema>/<start>__<end>.parquet``."""
    def _ts_slug(iso: str) -> str:
        return iso.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")

    return _CACHE_ROOT / symbol / schema / f"{_ts_slug(start)}__{_ts_slug(end)}.parquet"


def _load_manifest() -> dict[str, Any]:
    if _CACHE_MANIFEST.exists():
        return json.loads(_CACHE_MANIFEST.read_text(encoding="utf-8"))
    return {"entries": {}}


def _save_manifest(manifest: dict[str, Any]) -> None:
    _CACHE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _read_cache(path: Path) -> pd.DataFrame | None:
    """Read from Parquet cache if file exists."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            logger.warning("Cache file corrupt, re-fetching: %s", path)
    return None


def _write_cache(df: pd.DataFrame, path: Path, key: str, symbol: str,
                 schema: str, start: str, end: str, dataset: str) -> None:
    """Write DataFrame to Parquet and update manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, compression="zstd", index=False)
    except Exception:
        # zstd may not be available — fall back to snappy
        df.to_parquet(path, compression="snappy", index=False)

    manifest = _load_manifest()
    manifest["entries"][key] = {
        "symbol": symbol,
        "schema": schema,
        "start": start,
        "end": end,
        "dataset": dataset,
        "file": str(path.relative_to(_CACHE_ROOT)),
        "rows": len(df),
        "file_size_bytes": path.stat().st_size,
        "cached_at": datetime.utcnow().isoformat() + "Z",
    }
    _save_manifest(manifest)
    logger.info("Cached %d rows → %s", len(df), path)


@dataclass(frozen=True)
class ReplayStats:
    ticks_processed: int
    start: str
    end: str
    symbol: str


class DatabentoReplayProvider:
    """Fetch and replay historical Databento futures trade data."""

    def __init__(
        self,
        api_key: str | None = None,
        dataset: str = DATABENTO_DATASET,
        stype_in: str = DATABENTO_STYPE_IN,
    ) -> None:
        self.api_key = api_key or os.getenv("DATABENTO_API_KEY", "")
        if not self.api_key:
            raise ValueError("DATABENTO_API_KEY is required for Databento replay")
        self.dataset = dataset
        self.stype_in = stype_in

    def fetch_trades(
        self,
        *,
        start: str,
        end: str,
        symbol: str = DATABENTO_SYMBOL,
    ) -> pd.DataFrame:
        """Return normalized trades DataFrame: timestamp, price, size.

        Uses a local Parquet disk cache keyed by (symbol, schema, start, end).
        Cache hits return immediately without touching the Databento API.
        """
        schema = "trades"
        key = _cache_key(symbol, schema, start, end, self.dataset)
        cache_file = _cache_path(symbol, schema, start, end)

        # ── Cache hit ───────────────────────────────────────────────────
        cached = _read_cache(cache_file)
        if cached is not None:
            logger.info("Cache HIT (%d rows): %s %s→%s", len(cached), symbol, start, end)
            return cached

        # ── Cache miss — fetch from API ─────────────────────────────────
        logger.info("Cache MISS — fetching from Databento: %s %s→%s", symbol, start, end)
        import databento as db  # type: ignore[import-untyped]

        t0 = time.monotonic()
        client = db.Historical(self.api_key)
        raw = client.timeseries.get_range(
            dataset=self.dataset,
            symbols=[symbol],
            schema=schema,
            stype_in=self.stype_in,
            start=start,
            end=end,
        )
        df = raw.to_df()
        fetch_sec = time.monotonic() - t0
        logger.info("Databento fetch took %.1fs, %d raw rows", fetch_sec, len(df))

        if df.empty:
            return pd.DataFrame(columns=["timestamp", "price", "size"])

        ts_col = "ts_event" if "ts_event" in df.columns else df.index.name or "index"
        if ts_col in df.columns:
            timestamps = pd.to_datetime(df[ts_col], utc=True)
        else:
            timestamps = pd.to_datetime(df.index, utc=True)

        if "price" in df.columns:
            price_series = pd.to_numeric(df["price"], errors="coerce")
        else:
            price_series = pd.Series(index=df.index, dtype="float64")

        if "size" in df.columns:
            size_series = pd.to_numeric(df["size"], errors="coerce").fillna(1.0)
        else:
            size_series = pd.Series(1.0, index=df.index, dtype="float64")

        normalized = pd.DataFrame(
            {
                "timestamp": timestamps,
                "price": price_series,
                "size": size_series,
            }
        ).dropna(subset=["timestamp", "price"])

        normalized = normalized[normalized["price"] > 0]
        normalized = normalized.sort_values("timestamp").reset_index(drop=True)

        # ── Write to cache ──────────────────────────────────────────────
        _write_cache(normalized, cache_file, key, symbol, schema, start, end, self.dataset)

        return normalized

    def replay_trades(
        self,
        *,
        start: str,
        end: str,
        on_tick: Callable[[datetime, float, float], None],
        symbol: str = DATABENTO_SYMBOL,
        max_ticks: int | None = None,
    ) -> ReplayStats:
        """Replay historical trades into a callback signature used by bar aggregation."""
        trades = self.fetch_trades(start=start, end=end, symbol=symbol)
        if max_ticks is not None:
            trades = trades.head(max_ticks)

        ts_series = pd.to_datetime(trades["timestamp"], utc=True).dt.floor("us")
        px_series = pd.to_numeric(trades["price"], errors="coerce").fillna(0.0)
        sz_series = pd.to_numeric(trades["size"], errors="coerce").fillna(1.0)

        for ts, px, sz in zip(ts_series, px_series, sz_series):
            if px <= 0:
                continue
            on_tick(ts.to_pydatetime(), float(px), float(sz))

        logger.info(
            "Databento replay complete: %d ticks (%s → %s, %s)",
            len(trades),
            start,
            end,
            symbol,
        )
        return ReplayStats(
            ticks_processed=int(len(trades)),
            start=start,
            end=end,
            symbol=symbol,
        )
