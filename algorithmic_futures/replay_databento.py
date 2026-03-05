"""
replay_databento.py — Run Databento historical trade replay through bar aggregation.

Example:
  python replay_databento.py --start 2026-02-18T13:30:00Z --end 2026-02-18T20:00:00Z
"""

from __future__ import annotations

import argparse
from collections import deque

from dotenv import load_dotenv

from config import DATABENTO_SYMBOL, VWAP_BAR_INTERVAL_MIN
from data.databento_provider import DatabentoReplayProvider
from data.market_data import Bar, IntradayBarAggregator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Databento trades into 5m bars")
    parser.add_argument("--start", required=True, help="ISO timestamp, e.g. 2026-02-18T13:30:00Z")
    parser.add_argument("--end", required=True, help="ISO timestamp, e.g. 2026-02-18T20:00:00Z")
    parser.add_argument("--symbol", default=DATABENTO_SYMBOL, help="Databento symbol (default MES.c.0)")
    parser.add_argument("--max-ticks", type=int, default=None, help="Optional cap for quick smoke tests")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    emitted: deque[Bar] = deque(maxlen=5)

    def on_bar(bar: Bar) -> None:
        emitted.append(bar)

    aggregator = IntradayBarAggregator(
        interval_minutes=VWAP_BAR_INTERVAL_MIN,
        on_bar_callback=on_bar,
    )

    provider = DatabentoReplayProvider()
    stats = provider.replay_trades(
        start=args.start,
        end=args.end,
        symbol=args.symbol,
        max_ticks=args.max_ticks,
        on_tick=aggregator.on_tick,
    )
    aggregator.flush()

    print("Databento replay summary")
    print("========================")
    print(f"Symbol        : {stats.symbol}")
    print(f"Window        : {stats.start} -> {stats.end}")
    print(f"Ticks replayed: {stats.ticks_processed}")
    print(f"Bars emitted  : {len(aggregator.bars)}")

    if emitted:
        last = emitted[-1]
        print(
            "Last bar      : "
            f"{last.timestamp} O={last.open:.2f} H={last.high:.2f} "
            f"L={last.low:.2f} C={last.close:.2f} V={last.volume:.0f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
