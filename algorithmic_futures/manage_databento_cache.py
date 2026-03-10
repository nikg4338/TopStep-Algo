"""Inspect cached Databento coverage and backfill older historical sessions."""

from __future__ import annotations

import argparse
from dotenv import load_dotenv

from config import DATABENTO_SYMBOL
from data.cache_inventory import summarize_cached_sessions
from data.databento_provider import DatabentoReplayProvider, _cache_path
from validation.session_generator import generate_sessions_for_range


def _print_summary(summary: dict[str, object]) -> None:
    print("Databento cache summary")
    print("=======================")
    print(f"Symbol          : {summary['symbol']}")
    print(f"Schema          : {summary['schema']}")
    print(f"Cache dir       : {summary['cache_dir']}")
    print(f"Session count   : {summary['session_count']}")
    print(f"Earliest session: {summary['earliest_session'] or 'N/A'}")
    print(f"Latest session  : {summary['latest_session'] or 'N/A'}")
    gap_count = int(summary.get("gap_count", 0) or 0)
    print(f"Gap count       : {gap_count}")
    gap_session_ids = list(summary.get("gap_session_ids", []))
    if gap_session_ids:
        preview = ", ".join(gap_session_ids[:10])
        suffix = " ..." if len(gap_session_ids) > 10 else ""
        print(f"Gap preview     : {preview}{suffix}")


def _validate_backfill_range(start_date: str, end_date: str, before: dict[str, object], allow_overlap: bool) -> None:
    earliest_date = str(before.get("earliest_date", "") or "")
    if earliest_date and end_date >= earliest_date and not allow_overlap:
        raise SystemExit(
            "Requested backfill range overlaps existing cache coverage. "
            f"Requested end-date {end_date} must be strictly earlier than current earliest cached date {earliest_date}. "
            "Use --allow-overlap only when you intentionally want exact-session skips inside an overlapping request."
        )


def _backfill(args: argparse.Namespace) -> int:
    before = summarize_cached_sessions(symbol=args.symbol, include_gaps=True)
    _print_summary(before)
    _validate_backfill_range(args.start_date, args.end_date, before, args.allow_overlap)

    sessions = generate_sessions_for_range(args.start_date, args.end_date)
    if not sessions:
        raise SystemExit("No valid CME Equity sessions found in the requested date range.")

    if args.dry_run:
        print(f"\nDry run: {len(sessions)} session(s) selected for backward extension.")
        return 0

    load_dotenv()
    provider = DatabentoReplayProvider()
    fetched = 0
    skipped = 0
    for session in sessions:
        cache_file = _cache_path(args.symbol, "trades", session["start"], session["end"])
        if cache_file.exists():
            skipped += 1
            print(f"SKIP  {session['session_id']} already cached -> {cache_file.name}")
            continue
        trades = provider.fetch_trades(start=session["start"], end=session["end"], symbol=args.symbol)
        if trades.empty:
            raise SystemExit(f"No data returned for {session['session_id']} ({session['start']} -> {session['end']}).")
        fetched += 1
        print(f"FETCH {session['session_id']} rows={len(trades)}")

    if fetched == 0:
        raise SystemExit("No new sessions were fetched. Check the requested range or use --allow-overlap if needed.")

    after = summarize_cached_sessions(symbol=args.symbol, include_gaps=True)
    print(f"\nFetched {fetched} new session(s); skipped {skipped} existing session(s).\n")
    _print_summary(after)
    return 0


def _inspect(args: argparse.Namespace) -> int:
    summary = summarize_cached_sessions(symbol=args.symbol, include_gaps=not args.no_gaps)
    _print_summary(summary)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect or backward-extend the Databento MES cache.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Report cached Databento session coverage")
    inspect_parser.add_argument("--symbol", default=DATABENTO_SYMBOL)
    inspect_parser.add_argument("--no-gaps", action="store_true", help="Skip obvious gap detection")
    inspect_parser.set_defaults(handler=_inspect)

    backfill_parser = subparsers.add_parser("backfill", help="Fetch older historical sessions into the local cache")
    backfill_parser.add_argument("--start-date", required=True, help="Inclusive YYYY-MM-DD")
    backfill_parser.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD")
    backfill_parser.add_argument("--symbol", default=DATABENTO_SYMBOL)
    backfill_parser.add_argument("--allow-overlap", action="store_true", help="Permit overlapping requests and skip exact cached sessions")
    backfill_parser.add_argument("--dry-run", action="store_true", help="Print the selected sessions without fetching")
    backfill_parser.set_defaults(handler=_backfill)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())