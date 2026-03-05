"""
main.py — Entry point for the Algorithmic Futures trading system.

Responsibilities:
  - Load environment variables from .env
  - Configure logging (console + rotating file)
  - Optionally run Monte Carlo validation only (--validate)
  - Create SessionManager and start the daily lifecycle
  - Handle KeyboardInterrupt for clean shutdown
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


def _setup_logging() -> None:
    """Configure root logger with console + file handlers."""
    from config import LOG_DIR, TIMEZONE

    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"session_{today}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    # File handler — DEBUG and above (full detail)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "websockets", "hmmlearn", "yfinance", "apscheduler"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.info("Logging initialised → console (INFO) + %s (DEBUG)", log_file)


def _run_validation() -> None:
    """Run Monte Carlo validation only — no trading, no API calls."""
    from config import RISK_PER_TRADE
    from risk.monte_carlo import MonteCarloValidator

    logger = logging.getLogger("validation")
    logger.info("=" * 60)
    logger.info("  Monte Carlo Validation Mode")
    logger.info("=" * 60)

    # Conservative parameters matching strategy expectations
    # VWAP Mean Reversion: ~55% win rate, 1:1.5 R:R
    # ORB Breakout:        ~45% win rate, 1:1.5 R:R
    scenarios = [
        {
            "name": "VWAP Mean Reversion (State 0)",
            "win_rate": 0.55,
            "avg_win": 30.0,
            "avg_loss": -22.0,
        },
        {
            "name": "ORB Breakout (State 1)",
            "win_rate": 0.45,
            "avg_win": 33.0,
            "avg_loss": -22.0,
        },
        {
            "name": "Blended (weighted by regime frequency)",
            "win_rate": 0.50,
            "avg_win": 30.0,
            "avg_loss": -22.0,
        },
    ]

    mc = MonteCarloValidator()
    all_passed = True

    for scenario in scenarios:
        logger.info("-" * 40)
        logger.info("Scenario: %s", scenario["name"])
        result = mc.run(
            win_rate=scenario["win_rate"],
            avg_win=scenario["avg_win"],
            avg_loss=scenario["avg_loss"],
            seed=42,
        )
        logger.info("\n%s", result.summary())
        if not result.accepted:
            all_passed = False

    logger.info("=" * 60)
    if all_passed:
        logger.info("ALL SCENARIOS ACCEPTED — risk parameters viable")
    else:
        logger.warning("ONE OR MORE SCENARIOS REJECTED — review risk parameters before trading")
    logger.info("=" * 60)

    sys.exit(0 if all_passed else 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="algorithmic_futures",
        description="Algorithmic Futures — Topstep Challenge Trading System",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run Monte Carlo validation only (no trading, no API calls)",
    )
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    # 1. Load environment variables (.env in project root)
    load_dotenv()

    # 2. Parse CLI arguments
    args = _parse_args()

    # 3. Set up logging
    _setup_logging()

    logger = logging.getLogger("main")
    logger.info("Algorithmic Futures starting…")

    # 4. Validation-only mode
    if args.validate:
        _run_validation()
        return  # _run_validation calls sys.exit, but guard anyway

    # 5. Normal trading mode
    from session_manager import SessionManager

    manager = SessionManager()

    try:
        manager.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — initiating clean shutdown")
    except Exception:
        logger.exception("Unhandled exception in SessionManager")
    finally:
        manager.stop()
        logger.info("Algorithmic Futures terminated")


if __name__ == "__main__":
    main()
