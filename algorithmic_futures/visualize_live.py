"""
visualize_live.py — Matplotlib live visualization for Monte Carlo and Databento replay.

Usage examples:
  # Monte Carlo live chart
  python visualize_live.py mc --simulations 10000 --chunk-size 250

  # Replay live chart
  python visualize_live.py replay --start 2026-02-18T14:30:00Z --end 2026-02-18T15:00:00Z --max-ticks 5000

  # Headless save (no window)
  python visualize_live.py mc --no-show --save-path logs/mc_live.png
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from dotenv import load_dotenv

from config import (
    MAX_LOSS_LIMIT,
    MC_DRAWDOWN_P95_MAX,
    MC_READINESS_MAX_TRADES,
    MC_READINESS_STREAK_P95_MAX,
    MC_RUIN_THRESHOLD,
    MC_SIMULATIONS,
    MC_TARGET_THRESHOLD,
    PROFIT_TARGET,
    RISK_PER_TRADE,
)
from data.databento_provider import DatabentoReplayProvider
from data.market_data import Bar, IntradayBarAggregator


@dataclass(frozen=True)
class _BatchMetrics:
    ruined: int
    targeted: int
    drawdowns: np.ndarray
    streaks: np.ndarray


def _simulate_batch(
    *,
    n_paths: int,
    max_trades: int,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    starting_capital: float,
    target_boundary: float,
    rng: np.random.Generator,
) -> _BatchMetrics:
    is_win = rng.random((n_paths, max_trades)) < win_rate
    outcomes = np.where(is_win, avg_win, avg_loss)

    capital = np.full(n_paths, starting_capital, dtype=np.float64)
    peak = capital.copy()
    max_dd = np.zeros(n_paths, dtype=np.float64)
    streak = np.zeros(n_paths, dtype=np.int64)
    max_streak = np.zeros(n_paths, dtype=np.int64)

    hit_ruin = np.zeros(n_paths, dtype=bool)
    hit_target = np.zeros(n_paths, dtype=bool)
    active = np.ones(n_paths, dtype=bool)

    for t in range(max_trades):
        capital[active] += outcomes[active, t]

        peak[active] = np.maximum(peak[active], capital[active])
        dd = peak[active] - capital[active]
        max_dd[active] = np.maximum(max_dd[active], dd)

        is_loss = outcomes[active, t] < 0
        streak_active = streak[active]
        streak_active[~is_loss] = 0
        streak_active[is_loss] += 1
        streak[active] = streak_active
        max_streak[active] = np.maximum(max_streak[active], streak[active])

        newly_ruined = active & (capital <= 0.0)
        newly_targeted = active & (capital >= target_boundary)

        hit_ruin |= newly_ruined
        hit_target |= newly_targeted
        active &= ~(newly_ruined | newly_targeted)

        if not active.any():
            break

    return _BatchMetrics(
        ruined=int(hit_ruin.sum()),
        targeted=int(hit_target.sum()),
        drawdowns=max_dd,
        streaks=max_streak,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Matplotlib visuals for MC + replay")
    parser.add_argument("--no-show", action="store_true", help="Run without opening a window")
    parser.add_argument("--save-path", default="", help="Optional output image path")

    sub = parser.add_subparsers(dest="mode", required=True)

    mc = sub.add_parser("mc", help="Live Monte Carlo visualization")
    mc.add_argument("--simulations", type=int, default=MC_SIMULATIONS)
    mc.add_argument("--max-trades", type=int, default=MC_READINESS_MAX_TRADES)
    mc.add_argument("--chunk-size", type=int, default=250)
    mc.add_argument("--win-rate", type=float, default=0.50)
    mc.add_argument("--avg-win", type=float, default=30.0)
    mc.add_argument("--avg-loss", type=float, default=-22.0)
    mc.add_argument("--seed", type=int, default=42)
    mc.add_argument("--pause", type=float, default=0.03)

    replay = sub.add_parser("replay", help="Live Databento replay visualization")
    replay.add_argument("--start", required=True)
    replay.add_argument("--end", required=True)
    replay.add_argument("--symbol", default="MES.c.0")
    replay.add_argument("--max-ticks", type=int, default=None)
    replay.add_argument("--update-every", type=int, default=100)
    replay.add_argument("--pause", type=float, default=0.001)

    return parser


def _prepare_matplotlib(no_show: bool) -> None:
    if no_show:
        import matplotlib

        matplotlib.use("Agg")


def _run_mc(args: argparse.Namespace) -> int:
    _prepare_matplotlib(args.no_show)
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(args.seed)

    starting_capital = float(MAX_LOSS_LIMIT)
    target_boundary = starting_capital + float(PROFIT_TARGET)
    risk_per_trade = float(RISK_PER_TRADE)

    print("MC config (same as readiness_check):")
    print(f"  starting_capital  = ${starting_capital:,.0f}")
    print(f"  target_boundary   = ${target_boundary:,.0f}  (profit_target=${PROFIT_TARGET:,})")
    print(f"  risk_per_trade    = ${risk_per_trade:,.0f}")
    print(f"  max_trades        = {args.max_trades}  (MC_READINESS_MAX_TRADES={MC_READINESS_MAX_TRADES})")
    print(f"  avg_win           = ${args.avg_win:.2f}  avg_loss = ${args.avg_loss:.2f}  (dollar units)")
    print(f"  win_rate          = {args.win_rate:.0%}")
    print(f"  paths are in DOLLARS — target is in DOLLARS")
    print()

    total = 0
    ruined_total = 0
    targeted_total = 0
    dd_samples: list[float] = []
    streak_samples: list[float] = []

    x_vals: list[int] = []
    ruin_vals: list[float] = []
    target_vals: list[float] = []
    dd_vals: list[float] = []
    streak_vals: list[float] = []

    fig, (ax_prob, ax_risk) = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    line_ruin, = ax_prob.plot([], [], label="Ruin Probability", color="tab:red")
    line_target, = ax_prob.plot([], [], label="Target Probability", color="tab:green")
    ax_prob.set_ylim(0, 1)
    ax_prob.set_title("Monte Carlo Live Progress")
    ax_prob.set_xlabel("Simulations Processed")
    ax_prob.set_ylabel("Probability")
    ax_prob.legend(loc="best")

    line_dd, = ax_risk.plot([], [], label="Max Drawdown p95", color="tab:blue")
    line_streak, = ax_risk.plot([], [], label="Losing Streak p95", color="tab:orange")
    ax_risk.set_xlabel("Simulations Processed")
    ax_risk.set_ylabel("Risk Metrics")
    ax_risk.legend(loc="best")

    # Config annotation on chart (matches readiness_check params)
    config_text = (
        f"capital=${starting_capital:,.0f}  profit_target=${PROFIT_TARGET:,}  "
        f"boundary=${target_boundary:,.0f}  trades={args.max_trades}  "
        f"risk/trade=${risk_per_trade:,.0f}\n"
        f"win={args.win_rate:.0%}  avg_w=${args.avg_win:.0f}  avg_l=${args.avg_loss:.0f}  "
        f"unit=DOLLARS  (50K Combine, MLL=${MAX_LOSS_LIMIT:,})"
    )
    fig.text(0.5, 0.01, config_text, ha="center", fontsize=8, color="gray")

    if not args.no_show:
        plt.ion()
        plt.show(block=False)

    while total < args.simulations:
        n_batch = min(args.chunk_size, args.simulations - total)
        batch = _simulate_batch(
            n_paths=n_batch,
            max_trades=args.max_trades,
            win_rate=args.win_rate,
            avg_win=args.avg_win,
            avg_loss=args.avg_loss,
            starting_capital=starting_capital,
            target_boundary=target_boundary,
            rng=rng,
        )

        total += n_batch
        ruined_total += batch.ruined
        targeted_total += batch.targeted
        dd_samples.extend(batch.drawdowns.tolist())
        streak_samples.extend(batch.streaks.tolist())

        ruin_prob = ruined_total / total
        target_prob = targeted_total / total
        dd_p95 = float(np.percentile(dd_samples, 95)) if dd_samples else 0.0
        streak_p95 = float(np.percentile(streak_samples, 95)) if streak_samples else 0.0

        x_vals.append(total)
        ruin_vals.append(ruin_prob)
        target_vals.append(target_prob)
        dd_vals.append(dd_p95)
        streak_vals.append(streak_p95)

        line_ruin.set_data(x_vals, ruin_vals)
        line_target.set_data(x_vals, target_vals)
        line_dd.set_data(x_vals, dd_vals)
        line_streak.set_data(x_vals, streak_vals)

        ax_prob.relim()
        ax_prob.autoscale_view(scaley=False)
        ax_risk.relim()
        ax_risk.autoscale_view()

        ax_prob.set_title(
            "Monte Carlo Live Progress | "
            f"sim={total}/{args.simulations} | ruin={ruin_prob:.2%} | target={target_prob:.2%}"
        )

        if not args.no_show:
            plt.pause(args.pause)

    accepted = (
        ruin_vals[-1] <= MC_RUIN_THRESHOLD
        and target_vals[-1] >= MC_TARGET_THRESHOLD
        and dd_vals[-1] <= MC_DRAWDOWN_P95_MAX
        and streak_vals[-1] <= MC_READINESS_STREAK_P95_MAX
    )

    print(
        "MC final | "
        f"ruin={ruin_vals[-1]:.2%} (threshold<={MC_RUIN_THRESHOLD:.0%}) | "
        f"target={target_vals[-1]:.2%} (threshold>={MC_TARGET_THRESHOLD:.0%}) | "
        f"dd95=${dd_vals[-1]:,.0f} (max=${MC_DRAWDOWN_P95_MAX:,.0f}) | "
        f"streak95={streak_vals[-1]:.0f} (max={MC_READINESS_STREAK_P95_MAX}) | "
        f"accepted={accepted}"
    )

    if args.save_path:
        fig.savefig(args.save_path, dpi=140)
        print(f"Saved figure: {args.save_path}")

    if not args.no_show:
        plt.ioff()
        plt.show()

    return 0


def _run_replay(args: argparse.Namespace) -> int:
    _prepare_matplotlib(args.no_show)
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    provider = DatabentoReplayProvider()
    trades = provider.fetch_trades(start=args.start, end=args.end, symbol=args.symbol)
    if args.max_ticks is not None:
        trades = trades.head(args.max_ticks)

    fig, (ax_px, ax_bar) = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    line_px, = ax_px.plot([], [], color="tab:blue", label="Trade Price")
    line_bar, = ax_bar.plot([], [], color="tab:purple", label="5m Close")

    # Datetime x-axis formatting (no ugly offset / scientific notation)
    for ax in (ax_px, ax_bar):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    ax_px.set_title("Databento Replay Live")
    ax_px.set_xlabel("Time")
    ax_px.set_ylabel("Price")
    ax_px.legend(loc="best")

    ax_bar.set_xlabel("Bar Time")
    ax_bar.set_ylabel("Close")
    ax_bar.legend(loc="best")

    bar_times: list = []
    bar_opens: list[float] = []
    bar_highs: list[float] = []
    bar_lows: list[float] = []
    bar_closes: list[float] = []
    bar_volumes: list[float] = []
    bar_is_partial: list[bool] = []  # track which bars came from flush
    bars_closed = 0
    bars_partial_flushed = 0
    unique_buckets: set[str] = set()  # bucket keys seen across all ticks

    def on_bar(bar: Bar, *, is_flush: bool = False) -> None:
        nonlocal bars_closed, bars_partial_flushed
        if is_flush:
            bars_partial_flushed += 1
        else:
            bars_closed += 1
        bar_times.append(bar.timestamp)
        bar_opens.append(bar.open)
        bar_highs.append(bar.high)
        bar_lows.append(bar.low)
        bar_closes.append(bar.close)
        bar_volumes.append(bar.volume)
        bar_is_partial.append(is_flush)
        n = bars_closed + bars_partial_flushed
        label = "PARTIAL-FLUSH" if is_flush else "CLOSED"
        print(
            f"  [BAR #{n} {label}] "
            f"bar_time={bar.timestamp}  "
            f"O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} V={bar.volume:.0f}"
        )

    agg = IntradayBarAggregator(interval_minutes=5, on_bar_callback=on_bar)

    if not args.no_show:
        plt.ion()
        plt.show(block=False)

    times: list = []
    prices: list[float] = []
    _first_bucket_logged = False

    ts_series = np.array(trades["timestamp"], dtype="datetime64[us]")
    px_series = np.asarray(trades["price"], dtype=float)
    sz_series = np.asarray(trades["size"], dtype=float)

    for idx, (ts64, px, sz) in enumerate(zip(ts_series, px_series, sz_series), start=1):
        ts = np.datetime64(ts64, "us").astype("datetime64[us]").astype(object)

        # Track unique bucket keys
        bucket_key = ts.replace(
            minute=(ts.minute // 5) * 5, second=0, microsecond=0
        ).strftime("%H:%M")
        unique_buckets.add(bucket_key)

        # Debug: log bucket key for first few ticks
        if not _first_bucket_logged and idx <= 3:
            print(f"  [TICK #{idx}] ts={ts}  bucket={bucket_key}  px={float(px):.2f}")
            if idx == 3:
                _first_bucket_logged = True

        agg.on_tick(ts, float(px), float(sz))

        times.append(ts)
        prices.append(float(px))

        if idx % args.update_every == 0 or idx == len(trades):
            line_px.set_data(times, prices)
            line_bar.set_data(bar_times, bar_closes)

            ax_px.relim()
            ax_px.autoscale_view()
            ax_bar.relim()
            ax_bar.autoscale_view()

            has_partial = agg._bar_start is not None and agg._current_open is not None
            total_bars = bars_closed + bars_partial_flushed
            ax_px.set_title(
                "Databento Replay Live | "
                f"ticks={idx}/{len(trades)}  last={float(px):.2f}  "
                f"bars_closed={bars_closed}  partial={'yes' if has_partial else 'no'}"
            )

            if not args.no_show:
                plt.pause(args.pause)

    # Flush partial bar at end (replay/debug only — not for live signal use)
    has_partial_before_flush = (
        agg._bar_start is not None and agg._current_open is not None
    )
    print(f"\n  [FLUSH] partial_bar_exists={has_partial_before_flush}  (replay-only; not used for signals)")
    if has_partial_before_flush:
        # Temporarily swap callback to tag partial bar
        _orig_cb = agg.on_bar
        agg.on_bar = lambda bar: on_bar(bar, is_flush=True)
        agg.flush()
        agg.on_bar = _orig_cb
    else:
        agg.flush()

    # ── Bar plot: handle 1-bar edge case ────────────────────────────────
    if len(bar_closes) == 1:
        # Scatter point + annotation instead of invisible 1-point line
        ax_bar.scatter(bar_times, bar_closes, color="tab:purple", s=80, zorder=5,
                       label=f"5m Close ({'partial' if bar_is_partial[0] else 'closed'})")
        ax_bar.annotate(
            f"{bar_closes[0]:.2f}\n{bar_times[0]}",
            xy=(bar_times[0], bar_closes[0]),
            xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=8, color="tab:purple",
        )
        ax_bar.set_ylim(bar_closes[0] - 5, bar_closes[0] + 5)  # ±5 points
        ax_bar.legend(loc="best")
    elif len(bar_closes) > 1:
        line_bar.set_data(bar_times, bar_closes)
        ax_bar.relim()
        ax_bar.autoscale_view()
    # else: no bars — leave empty

    # Re-draw tick chart
    line_px.set_data(times, prices)
    ax_px.relim()
    ax_px.autoscale_view()

    # Status footer on bar subplot
    last_bar_ts = bar_times[-1] if bar_times else "none"
    ax_bar.set_title(
        f"5m Bars | bars_closed={bars_closed}  bars_partial_flushed={bars_partial_flushed}  "
        f"unique_buckets={len(unique_buckets)}  "
        f"last_bar_ts={last_bar_ts}"
    )

    # Debug: pre-plot array lengths + bucket diagnostic
    total_bars = bars_closed + bars_partial_flushed
    print(f"  [PLOT] len(bar_times)={len(bar_times)}  len(bar_closes)={len(bar_closes)}")
    print(f"  [BUCKETS] unique_buckets={len(unique_buckets)}  keys={sorted(unique_buckets)}")
    if len(unique_buckets) > 1 and total_bars <= 1:
        print("  ⚠️  Multiple buckets seen but ≤1 bar emitted — possible bucket-close bug!")
    elif len(unique_buckets) == 1:
        print("  ℹ️  Only 1 bucket seen — replay didn't span enough market time for multiple bars")
    if bar_closes:
        print(f"  [PLOT] last_bar_close={bar_closes[-1]:.2f}  last_bar_time={bar_times[-1]}")

    print(
        f"\nReplay final | "
        f"ticks={len(trades)}  bars_closed={bars_closed}  "
        f"bars_partial_flushed={bars_partial_flushed}  "
        f"unique_buckets={len(unique_buckets)}  "
        f"last_price={prices[-1]:.2f}"
    )

    if args.save_path:
        fig.savefig(args.save_path, dpi=140)
        print(f"Saved figure: {args.save_path}")

    if not args.no_show:
        plt.ioff()
        plt.show()

    return 0


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "mc":
        return _run_mc(args)
    if args.mode == "replay":
        return _run_replay(args)
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
