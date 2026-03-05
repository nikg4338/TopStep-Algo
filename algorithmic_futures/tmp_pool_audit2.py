#!/usr/bin/env python3
"""Complete pool audit: ADX classification + trade availability per session."""
import csv
import os
import statistics
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
val_dir = os.path.join(ROOT, "artifacts/validation_runs")
cache_dir = os.path.join(ROOT, "data/cache/MES.c.0/trades")

# ── 1. Cached trading days ──
cached_dates = set()
for f in os.listdir(cache_dir):
    if f.endswith(".parquet"):
        cached_dates.add(f[:8])

# ── 2. Scan ALL validation runs for full-RTH session_* entries ──
# For each session date, collect: best ADX data, best trade data (may be from different runs)
session_info = {}  # session_id -> {adx_median, atr_median, trades: [{pnl_dollars, ...}], run_with_trades, run_with_adx}

for run_id in sorted(os.listdir(val_dir)):
    sess_dir = os.path.join(val_dir, run_id, "sessions")
    if not os.path.isdir(sess_dir):
        continue
    for sess in sorted(os.listdir(sess_dir)):
        if not sess.startswith("session_"):
            continue
        sp = os.path.join(sess_dir, sess)

        # Features / ADX
        feat = os.path.join(sp, "features_snapshot.csv")
        if os.path.isfile(feat):
            with open(feat) as f:
                total_rows = sum(1 for _ in f) - 1
            if total_rows >= 60:
                adx_vals = []
                atr_vals = []
                with open(feat) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        a = float(row.get("adx", 0))
                        if a > 0:
                            adx_vals.append(a)
                        t = float(row.get("atr", 0))
                        if t > 0:
                            atr_vals.append(t)
                adx_med = statistics.median(adx_vals) if adx_vals else 0.0
                atr_med = statistics.median(atr_vals) if atr_vals else 0.0
                if sess not in session_info:
                    session_info[sess] = {"adx_median": 0, "atr_median": 0, "trades": [], "run_with_trades": None, "run_with_adx": None}
                if len(adx_vals) > 0 and (session_info[sess]["adx_median"] == 0 or len(adx_vals) > session_info[sess].get("n_adx_bars", 0)):
                    session_info[sess]["adx_median"] = adx_med
                    session_info[sess]["atr_median"] = atr_med
                    session_info[sess]["n_adx_bars"] = len(adx_vals)
                    session_info[sess]["run_with_adx"] = run_id

        # Trades
        trades_file = os.path.join(sp, "trades.csv")
        if os.path.isfile(trades_file):
            with open(trades_file) as f:
                reader = csv.DictReader(f)
                trades = []
                for row in reader:
                    trades.append({
                        "pnl_dollars": float(row.get("pnl_dollars", 0)),
                        "pnl_r": float(row.get("pnl_r", 0)),
                        "session_id": sess,
                        "regime_at_entry": row.get("regime_at_entry", ""),
                    })
            if trades and (sess not in session_info or len(trades) > len(session_info.get(sess, {}).get("trades", []))):
                if sess not in session_info:
                    session_info[sess] = {"adx_median": 0, "atr_median": 0, "trades": [], "run_with_trades": None, "run_with_adx": None}
                session_info[sess]["trades"] = trades
                session_info[sess]["run_with_trades"] = run_id

# ── 3. Summary ──
print(f"Cached trading days: {len(cached_dates)}")
print(f"Sessions scanned:    {len(session_info)}")

has_adx = {k: v for k, v in session_info.items() if v["adx_median"] > 0}
has_trades = {k: v for k, v in session_info.items() if len(v["trades"]) > 0}
has_both = {k: v for k, v in session_info.items() if v["adx_median"] > 0 and len(v["trades"]) > 0}
print(f"Sessions with ADX:   {len(has_adx)}")
print(f"Sessions with trades:{len(has_trades)}")
print(f"Sessions with BOTH:  {len(has_both)}")

# Dates not covered at all
session_dates = set()
for s in session_info:
    session_dates.add(s.replace("session_", ""))
missing = sorted(cached_dates - session_dates)
print(f"Cached dates NOT run: {len(missing)}  {missing}")

# ── 4. ADX tertile classification (sessions with BOTH adx + trades) ──
if has_both:
    adx_meds = sorted(v["adx_median"] for v in has_both.values())
    n = len(adx_meds)
    t1 = adx_meds[n // 3]
    t2 = adx_meds[2 * n // 3]
    print(f"\n--- Sessions with BOTH ADX + trades (N={n}) ---")
    print(f"ADX: min={adx_meds[0]:.1f} p25={adx_meds[n//4]:.1f} med={adx_meds[n//2]:.1f} p75={adx_meds[3*n//4]:.1f} max={adx_meds[-1]:.1f}")
    print(f"Tertile thresholds: <{t1:.1f} = range, {t1:.1f}-{t2:.1f} = mixed, >{t2:.1f} = trend")

    range_s = {k: v for k, v in has_both.items() if v["adx_median"] < t1}
    mixed_s = {k: v for k, v in has_both.items() if t1 <= v["adx_median"] <= t2}
    trend_s = {k: v for k, v in has_both.items() if v["adx_median"] > t2}
    print(f"Range: {len(range_s)}, Mixed: {len(mixed_s)}, Trend: {len(trend_s)}")

    total_trades = sum(len(v["trades"]) for v in has_both.values())
    print(f"Total trades across all sessions: {total_trades}")
    print(f"  Range trades: {sum(len(v['trades']) for v in range_s.values())}")
    print(f"  Mixed trades: {sum(len(v['trades']) for v in mixed_s.values())}")
    print(f"  Trend trades: {sum(len(v['trades']) for v in trend_s.values())}")

    for label, group in [("Range", range_s), ("Mixed", mixed_s), ("Trend", trend_s)]:
        print(f"\n── {label} ──")
        for k, v in sorted(group.items(), key=lambda x: x[1]["adx_median"]):
            pnl = sum(t["pnl_dollars"] for t in v["trades"])
            print(f"  {k:25s}  ADX={v['adx_median']:5.1f}  ATR={v['atr_median']:5.1f}  "
                  f"trades={len(v['trades']):2d}  pnl=${pnl:+.0f}  run={v['run_with_trades']}")

# ── 5. Also show sessions with ADX but NO trades ──
adx_no_trades = {k: v for k, v in has_adx.items() if len(v["trades"]) == 0}
if adx_no_trades:
    print(f"\n--- Sessions with ADX but NO trades (N={len(adx_no_trades)}) ---")
    for k, v in sorted(adx_no_trades.items(), key=lambda x: x[1]["adx_median"]):
        print(f"  {k:25s}  ADX={v['adx_median']:5.1f}  run={v['run_with_adx']}")
