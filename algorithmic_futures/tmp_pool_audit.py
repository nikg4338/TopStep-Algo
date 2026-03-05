#!/usr/bin/env python3
"""Find cached dates without a validation run and classify all sessions by ADX."""
import csv
import os
import statistics

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── 1. Cached trading days ──
cache_dir = os.path.join(ROOT, "data/cache/MES.c.0/trades")
cached_dates = set()
for f in os.listdir(cache_dir):
    if f.endswith(".parquet"):
        cached_dates.add(f[:8])

# ── 2. Sessions with full features in validation runs ──
val_dir = os.path.join(ROOT, "artifacts/validation_runs")
sessions = {}  # session_id -> best info dict

for run_id in os.listdir(val_dir):
    sess_dir = os.path.join(val_dir, run_id, "sessions")
    if not os.path.isdir(sess_dir):
        continue
    for sess in os.listdir(sess_dir):
        sp = os.path.join(sess_dir, sess)
        feat = os.path.join(sp, "features_snapshot.csv")
        trades_file = os.path.join(sp, "trade_log.csv")
        if not os.path.isfile(feat):
            continue
        with open(feat) as f:
            total_rows = sum(1 for _ in f) - 1
        if total_rows < 60:
            continue  # Skip short/replay sessions
        n_trades = 0
        if os.path.isfile(trades_file):
            with open(trades_file) as f:
                n_trades = max(0, sum(1 for _ in f) - 1)
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
        # Keep best version (most ADX bars)
        if sess not in sessions or len(adx_vals) > sessions[sess].get("n_adx_bars", 0):
            sessions[sess] = {
                "n_trades": n_trades,
                "adx_median": adx_med,
                "atr_median": atr_med,
                "n_adx_bars": len(adx_vals),
                "total_rows": total_rows,
                "run_id": run_id,
                "path": sp,
            }

# ── 3. Map session IDs to dates ──
session_dates = set()
for s in sessions:
    if s.startswith("session_"):
        session_dates.add(s.replace("session_", ""))

missing = cached_dates - session_dates
print(f"Cached trading days:    {len(cached_dates)}")
print(f"Full-RTH sessions run:  {len(sessions)}")
print(f"Session dates covered:  {len(session_dates)}")
print(f"Cached dates NOT run:   {len(missing)}")
if missing:
    for d in sorted(missing):
        print(f"  {d}")

# ── 4. ADX classification of run sessions ──
with_adx = {k: v for k, v in sessions.items() if v["adx_median"] > 0}
print(f"\nSessions with ADX>0: {len(with_adx)}")
adx_meds = sorted(v["adx_median"] for v in with_adx.values())
if adx_meds:
    n = len(adx_meds)
    print(f"ADX median: min={adx_meds[0]:.1f} p25={adx_meds[n//4]:.1f} "
          f"med={adx_meds[n//2]:.1f} p75={adx_meds[3*n//4]:.1f} max={adx_meds[-1]:.1f}")

# Tertile split for balanced strata
if len(adx_meds) >= 3:
    t1 = adx_meds[len(adx_meds) // 3]
    t2 = adx_meds[2 * len(adx_meds) // 3]
    print(f"\nTertile thresholds: <{t1:.1f} = range, {t1:.1f}-{t2:.1f} = mixed, >{t2:.1f} = trend")
    range_s = [(k, v) for k, v in with_adx.items() if v["adx_median"] < t1]
    mixed_s = [(k, v) for k, v in with_adx.items() if t1 <= v["adx_median"] <= t2]
    trend_s = [(k, v) for k, v in with_adx.items() if v["adx_median"] > t2]
    print(f"  Range: {len(range_s)}, Mixed: {len(mixed_s)}, Trend: {len(trend_s)}")

    print("\n── Range sessions ──")
    for k, v in sorted(range_s, key=lambda x: x[1]["adx_median"]):
        print(f"  {k:25s}  ADX={v['adx_median']:5.1f}  ATR={v['atr_median']:5.1f}  trades={v['n_trades']}")
    print("\n── Mixed sessions ──")
    for k, v in sorted(mixed_s, key=lambda x: x[1]["adx_median"]):
        print(f"  {k:25s}  ADX={v['adx_median']:5.1f}  ATR={v['atr_median']:5.1f}  trades={v['n_trades']}")
    print("\n── Trend sessions ──")
    for k, v in sorted(trend_s, key=lambda x: x[1]["adx_median"]):
        print(f"  {k:25s}  ADX={v['adx_median']:5.1f}  ATR={v['atr_median']:5.1f}  trades={v['n_trades']}")
