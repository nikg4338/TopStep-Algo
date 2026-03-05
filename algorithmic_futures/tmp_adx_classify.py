#!/usr/bin/env python3
"""Classify sessions by median non-zero ADX."""
import csv, os, statistics

base = "artifacts/validation_runs"
sessions = {}

for run_dir in os.listdir(base):
    run_path = os.path.join(base, run_dir)
    sess_root = os.path.join(run_path, "sessions")
    if not os.path.isdir(sess_root):
        continue
    for sess in os.listdir(sess_root):
        sp = os.path.join(sess_root, sess)
        feat = os.path.join(sp, "features_snapshot.csv")
        trades = os.path.join(sp, "trade_log.csv")
        if not os.path.isfile(feat):
            continue
        n_trades = 0
        if os.path.isfile(trades):
            with open(trades) as f:
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
        if sess not in sessions or len(adx_vals) > len(sessions.get(sess, {}).get("adx_vals", [])):
            sessions[sess] = {
                "n_trades": n_trades,
                "adx_vals": adx_vals,
                "adx_median": statistics.median(adx_vals) if adx_vals else 0.0,
                "adx_mean": statistics.mean(adx_vals) if adx_vals else 0.0,
                "atr_median": statistics.median(atr_vals) if atr_vals else 0.0,
                "n_adx_bars": len(adx_vals),
                "path": sp,
            }

print(f"Total sessions: {len(sessions)}")
with_adx = {k: v for k, v in sessions.items() if v["adx_median"] > 0}
print(f"Sessions with ADX > 0: {len(with_adx)}")
no_adx = {k: v for k, v in sessions.items() if v["adx_median"] == 0}
print(f"Sessions with ADX = 0: {len(no_adx)}")

if with_adx:
    adx_meds = [v["adx_median"] for v in with_adx.values()]
    adx_sorted = sorted(adx_meds)
    n = len(adx_sorted)
    print(f"\nADX median stats (N={n}):")
    print(f"  min={min(adx_meds):.1f}, p25={adx_sorted[n//4]:.1f}, "
          f"med={statistics.median(adx_meds):.1f}, p75={adx_sorted[3*n//4]:.1f}, "
          f"max={max(adx_meds):.1f}")

    # Classify by ADX: <20 range, 20-30 mixed, >30 trend
    range_s = [k for k, v in with_adx.items() if v["adx_median"] < 20]
    mixed_s = [k for k, v in with_adx.items() if 20 <= v["adx_median"] <= 30]
    trend_s = [k for k, v in with_adx.items() if v["adx_median"] > 30]
    print(f"\nClassification (ADX<20=range, 20-30=mixed, >30=trend):")
    print(f"  Range: {len(range_s)}")
    print(f"  Mixed: {len(mixed_s)}")
    print(f"  Trend: {len(trend_s)}")

    # Try different thresholds
    print("\nAlternative thresholds:")
    for lo, hi in [(18, 25), (15, 25), (20, 35), (25, 40)]:
        r = sum(1 for v in with_adx.values() if v["adx_median"] < lo)
        m = sum(1 for v in with_adx.values() if lo <= v["adx_median"] <= hi)
        t = sum(1 for v in with_adx.values() if v["adx_median"] > hi)
        print(f"  {lo}/{hi}: range={r}, mixed={m}, trend={t}")

    # Print distribution
    print("\nADX median histogram:")
    for bucket_lo in range(10, 65, 5):
        bucket_hi = bucket_lo + 5
        count = sum(1 for v in with_adx.values() if bucket_lo <= v["adx_median"] < bucket_hi)
        bar = "#" * count
        print(f"  [{bucket_lo:2d}-{bucket_hi:2d}): {count:3d} {bar}")

# Also check the no-ADX sessions
if no_adx:
    print(f"\n--- Sessions with ADX=0 ---")
    for k, v in sorted(no_adx.items()):
        print(f"  {k}: trades={v['n_trades']}, atr_med={v['atr_median']:.1f}, adx_bars={v['n_adx_bars']}")
        # Check total rows in features_snapshot
        feat = os.path.join(v["path"], "features_snapshot.csv")
        with open(feat) as f:
            total_rows = sum(1 for _ in f) - 1
        print(f"    total_feat_rows={total_rows}")
