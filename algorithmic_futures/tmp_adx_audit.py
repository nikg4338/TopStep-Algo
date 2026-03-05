"""ADX Warmup Audit — Extended 60d forward replay sessions.

Analyzes what the allocator V2 WOULD have decided if ADX were available,
by reading the full-session ADX values (post-warmup) from features_snapshot.csv.
"""
import csv
import statistics
from pathlib import Path

RUN_DIR = Path("artifacts/validation_runs/extended_60d_20260303_175155/sessions")


def analyze():
    sessions = sorted(d.name for d in RUN_DIR.iterdir() if d.is_dir())
    print(f"Total sessions: {len(sessions)}")
    print()
    hdr = (f"{'Session':<25} {'1st_ADX_Bar':>12} {'1st_ADX':>8} "
           f"{'Med_ADX':>8} {'Max_ADX':>8} {'V2_Route':>10}")
    print(hdr)
    print("-" * 80)

    orb_count = 0
    trend_sessions = []
    for sid in sessions:
        feat = RUN_DIR / sid / "features_snapshot.csv"
        if not feat.exists():
            continue
        adx_vals: list[float] = []
        first_bar = None
        first_val = None
        with open(feat) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                a = float(row.get("adx", 0))
                if a > 0:
                    adx_vals.append(a)
                    if first_bar is None:
                        first_bar = i
                        first_val = a
        if adx_vals:
            med = statistics.median(adx_vals)
            mx = max(adx_vals)
            # Emulate V2 allocator with first 12 non-zero ADX values
            early12 = adx_vals[:12]
            trend_open = any(v >= 25.0 for v in early12)
            rising = early12[-3:]
            rising_ok = (
                len(rising) >= 3
                and all(v > 20.0 for v in rising)
                and all(rising[j] < rising[j + 1] for j in range(len(rising) - 1))
            )
            would_orb = trend_open or rising_ok
            if would_orb:
                orb_count += 1
                trend_sessions.append(sid)
            route = "ORB" if would_orb else "MR"
            print(f"{sid:<25} {first_bar:>12} {first_val:>8.1f} "
                  f"{med:>8.1f} {mx:>8.1f} {route:>10}")
        else:
            print(f"{sid:<25} {'N/A':>12} {'N/A':>8} "
                  f"{'N/A':>8} {'N/A':>8} {'MR':>10}")

    print()
    print(f"Would-be ORB sessions: {orb_count}/{len(sessions)}")
    if trend_sessions:
        print(f"\nMissed ORB days: {', '.join(trend_sessions)}")


if __name__ == "__main__":
    analyze()
