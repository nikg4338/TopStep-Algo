"""Inventory all available sessions and classify by ADX regime.

Scans all validation run directories to find unique sessions,
extracts early-session ADX, and reports the distribution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import pandas as pd
import numpy as np
from collections import defaultdict

ARTIFACTS = Path("artifacts/validation_runs")


def get_session_adx(session_dir: Path, max_bars: int = 12) -> list[float]:
    """Return early-session ADX values."""
    features_path = session_dir / "features_snapshot.csv"
    if not features_path.is_file():
        return []
    try:
        df = pd.read_csv(features_path, nrows=max_bars)
        if "adx" in df.columns:
            return df["adx"].dropna().astype(float).tolist()
    except Exception:
        pass
    return []


def get_session_atr(session_dir: Path, max_bars: int = 12) -> float:
    features_path = session_dir / "features_snapshot.csv"
    if not features_path.is_file():
        return 0.0
    try:
        df = pd.read_csv(features_path, nrows=max_bars)
        if "atr" in df.columns:
            vals = df["atr"].dropna()
            if len(vals) > 0:
                return float(vals.median())
    except Exception:
        pass
    return 0.0


def get_session_regime(session_dir: Path) -> str:
    summary_path = session_dir / "session_summary.json"
    if not summary_path.is_file():
        return "unknown"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        regime_dist = summary.get("regime_distribution", {})
        if regime_dist:
            top = max(regime_dist, key=regime_dist.get)
            lower = top.lower()
            if "range" in lower:
                return "range"
            elif "trend" in lower:
                return "trend"
            elif "chop" in lower:
                return "chop"
    except Exception:
        pass
    return "unknown"


def get_session_trade_count(session_dir: Path) -> int:
    trades_csv = session_dir / "trades.csv"
    if not trades_csv.is_file():
        return 0
    try:
        df = pd.read_csv(trades_csv)
        return len(df)
    except Exception:
        return 0


def get_session_pnl(session_dir: Path) -> float:
    trades_csv = session_dir / "trades.csv"
    if not trades_csv.is_file():
        return 0.0
    try:
        df = pd.read_csv(trades_csv)
        if "pnl_dollars" in df.columns:
            return float(df["pnl_dollars"].sum())
    except Exception:
        pass
    return 0.0


# Scan all run directories
run_dirs = sorted(ARTIFACTS.iterdir())
session_data = {}  # session_id -> best info
session_sources = defaultdict(set)  # session_id -> set of source runs

for run_dir in run_dirs:
    if not run_dir.is_dir():
        continue
    sessions_dir = run_dir / "sessions"
    if not sessions_dir.is_dir():
        continue
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        continue

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        continue

    successful_sids = {
        s["session_id"] for s in manifest.get("sessions", [])
        if s.get("success", True)
    }

    for session_dir in sorted(sessions_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        sid = session_dir.name
        if sid not in successful_sids:
            continue
        if not (session_dir / "trades.csv").is_file():
            continue

        session_sources[sid].add(run_dir.name)

        # Only process if we haven't seen this session yet
        if sid in session_data:
            continue

        adx_vals = get_session_adx(session_dir)
        adx_mean = float(np.mean(adx_vals)) if adx_vals else 0.0
        adx_open = adx_vals[0] if adx_vals else 0.0
        atr_med = get_session_atr(session_dir)
        regime = get_session_regime(session_dir)
        trade_count = get_session_trade_count(session_dir)
        pnl = get_session_pnl(session_dir)

        session_data[sid] = {
            "session_id": sid,
            "adx_open": round(adx_open, 2),
            "adx_mean": round(adx_mean, 2),
            "atr_median": round(atr_med, 2),
            "regime": regime,
            "trade_count": trade_count,
            "pnl_dollars": round(pnl, 2),
            "source_run": sorted(session_sources[sid])[0],
            "session_dir": str(session_dir),
        }

# Build DataFrame
df = pd.DataFrame(session_data.values())
df = df.sort_values("session_id")

print(f"\n{'='*80}")
print(f"  SESSION INVENTORY")
print(f"{'='*80}")
print(f"  Unique sessions: {len(df)}")
print(f"  Sessions with trades: {(df.trade_count > 0).sum()}")
print(f"  Total trades: {df.trade_count.sum()}")
print(f"  Total PnL: ${df.pnl_dollars.sum():.2f}")

print(f"\n  --- ADX Distribution ---")
print(f"  ADX(open) mean={df.adx_open.mean():.1f}  med={df.adx_open.median():.1f}  "
      f"p25={df.adx_open.quantile(0.25):.1f}  p75={df.adx_open.quantile(0.75):.1f}")
print(f"  ADX(mean) mean={df.adx_mean.mean():.1f}  med={df.adx_mean.median():.1f}  "
      f"p25={df.adx_mean.quantile(0.25):.1f}  p75={df.adx_mean.quantile(0.75):.1f}")

print(f"\n  --- ATR Distribution ---")
print(f"  ATR(med) mean={df.atr_median.mean():.1f}  med={df.atr_median.median():.1f}  "
      f"p25={df.atr_median.quantile(0.25):.1f}  p75={df.atr_median.quantile(0.75):.1f}")

print(f"\n  --- Regime Distribution ---")
for regime, cnt in df.regime.value_counts().items():
    pnl = df.loc[df.regime == regime, "pnl_dollars"].sum()
    trades = df.loc[df.regime == regime, "trade_count"].sum()
    print(f"  {regime:>10}: {cnt:3d} sessions, {trades:4d} trades, PnL=${pnl:8.2f}")

# Classify by ADX: range-biased (ADX < 20), trend-biased (ADX >= 25), mixed (20-25)
df["adx_class"] = pd.cut(df.adx_mean, bins=[0, 20, 25, 100], labels=["range", "mixed", "trend"])
print(f"\n  --- ADX Classification (range <20, mixed 20-25, trend >25) ---")
for cls, cnt in df.adx_class.value_counts().sort_index().items():
    pnl = df.loc[df.adx_class == cls, "pnl_dollars"].sum()
    trades = df.loc[df.adx_class == cls, "trade_count"].sum()
    print(f"  {cls:>10}: {cnt:3d} sessions, {trades:4d} trades, PnL=${pnl:8.2f}")

# How many unique sessions do we have?
print(f"\n  --- Source run coverage ---")
for run_id in sorted(set(s for sources in session_sources.values() for s in sources)):
    sids = [sid for sid, sources in session_sources.items() if run_id in sources]
    print(f"  {run_id}: {len(sids)} sessions")

# Save inventory CSV
out_path = Path("/tmp/session_inventory.csv")
df.to_csv(out_path, index=False)
print(f"\n  Inventory saved → {out_path}")

# Check extended_60d pack
print(f"\n  --- Extended 60d pack ---")
from validation.validation_pack import load_pack
ext = load_pack("extended_60d")
ext_sids = {s.session_id for s in ext.sessions}
print(f"  Extended pack sessions: {len(ext_sids)}")
in_inventory = ext_sids & set(df.session_id)
print(f"  Overlapping with inventory: {len(in_inventory)}")
missing = ext_sids - set(df.session_id)
print(f"  In extended but not in inventory: {len(missing)}")
if missing:
    print(f"    Examples: {sorted(missing)[:5]}")
