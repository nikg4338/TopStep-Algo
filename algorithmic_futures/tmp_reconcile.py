"""Reconcile old vs new 100-draw results by comparing pnl_r vs pnl_dollars."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

base_runs = [
    "pilot_20d_20260227_005250",
    "random20_01_20260227_005957",
    "random20_02_20260227_010608",
    "random20_03_20260227_011226",
    "trend20_adx_20260226_232222",
]

all_trades = []
for run_id in base_runs:
    agg_path = Path("artifacts/validation_runs") / run_id / "aggregate_trades.csv"
    if agg_path.is_file():
        df = pd.read_csv(agg_path)
        df["source_run"] = run_id
        all_trades.append(df)
        print(f"  {run_id}: {len(df)} trades")

df = pd.concat(all_trades, ignore_index=True)
print(f"\nTotal pool: {len(df)} trades")
print(f"Unique sessions: {df['session_id'].nunique()}")

print(f"\npnl_r:       mean={df.pnl_r.mean():.4f}  std={df.pnl_r.std():.4f}  sum={df.pnl_r.sum():.2f}")
print(f"pnl_dollars: mean={df.pnl_dollars.mean():.2f}  std={df.pnl_dollars.std():.2f}  sum={df.pnl_dollars.sum():.2f}")

# Implied risk_per_trade = pnl_dollars / pnl_r
mask = df.pnl_r.abs() > 0.01
ratios = df.loc[mask, "pnl_dollars"] / df.loc[mask, "pnl_r"]
print(f"\nImplied RISK_PER_TRADE (pnl_dollars / pnl_r):")
print(f"  mean={ratios.mean():.2f}  median={ratios.median():.2f}  std={ratios.std():.2f}")
print(f"  min={ratios.min():.2f}  max={ratios.max():.2f}")

# Compare MC views
print(f"\n--- MC view comparison ---")
# Old MC: used pnl_r with use_dollar_values=False → MC converts: pnl_r * RISK_PER_TRADE(=20)
mc_old_pnl = df.pnl_r * 20.0
print(f"Old MC (pnl_r × $20):  per-trade mean=${mc_old_pnl.mean():.2f}  total=${mc_old_pnl.sum():.2f}")
print(f"New MC (pnl_dollars):  per-trade mean=${df.pnl_dollars.mean():.2f}  total=${df.pnl_dollars.sum():.2f}")

if df.pnl_dollars.sum() != 0:
    ratio = mc_old_pnl.sum() / df.pnl_dollars.sum()
    print(f"Ratio (old/new): {ratio:.2f}x")

# What per-trade edge does each view see?
print(f"\n--- Per-draw (20 sessions, ~32 trades) expected edge ---")
avg_trades = 32
old_edge = mc_old_pnl.mean() * avg_trades
new_edge = df.pnl_dollars.mean() * avg_trades
print(f"Old MC: ${old_edge:.2f} per 20-session draw")
print(f"New MC: ${new_edge:.2f} per 20-session draw")

# Check how old test actually scored — did it use pnl_r or pnl_dollars?
# Read a per-draw entry from old robustness
import json
old_results = json.loads(Path("artifacts/validation_runs/robustness_100draw_20260301_045454.json").read_text())
draw0 = [d for d in old_results["draws"] if d["draw_idx"] == 0]
for d in draw0:
    print(f"\nOld draw 0 @ {d['contracts']}c: P_hit={d['p_hit']:.4f}  P_ruin={d['p_ruin']:.4f}  "
          f"dd_p95=${d['dd_p95']:.0f}  eq_p50=${d['equity_p50']:.0f}  trades={d['trade_count']}")
