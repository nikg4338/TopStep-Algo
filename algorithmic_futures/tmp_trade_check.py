#!/usr/bin/env python3
"""Check trade availability per session across runs."""
import os
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))
val_dir = os.path.join(ROOT, "artifacts/validation_runs")

session_trades = {}  # session_id -> {run_id: n_trades}

for run_id in sorted(os.listdir(val_dir)):
    sess_dir = os.path.join(val_dir, run_id, "sessions")
    if not os.path.isdir(sess_dir):
        continue
    for sess in sorted(os.listdir(sess_dir)):
        sp = os.path.join(sess_dir, sess)
        trades_file = os.path.join(sp, "trade_log.csv")
        if not os.path.isfile(trades_file):
            continue
        with open(trades_file) as f:
            lines = f.readlines()
        n_trades = max(0, len(lines) - 1)
        if sess not in session_trades:
            session_trades[sess] = {}
        session_trades[sess][run_id] = n_trades

# Show sessions starting with "session_" that have trades in ANY run
has_trades = {}
for sess, runs in sorted(session_trades.items()):
    if not sess.startswith("session_"):
        continue
    max_trades = max(runs.values())
    best_run = max(runs, key=runs.get)
    if max_trades > 0:
        has_trades[sess] = {"max_trades": max_trades, "best_run": best_run, "n_runs": len(runs)}

print(f"Total session_* with trade_log: {sum(1 for s in session_trades if s.startswith('session_'))}")
print(f"Sessions with trades > 0: {len(has_trades)}")
print()
for sess, info in sorted(has_trades.items()):
    print(f"  {sess:30s}  trades={info['max_trades']:3d}  best_run={info['best_run']}")

# Now show which have 0 trades everywhere
no_trades = []
for sess, runs in sorted(session_trades.items()):
    if not sess.startswith("session_"):
        continue
    if max(runs.values()) == 0:
        no_trades.append(sess)

print(f"\nSessions with 0 trades in all runs: {len(no_trades)}")
for s in no_trades:
    print(f"  {s}")
