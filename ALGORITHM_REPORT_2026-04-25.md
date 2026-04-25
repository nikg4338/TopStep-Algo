# Algorithm Report: Topstep Algo Current State

Date: 2026-04-25  
Workspace: `/Users/nikghaderi/Desktop/TopStep Algo`  
Primary account assumption: Topstep 50K Trading Combine, MES

## Executive Summary

The current system is technically healthy but not challenge-ready. The test suite passes, the validation framework is strong, and the algorithm has a serious research pipeline around replay, Monte Carlo, stress testing, sizing, and allocation. The limiting factor is not plumbing. The limiting factor is the trade distribution: the current configurations either preserve the account but do not earn fast enough, or reach for target probability while creating unacceptable trailing drawdown and ruin risk.

The most recent strong safety variant, `route_sensitivity_16_20260411_135051`, produced positive expectancy, low drawdown, and zero base-case simulated ruin, but only a 2.23% probability of hitting the $3,000 target within the 20-day horizon. The more aggressive recent pilot, `pilot_20d_20260411_132123`, produced a 17.89% target probability, but with 66.90% simulated ruin and 95th percentile drawdown above the $2,000 Maximum Loss Limit.

That means the algorithm is currently stuck between two bad modes:

1. Too conservative to pass within the challenge horizon.
2. Too volatile to survive Topstep's trailing Maximum Loss Limit.

Markov chains could help only as a regime-routing layer. A plain Markov chain will not solve the core expectancy and payoff-shape problem. A properly walk-forward Hidden Markov Model or other probabilistic state model may improve routing between mean reversion, ORB, and no-trade states, but it cannot manufacture edge from trades whose payoff distribution is too thin.

## Current Topstep Constraints

Topstep's current materials describe the Trading Combine as having one hard rule: do not let the account balance hit or fall below the Maximum Loss Limit. The objectives are to reach and maintain the profit target and meet the consistency target. For the 50K Combine, the Maximum Loss Limit is $2,000, and it trails the highest end-of-day balance, then locks once it reaches the starting balance. Importantly, Topstep states the MLL is monitored in real time and includes realized and unrealized P&L.

For the Trading Combine consistency target, Topstep says the single best day must stay at or below 50% of the Profit Target. For a 50K account, the recommended best day is less than $1,500. If the best day exceeds the threshold, the profit target increases.

Topstep also states automated strategies are permitted, but they will not troubleshoot or make exceptions for errant automated trades. Maximum position size for the 50K Trading Combine is 5 full contracts or 50 micros.

Daily loss behavior has changed on TopstepX. The Daily Loss Limit can be a platform risk setting rather than the single disqualifying Combine rule; the account boundary remains the Maximum Loss Limit if no daily loss safety net is enabled. Live Funded Accounts still have automatic daily loss limits. Practically, the algorithm should keep its own internal daily loss stop regardless of the platform default.

Sources are listed at the end of this report.

## Current Architecture

The repo contains two overlapping versions of the strategy stack.

The intended architecture in the docs is:

- MES-only Topstep Combine engine.
- VWAP mean reversion in range regimes.
- ORB breakout or pullback in directional regimes.
- HMM-style regime classifier.
- Monte Carlo validation and promotion gates.
- Internal daily loss/profit circuit breakers.

The active research architecture is:

- `MRSignalEngine` for VWAP mean-reversion signal formation.
- `MRExitSimulator` for deterministic offline exits and trade logs.
- `HybridThresholdRegimeClassifier` and `open_proxy_v1` allocator variants for routing.
- Validation packs, scorecards, promotion gates, route sensitivity, sizing policy experiments, and Monte Carlo day-horizon testing.

The live/session-manager architecture is:

- Nightly `RegimeClassifier` HMM predicts one regime for the next session.
- `SessionManager` dispatches bars to legacy `VWAPMeanReversion` when the regime is `BALANCED`, or legacy `ORBBreakout` when `DIRECTIONAL`.
- `OrderManager`, `CircuitBreakers`, and `PositionSizer` enforce pre-trade checks and live order flow.

This live/research mismatch is one of the main implementation risks. The research path has the newest MR candidate logic, open-proxy routing, pullback ORB experiments, sizing experiments, and diagnostics. The live path still routes through the older VWAP/ORB classes and a day-level HMM regime decision. In other words, the latest validated research behavior is not cleanly identical to the live execution behavior.

## Latest Validation Read

### Latest Safer Variant

Artifact: `algorithmic_futures/artifacts/validation_runs/route_sensitivity_16_20260411_135051`

- Sessions: 16
- Trades: 37
- Sessions with trades: 14
- Trades/session mean: 2.64
- Win rate: 45.95%
- Avg R: +0.0360
- Avg win: +0.8723R
- Avg loss: -0.6747R
- Avg trade P&L in MC: +$17.06
- Base target probability: 2.23%
- Base ruin probability: 0.00%
- DD p95: $1,030.63
- Losing streak p95: 14
- Stress mild target probability: about 0.47%
- Stress severe target probability: effectively 0%
- Stress severe ruin: 17.60%
- Promotion gate: FAIL

Interpretation: this is a capital-preserving configuration, not a pass configuration. It protects the account but does not generate enough profit velocity to reach $3,000 in the target horizon.

### Recent Pilot Variant

Artifact: `algorithmic_futures/artifacts/validation_runs/pilot_20d_20260411_132123`

- Sessions: 20
- Trades: 36
- Sessions with trades: 7
- Win rate: 44.44%
- Avg R: -0.0216
- Avg win: +0.8138R
- Avg loss: -0.6899R
- Avg trade P&L in MC: -$3.77
- Base target probability: 17.89%
- Base ruin probability: 66.90%
- DD p95: $2,174.00
- Losing streak p95: 13
- Mild stress target probability: 3.98%
- Severe stress target probability: 0.03%
- Tilt bad-week ruin: 99.66%
- Promotion gate: FAIL

Interpretation: this configuration has more target reach, but the path dependency is unacceptable. It breaches the survival premise of the strategy.

### Older MR Best-So-Far Context

The project notes identify an earlier mean-reversion variant around:

- Sigma: 1.4
- No time stop
- No runner
- Thesis-break plus hard stop
- Trades/day: about 7.25
- Win rate: about 50%
- Avg win: about +0.86R
- Avg loss: about -0.69R
- Expectancy: about +0.0919R
- DD p95: about $717

But subsequent 20-day ablations and current artifacts show that candidate formation changed materially. Current A0-style ablation was far weaker: 2.79 trades/day, 41.5% win rate, and -0.0504R avg R. The current trouble note correctly identifies upstream candidate formation, especially reclaim-only formation, as the immediate bottleneck.

## What Is Limiting The Algorithm From Passing?

### 1. The Edge Is Too Thin Relative To The Profit Target

The 50K Combine is not really a $50,000 risk problem. It is a $2,000 Maximum Loss Limit problem with a $3,000 profit objective. The current safe configuration averages about $17 per trade in Monte Carlo. At that rate, the expected trades-to-target is around 176 trades, while the validation horizon and actual trade frequency are far lower.

The current safe variant survives, but it does not climb fast enough.

### 2. The Aggressive Variants Consume Too Much Drawdown

The recent pilot reaches the target more often than the safety variant, but its simulated ruin probability is 66.90% and DD p95 is $2,174, above the $2,000 MLL. Because Topstep monitors realized and unrealized P&L in real time, a strategy whose p95 path touches or exceeds the MLL is not practically passable even if some simulations hit the target.

### 3. MR Payoff Distribution Has A Thin Right Tail

The project notes already found that VWAP MR profits are capped near VWAP:

- Beyond VWAP +0.8R: 2.76%
- Beyond VWAP +1.2R: 1.38%
- Beyond VWAP +1.4R: 0.00%

That is the core reason runner exits failed. Mean reversion to VWAP can produce a modest positive edge, but it does not naturally create the large positive tail needed to reach a $3,000 target quickly while keeping daily losses small.

### 4. Candidate Formation Is The Current Upstream Bottleneck

The newest ablation note says A0 itself collapsed because the candidate pool was only 56 across the pack. The tested quality filters removed only 7-10 candidates each, so the filters were not the main problem. Reclaim-only candidate formation appears to be suppressing too much flow and/or selecting a worse subset than the historical best.

The repo already supports `reclaim_mode` values of `on`, `off`, `soft`, and `touch`. That should be the next focused experimental axis, not more broad filter stacking.

### 5. Regime Routing Is Not Yet Stable Enough

There are three regime/routing ideas in the codebase:

- Nightly HMM in `RegimeClassifier`.
- Intraday hybrid threshold classifier in `regime_v1`.
- Open-proxy price-action allocator in `open_proxy_allocator`.

The preset notes explicitly say an ADX-14 warmup bug prevented ORB routing in live replay because early ADX is structurally unavailable during the allocator decision window. Open-proxy routing was created to address this. That is good progress, but it also means the architecture is still in transition.

Current routing has not yet proven it can consistently put MR only in mean-reverting sessions and ORB only in continuation sessions.

### 6. Research And Live Execution Are Not The Same System

The current research stack validates `MRSignalEngine`, route sensitivity, open-proxy allocation, and dynamic sizing. The live `SessionManager` still instantiates legacy `VWAPMeanReversion` and `ORBBreakout`.

Specific mismatch examples:

- `MRSignalEngine` uses sigma 1.4 candidate logic and has reclaim-mode experiments.
- Legacy `VWAPMeanReversion` still references 2.5-3.0 sigma style logic and CVD divergence.
- `CVDProxyFilter` in the research MR engine is still a stub/pass-through.
- Config says ORB trigger mode can be `pullback_v3`, but the live `ORBBreakout` class shown in the repo uses close-outside-range breakout logic, not the full pullback-v3 research logic.

This matters because even a passing research configuration would not be safe to deploy until the live execution path is brought into parity with the validated path.

### 7. Position Sizing Can Exceed The Intended Risk In Wide-Stop Conditions

`PositionSizer.calculate()` always returns at least 1 MES contract. If the stop distance is wide enough, one MES contract can exceed the intended $20-$40 risk budget. Example: with MES at $5/point, an 8-point stop is already $40 for one contract. A 20-point stop is $100 for one contract before slippage.

This is unavoidable if the system insists on always trading at least one micro. The mitigation is not fractional sizing; it is a hard "no trade if minimum contract risk exceeds budget" rule. The current code does not appear to enforce that.

### 8. Consistency Logic Is Incomplete For Real Passing

The repo has a stronger `ConsistencyCapEngine`, but the live `CircuitBreakers` use a simpler pre-trade check based on current daily and cumulative P&L. It does not project whether the next trade could push the best day over the cap. It can also behave awkwardly early in the Combine because cumulative P&L includes today's P&L.

For Topstep, consistency is not just a daily halt; it is a pass-state condition. The system should maintain a pass calculator that answers: "If we stop now, do we pass?" and "How much can we make today without increasing the required total profit?"

## Would Markov Chains Improve The Algorithm?

Short answer: not by themselves.

A simple Markov chain can model transition probabilities between observed states like range, trend, chop, and stress. That could improve regime routing if the current state labels are high quality. But a plain Markov chain will not fix the core problem: the validated trade distribution is either too small to hit target or too volatile to survive.

A Hidden Markov Model is more appropriate than a simple Markov chain because market regimes are latent. The repo already contains an HMM classifier, and the strategy paper correctly describes why walk-forward training matters. The issue is not "should we add Markov logic?" The issue is "can we prove a regime model improves the joint distribution of trades after costs, slippage, and Topstep constraints?"

Recommended use:

- Use Markov/HMM only for session routing and no-trade filtering.
- Train and evaluate walk-forward only.
- Measure improvement by target probability, ruin probability, DD p95, losing-streak p95, and trade-count sufficiency.
- Do not use it to predict direction trade-by-trade.
- Do not promote it unless it improves out-of-sample trade distribution by regime and survives stress cases.

The best probabilistic-regime upgrade would be a "probability-weighted allocator":

- If P(range) is high and volatility is normal, allow MR.
- If P(trend) is high and opening impulse confirms, allow ORB.
- If P(stress/chop) is high, stay flat.
- If probabilities are mixed, reduce size or require stronger trade evidence.

This could reduce bad trades and improve routing, but it will not solve the need for a second return source with more convexity.

## Recommended Improvement Plan

### Priority 1: Reconcile Research And Live Paths

Make the live path execute the same strategy logic that is validated offline.

Actions:

- Promote `MRSignalEngine` into the live route or create a shared signal adapter used by both replay and live.
- Ensure the live ORB implementation matches the validated ORB variant, especially pullback-v3 if that remains the chosen trend engine.
- Keep one source of truth for strategy parameters, preferably preset-driven.
- Add a parity test: same bars + same config must generate the same signals in replay and live-style dispatch.

### Priority 2: Add A Minimum-Risk Gate

Before placing any trade, compute the one-contract stop risk. If one MES contract exceeds the allowed risk budget or current drawdown headroom, reject the trade.

Suggested rule:

- `min_contract_risk = stop_distance_points * 5`
- Reject if `min_contract_risk > max_allowed_trade_risk`
- Reject if `min_contract_risk > remaining_daily_loss_budget * safety_fraction`
- Reject if `min_contract_risk > remaining_mll_headroom * safety_fraction`

This will reduce frequency, but it prevents wide-stop trades from silently violating the survival premise.

### Priority 3: Reclaim-Mode Ablation

Run the clean comparison already implied by the current trouble note:

- `reclaim_mode=on`
- `reclaim_mode=off`
- `reclaim_mode=soft`
- `reclaim_mode=touch`

Keep every other variable fixed. Report candidate pool, approved trades, win rate, avg R, P(target), P(ruin), DD p95, losing-streak p95, and trade frequency.

The goal is not maximum approval. The goal is restoring enough candidate flow without importing trend-contaminated losses.

### Priority 4: Stop Trying To Force MR Into A Passing Strategy Alone

MR should be treated as a low-dispersion base engine, not the entire passing engine. The current evidence says MR's right tail is too thin. Keep improving it, but do not expect exit shaping or runners to turn it into a $3,000-in-20-days machine.

The passing architecture probably needs:

- MR for controlled range-day income.
- Selective ORB/pullback trend engine for convexity.
- Strong no-trade state for chop/stress.
- Dynamic but conservative sizing that only increases after earned cushion.

### Priority 5: Improve ORB Selectivity Instead Of ORB Frequency

The ORB research memo found only 14 rows, with 4 good ORB, 1 bad ORB, and 9 neutral. It identified opening-range width, ATR, and one-sidedness as candidate discriminators.

Next ORB experiment:

- Require a continuation-quality discriminator, not just any breakout.
- Penalize high ATR contexts where raw volatility overwhelms selectivity.
- Use more labeled sessions before trusting the filter.

### Priority 6: Upgrade Consistency And Pass-State Accounting

Add a Combine-aware pass calculator:

- Current cumulative profit.
- Current best day.
- Today's realized P&L.
- Required total profit under Topstep consistency.
- Remaining safe profit today before target inflation.
- Whether stopping now passes.

Then wire this into live trading so the algorithm can stop itself when the account is in a passing state.

### Priority 7: Use Markov/HMM As A Risk Filter, Not A Magic Edge Engine

Recommended experiment:

- Build daily/session labels from actual historical performance: MR-good, ORB-good, no-trade/stress.
- Train a walk-forward HMM or Markov transition model on non-leaky features.
- Compare against open-proxy routing and hybrid threshold routing.
- Promotion criterion: improved base target probability without increasing DD p95 or stress ruin.

If it only improves classification accuracy but not Topstep survival metrics, do not promote it.

## Practical Go/No-Go

Current go/no-go status: NO-GO for live Topstep challenge attempt.

Reason:

- The safest current variant has only 2.23% target probability in the latest route sensitivity artifact.
- The more aggressive recent pilot has 66.90% ruin probability and DD p95 above the MLL.
- Stress cases are still poor.
- Trade counts are too low for high-confidence Monte Carlo conclusions.
- Research and live execution behavior are not yet unified.

Minimum promotion target before attempting a Combine:

- Base P(target before ruin): at least 60%.
- Base P(ruin): below 15%.
- DD p95: below $1,200.
- Losing-streak p95: below 8, or sizing reduced enough to make that streak survivable.
- Stress mild P(target): at least 50%.
- Severe stress P(target): at least 30%, or explicitly accepted as an advisory fail with much lower ruin.
- At least 200 trades in the validation input before trusting Monte Carlo.
- Live/replay signal parity demonstrated.

## Verification Performed

Command:

```bash
/Users/nikghaderi/Desktop/TopStep\ Algo/.venv/bin/python -m pytest -q
```

Result:

```text
292 passed in 9.58s
```

## Sources

- Topstep Help Center, Trading Combine Parameters: https://help.topstep.com/en/articles/8284197-trading-combine-parameters
- Topstep Help Center, Maximum Loss Limit: https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit
- Topstep Help Center, Consistency at Topstep: https://help.topstep.com/en/articles/8284208-consistency-at-topstep
- Topstep Help Center, Daily Loss Limit in the Trading Combine and Express Funded Account: https://help.topstep.com/en/articles/10490293-daily-loss-limit-in-the-trading-combine-and-express-funded-account
- Topstep Help Center, Express Funded Account Parameters: https://help.topstep.com/en/articles/8284215-express-funded-account-parameters

## Internal Evidence Reviewed

- `/Users/nikghaderi/Desktop/TopStep Algo/ContextFiles/Current Troubles.txt`
- `/Users/nikghaderi/Desktop/TopStep Algo/ContextFiles/StrategyPaper.txt`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/config.py`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/strategies/mr_signal_engine.py`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/strategies/vwap_mean_reversion.py`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/strategies/orb_breakout.py`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/session_manager.py`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/risk/position_sizer.py`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/execution/circuit_breakers.py`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/validation/sizing_policy.py`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/validation/open_proxy_allocator.py`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/artifacts/validation_runs/route_sensitivity_16_20260411_135051`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/artifacts/validation_runs/pilot_20d_20260411_132123`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/artifacts/candidate_reports/mr_candidate_formation_sweep_pilot_20d_20260404_130941`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/artifacts/candidate_reports/mr_edge_iteration_pilot_20d_20260411_131854`
- `/Users/nikghaderi/Desktop/TopStep Algo/algorithmic_futures/artifacts/candidate_reports/orb_selectivity_research_memo_20260306_170255`
