# TopStep-Algo

## Allocator openfix candidate

This repo keeps `mainline_combine_v1` frozen as the profitable baseline.

The versioned investigation candidate `mainline_combine_v1_1_allocator_openfix`
exists to fix the open-window allocator mismatch: ADX-14 does not populate in
time for the 09:30–09:45 ET route decision, so live-style replay cannot rely on
open-session ADX for MR vs ORB routing. The candidate replaces that invalid
open-window dependency with `open_proxy_v1`, a deterministic price-action proxy
that uses only data available by the end of the opening range.

### Candidate workflow

1. Run side-by-side comparison
	- `python run_allocator_openfix_compare.py --packs pilot_20d extended_60d trend20`
2. Run per-session attribution on a chosen pack
	- `python run_allocator_openfix_attribution.py --baseline-run <run_id> --candidate-run <run_id>`
3. Run robustness checks
	- `python run_allocator_openfix_robustness.py --run-id <candidate_run_id>`
4. Run live-sim integrity audit
	- `python run_allocator_openfix_live_sim.py --run-id <candidate_run_id>`
5. Generate final go / no-go summary
	- `python generate_allocator_openfix_gonogo.py --baseline-run <run_id> --candidate-run <run_id>`
6. Run forward shadow on unseen sessions
	- `python run_allocator_openfix_forward_shadow.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD --exclude-run-id <prior_run_id>`

### Go / no-go semantics

The final report now separates:

- `Engineering verdict`
  - routing, auditability, EOD flatten, no-lookahead evidence, and core risk plumbing
- `Promotion verdict`
  - `PASS`, `HOLD`, or `FAIL`

`HOLD` is the intended state when the candidate is structurally sound but still
falls short of promotion-quality deployment standards, especially the configured
`P(target)` threshold.

### New artifacts

- `allocator_debug.csv`
  - one row per session with allocator route, OR width, ATR, width/ATR,
	 impulse, persistence, confidence proxy, notes, and realized session PnL
- `artifacts/candidate_reports/...`
  - comparison, attribution, robustness, live-sim, and go/no-go reports

## Historical holdout workflow

Historical holdout is separate from forward-shadow. Use it to extend the local
Databento MES cache backward in time and test frozen presets on older untouched
history.

### Inspect cache coverage

- `python manage_databento_cache.py inspect`

This reports earliest cached session, latest cached session, total session
count, and obvious gaps based on the CME Equity calendar.

### Backward-extend the cache

- `python manage_databento_cache.py backfill --start-date 2025-11-03 --end-date 2025-11-28`

This fetches older full-RTH sessions, skips exact existing parquet files,
prints the cache range before and after, and fails loudly if the requested
range overlaps current coverage without `--allow-overlap`.

### Reproducible historical holdout pack

- Pack name: `historical_holdout_20d`
- Window: 2025-11-03 → 2025-11-28

Dry-run the resolved session list:

- `python run_validation_pack.py --pack historical_holdout_20d --dry-run`

Run the frozen candidate unchanged:

- `python run_validation_pack.py --preset mainline_combine_v1_1_allocator_openfix --pack historical_holdout_20d`

Run the frozen baseline on the same holdout:

- `python run_validation_pack.py --preset mainline_combine_v1 --pack historical_holdout_20d`

Compare candidate vs baseline using the existing reporting pipeline:

- `python run_allocator_openfix_compare.py --packs historical_holdout_20d`

This is additional historical out-of-sample evidence only. It is not a
substitute for future forward-shadow evidence.

## ORB autopsy research

This phase is for false-positive ORB analysis only. It reuses older-data
holdouts and recent unseen runs to study why `open_proxy_v1` over-routes to ORB
in some regimes. It does not mutate frozen presets or strategy behavior.

Build an ORB autopsy dataset from existing run artifacts:

- `python run_orb_autopsy_dataset.py --candidate-window recent_forward=forward_shadow_20260223_20260305_20260306_142437 --baseline-window recent_forward=forward_shadow_20260223_20260305_20260306_144150 --candidate-window historical_holdout=historical_holdout_20d_20260306_150624 --baseline-window historical_holdout=historical_holdout_20d_20260306_150145`

Generate the good-vs-bad ORB report:

- `python run_orb_autopsy_report.py --dataset artifacts/candidate_reports/<orb_autopsy_dataset_dir>/orb_autopsy_dataset.csv`

Generate the research handoff memo for the next candidate branch:

- `python generate_orb_selectivity_research_memo.py --dataset artifacts/candidate_reports/<orb_autopsy_dataset_dir>/orb_autopsy_dataset.csv`

Suggested future experiment name:

- `mainline_combine_v1_2_orb_selectivity`

## Pairwise conditional edge analysis

This workflow is descriptive research only. It does not tune strategy logic or
promote a routing rule by itself. It asks whether combinations of two existing
features concentrate expectancy more clearly than one-dimensional buckets.

Run pairwise analysis from an existing validation run:

- `python run_pairwise_edge_analysis.py --run-id extended_60d_20260306_161717`

Optional thresholds:

- `python run_pairwise_edge_analysis.py --run-id extended_60d_20260306_161717 --reporting-min-sample 10 --candidate-min-sample 20`

Interpretation:

- reporting threshold: rows below this trade count are still computed, but they
	should not be treated as summary-grade evidence
- candidate threshold: rows meeting this higher bar are the most useful inputs
	for future routing hypotheses
- outputs remain descriptive; they are not strategy optimization or proof of a
	production-ready rule

## v1.2 ORB selectivity candidate

`mainline_combine_v1_2_orb_selectivity` is a research candidate layered on top
of the frozen `mainline_combine_v1_1_allocator_openfix` allocator fix.

Hypothesis tested:

- sharp early moves in lower-volatility or weak-persistence contexts are being
	over-routed to ORB and should require stronger confirmation.

What changes in v1.2:

- low-ATR caution can require stronger persistence before ORB routing
- very high impulse with weak persistence can be downgraded back to MR

What does not change:

- `mainline_combine_v1_1_allocator_openfix` remains frozen
- MR and ORB engines are unchanged
- this is still a research candidate, not a promoted deployment preset

Core comparison workflow:

- `python run_allocator_openfix_compare.py --baseline-preset mainline_combine_v1_1_allocator_openfix --candidate-preset mainline_combine_v1_2_orb_selectivity --packs extended_60d trend20 historical_holdout_20d`

To compare an existing recent forward window by run id:

- `python run_allocator_openfix_compare.py --baseline-preset mainline_combine_v1_1_allocator_openfix --candidate-preset mainline_combine_v1_2_orb_selectivity --packs recent_forward --baseline-run recent_forward=<v1_1_run_id> --candidate-run recent_forward=<v1_2_run_id>`

## v1.3 ORB selectivity refine candidate

`mainline_combine_v1_3_orb_selectivity_refine` is a research candidate layered
on top of frozen `mainline_combine_v1_2_orb_selectivity` behavior.

Hypothesis tested:

- medium opening-impulse ORB conditions with weak persistence are a negative
	ORB slice and should be suppressed before final routing

How v1.3 differs from v1.2:

- v1.2 keeps the low-ATR and high-impulse selectivity refinement
- v1.3 adds one narrow post-decision filter for medium-impulse + weak-persistence
	ORB conditions
- v1.3 does not add a second medium-impulse heuristic; the experiment stays
	single-reason and audit-friendly

Compare v1.2 vs v1.3 on the core packs:

- `python run_allocator_openfix_compare.py --baseline-preset mainline_combine_v1_2_orb_selectivity --candidate-preset mainline_combine_v1_3_orb_selectivity_refine --packs extended_60d trend20 historical_holdout_20d`

Compare an existing recent-forward window by run id:

- `python run_allocator_openfix_compare.py --baseline-preset mainline_combine_v1_2_orb_selectivity --candidate-preset mainline_combine_v1_3_orb_selectivity_refine --packs recent_forward --baseline-run recent_forward=<v1_2_run_id> --candidate-run recent_forward=<v1_3_run_id>`

This remains descriptive research only. It does not modify frozen v1.1 or v1.2
behavior and is not a production promotion claim.

### Out of scope

- Rewriting MR or ORB engines
- Changing the frozen baseline preset in place
- Introducing opaque or adaptive allocator logic
- Modeling missed-fill / delayed-fill execution unless the simulator gains a
  native execution-latency hook
