"""
config.py — Central parameters file for Algorithmic Futures.

All hardcoded risk parameters, session timing, strategy thresholds,
and HMM settings live here. No magic numbers elsewhere in the codebase.

Timezone convention: ALL times stored as US/Eastern (ET).
Broker/Topstep CT rules are converted to ET at the edges.
Active preset: mainline_combine_v1 (frozen 2026-03-02)
See presets/mainline_combine_v1.json for full specification."""

from __future__ import annotations

# ── Provider Selection ─────────────────────────────────────────────────
# DATA_PROVIDER: source for historical/replay market data
# EXECUTION_PROVIDER: source for order routing/account operations
DATA_PROVIDER: str = "databento"
EXECUTION_PROVIDER: str = "projectx"
REQUIRE_BROKER_ENV_FOR_READINESS: bool = False

# Databento defaults (CME Globex)
DATABENTO_DATASET: str = "GLBX.MDP3"
DATABENTO_STYPE_IN: str = "continuous"
DATABENTO_SYMBOL: str = "MES.c.0"

# open_proxy_v1 selectivity refinement defaults (research-only; disabled by default)
ALLOC_OPENPROXY_SELECTIVITY_ENABLED: bool = False
ALLOC_OPENPROXY_LOW_ATR_THRESHOLD: float = 10.0
ALLOC_OPENPROXY_MIN_PERSISTENCE_IN_LOW_ATR: int = 2
ALLOC_OPENPROXY_HIGH_IMPULSE_THRESHOLD: float = 2.4
ALLOC_OPENPROXY_MIN_PERSISTENCE_WHEN_HIGH_IMPULSE: int = 1
ALLOC_OPENPROXY_MEDIUM_IMPULSE_WEAK_PERSISTENCE_FILTER_ENABLED: bool = False

# ── Execution Mode ──────────────────────────────────────────────────────
# "projectx_native"  — use ProjectX OCO / native CVD when available
# "client_fallback"  — client-side bracket emulation + proxy CVD
EXECUTION_MODE: str = "client_fallback"

# ── Account Constraints ─────────────────────────────────────────────────
NOMINAL_ACCOUNT_SIZE: int = 50_000       # Ignore for sizing decisions
MAX_LOSS_LIMIT: int       = 2_000        # True risk capital (MLL)
PROFIT_TARGET: int        = 3_000        # Upper boundary for challenge

# ── Account Mode ────────────────────────────────────────────────────────
# "combine"  → 50 % consistency cap
# "express_funded" → 40 % consistency cap
ACCOUNT_MODE: str = "combine"

# ── Daily Circuit Breakers ──────────────────────────────────────────────
DAILY_LOSS_LIMIT_EXTERNAL: int = 1_000   # Topstep hard daily loss
DAILY_LOSS_LIMIT_INTERNAL: int = 240     # Our internal halt (5-6 losses)
DAILY_PROFIT_HALT: int         = 1_200   # Consistency protection cap
MLL_PROXIMITY_BUFFER: int      = 400     # Reduce risk when within $400 of MLL

# Consistency caps by mode
CONSISTENCY_CAP: dict[str, float] = {
    "combine": 0.50,
    "express_funded": 0.40,
}

# ── Position Sizing (Fixed Fractional) ──────────────────────────────────
RISK_PER_TRADE_MIN: int = 20             # 1 % of MLL
RISK_PER_TRADE_MAX: int = 40             # 2 % of MLL
RISK_PER_TRADE: int     = 20             # Default starting risk

# ── Instrument ──────────────────────────────────────────────────────────
INSTRUMENT: str     = "MES"
TICK_SIZE: float    = 0.25               # MES points per tick
TICK_VALUE: float   = 1.25               # USD per tick for MES
POINT_VALUE: float  = 5.00               # USD per point for MES (4 ticks)

# ── Session Timing (ALL in ET) ──────────────────────────────────────────
# CT → ET conversion: add 1 hour (CT + 1h = ET)
TIMEZONE: str               = "US/Eastern"
SESSION_OPEN: str            = "09:30"    # Pre-market data collection start (was 08:30 CT)
RTH_OPEN: str                = "09:30"    # Regular trading hours open
ORB_END: str                 = "09:45"    # End of 15-minute opening range
LAST_ENTRY_CUTOFF: str       = "15:50"    # No new entries within 15 min of close (16:05 ET)
EOD_CLOSE: str               = "16:05"    # Hard position close deadline (3:05 PM CT = 4:05 PM ET)
NIGHTLY_REFIT_TIME: str      = "18:30"    # HMM refit (17:30 CT = 18:30 ET)

# ── Order Execution ─────────────────────────────────────────────────────
ORDER_FILL_TIMEOUT_SEC: int  = 30         # Cancel unfilled order after 30s
WS_RECONNECT_TIMEOUT_SEC: int = 60       # Flatten if WS down > 60s
WS_RECONNECT_INTERVAL_SEC: int = 5       # Retry connect every 5s
API_MAX_RETRIES: int          = 3         # Exponential backoff retries
API_BACKOFF_BASE_SEC: float   = 1.0       # Base delay for backoff

# ── VWAP Strategy (State 0 — Balanced) ──────────────────────────────────
# NOTE: These VWAP_SD_* thresholds are for the legacy VWAPMeanReversion
# strategy ONLY.  The MR pipeline (MRSignalEngine + MRExitSimulator)
# uses MR_SIGMA_ENTRY / MR_SIGMA_EXTREME below — NOT these values.
VWAP_SD_ENTRY_MIN: float     = 2.5       # Min SD band for entry (legacy VWAP strategy)
VWAP_SD_ENTRY_MAX: float     = 3.0       # Max SD band for entry (legacy VWAP strategy)
VWAP_STOP_ATR_MULT: float   = 1.5       # Stop = 1.5× 5-min ATR beyond entry
VWAP_TRADES_PER_DAY: int    = 3          # Max trade attempts per session
VWAP_BAR_INTERVAL_MIN: int  = 5          # Strategy eval bar size

# ── ORB Strategy (State 1 — Directional) ────────────────────────────────
ORB_MINUTES: int             = 15        # Opening range window
ORB_RR_RATIO: float          = 1.5       # Risk:Reward target (1:1.5)
ORB_TRADES_PER_DAY: int     = 2          # Max trade attempts per session
ORB_STALE_CUTOFF: str       = "11:00"    # Reject ORB signals after this (ET)
ORB_TRIGGER_MODE: str       = "pullback_v3"  # "break" | "pullback" | "either" | "pullback_v3"  [mainline_combine_v1]
ORB_PULLBACK_CONFIRM_BARS: int = 3        # Bars to wait for pullback confirmation after break

# ── ORB Pullback v3 (empirical trend-day entry) ────────────────────────
# breakout → pullback to OR level → continuation entry
ORB_PULLBACK_V3_MAX_BARS: int       = 3     # Max bars after breakout to wait for pullback  [mainline: tol=5, bars=3]
ORB_PULLBACK_V3_TOLERANCE_PTS: float = 5.0  # Points from OR level for pullback detection  [mainline: calibrated best-cell]
ORB_PULLBACK_V3_ENTRY_MODE: str     = "touch_only"  # "touch_only" | "touch_recovery"  [mainline_combine_v1]

# ── HMM Regime Classifier ──────────────────────────────────────────────
HMM_N_STATES: int            = 3         # 0=Balanced, 1=Directional, 2=Crisis
HMM_MIN_LOOKBACK_DAYS: int  = 504        # ~2 years minimum training data
HMM_COVARIANCE_TYPE: str     = "full"
HMM_N_ITER: int              = 200       # EM algorithm iterations
HMM_TRAINING_MODE: str       = "expanding"  # "expanding" walk-forward only

# ── Monte Carlo ─────────────────────────────────────────────────────────
MC_SIMULATIONS: int          = 10_000     # Simulation runs per validation
MC_MAX_TRADES: int           = 800        # Max trades per simulated run (must exceed PROFIT_TARGET / (avg_r * RISK_PER_TRADE))
MC_RUIN_THRESHOLD: float     = 0.15       # Reject if ruin prob > 15 %
MC_TARGET_THRESHOLD: float   = 0.60       # Reject if target prob < 60 %
MC_DRAWDOWN_P95_MAX: float   = 1_200.0    # Reject if 95th pct DD > $1,200
MC_LOSING_STREAK_P95_MAX: int = 8         # Reject if 95th pct streak ≥ 8

# ── Candidate Promotion Governance ─────────────────────────────────────
# Separate from engineering integrity: a candidate may be structurally sound
# yet still held back from combine deployment if target probability is weak.
CANDIDATE_PROMOTION_TARGET_THRESHOLD: float = MC_TARGET_THRESHOLD

# ── Monte Carlo Day-Horizon Gate ────────────────────────────────────────
# Primary MC objective: P(hit target within N trading days) before ruin.
# Block-mode only — samples session blocks and counts "days used".
MC_MAX_DAYS: int             = 20         # Combine challenge day budget
MC_DAY_HORIZON_ENABLED: bool = True       # Enable day-horizon mode

# ── Monte Carlo Mode ────────────────────────────────────────────────────
MC_MODE: str             = "block"     # "iid" or "block" bootstrap
MC_BLOCK_TYPE: str       = "session"   # "session" or "day" (block mode only)
                                        # Consider "day" once trades/day ≥ ~8

# ── Monte Carlo Stress Testing ──────────────────────────────────────────
MC_STRESS_LOSS_MULTIPLIER: float   = 1.0   # Multiply negative PnL
MC_STRESS_WIN_MULTIPLIER: float    = 1.0   # Multiply positive PnL
MC_STRESS_WIN_RATE_SHIFT: float    = 0.0   # Shift win rate (e.g. -0.1)
MC_STRESS_SLIPPAGE_TICKS: int      = 0     # Per-trade slippage in ticks

# Stress scenario thresholds for promotion gate
MC_STRESS_MILD_TARGET_THRESHOLD: float   = 0.50
MC_STRESS_SEVERE_TARGET_THRESHOLD: float = 0.30
MC_STRESS_SEVERE_FLOOR: float            = 0.20   # Worst-batch / CI lower-bound floor

# ── Monte Carlo Confidence Interval ─────────────────────────────────────
MC_N_BATCHES: int        = 5            # Independent MC batches for CI estimation
MC_CI_LEVEL: float       = 0.95         # Confidence level (Wilson interval)

# ── Monte Carlo Tilt (Bad-Week Stress) ──────────────────────────────────
MC_TILT_BAD_FRAC: float  = 0.50         # Fraction of blocks drawn from worst sessions
MC_TILT_BAD_QUANTILE: float = 0.20      # Bottom N% of sessions = "bad" bucket

# ── Monte Carlo Readiness Calibration ─────────────────────────────────
# Readiness checks use a longer trade horizon and blended-path gate to
# match challenge-level progression rather than single-strategy slices.
MC_READINESS_MAX_TRADES: int = 800
MC_READINESS_STREAK_P95_MAX: int = 12
MC_READINESS_REQUIRED_SCENARIOS: tuple[str, ...] = ("blended",)
MC_CALIBRATION_CANDIDATE_TRADES: tuple[int, ...] = (200, 300, 400, 500, 600, 800, 1000)

# ── CVD Proxy ───────────────────────────────────────────────────────────
CVD_DIVERGENCE_LOOKBACK: int = 3          # Bars to evaluate divergence

# ── Regime Classifier V1 (Hybrid Threshold) ─────────────────────────────
REGIME_CLASSIFIER_TYPE: str      = "hybrid_threshold"
REGIME_ADX_TREND_THRESHOLD: float = 25.0      # ADX > 25 → trending
REGIME_ATR_EXTREME_PCTILE: float  = 90.0      # ATR pctile > 90 → extreme
REGIME_VOL_EXTREME_PCTILE: float  = 90.0      # Realized vol pctile > 90 → extreme
REGIME_WARMUP_BARS: int           = 10        # Min bars before classification (was 14; 10 = 50 min)

# ── MR Signal Engine ──────────────────────────────────────────────
# These are the ACTIVE thresholds for the MR pipeline (signal + exit sim).
# VWAP_SD_ENTRY_MIN/MAX above are NOT used here.──────
MR_SIGMA_ENTRY: float           = 1.4        # Ablation control baseline (A0)
MR_SIGMA_EXTREME: float         = 2.5        # Optional deeper-band threshold (diagnosis: was 3.0)
MR_RECLAIM_TICKS: int           = 0          # Ticks back inside band to confirm (diagnosis: was 2)
MR_COOLDOWN_BARS: int           = 1          # Bars after signal before next allowed (diagnosis: was 3)
MR_MAX_ATTEMPTS_PER_SIDE: int   = 4          # Per-side per session (diagnosis: was 2)
MR_MIN_DISTANCE_VWAP_TICKS: int = 5          # Min ticks from VWAP to consider (diagnosis: was 10)
MR_ATTEMPT_CAP_MODE: str        = "soft"     # "hard" | "soft"
MR_SOFT_CAP_COOLDOWN_BARS: int  = 4          # Bars required after per-side cap before soft override
MR_SOFT_CAP_MIN_ZSCORE: float   = 2.2        # Min |z| required to soft-override cap

# ── Attempt Clustering Reset ────────────────────────────────────────────
# Per-side attempt caps reset when z-score returns inside this band
# (i.e. a fresh excursion, not the same move printing signal after signal).
MR_CLUSTER_RESET_ZSCORE: float  = 0.5        # |z| must drop below this to reset attempts
MR_CLUSTER_RESET_ENABLED: bool  = True       # Enable cluster-based attempt reset
MR_CLUSTER_RESET_MODE: str      = "retrace"  # "legacy" | "retrace"
MR_CLUSTER_RETRACE_FRACTION: float = 0.50    # Reset side when |z| retraces this fraction toward VWAP
MR_CLUSTER_RESET_MIN_PEAK_Z: float = 1.4     # Minimum peak |z| to establish a resettable cluster

# ── MR Setup Quality Filters (Path A) ──────────────────────────────────
MR_FILTER_DISTANCE_ENABLED: bool      = False
MR_QUALITY_MIN_EXCURSION_ATR: float = 0.6
MR_FILTER_VWAP_FLAT_ENABLED: bool    = False
MR_QUALITY_VWAP_FLAT_LOOKBACK: int  = 6      # Bars for VWAP drift flatness check
MR_QUALITY_VWAP_FLAT_MAX_ATR: float = 0.45   # A2: only VWAP flatness filter active
MR_FILTER_RECLAIM_STRENGTH_ENABLED: bool = False
MR_QUALITY_RECLAIM_CLOSE_LOC_MIN: float = 0.60  # BUY reclaim close >=60% bar; SELL <=40%
MR_SOFT_RECLAIM_RANGE_IMPULSE_K: float = 1.2    # Soft-v3 range-impulse ATR threshold (sweep: 1.0/1.2/1.4)
MR_SOFT_RECLAIM_IMPULSE_K: float = MR_SOFT_RECLAIM_RANGE_IMPULSE_K  # Backward-compatible alias
MR_FIRST_OUTSIDE_ENABLED: bool         = False   # A: emit one candidate when first eligible bar is already outside
MR_TOUCH_LATCH_RESET_BUFFER: float     = 0.2     # B: latch reset when |z| <= (entry - buffer)
MR_DEDUPE_WINDOW_BARS: int             = 1       # Smarter dedupe: suppress rapid repeats inside this window
MR_DEDUPE_MIN_DELTA_Z: float           = 0.35    # Allow repeat if excursion deepens by at least this |z|

# ── One-Trade-Per-Excursion Dedupe ─────────────────────────────────────
MR_EXCURSION_DEDUPE_ENABLED: bool      = False
MR_EXCURSION_RESET_ZSCORE: float       = 1.0   # Reset excursion when |z| returns inside 1.0σ
MR_EXCURSION_RESET_VWAP_TICKS: int     = 1     # Or when price touches VWAP within this many ticks

# ── Trend Contamination Filter ──────────────────────────────────────────
# Block MR signals when strong trend is detected, regardless of regime label.
# Requires BOTH conditions met to block.
TREND_CONTAM_ADX_THRESHOLD: float    = 30.0  # ADX above this → trending
TREND_CONTAM_VWAP_SLOPE_MIN: float   = 0.15  # |VWAP slope| per bar (points) above this → trending
TREND_CONTAM_ENABLED: bool           = True   # Enable trend contamination filter
TREND_CONTAM_SLOPE_LOOKBACK: int     = 6      # Bars to compute VWAP slope over
MR_REGIME_ADX_BUCKETS: tuple[float, ...] = (20.0, 30.0, 40.0, 50.0)  # Diagnostics buckets for regime rejects

# ── Risk Governor (Research / Replay) ───────────────────────────────────
RG_DAILY_LOSS_HALT: float         = 240.0    # $ — mirrors DAILY_LOSS_LIMIT_INTERNAL
RG_STRATEGY_DAILY_PROFIT_CAP: float = 1_200.0  # $ — mirrors DAILY_PROFIT_HALT
RG_MAX_TRADES_PER_DAY: int        = 5        # Total across strategies
RG_FLATTEN_CUTOFF_TIME: str       = "15:50"  # ET — no new entries
RG_NO_TRADE_WINDOWS: list[tuple[str, str]] = [
    ("09:30", "09:32"),                       # first 2 min chop
]

# ── Consistency Cap ─────────────────────────────────────────────────────
# mode is inherited from ACCOUNT_MODE
CC_SOFT_CAP_ENABLED: bool     = True         # xfa_standard soft cap
CC_SOFT_CAP_AMOUNT: float     = 1_500.0      # Optional soft daily cap

# ── Replay Plot Overlays ────────────────────────────────────────────────
REPLAY_SHOW_VWAP: bool           = True
REPLAY_SHOW_SIGMA_BANDS: bool    = True
REPLAY_SHOW_ORB: bool            = True
REPLAY_SHOW_SIGNAL_MARKERS: bool = True
REPLAY_SHOW_REGIME_LABEL: bool   = True

# ── Reporting ───────────────────────────────────────────────────────────
ARTIFACTS_DIR: str               = "artifacts/replay_runs"
EXPORT_FEATURES_SNAPSHOT: bool   = True

# ── Validation Pack ─────────────────────────────────────────────────────
VALIDATION_ARTIFACTS_ROOT: str       = "artifacts/validation_runs"
VALIDATION_CONTINUE_ON_ERROR: bool   = True
VALIDATION_MIN_TRADES_FOR_MC: int    = 200   # Warn if total trades < this

# ── MR Exit Simulator ──────────────────────────────────────────────────
MR_EXIT_SIM_BAR_INTERVAL_SEC: int    = 300       # 5-min bars
MR_EXIT_SIM_STOP_ATR_MULT: float     = 1.5       # Matches VWAP_STOP_ATR_MULT
MR_EXIT_SIM_TARGET_MODE: str         = "vwap"    # "vwap" | "fixed_r"
MR_EXIT_SIM_FIXED_R_MULT: float      = 2.0       # Only used when target_mode="fixed_r"
MR_EXIT_SIM_TIME_STOP_BARS: int      = 1000      # Path A baseline: effectively disable time-stop
MR_EXIT_SIM_SLIPPAGE_TICKS: int      = 0         # Conservative: no slippage
MR_EXIT_SIM_SESSION_CUTOFF: str      = "15:50"   # Mirrors RG_FLATTEN_CUTOFF_TIME
MR_EXIT_SIM_RUNNER_ENABLED: bool     = False     # Path A baseline: no runner
MR_EXIT_SIM_RUNNER_PRIMARY_PCT: float = 0.50
MR_EXIT_SIM_RUNNER_TARGET_R: float   = 1.4
MR_EXIT_SIM_RUNNER_TRAIL_R: float    = 10.0
MR_EXIT_SIM_RUNNER_STEP_ENABLED: bool = True
MR_EXIT_SIM_RUNNER_STEP_TRIGGER_R: float = 1.0   # Arm lock once runner reaches +1.0R
MR_EXIT_SIM_RUNNER_STEP_LOCK_R: float = 0.5      # Lock runner stop at +0.5R once armed

# ── Thesis-Break Exit (replaces blind time-stop) ───────────────────────
# Instead of a flat time-stop, exit when the MR thesis is invalidated:
#   1. z-score mean-reverts below THESIS_BREAK_ZSCORE but price stalls
#   2. MAE exceeds a fraction of stop distance early in trade
#   3. ADX rises while VWAP slope turns adverse
THESIS_BREAK_ENABLED: bool           = True
THESIS_BREAK_ZSCORE: float           = 0.7       # z reverts inside this → thesis weakening
THESIS_BREAK_STALL_BARS: int         = 4         # Bars to wait after z reverts before cutting
THESIS_BREAK_MAE_FRAC: float         = 0.50      # If MAE > 50% of stop distance by bar N → cut
THESIS_BREAK_MAE_BAR_LIMIT: int      = 4         # Check MAE fraction after this many bars
THESIS_BREAK_ADX_THRESHOLD: float    = 30.0      # ADX above this + adverse slope → cut

# ── Scorecard Aggregation ───────────────────────────────────────────────
SCORECARD_MIN_TRADES: int            = 10        # Warn if fewer trades

# ── MC Profile Bridge ──────────────────────────────────────────────────
MC_PROFILE_MIN_TRADE_COUNT: int      = 10        # Min trades for reliable profile

# ── Promotion Gate ──────────────────────────────────────────────────────
PROMOTION_MIN_SESSION_SUCCESS_RATE: float = 0.80  # 80 % sessions must succeed
PROMOTION_APPROVAL_RATE_MIN: float       = 0.05   # ≥ 5 % approval rate
PROMOTION_APPROVAL_RATE_MAX: float       = 0.40   # ≤ 40 % approval rate
PROMOTION_MIN_EXPECTANCY: float          = 0.10   # ≥ $0.10 per trade (points)
PROMOTION_MC_MAX_DD_P95: float           = 1200.0 # 95th pct MC drawdown ≤ $1,200
PROMOTION_REQUIRE_ARTIFACTS: list[str]   = [
    "aggregate_metrics.json",
    "trades.csv",
]

# ── State Persistence ──────────────────────────────────────────────────
STATE_FILE: str              = "state/session_state.json"
HMM_STATE_FILE: str          = "state/hmm_state.json"
LOG_DIR: str                 = "logs"
