"""
validation_pack.py — Validation pack data structures and runner logic.

Defines reusable "packs" of replay sessions that can be executed as a batch
to validate strategy behaviour across diverse market conditions (range, trend,
chop, event).  The runner invokes ``run_debug_replay`` for each session,
collects results, and writes a ``manifest.json`` for downstream scorecard
and Monte-Carlo profile aggregation.

Usage (programmatic):
    from validation.validation_pack import load_pack, ValidationPackRunner

    pack   = load_pack("baseline_v1")
    runner = ValidationPackRunner(pack)
    manifest = runner.run()

See also: ``run_validation_pack.py`` for the CLI entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import csv
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from validation.preset_utils import normalize_allocator_policy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════

VALID_CATEGORIES = frozenset({"range", "trend", "event", "chop", "unlabeled"})


@dataclass
class SessionEntry:
    """A single replay session within a validation pack."""

    session_id: str
    start: str  # ISO-8601 timestamp
    end: str  # ISO-8601 timestamp
    category: str  # one of VALID_CATEGORIES
    symbol: str = "MES.c.0"
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category '{self.category}'; "
                f"must be one of {sorted(VALID_CATEGORIES)}"
            )


@dataclass
class ValidationPack:
    """An ordered collection of replay sessions to run as a batch."""

    pack_id: str
    description: str
    sessions: list[SessionEntry]


@dataclass
class SessionResult:
    """Outcome of a single session replay execution."""

    session_id: str
    success: bool
    category: str = "default"
    error_message: str = ""
    runtime_seconds: float = 0.0
    artifact_dir: str = ""


@dataclass
class ValidationRunManifest:
    """Top-level manifest written once a full pack run completes."""

    run_id: str
    pack_id: str
    timestamp: str
    config_hash: str
    sessions: list[SessionResult]
    total_runtime_seconds: float
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  Built-in packs
# ═══════════════════════════════════════════════════════════════════════

BUILTIN_PACKS: dict[str, ValidationPack] = {
    "baseline_v1": ValidationPack(
        pack_id="baseline_v1",
        description="Baseline validation pack — mixed session types (4h windows)",
        sessions=[
            SessionEntry(
                session_id="range_session_1",
                start="2026-02-18T14:30:00Z",
                end="2026-02-18T18:30:00Z",
                category="range",
            ),
            SessionEntry(
                session_id="trend_session_1",
                start="2026-02-19T14:30:00Z",
                end="2026-02-19T18:30:00Z",
                category="trend",
            ),
            SessionEntry(
                session_id="chop_session_1",
                start="2026-02-20T14:30:00Z",
                end="2026-02-20T18:30:00Z",
                category="chop",
            ),
        ],
    ),
}


def _build_generated_pack(
    pack_id: str,
    description: str,
    start_date: str,
    end_date: str,
    category: str = "unlabeled",
) -> ValidationPack:
    """Build a ValidationPack from auto-generated sessions via the calendar."""
    from validation.session_generator import generate_sessions_for_range

    raw = generate_sessions_for_range(start_date, end_date, category=category)
    entries = [
        SessionEntry(
            session_id=s["session_id"],
            start=s["start"],
            end=s["end"],
            category=s["category"],
            symbol=s["symbol"],
        )
        for s in raw
    ]
    return ValidationPack(pack_id=pack_id, description=description, sessions=entries)


# Register generated packs lazily (built on first access)
_GENERATED_PACK_SPECS: dict[str, dict] = {
    "historical_holdout_20d": {
        "description": "Historical holdout 20-day pack — Nov 3 to Nov 28, 2025 (full RTH sessions, pre-extended window)",
        "start_date": "2025-11-03",
        "end_date": "2025-11-28",
    },
    "pilot_20d": {
        "description": "Pilot 20-day pack — Jan 26 to Feb 20, 2026 (full RTH sessions)",
        "start_date": "2026-01-26",
        "end_date": "2026-02-20",
    },
    "extended_60d": {
        "description": "Extended 60-day pack — Dec 1, 2025 to Feb 20, 2026 (full RTH sessions)",
        "start_date": "2025-12-01",
        "end_date": "2026-02-20",
    },
}

_SESSION_ID_PACK_SPECS: dict[str, dict[str, Any]] = {
    "route_sensitivity_16": {
        "description": (
            "Route-sensitive 16-session pack — medium-impulse boundary days plus "
            "clean ORB controls sourced from extended_60d"
        ),
        "source_pack": "extended_60d",
        "session_ids": [
            "session_20251204",
            "session_20251208",
            "session_20251215",
            "session_20251217",
            "session_20251218",
            "session_20251223",
            "session_20251229",
            "session_20260108",
            "session_20260120",
            "session_20260202",
            "session_20260209",
            "session_20260210",
            "session_20260211",
            "session_20260212",
            "session_20260213",
            "session_20260112",
        ],
    },
}

_TREND20_SOURCE_RUN_ID = "trend20_adx_20260226_232222"


def _pack_from_session_ids(
    *,
    pack_id: str,
    description: str,
    source_pack: ValidationPack,
    session_ids: list[str],
) -> ValidationPack:
    by_id = {s.session_id: s for s in source_pack.sessions}
    entries = [
        SessionEntry(
            session_id=s.session_id,
            start=s.start,
            end=s.end,
            category=s.category,
            symbol=s.symbol,
            tags=list(s.tags),
            notes=s.notes,
        )
        for sid in session_ids
        if (s := by_id.get(sid)) is not None
    ]
    return ValidationPack(pack_id=pack_id, description=description, sessions=entries)


def _build_trend20_pack(source_run_id: str = _TREND20_SOURCE_RUN_ID) -> ValidationPack:
    run_dir = Path("artifacts/validation_runs") / source_run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"trend20 source manifest not found: {manifest_path}. "
            "Use pilot_20d/extended_60d or restore the trend20 source run."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session_ids = [str(s.get("session_id")) for s in manifest.get("sessions", []) if s.get("session_id")]
    extended = load_pack("extended_60d")
    return _pack_from_session_ids(
        pack_id="trend20",
        description=f"Trend-selected 20-session pack sourced from {source_run_id}",
        source_pack=extended,
        session_ids=session_ids,
    )


def _build_session_id_pack(pack_name: str) -> ValidationPack:
    spec = _SESSION_ID_PACK_SPECS[pack_name]
    source_pack = load_pack(str(spec["source_pack"]))
    return _pack_from_session_ids(
        pack_id=pack_name,
        description=str(spec["description"]),
        source_pack=source_pack,
        session_ids=[str(sid) for sid in spec["session_ids"]],
    )


# ═══════════════════════════════════════════════════════════════════════
#  Pack loader
# ═══════════════════════════════════════════════════════════════════════


def load_pack(pack_name: str) -> ValidationPack:
    """Load a validation pack by name from ``BUILTIN_PACKS`` or generated specs.

    Parameters
    ----------
    pack_name:
        Key into ``BUILTIN_PACKS`` or ``_GENERATED_PACK_SPECS``.

    Raises
    ------
    ValueError
        If *pack_name* is not found.
    """
    if pack_name in BUILTIN_PACKS:
        return BUILTIN_PACKS[pack_name]

    if pack_name in _GENERATED_PACK_SPECS:
        spec = _GENERATED_PACK_SPECS[pack_name]
        return _build_generated_pack(
            pack_id=pack_name,
            description=spec["description"],
            start_date=spec["start_date"],
            end_date=spec["end_date"],
        )

    if pack_name in _SESSION_ID_PACK_SPECS:
        return _build_session_id_pack(pack_name)

    if pack_name == "trend20":
        return _build_trend20_pack()

    available = sorted(
        set(BUILTIN_PACKS.keys())
        | set(_GENERATED_PACK_SPECS.keys())
        | set(_SESSION_ID_PACK_SPECS.keys())
        | {"trend20"}
    )
    raise ValueError(
        f"Unknown pack '{pack_name}'. Available packs: {', '.join(available)}"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _compute_config_hash(pack: ValidationPack) -> str:
    """Deterministic hash of the pack definition for provenance tracking."""
    payload = json.dumps(asdict(pack), sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════


class ValidationPackRunner:
    """Execute every session in a :class:`ValidationPack` and collect results.

    Parameters
    ----------
    pack:
        The validation pack to run.
    artifacts_root:
        Base directory for validation run outputs.  Each run creates a
        subfolder ``{run_id}/`` underneath this root.
    continue_on_error:
        If *True* (default), keep running remaining sessions when one fails.
        If *False*, abort immediately on the first failure.
    mr_reclaim_mode:
        MR candidate mode. ``"on"`` requires reclaim confirmation;
        ``"off"`` uses threshold-cross entries without reclaim;
        ``"soft"`` applies threshold-cross with light momentum confirmation.
    """

    def __init__(
        self,
        pack: ValidationPack,
        artifacts_root: str = "artifacts/validation_runs",
        continue_on_error: bool = True,
        mr_reclaim_mode: str = "on",
        mr_sigma_entry: float = 1.4,
        mr_soft_impulse_k: float = 0.25,
        mr_dedupe_enabled: bool = False,
        mr_attempt_cap_enabled: bool = True,
        mr_cooldown_bars: int = 1,
        mr_first_outside_enabled: bool = False,
        mr_touch_latch_reset_buffer: float = 0.2,
        mr_dedupe_window_bars: int = 1,
        mr_dedupe_min_delta_z: float = 0.35,
        mr_regime_enabled: bool = True,
        engine_mode: str = "both",
        allocator_policy: str = "none",
        allocator_v1_adx_threshold: float = 25.0,
        allocator_v2_trend_open_threshold: float = 25.0,
        allocator_v2_rising_threshold: float = 20.0,
        allocator_v2_rising_bars: int = 3,
        allocator_v2_range_threshold: float = 18.0,
        allocator_v2_range_bars: int = 3,
        alloc_openproxy_or_width_atr: float = 2.2,
        alloc_openproxy_impulse_atr: float = 0.9,
        alloc_openproxy_persist_bars: int = 1,
        alloc_openproxy_require_break: bool = False,
        alloc_openproxy_enable_orb_selectivity_refinement: bool = False,
        alloc_openproxy_low_atr_threshold: float = 10.0,
        alloc_openproxy_min_persistence_in_low_atr: int = 2,
        alloc_openproxy_high_impulse_threshold: float = 2.4,
        alloc_openproxy_min_persistence_when_high_impulse: int = 1,
        alloc_openproxy_medium_impulse_weak_persistence_filter_enabled: bool = False,
        alloc_openproxy_medium_impulse_decay_filter_enabled: bool = False,
        alloc_openproxy_medium_impulse_min_atr: float = 8.0,
        alloc_openproxy_medium_impulse_max_atr: float = 15.0,
        alloc_openproxy_medium_impulse_min: float = 0.9,
        alloc_openproxy_medium_impulse_max: float = 2.0,
        alloc_openproxy_medium_impulse_min_persistence: int = 2,
        orb_enabled: bool = False,
        orb_trigger_mode: str = "either",
        orb_pullback_confirm_bars: int = 3,
        orb_pullback_max_bars: int = 5,
        orb_pullback_tolerance_pts: float = 5.4,
        orb_pullback_entry_mode: str = "touch_only",
        orb_allocator_enabled: bool = False,
        orb_day_pnl_floor: float = -120.0,
        orb_max_loss_cluster: int = 2,
        sizing_policy: str = "fixed",
        fixed_contracts: int = 2,
        dyn_up_trail_headroom: float = 1400.0,
        dyn_up_day_headroom: float = 700.0,
        dyn_down_trail_headroom: float = 1200.0,
        dyn_down_day_headroom: float = 600.0,
        dyn_loss_streak_up_max: int = 1,
        dyn_loss_streak_down_min: int = 2,
        dyn_profit_lock: float = 2000.0,
        dyn_shock_loss_frac: float = 0.6,
        dyn_vol_atr_cap: float = 14.0,
        dyn_earned_traction: float = 150.0,
        dyn_earned_giveback: float = 50.0,
        # Dynamic v3 params
        dyn_v3_earned_traction: float = 75.0,
        dyn_v3_giveback_floor: float = 25.0,
        dyn_v3_orb_upsize_allowed: bool = False,
        dyn_v3_day_headroom_up: float = 800.0,
        dyn_v3_day_headroom_down: float = 600.0,
        dyn_v3_trail_headroom_up: float = 1400.0,
        dyn_v3_trail_headroom_down: float = 1200.0,
        dyn_v3_atr_traction_scale_enabled: bool = False,
        dyn_v3_atr_traction_baseline: float = 12.0,
        dyn_v3_atr_traction_min_scale: float = 0.75,
        dyn_v3_atr_traction_max_scale: float = 1.25,
        dyn_v3_consistency_brake_enabled: bool = False,
        dyn_v3_consistency_cap_pct: float = 0.50,
        dyn_v3_consistency_loss_buffer_mult: float = 2.0,
        batch_fast_mode: bool = False,
    ) -> None:
        self.pack = pack
        self.artifacts_root = artifacts_root
        self.continue_on_error = continue_on_error
        if mr_reclaim_mode not in {"on", "off", "soft", "touch"}:
            raise ValueError("mr_reclaim_mode must be 'on', 'off', 'soft', or 'touch'")
        self.mr_reclaim_mode = mr_reclaim_mode
        self.mr_sigma_entry = max(0.1, float(mr_sigma_entry))
        self.mr_soft_impulse_k = max(0.0, float(mr_soft_impulse_k))
        self.mr_dedupe_enabled = bool(mr_dedupe_enabled)
        self.mr_attempt_cap_enabled = bool(mr_attempt_cap_enabled)
        self.mr_cooldown_bars = max(0, int(mr_cooldown_bars))
        self.mr_first_outside_enabled = bool(mr_first_outside_enabled)
        self.mr_touch_latch_reset_buffer = max(0.0, float(mr_touch_latch_reset_buffer))
        self.mr_dedupe_window_bars = max(0, int(mr_dedupe_window_bars))
        self.mr_dedupe_min_delta_z = max(0.0, float(mr_dedupe_min_delta_z))
        self.mr_regime_enabled = bool(mr_regime_enabled)
        if engine_mode not in {"mr", "orb", "both"}:
            raise ValueError("engine_mode must be 'mr', 'orb', or 'both'")
        self.engine_mode = engine_mode
        self.allocator_policy = normalize_allocator_policy(allocator_policy)
        self.allocator_v1_adx_threshold = float(allocator_v1_adx_threshold)
        self.allocator_v2_trend_open_threshold = float(allocator_v2_trend_open_threshold)
        self.allocator_v2_rising_threshold = float(allocator_v2_rising_threshold)
        self.allocator_v2_rising_bars = max(1, int(allocator_v2_rising_bars))
        self.allocator_v2_range_threshold = float(allocator_v2_range_threshold)
        self.allocator_v2_range_bars = max(1, int(allocator_v2_range_bars))
        self.alloc_openproxy_or_width_atr = float(alloc_openproxy_or_width_atr)
        self.alloc_openproxy_impulse_atr = float(alloc_openproxy_impulse_atr)
        self.alloc_openproxy_persist_bars = max(0, int(alloc_openproxy_persist_bars))
        self.alloc_openproxy_require_break = bool(alloc_openproxy_require_break)
        self.alloc_openproxy_enable_orb_selectivity_refinement = bool(alloc_openproxy_enable_orb_selectivity_refinement)
        self.alloc_openproxy_low_atr_threshold = float(alloc_openproxy_low_atr_threshold)
        self.alloc_openproxy_min_persistence_in_low_atr = max(0, int(alloc_openproxy_min_persistence_in_low_atr))
        self.alloc_openproxy_high_impulse_threshold = float(alloc_openproxy_high_impulse_threshold)
        self.alloc_openproxy_min_persistence_when_high_impulse = max(0, int(alloc_openproxy_min_persistence_when_high_impulse))
        self.alloc_openproxy_medium_impulse_weak_persistence_filter_enabled = bool(alloc_openproxy_medium_impulse_weak_persistence_filter_enabled)
        self.alloc_openproxy_medium_impulse_decay_filter_enabled = bool(alloc_openproxy_medium_impulse_decay_filter_enabled)
        self.alloc_openproxy_medium_impulse_min_atr = float(alloc_openproxy_medium_impulse_min_atr)
        self.alloc_openproxy_medium_impulse_max_atr = float(alloc_openproxy_medium_impulse_max_atr)
        self.alloc_openproxy_medium_impulse_min = float(alloc_openproxy_medium_impulse_min)
        self.alloc_openproxy_medium_impulse_max = float(alloc_openproxy_medium_impulse_max)
        self.alloc_openproxy_medium_impulse_min_persistence = max(0, int(alloc_openproxy_medium_impulse_min_persistence))
        self.orb_enabled = bool(orb_enabled)
        self.orb_trigger_mode = orb_trigger_mode if orb_trigger_mode in {"break", "pullback", "either", "pullback_v3"} else "either"
        self.orb_pullback_confirm_bars = max(1, int(orb_pullback_confirm_bars))
        self.orb_pullback_max_bars = max(1, int(orb_pullback_max_bars))
        self.orb_pullback_tolerance_pts = max(0.0, float(orb_pullback_tolerance_pts))
        self.orb_pullback_entry_mode = orb_pullback_entry_mode if orb_pullback_entry_mode in {"touch_only", "touch_recovery"} else "touch_only"
        self.orb_allocator_enabled = bool(orb_allocator_enabled)
        self.orb_day_pnl_floor = float(orb_day_pnl_floor)
        self.orb_max_loss_cluster = max(0, int(orb_max_loss_cluster))

        # ── Sizing policy ───────────────────────────────────────────────
        self.sizing_policy_name = sizing_policy if sizing_policy in {"fixed", "dynamic_v1", "dynamic_v2", "dynamic_v3"} else "fixed"
        self.fixed_contracts = max(1, int(fixed_contracts))
        self.dyn_up_trail_headroom = float(dyn_up_trail_headroom)
        self.dyn_up_day_headroom = float(dyn_up_day_headroom)
        self.dyn_down_trail_headroom = float(dyn_down_trail_headroom)
        self.dyn_down_day_headroom = float(dyn_down_day_headroom)
        self.dyn_loss_streak_up_max = max(0, int(dyn_loss_streak_up_max))
        self.dyn_loss_streak_down_min = max(1, int(dyn_loss_streak_down_min))
        self.dyn_profit_lock = float(dyn_profit_lock)
        self.dyn_shock_loss_frac = float(dyn_shock_loss_frac)
        self.dyn_vol_atr_cap = float(dyn_vol_atr_cap)
        self.dyn_earned_traction = float(dyn_earned_traction)
        self.dyn_earned_giveback = float(dyn_earned_giveback)
        # v3 params
        self.dyn_v3_earned_traction = float(dyn_v3_earned_traction)
        self.dyn_v3_giveback_floor = float(dyn_v3_giveback_floor)
        self.dyn_v3_orb_upsize_allowed = bool(dyn_v3_orb_upsize_allowed)
        self.dyn_v3_day_headroom_up = float(dyn_v3_day_headroom_up)
        self.dyn_v3_day_headroom_down = float(dyn_v3_day_headroom_down)
        self.dyn_v3_trail_headroom_up = float(dyn_v3_trail_headroom_up)
        self.dyn_v3_trail_headroom_down = float(dyn_v3_trail_headroom_down)
        self.dyn_v3_atr_traction_scale_enabled = bool(dyn_v3_atr_traction_scale_enabled)
        self.dyn_v3_atr_traction_baseline = float(dyn_v3_atr_traction_baseline)
        self.dyn_v3_atr_traction_min_scale = float(dyn_v3_atr_traction_min_scale)
        self.dyn_v3_atr_traction_max_scale = float(dyn_v3_atr_traction_max_scale)
        self.dyn_v3_consistency_brake_enabled = bool(dyn_v3_consistency_brake_enabled)
        self.dyn_v3_consistency_cap_pct = float(dyn_v3_consistency_cap_pct)
        self.dyn_v3_consistency_loss_buffer_mult = float(dyn_v3_consistency_loss_buffer_mult)
        self.batch_fast_mode = bool(batch_fast_mode)

    # ── Public API ──────────────────────────────────────────────────────

    def run(self) -> ValidationRunManifest:
        """Run all sessions in the pack sequentially.

        Returns
        -------
        ValidationRunManifest
            Summary of the entire run, also persisted as ``manifest.json``.
        """
        from dotenv import load_dotenv

        load_dotenv()

        import config  # noqa: F811  (module-level re-import for monkey-patching)

        run_id = f"{self.pack.pack_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = Path(self.artifacts_root) / run_id
        sessions_dir = run_dir / "sessions"
        scorecard_dir = run_dir / "scorecard"
        mc_profile_dir = run_dir / "mc_profile"

        # Create directory skeleton
        for d in (sessions_dir, scorecard_dir, mc_profile_dir):
            d.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory: %s", d)

        config_hash = _compute_config_hash(self.pack)
        results: list[SessionResult] = []
        run_start = time.monotonic()

        original_artifacts_dir = config.ARTIFACTS_DIR
        allocator = None
        allocator_decisions: list[dict[str, Any]] = []
        if self.orb_allocator_enabled:
            from validation.day_allocator import DayAllocator, DayAllocatorConfig
            allocator = DayAllocator(
                DayAllocatorConfig(
                    orb_day_pnl_floor=self.orb_day_pnl_floor,
                    orb_max_loss_cluster=self.orb_max_loss_cluster,
                )
            )

        # ── Sizing policy ───────────────────────────────────────────────
        from validation.sizing_policy import SizingConfig, SizingPolicy, apply_sizing_to_trades
        sizing_cfg = SizingConfig(
            policy=self.sizing_policy_name,
            fixed_contracts=self.fixed_contracts,
            up_trail_headroom=self.dyn_up_trail_headroom,
            up_day_headroom=self.dyn_up_day_headroom,
            down_trail_headroom=self.dyn_down_trail_headroom,
            down_day_headroom=self.dyn_down_day_headroom,
            loss_streak_up_max=self.dyn_loss_streak_up_max,
            loss_streak_down_min=self.dyn_loss_streak_down_min,
            shock_loss_frac=self.dyn_shock_loss_frac,
            profit_lock=self.dyn_profit_lock,
            daily_loss_limit=float(config.DAILY_LOSS_LIMIT_EXTERNAL),
            trail_dd_limit=float(config.MAX_LOSS_LIMIT),
            vol_atr_cap=self.dyn_vol_atr_cap,
            earned_traction=self.dyn_earned_traction,
            earned_giveback=self.dyn_earned_giveback,
            # v3 params
            v3_earned_traction=self.dyn_v3_earned_traction,
            v3_giveback_floor=self.dyn_v3_giveback_floor,
            v3_orb_upsize_allowed=self.dyn_v3_orb_upsize_allowed,
            v3_day_headroom_up=self.dyn_v3_day_headroom_up,
            v3_day_headroom_down=self.dyn_v3_day_headroom_down,
            v3_trail_headroom_up=self.dyn_v3_trail_headroom_up,
            v3_trail_headroom_down=self.dyn_v3_trail_headroom_down,
            v3_atr_traction_scale_enabled=self.dyn_v3_atr_traction_scale_enabled,
            v3_atr_traction_baseline=self.dyn_v3_atr_traction_baseline,
            v3_atr_traction_min_scale=self.dyn_v3_atr_traction_min_scale,
            v3_atr_traction_max_scale=self.dyn_v3_atr_traction_max_scale,
            v3_consistency_brake_enabled=self.dyn_v3_consistency_brake_enabled,
            v3_consistency_cap_pct=self.dyn_v3_consistency_cap_pct,
            v3_consistency_loss_buffer_mult=self.dyn_v3_consistency_loss_buffer_mult,
        )
        sizing_policy = SizingPolicy(sizing_cfg)
        print(f"  [SIZING] policy={sizing_cfg.policy} "
              f"contracts={sizing_cfg.fixed_contracts if sizing_cfg.policy == 'fixed' else '1↔2 dynamic'}")

        for idx, session in enumerate(self.pack.sessions, 1):
            header = (
                f"[{idx}/{len(self.pack.sessions)}] "
                f"{session.session_id} ({session.category})"
            )
            logger.info("Starting session: %s", header)
            print(f"\n{'─'*70}")
            print(f"  ▶ {header}")
            print(f"    {session.start} → {session.end}")
            print(f"{'─'*70}")

            orb_enabled_this_session = self.orb_enabled
            if allocator is not None:
                decision = allocator.decide(
                    session_id=session.session_id,
                    category=session.category,
                    orb_requested=self.orb_enabled,
                )
                orb_enabled_this_session = decision.orb_enabled
                allocator_decisions.append({
                    "session_id": decision.session_id,
                    "category": decision.category,
                    "day_pnl_before": decision.day_pnl_before,
                    "max_loss_cluster_before": decision.max_loss_cluster_before,
                    "volatility_ok": decision.volatility_ok,
                    "orb_enabled": decision.orb_enabled,
                    "reason": decision.reason,
                })

            result = self._run_single_session(
                session, sessions_dir, config, original_artifacts_dir, orb_enabled_this_session
            )

            # ── Run exit simulation on successful sessions ──────────
            if result.success:
                session_dir = sessions_dir / session.session_id
                try:
                    diag = self._run_exit_sim(session, session_dir)
                    logger.info(
                        "Exit sim for %s: %d signals → %d trades",
                        session.session_id,
                        diag.get("signals_received", 0),
                        diag.get("trades_emitted", 0),
                    )
                except Exception as exc:
                    logger.exception(
                        "Exit sim failed for %s: %s", session.session_id, exc
                    )
                    print(f"  [exit_sim] ERROR: {type(exc).__name__}: {exc}")
                    # Don't mark the session as failed — signals still valid
                if allocator is not None:
                    allocator.update_from_trades_csv(session_dir / "trades.csv")

                # ── Apply sizing policy to session trades ───────────
                try:
                    regime, active_engine = self._get_session_regime_engine(session_dir)
                    session_atr = self._get_session_atr_median(session_dir)
                    scaled = apply_sizing_to_trades(
                        session_dir / "trades.csv",
                        sizing_policy,
                        regime=regime,
                        active_engine=active_engine,
                        session_id=session.session_id,
                        day_index=idx,
                        session_atr_median=session_atr,
                    )
                    if scaled:
                        import pandas as pd
                        pd.DataFrame(scaled).to_csv(session_dir / "trades.csv", index=False)
                    else:
                        # No trades — still register day in sizing policy
                        pass
                    day_rec = sizing_policy.daily_log[-1] if sizing_policy.daily_log else None
                    if day_rec:
                        policy_extra = ""
                        if self.sizing_policy_name == "dynamic_v2":
                            policy_extra = f" atr={day_rec.session_atr_median:.1f}"
                            if day_rec.vol_throttled:
                                policy_extra += " VOL_CAP"
                            if day_rec.earned_upsize_triggered:
                                policy_extra += " EARNED_2c"
                        elif self.sizing_policy_name == "dynamic_v3":
                            policy_extra = f" engine={day_rec.allocator_engine}"
                            if day_rec.v3_orb_day:
                                policy_extra += " ORB_DAY"
                            if day_rec.v3_upsize_trigger:
                                policy_extra += f" UP:{day_rec.v3_upsize_trigger}"
                            policy_extra += f" day_hr={day_rec.day_headroom:.0f}"
                        print(
                            f"  [SIZING] day={idx} contracts={day_rec.contracts_start}→{day_rec.contracts_final} "
                            f"equity={day_rec.equity_after:.0f} trail_hr={day_rec.trail_headroom:.0f} "
                            f"lock={day_rec.profit_lock_triggered}"
                            f"{' DOWNSHIFT:' + day_rec.downshift_reason if day_rec.downshift_reason else ''}"
                            f"{policy_extra}"
                        )
                except Exception as exc:
                    logger.exception("Sizing failed for %s: %s", session.session_id, exc)
                    print(f"  [sizing] ERROR: {type(exc).__name__}: {exc}")

            results.append(result)

            status = "✓ OK" if result.success else f"✗ FAIL: {result.error_message}"
            logger.info(
                "Session %s finished in %.1fs — %s",
                session.session_id,
                result.runtime_seconds,
                status,
            )
            print(f"  {status}  ({result.runtime_seconds:.1f}s)")

            if not result.success and not self.continue_on_error:
                logger.warning("Aborting pack run (continue_on_error=False)")
                print("  ⚠ Aborting remaining sessions (continue_on_error=False)")
                break

        total_runtime = time.monotonic() - run_start

        manifest = ValidationRunManifest(
            run_id=run_id,
            pack_id=self.pack.pack_id,
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            sessions=results,
            total_runtime_seconds=round(total_runtime, 2),
        )

        self._write_manifest(manifest, run_dir)
        if allocator_decisions:
            decisions_path = run_dir / "allocator_decisions.json"
            decisions_path.write_text(json.dumps(allocator_decisions, indent=2) + "\n", encoding="utf-8")
            logger.info("Allocator decisions written -> %s", decisions_path)
        allocator_debug_path = self._stage_allocator_debug(run_dir)
        if allocator_debug_path is not None:
            logger.info("Allocator debug CSV written -> %s", allocator_debug_path)

        # ── Write sizing artifacts ──────────────────────────────────────
        if sizing_policy.daily_log:
            sizing_log_path = run_dir / "sizing_decisions.json"
            sizing_policy.write_daily_log(sizing_log_path)
            logger.info("Sizing decisions written -> %s", sizing_log_path)
            print(f"  [SIZING] daily log → {sizing_log_path}")

        sizing_config_path = run_dir / "sizing_config.json"
        sizing_config_path.write_text(
            json.dumps(sizing_policy.config_snapshot(), indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("Sizing config written -> %s", sizing_config_path)

        # ════════════════════════════════════════════════════════════════
        #  Post-session pipeline: aggregate → bridge → MC → gate
        # ════════════════════════════════════════════════════════════════
        passed_sessions = sum(1 for r in results if r.success)
        print(f"\n{'═'*70}")
        print(f"  Post-session pipeline  ({passed_sessions}/{len(results)} sessions succeeded)")
        print(f"{'═'*70}")

        agg_metrics = self._stage_aggregate(run_dir)
        scorecard_metrics = self._stage_scorecard(run_dir, agg_metrics)
        if scorecard_metrics is not None:
            agg_metrics = self._merge_scorecard_into_aggregate(run_dir, agg_metrics, scorecard_metrics)
        mc_profile = self._stage_bridge(run_dir)
        mc_results = self._stage_mc_survival(run_dir, agg_metrics)
        gate_result = self._stage_promotion_gate(run_dir)

        # ── End-of-run summary ──────────────────────────────────────────
        total_trades = agg_metrics.get("trade_count_total", 0) if agg_metrics else 0
        mc_ready = bool(agg_metrics.get("readiness")) if agg_metrics else False
        p_target = mc_results.get("p_target_before_ruin", "N/A") if mc_results else "N/A"
        gate_pass = gate_result.overall_pass if gate_result else "N/A"

        print(f"\n{'═'*70}")
        print(f"  END-OF-RUN SUMMARY")
        print(f"{'═'*70}")
        print(f"  Sessions run            : {len(results)}")
        print(f"  Sessions succeeded      : {passed_sessions}")
        print(f"  Total trades aggregated : {total_trades}")
        print(f"  MC readiness            : {mc_ready}")
        print(f"  p_target_before_ruin    : {p_target}")
        print(f"  Gate pass/fail          : {gate_pass}")

        # Sizing summary
        sizing_days = len(sizing_policy.daily_log)
        sizing_2c_days = sum(1 for r in sizing_policy.daily_log if r.contracts_start == 2)
        sizing_downshifts = sum(1 for r in sizing_policy.daily_log if r.downshift_reason)
        sizing_lock = sizing_policy.profit_lock_triggered
        print(f"  Sizing policy           : {sizing_policy.config.policy}")
        print(f"  Sizing days tracked     : {sizing_days}")
        if sizing_policy.config.policy in ("dynamic_v1", "dynamic_v2"):
            print(f"  Days at 2c (start)      : {sizing_2c_days}/{sizing_days}")
            print(f"  Intraday downshifts     : {sizing_downshifts}")
            print(f"  Profit lock triggered   : {sizing_lock}")
        if sizing_policy.config.policy == "dynamic_v2":
            vol_throttled_days = sum(
                1 for r in sizing_policy.daily_log if getattr(r, "vol_throttled", False)
            )
            earned_upsize_days = sum(
                1 for r in sizing_policy.daily_log if getattr(r, "earned_upsize_triggered", False)
            )
            print(f"  Vol-throttled days      : {vol_throttled_days}/{sizing_days}")
            print(f"  Earned upsize days      : {earned_upsize_days}/{sizing_days}")
        if sizing_policy.config.policy == "dynamic_v3":
            days_started_2c = sum(1 for r in sizing_policy.daily_log if r.contracts_start == 2)
            days_ever_2c = sum(
                1 for r in sizing_policy.daily_log
                if r.contracts_start == 2 or r.v3_upsize_trigger != ""
            )
            orb_sessions = sum(1 for r in sizing_policy.daily_log if r.v3_orb_day)
            traction_triggers = sum(1 for r in sizing_policy.daily_log if r.v3_upsize_trigger == "traction")
            first_win_triggers = sum(1 for r in sizing_policy.daily_log if r.v3_upsize_trigger == "first_trade_win")
            orb_triggers = sum(1 for r in sizing_policy.daily_log if r.v3_upsize_trigger == "orb_day")
            print(f"  Days started at 2c      : {days_started_2c}/{sizing_days}")
            print(f"  Days ever at 2c         : {days_ever_2c}/{sizing_days}")
            print(f"  ORB sessions            : {orb_sessions}/{sizing_days}")
            print(f"  Intraday downshifts     : {sizing_downshifts}")
            print(f"  Upsize triggers         : traction={traction_triggers} first_win={first_win_triggers} orb={orb_triggers}")
            print(f"  Profit lock triggered   : {sizing_lock}")
        print(f"  Final equity            : {sizing_policy.equity:.2f}")
        print(f"  Peak equity             : {sizing_policy.peak_equity:.2f}")
        print(f"  Trailing DD used        : {sizing_policy.trailing_dd_used:.2f}")
        print(f"{'═'*70}\n")

        return manifest

    # ── Post-session pipeline stages ────────────────────────────────

    def _stage_aggregate(self, run_dir: Path) -> dict | None:
        """Stage 1: aggregate trades across sessions."""
        print("\n  ▶ Stage 1: Trade Aggregation")
        try:
            from validation.trade_aggregator import aggregate_trades
            import config as _cfg
            metrics = aggregate_trades(
                run_dir,
                min_trade_count=_cfg.MC_PROFILE_MIN_TRADE_COUNT,
            )
            print(f"    ✓ {metrics.get('trade_count_total', 0)} trades aggregated")
            return metrics
        except Exception as exc:
            logger.exception("Aggregation stage failed: %s", exc)
            print(f"    ✗ Aggregation FAILED: {type(exc).__name__}: {exc}")
            return None

    def _stage_bridge(self, run_dir: Path) -> dict | None:
        """Stage 3: build MC profile via ReplayProfileBridge."""
        print("\n  ▶ Stage 3: MC Profile Bridge")
        try:
            from validation.replay_profile_bridge import ReplayProfileBridge
            import config as _cfg
            bridge = ReplayProfileBridge(
                str(run_dir),
                min_trade_count=_cfg.MC_PROFILE_MIN_TRADE_COUNT,
            )
            profile = bridge.build_profile_from_aggregate()
            bridge.write_profile(profile)
            print(f"    ✓ profile: {profile.sample_size_trades} trades, "
                  f"win_rate={profile.win_rate:.2%}, "
                  f"expectancy_r={profile.expectancy_r:.4f}")
            return {"sample_size_trades": profile.sample_size_trades}
        except Exception as exc:
            logger.exception("Bridge stage failed: %s", exc)
            print(f"    ✗ Bridge FAILED: {type(exc).__name__}: {exc}")
            return None

    def _stage_mc_survival(
        self, run_dir: Path, agg_metrics: dict | None
    ) -> dict | None:
        """Stage 4: Monte Carlo combine-survival simulation (base + stress)."""
        print("\n  ▶ Stage 4: Monte Carlo Survival Simulation")

        if agg_metrics is None or not agg_metrics.get("readiness"):
            print("    ⚠ Skipped — insufficient trades for MC readiness")
            return None

        try:
            import pandas as pd
            from simulation.mc_survival import MonteCarloSurvivalSimulator

            agg_csv = run_dir / "aggregate_trades.csv"
            if not agg_csv.is_file():
                print("    ✗ aggregate_trades.csv not found")
                return None

            df = pd.read_csv(agg_csv)

            # Use dollar-scaled PnL when sizing is active so MC reflects
            # the actual contract count (pnl_r is never scaled by contracts).
            use_dollars = "contracts" in df.columns
            if use_dollars:
                pnl_values = df["pnl_dollars"].dropna().tolist()
                val_col = "pnl_dollars"
            else:
                pnl_values = df["pnl_r"].dropna().tolist()
                val_col = "pnl_r"

            if not pnl_values:
                print(f"    ✗ No {val_col} values in aggregate_trades.csv")
                return None

            # Extract session_ids if available
            session_ids = None
            if "session_id" in df.columns:
                session_ids = df["session_id"].tolist()

            sim = MonteCarloSurvivalSimulator()

            # Run base + mild + severe + tilt_bad_week scenarios
            all_results = sim.run_all_scenarios(
                pnl_values, seed=42, session_ids=session_ids,
                use_dollar_values=use_dollars,
            )
            sim.write_all_results(all_results, run_dir)

            base = all_results["base"]
            print(f"    ✓ base:   p_target={base.p_target_before_ruin:.2%}, "
                  f"p_ruin={base.p_ruin:.2%}, dd_p95=${base.dd_p95:,.0f}")

            for name in ("mild", "severe", "tilt_bad_week"):
                if name not in all_results:
                    continue
                r = all_results[name]
                label = name.replace("_", " ")
                print(f"    ✓ {label:14s}: p_target={r.p_target_before_ruin:.2%}, "
                      f"p_ruin={r.p_ruin:.2%}, dd_p95=${r.dd_p95:,.0f}")

            # Print stress comparison table
            sim.log_stress_comparison(all_results)

            return {
                "p_target_before_ruin": base.p_target_before_ruin,
                "p_ruin": base.p_ruin,
                "dd_p95": base.dd_p95,
            }
        except Exception as exc:
            logger.exception("MC survival stage failed: %s", exc)
            print(f"    ✗ MC FAILED: {type(exc).__name__}: {exc}")
            return None

    def _stage_promotion_gate(self, run_dir: Path) -> Any:
        """Stage 5: PromotionGate evaluation."""
        print("\n  ▶ Stage 5: Promotion Gate")
        try:
            from validation.promotion_gate import PromotionGate, PromotionGateResult  # noqa: F811
            gate = PromotionGate(str(run_dir))
            result = gate.evaluate()
            gate.write_summary(result)
            verdict = "PASS" if result.overall_pass else "FAIL"
            print(f"    {'✓' if result.overall_pass else '✗'} Gate: {verdict}")
            for c in result.checks:
                status = "✓" if c.passed else "✗"
                print(f"      {status} {c.name}: {c.current_value} (threshold: {c.threshold}) {c.notes}")
            return result
        except Exception as exc:
            logger.exception("Promotion gate failed: %s", exc)
            print(f"    ✗ Gate FAILED: {type(exc).__name__}: {exc}")
            return None

    def _stage_scorecard(
        self,
        run_dir: Path,
        agg_metrics: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Stage 2: generate scorecard artifacts from session outputs."""
        print("\n  ▶ Stage 2: Scorecard Aggregation")

        if agg_metrics is None:
            print("    ⚠ Skipped — trade aggregation did not complete")
            return None

        try:
            from validation.scorecard import ScorecardAggregator

            aggregator = ScorecardAggregator(str(run_dir))
            scorecard = aggregator.generate()

            approval_rate = scorecard.get("approval_rate", scorecard.get("aggregate_approval_rate", 0.0))
            expectancy_r = scorecard.get("expectancy_r")
            if expectancy_r is None:
                expectancy_r = (scorecard.get("trade_metrics") or {}).get("expectancy_r")

            print(
                "    ✓ scorecard: "
                f"approval_rate={float(approval_rate or 0.0):.2%}, "
                f"expectancy_r={float(expectancy_r or 0.0):.4f}"
            )
            return scorecard
        except Exception as exc:
            logger.exception("Scorecard stage failed: %s", exc)
            print(f"    ✗ Scorecard FAILED: {type(exc).__name__}: {exc}")
            return None

    def _merge_scorecard_into_aggregate(
        self,
        run_dir: Path,
        agg_metrics: dict[str, Any] | None,
        scorecard_metrics: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Backfill promotion-gate fields into run-level aggregate metrics."""
        aggregate_path = run_dir / "aggregate_metrics.json"
        merged = dict(agg_metrics or {})

        approval_rate = scorecard_metrics.get("approval_rate")
        if approval_rate is None:
            approval_rate = scorecard_metrics.get("aggregate_approval_rate")
        if approval_rate is not None:
            merged["approval_rate"] = approval_rate

        if "expectancy_r" not in merged:
            trade_metrics = scorecard_metrics.get("trade_metrics") or {}
            expectancy_r = scorecard_metrics.get("expectancy_r", trade_metrics.get("expectancy_r"))
            if expectancy_r is not None:
                merged["expectancy_r"] = expectancy_r

        aggregate_payload = json.dumps(merged, indent=2, default=str) + "\n"
        aggregate_path.write_text(aggregate_payload, encoding="utf-8")
        return merged

    def _stage_allocator_debug(self, run_dir: Path) -> Path | None:
        """Stage allocator session-level diagnostics as a single CSV artifact."""
        sessions_dir = run_dir / "sessions"
        if not sessions_dir.is_dir():
            return None

        rows: list[dict[str, Any]] = []
        for session_dir in sorted(p for p in sessions_dir.iterdir() if p.is_dir()):
            summary_path = session_dir / "session_summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                logger.exception("Failed reading session summary for allocator debug: %s", summary_path)
                continue

            orb_funnel = summary.get("orb_funnel", {}) or {}
            open_proxy = orb_funnel.get("open_proxy_diagnostics", {}) or {}
            session_pnl = self._read_session_pnl_dollars(session_dir)
            trade_count = self._read_session_trade_count(session_dir)
            route = orb_funnel.get("allocator_decision") or orb_funnel.get("engine_mode") or self.engine_mode
            notes = orb_funnel.get("allocator_reason", "")
            rows.append(
                {
                    "session_id": summary.get("session_id", session_dir.name),
                    "date": session_dir.name.replace("session_", ""),
                    "allocator_policy": orb_funnel.get("allocator_policy", self.allocator_policy),
                    "route": route,
                    "engine_mode": orb_funnel.get("engine_mode", self.engine_mode),
                    "opening_range_width": open_proxy.get("opening_range_width_pts", 0.0),
                    "atr": open_proxy.get("atr_at_decision", 0.0),
                    "width_atr": open_proxy.get("opening_range_width_atr", 0.0),
                    "impulse": open_proxy.get("first_3bar_directional_impulse", 0.0),
                    "persistence": open_proxy.get("persist_bars_observed", 0),
                    "close_location": open_proxy.get("close_location_in_opening_range", ""),
                    "one_sidedness": open_proxy.get("one_sidedness", open_proxy.get("signed_imbalance", 0.0)),
                    "confidence_score": self._derive_allocator_confidence(open_proxy),
                    "breakout_direction": open_proxy.get("breakout_direction", ""),
                    "trigger_width": open_proxy.get("trigger_width", False),
                    "trigger_impulse": open_proxy.get("trigger_impulse", False),
                    "trigger_persist": open_proxy.get("trigger_persist", False),
                    "breakout_persistence": open_proxy.get("breakout_persistence", False),
                    "selectivity_refinement_enabled": open_proxy.get("selectivity_refinement_enabled", False),
                    "selectivity_low_atr_caution": open_proxy.get("selectivity_low_atr_caution", False),
                    "selectivity_high_impulse_caution": open_proxy.get("selectivity_high_impulse_caution", False),
                    "selectivity_orb_blocked": open_proxy.get("selectivity_orb_blocked", False),
                    "selectivity_block_reason": open_proxy.get("selectivity_block_reason", ""),
                    "pre_selectivity_decision": open_proxy.get("pre_selectivity_decision", ""),
                    "selectivity_medium_impulse_weak_persistence_caution": open_proxy.get("selectivity_medium_impulse_weak_persistence_caution", False),
                    "selectivity_v3_orb_blocked": open_proxy.get("selectivity_v3_orb_blocked", False),
                    "selectivity_v3_block_reason": open_proxy.get("selectivity_v3_block_reason", ""),
                    "pre_v3_selectivity_decision": open_proxy.get("pre_v3_selectivity_decision", ""),
                    "post_v3_selectivity_decision": open_proxy.get("post_v3_selectivity_decision", ""),
                    "trade_count": trade_count,
                    "session_pnl_dollars": session_pnl,
                    "notes": notes,
                }
            )

        if not rows:
            return None

        out_path = run_dir / "allocator_debug.csv"
        fieldnames = list(rows[0].keys())
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return out_path

    @staticmethod
    def _read_session_trade_count(session_dir: Path) -> int:
        trades_path = session_dir / "trades.csv"
        if not trades_path.is_file():
            return 0
        try:
            with trades_path.open("r", encoding="utf-8") as fh:
                return max(sum(1 for _ in fh) - 1, 0)
        except Exception:
            return 0

    @staticmethod
    def _read_session_pnl_dollars(session_dir: Path) -> float:
        trades_path = session_dir / "trades.csv"
        if not trades_path.is_file():
            return 0.0
        pnl = 0.0
        try:
            with trades_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    pnl += float(row.get("pnl_dollars", 0.0) or 0.0)
        except Exception:
            return 0.0
        return round(pnl, 2)

    @staticmethod
    def _derive_allocator_confidence(open_proxy: dict[str, Any]) -> float:
        if not open_proxy:
            return 0.0
        score = 0.0
        score += min(float(open_proxy.get("opening_range_width_atr", 0.0) or 0.0) / 3.0, 1.0)
        score += min(float(open_proxy.get("first_3bar_directional_impulse", 0.0) or 0.0) / 1.5, 1.0)
        score += min(float(open_proxy.get("persist_bars_observed", 0) or 0) / 2.0, 1.0)
        return round(score / 3.0, 4)

    # ── Sizing helpers ──────────────────────────────────────────────

    def _get_session_regime_engine(self, session_dir: Path) -> tuple[str, str]:
        """Extract regime label and active engine from session artifacts.

        Returns
        -------
        (regime, active_engine)
            regime: "trend", "range", "chop", or "unknown"
            active_engine: "mr", "orb", "both", or the allocator decision
        """
        summary_path = session_dir / "session_summary.json"
        regime = "unknown"
        active_engine = self.engine_mode  # default to runner-level

        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))

                # Active engine from allocator decision (in orb_funnel)
                orb_funnel = summary.get("orb_funnel", {})
                alloc_decision = orb_funnel.get("allocator_decision")
                if alloc_decision and alloc_decision in {"mr", "orb", "both"}:
                    active_engine = alloc_decision

                # Regime from config_snapshot → ALLOCATOR_POLICY decision,
                # or from regime_distribution majority
                regime_dist = summary.get("regime_distribution", {})
                if regime_dist:
                    # Pick the majority regime label
                    regime = max(regime_dist, key=regime_dist.get)  # type: ignore[arg-type]
                    # Normalize labels
                    regime_lower = regime.lower()
                    if "range" in regime_lower:
                        regime = "range"
                    elif "trend" in regime_lower:
                        regime = "trend"
                    elif "chop" in regime_lower:
                        regime = "chop"
            except Exception:
                pass  # Fall back to defaults

        return regime, active_engine

    @staticmethod
    def _get_session_atr_median(session_dir: Path, max_bars: int = 12) -> float:
        """Compute median ATR over the first ``max_bars`` of the session.

        Reads ``features_snapshot.csv`` and returns the median of the first
        ``max_bars`` ATR values.  Returns 0.0 if the file is missing or empty.
        """
        import numpy as np
        import pandas as pd

        fp = session_dir / "features_snapshot.csv"
        if not fp.is_file():
            return 0.0
        try:
            df = pd.read_csv(fp)
            atr_vals = pd.to_numeric(df["atr"], errors="coerce").head(max_bars).dropna()
            if atr_vals.empty:
                return 0.0
            return float(np.median(atr_vals.to_numpy()))
        except Exception:
            return 0.0

    # ── Exit simulation ──────────────────────────────────────────────

    def _run_exit_sim(
        self,
        session: SessionEntry,
        session_dir: Path,
    ) -> dict:
        """Run MRExitSimulator on a completed session's signals.

        Returns a diagnostics dict suitable for ``exit_sim_diagnostics.json``.
        """
        from simulation.mr_exit_simulator import ExitSimConfig, MRExitSimulator

        signals_csv = session_dir / "signals.csv"
        if not signals_csv.is_file():
            msg = f"signals.csv not found in {session_dir}"
            logger.warning(msg)
            print(f"  [exit_sim] SKIP — {msg}")
            return {"error": msg, "signals_received": 0, "trades_emitted": 0}

        sim = MRExitSimulator()  # uses defaults from ExitSimConfig

        # Load approved signals to report count
        import pandas as pd
        sig_df = pd.read_csv(signals_csv)
        sig_df.columns = sig_df.columns.str.strip().str.lower()
        approved_mask = sig_df["approved"].astype(str).str.strip().str.lower() == "true"
        type_mask = sig_df["signal_type"].astype(str).str.strip().str.upper().isin({"MR", "ORB"})
        n_approved = int((approved_mask & type_mask).sum())
        print(f"  [exit_sim] session_id={session.session_id} approved_signals={n_approved}")

        # Run the full simulate_from_replay_artifacts pipeline
        trades = sim.simulate_from_replay_artifacts(
            session_dir=str(session_dir),
            replay_start=session.start,
            replay_end=session.end,
            symbol=session.symbol,
        )

        # Build diagnostics
        sim_diag = dict(getattr(sim, "last_run_diagnostics", {}))
        exit_reasons: dict[str, int] = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        skipped_no_entry_bar = int(sim_diag.get("skipped_no_entry_bar", 0))
        skipped_invalid_levels = int(sim_diag.get("skipped_invalid_levels", 0))

        diagnostics = {
            "session_id": session.session_id,
            "signals_received": int(sim_diag.get("signals_received", n_approved)),
            "entries_opened": int(sim_diag.get("entries_opened", len(trades))),
            "trades_emitted": int(sim_diag.get("trades_emitted", len(trades))),
            "forced_replay_exits": int(sim_diag.get("forced_replay_exits", 0)),
            "skipped_invalid_levels": skipped_invalid_levels,
            "exit_reason_breakdown": exit_reasons,
            "skips_by_reason": {
                "no_entry_bar": skipped_no_entry_bar,
                "invalid_levels": skipped_invalid_levels,
            },
        }

        # Write diagnostics JSON
        diag_path = session_dir / "exit_sim_diagnostics.json"
        diag_path.write_text(
            json.dumps(diagnostics, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"  [exit_sim] diagnostics → {diag_path}")

        # Write "I was here" proof file
        marker_path = session_dir / "exit_sim_called.txt"
        marker_path.write_text(
            f"MRExitSimulator ran for {session.session_id}\n"
            f"signals_received={diagnostics['signals_received']}\n"
            f"entries_opened={diagnostics['entries_opened']}\n"
            f"trades_emitted={diagnostics['trades_emitted']}\n"
            f"forced_replay_exits={diagnostics['forced_replay_exits']}\n"
            f"skipped_invalid_levels={diagnostics['skipped_invalid_levels']}\n",
            encoding="utf-8",
        )

        return diagnostics

    # ── Internal helpers ────────────────────────────────────────────────

    def _run_single_session(
        self,
        session: SessionEntry,
        sessions_dir: Path,
        config_module: Any,
        original_artifacts_dir: str,
        orb_enabled: bool,
    ) -> SessionResult:
        """Execute one replay session with proper artifact redirection.

        Temporarily overrides ``config.ARTIFACTS_DIR`` so that
        ``ReplaySessionReport.export()`` writes into the validation run's
        session directory instead of the default replay artifacts location.
        """
        from replay_debug import run_debug_replay

        args = self._build_replay_args(
            session,
            self.batch_fast_mode,
            self.mr_reclaim_mode,
            self.mr_sigma_entry,
            self.mr_soft_impulse_k,
            self.mr_dedupe_enabled,
            self.mr_attempt_cap_enabled,
            self.mr_cooldown_bars,
            self.mr_first_outside_enabled,
            self.mr_touch_latch_reset_buffer,
            self.mr_dedupe_window_bars,
            self.mr_dedupe_min_delta_z,
            self.mr_regime_enabled,
            self.engine_mode,
            self.allocator_policy,
            self.allocator_v1_adx_threshold,
            self.allocator_v2_trend_open_threshold,
            self.allocator_v2_rising_threshold,
            self.allocator_v2_rising_bars,
            self.allocator_v2_range_threshold,
            self.allocator_v2_range_bars,
            self.alloc_openproxy_or_width_atr,
            self.alloc_openproxy_impulse_atr,
            self.alloc_openproxy_persist_bars,
            self.alloc_openproxy_require_break,
            self.alloc_openproxy_enable_orb_selectivity_refinement,
            self.alloc_openproxy_low_atr_threshold,
            self.alloc_openproxy_min_persistence_in_low_atr,
            self.alloc_openproxy_high_impulse_threshold,
            self.alloc_openproxy_min_persistence_when_high_impulse,
            self.alloc_openproxy_medium_impulse_weak_persistence_filter_enabled,
            self.alloc_openproxy_medium_impulse_decay_filter_enabled,
            self.alloc_openproxy_medium_impulse_min_atr,
            self.alloc_openproxy_medium_impulse_max_atr,
            self.alloc_openproxy_medium_impulse_min,
            self.alloc_openproxy_medium_impulse_max,
            self.alloc_openproxy_medium_impulse_min_persistence,
            orb_enabled,
            self.orb_trigger_mode,
            self.orb_pullback_confirm_bars,
            self.orb_pullback_max_bars,
            self.orb_pullback_tolerance_pts,
            self.orb_pullback_entry_mode,
        )
        t0 = time.monotonic()

        # Redirect report export to the validation run's sessions/ folder
        config_module.ARTIFACTS_DIR = str(sessions_dir)

        try:
            retcode = run_debug_replay(args)
            elapsed = time.monotonic() - t0

            if retcode != 0:
                return SessionResult(
                    session_id=session.session_id,
                    success=False,
                    category=session.category,
                    error_message=f"run_debug_replay returned exit code {retcode}",
                    runtime_seconds=round(elapsed, 2),
                    artifact_dir=str(sessions_dir / session.session_id),
                )

            return SessionResult(
                session_id=session.session_id,
                success=True,
                category=session.category,
                runtime_seconds=round(elapsed, 2),
                artifact_dir=str(sessions_dir / session.session_id),
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.exception(
                "Session %s raised an exception", session.session_id
            )
            return SessionResult(
                session_id=session.session_id,
                success=False,
                category=session.category,
                error_message=f"{type(exc).__name__}: {exc}",
                runtime_seconds=round(elapsed, 2),
            )
        finally:
            # Always restore the original value
            config_module.ARTIFACTS_DIR = original_artifacts_dir

    @staticmethod
    def _build_replay_args(
        session: SessionEntry,
        batch_fast_mode: bool,
        mr_reclaim_mode: str,
        mr_sigma_entry: float,
        mr_soft_impulse_k: float,
        mr_dedupe_enabled: bool,
        mr_attempt_cap_enabled: bool,
        mr_cooldown_bars: int,
        mr_first_outside_enabled: bool,
        mr_touch_latch_reset_buffer: float,
        mr_dedupe_window_bars: int,
        mr_dedupe_min_delta_z: float,
        mr_regime_enabled: bool,
        engine_mode: str,
        allocator_policy: str,
        allocator_v1_adx_threshold: float,
        allocator_v2_trend_open_threshold: float,
        allocator_v2_rising_threshold: float,
        allocator_v2_rising_bars: int,
        allocator_v2_range_threshold: float,
        allocator_v2_range_bars: int,
        alloc_openproxy_or_width_atr: float,
        alloc_openproxy_impulse_atr: float,
        alloc_openproxy_persist_bars: int,
        alloc_openproxy_require_break: bool,
        alloc_openproxy_enable_orb_selectivity_refinement: bool,
        alloc_openproxy_low_atr_threshold: float,
        alloc_openproxy_min_persistence_in_low_atr: int,
        alloc_openproxy_high_impulse_threshold: float,
        alloc_openproxy_min_persistence_when_high_impulse: int,
        alloc_openproxy_medium_impulse_weak_persistence_filter_enabled: bool,
        alloc_openproxy_medium_impulse_decay_filter_enabled: bool,
        alloc_openproxy_medium_impulse_min_atr: float,
        alloc_openproxy_medium_impulse_max_atr: float,
        alloc_openproxy_medium_impulse_min: float,
        alloc_openproxy_medium_impulse_max: float,
        alloc_openproxy_medium_impulse_min_persistence: int,
        orb_enabled: bool,
        orb_trigger_mode: str,
        orb_pullback_confirm_bars: int,
        orb_pullback_max_bars: int,
        orb_pullback_tolerance_pts: float,
        orb_pullback_entry_mode: str,
    ) -> argparse.Namespace:
        """Build an ``argparse.Namespace`` compatible with ``run_debug_replay``."""
        return argparse.Namespace(
            start=session.start,
            end=session.end,
            symbol=session.symbol,
            max_ticks=None,
            update_every=9999,  # suppress chart updates (headless)
            pause=0.001,
            no_show=True,
            no_dashboard=batch_fast_mode,
            save_path="",
            session_id=session.session_id,
            no_report=False,
            mr_reclaim_mode=mr_reclaim_mode,
            mr_sigma_entry=mr_sigma_entry,
            mr_soft_impulse_k=mr_soft_impulse_k,
            mr_dedupe_enabled=("on" if mr_dedupe_enabled else "off"),
            mr_attempt_cap_enabled=("on" if mr_attempt_cap_enabled else "off"),
            mr_cooldown_bars=mr_cooldown_bars,
            mr_first_outside_enabled=("on" if mr_first_outside_enabled else "off"),
            mr_touch_latch_reset_buffer=mr_touch_latch_reset_buffer,
            mr_dedupe_window_bars=mr_dedupe_window_bars,
            mr_dedupe_min_delta_z=mr_dedupe_min_delta_z,
            mr_regime_enabled=("on" if mr_regime_enabled else "off"),
            engine_mode=engine_mode,
            allocator_policy=allocator_policy,
            allocator_v1_adx_threshold=allocator_v1_adx_threshold,
            allocator_v2_trend_open_threshold=allocator_v2_trend_open_threshold,
            allocator_v2_rising_threshold=allocator_v2_rising_threshold,
            allocator_v2_rising_bars=allocator_v2_rising_bars,
            allocator_v2_range_threshold=allocator_v2_range_threshold,
            allocator_v2_range_bars=allocator_v2_range_bars,
            orb_enabled=("on" if orb_enabled else "off"),
            orb_trigger_mode=orb_trigger_mode,
            orb_pullback_confirm_bars=orb_pullback_confirm_bars,
            orb_pullback_max_bars=orb_pullback_max_bars,
            orb_pullback_tolerance_pts=orb_pullback_tolerance_pts,
            orb_pullback_entry_mode=orb_pullback_entry_mode,
            alloc_openproxy_or_width_atr=alloc_openproxy_or_width_atr,
            alloc_openproxy_impulse_atr=alloc_openproxy_impulse_atr,
            alloc_openproxy_persist_bars=alloc_openproxy_persist_bars,
            alloc_openproxy_require_break=("on" if alloc_openproxy_require_break else "off"),
            alloc_openproxy_enable_orb_selectivity_refinement=("on" if alloc_openproxy_enable_orb_selectivity_refinement else "off"),
            alloc_openproxy_low_atr_threshold=alloc_openproxy_low_atr_threshold,
            alloc_openproxy_min_persistence_in_low_atr=alloc_openproxy_min_persistence_in_low_atr,
            alloc_openproxy_high_impulse_threshold=alloc_openproxy_high_impulse_threshold,
            alloc_openproxy_min_persistence_when_high_impulse=alloc_openproxy_min_persistence_when_high_impulse,
            alloc_openproxy_medium_impulse_weak_persistence_filter_enabled=("on" if alloc_openproxy_medium_impulse_weak_persistence_filter_enabled else "off"),
            alloc_openproxy_medium_impulse_decay_filter_enabled=("on" if alloc_openproxy_medium_impulse_decay_filter_enabled else "off"),
            alloc_openproxy_medium_impulse_min_atr=alloc_openproxy_medium_impulse_min_atr,
            alloc_openproxy_medium_impulse_max_atr=alloc_openproxy_medium_impulse_max_atr,
            alloc_openproxy_medium_impulse_min=alloc_openproxy_medium_impulse_min,
            alloc_openproxy_medium_impulse_max=alloc_openproxy_medium_impulse_max,
            alloc_openproxy_medium_impulse_min_persistence=alloc_openproxy_medium_impulse_min_persistence,
        )

    @staticmethod
    def _write_manifest(manifest: ValidationRunManifest, run_dir: Path) -> None:
        """Persist ``manifest.json`` in the run directory."""
        manifest_path = run_dir / "manifest.json"
        payload = asdict(manifest)
        manifest_path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("Manifest written → %s", manifest_path)
        print(f"\n  📄 Manifest written → {manifest_path}")
