"""
validation/promotion_gate.py — Promotion gate for validation sprint results.

Loads artifacts from a completed validation run (manifest, scorecard,
MC profile) and evaluates a set of gate checks that must all pass before
the build can be promoted.  Optionally compares against a baseline run
to detect regressions.

Usage:
    from validation.promotion_gate import run_promotion_gate, GateThresholds
    result = run_promotion_gate("artifacts/run_20260221", compare_to="artifacts/run_20260220")
    print(result.overall_pass)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Dataclasses
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GateThresholds:
    """Configurable thresholds for promotion gate checks."""

    min_session_success_rate: float = 0.9       # 90 % of sessions must succeed
    min_approval_rate: float | None = None       # optional minimum approval rate
    max_approval_rate: float | None = None       # optional maximum approval rate
    min_expectancy_r: float | None = None        # optional minimum expectancy in R
    max_p95_drawdown: float | None = None        # optional cap from MC results
    regression_tolerance: float = 0.05           # 5 % regression before flagging


@dataclass
class GateCheck:
    """Result of a single gate check."""

    name: str
    passed: bool
    current_value: float | str
    threshold: float | str
    notes: str = ""


@dataclass
class PromotionGateResult:
    """Aggregated result for all gate checks in a promotion evaluation."""

    run_id: str
    timestamp: str
    overall_pass: bool
    checks: list[GateCheck] = field(default_factory=list)
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists; return *None* otherwise."""
    if not path.exists():
        return None
    with path.open("r") as fh:
        return json.load(fh)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Coerce *value* to float, returning *default* on failure."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════════════
#  PromotionGate
# ═══════════════════════════════════════════════════════════════════════


class PromotionGate:
    """Evaluate promotion-readiness of a completed validation run.

    Parameters
    ----------
    run_dir:
        Path to the validation run directory (must contain manifest.json).
    thresholds:
        Gate thresholds.  Uses sensible defaults when *None*.
    compare_to_dir:
        Optional path to a prior run for regression detection.
    """

    def __init__(
        self,
        run_dir: str,
        thresholds: GateThresholds | None = None,
        compare_to_dir: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.thresholds = thresholds or GateThresholds()
        self.compare_to_dir = Path(compare_to_dir) if compare_to_dir else None

    # ── Public API ──────────────────────────────────────────────────────

    def evaluate(self) -> PromotionGateResult:
        """Run all gate checks and return a :class:`PromotionGateResult`."""
        checks: list[GateCheck] = []
        notes_parts: list[str] = []

        # -- Load artifacts --------------------------------------------------
        manifest = _load_json(self.run_dir / "manifest.json")
        # New run-root level artifacts
        aggregate_metrics = _load_json(self.run_dir / "aggregate_metrics.json")
        mc_profile = _load_json(self.run_dir / "mc_profile.json")
        mc_results = _load_json(self.run_dir / "mc_results.json")
        mc_results_mild = _load_json(self.run_dir / "mc_results_stress_mild.json")
        mc_results_severe = _load_json(self.run_dir / "mc_results_stress_severe.json")
        mc_results_tilt = _load_json(self.run_dir / "mc_results_stress_tilt_bad_week.json")

        # Legacy: fall back to scorecard/aggregate_metrics.json
        scorecard = aggregate_metrics or _load_json(
            self.run_dir / "scorecard" / "aggregate_metrics.json"
        )

        # Legacy: fall back to mc_profile/ dir
        if mc_profile is None:
            mc_profile = _load_json(
                self.run_dir / "mc_profile" / "replay_derived_profile.json"
            )

        run_id = ""
        if manifest:
            run_id = manifest.get("run_id", self.run_dir.name)
        else:
            run_id = self.run_dir.name
            notes_parts.append("manifest.json not found; using directory name as run_id")

        # -- 1. Required artifacts exist -----------------------------------
        checks.append(self._check_required_artifacts(
            aggregate_metrics, mc_profile, mc_results
        ))

        # -- 2. Session success rate ------------------------------------------
        checks.append(self._check_session_success_rate(manifest))

        # -- 3. Approval rate bounds ------------------------------------------
        checks.append(self._check_approval_rate(scorecard))

        # -- 4. Expectancy check (from aggregate_metrics) ---------------------
        checks.append(self._check_expectancy(aggregate_metrics))

        # -- 5. MC drawdown check (from mc_results) ---------------------------
        checks.append(self._check_mc_drawdown_from_results(mc_results))

        # -- 6. MC target probability (from mc_results) -----------------------
        checks.append(self._check_mc_target_prob(mc_results))

        # -- 7. MC ruin probability (from mc_results) -------------------------
        checks.append(self._check_mc_ruin_prob(mc_results))

        # -- 8. Minimum trade count -------------------------------------------
        checks.append(self._check_min_trade_count_agg(aggregate_metrics))

        # -- 9. Streak threshold (optional, from mc_results) ------------------
        checks.append(self._check_mc_streak(mc_results))

        # -- 10. Stress scenario checks (from stress result files) ----------
        checks.append(self._check_stress_mild_target(mc_results_mild))
        checks.append(self._check_stress_severe_target(mc_results_severe))
        checks.append(self._check_stress_severe_ruin(mc_results_severe))
        checks.append(self._check_stress_tilt_target(mc_results_tilt))

        # -- 11. Trade count warning for MC reliability --------------------
        checks.append(self._check_mc_trade_count_warning(aggregate_metrics))

        # -- 12. Artifact completeness (manifest) --------------------------
        checks.append(self._check_artifacts())

        # -- 13. Day-horizon MC check (when enabled) -----------------------
        checks.append(self._check_mc_day_horizon(mc_results))

        # -- 11. Regression checks (optional) ---------------------------------
        if self.compare_to_dir is not None:
            reg_checks = self._check_regressions()
            checks.extend(reg_checks)

        overall_pass = all(c.passed for c in checks)
        ts = datetime.now(timezone.utc).isoformat()

        result = PromotionGateResult(
            run_id=run_id,
            timestamp=ts,
            overall_pass=overall_pass,
            checks=checks,
            notes="; ".join(notes_parts) if notes_parts else "",
        )

        # Write gate_result.json
        self._write_gate_result(result)

        return result

    def write_summary(
        self,
        result: PromotionGateResult,
        output_path: str | None = None,
    ) -> str:
        """Write a Markdown promotion gate summary.

        Parameters
        ----------
        result:
            The evaluated gate result.
        output_path:
            File path for the summary.  Defaults to
            ``{run_dir}/promotion_gate_summary.md``.

        Returns
        -------
        str
            Absolute path to the written file.
        """
        dest = Path(output_path) if output_path else self.run_dir / "promotion_gate_summary.md"
        dest.parent.mkdir(parents=True, exist_ok=True)

        verdict = "PASS" if result.overall_pass else "FAIL"
        lines: list[str] = [
            "# Promotion Gate Summary",
            "",
            f"**Run:** {result.run_id}",
            f"**Timestamp:** {result.timestamp}",
            f"**Result:** {verdict}",
            "",
            "## Gate Checks",
            "",
            "| Check | Result | Value | Threshold | Notes |",
            "|-------|--------|-------|-----------|-------|",
        ]

        for chk in result.checks:
            r = "PASS" if chk.passed else "FAIL"
            lines.append(
                f"| {chk.name} | {r} | {chk.current_value} | {chk.threshold} | {chk.notes} |"
            )

        # Regression section
        regression_checks = [c for c in result.checks if c.name.startswith("regression:")]
        if regression_checks:
            lines.append("")
            lines.append("## Regressions")
            lines.append("")
            for rc in regression_checks:
                status = "OK" if rc.passed else "FLAGGED"
                lines.append(f"- **{rc.name}**: {status} — {rc.notes}")

        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append(result.notes if result.notes else "_(none)_")
        lines.append("")

        dest.write_text("\n".join(lines), encoding="utf-8")
        abs_path = str(dest.resolve())
        logger.info("Promotion gate summary written → %s", abs_path)
        return abs_path

    # ── Individual gate checks ──────────────────────────────────────────

    def _check_required_artifacts(
        self,
        aggregate_metrics: dict[str, Any] | None,
        mc_profile: dict[str, Any] | None,
        mc_results: dict[str, Any] | None,
    ) -> GateCheck:
        """Gate 0: required run-root artifacts must exist."""
        name = "required_artifacts"
        missing: list[str] = []
        if aggregate_metrics is None:
            missing.append("aggregate_metrics.json")
        if mc_profile is None:
            missing.append("mc_profile.json")
        if mc_results is None:
            missing.append("mc_results.json")

        if missing:
            return GateCheck(
                name=name,
                passed=False,
                current_value=f"missing {len(missing)}",
                threshold="all present",
                notes=f"Missing: {', '.join(missing)}",
            )
        return GateCheck(
            name=name,
            passed=True,
            current_value="all present",
            threshold="all present",
            notes="aggregate_metrics.json, mc_profile.json, mc_results.json found",
        )

    def _write_gate_result(self, result: PromotionGateResult) -> None:
        """Persist ``gate_result.json`` at run root."""
        out_path = self.run_dir / "gate_result.json"
        payload = {
            "run_id": result.run_id,
            "timestamp": result.timestamp,
            "overall_pass": result.overall_pass,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "current_value": c.current_value,
                    "threshold": c.threshold,
                    "notes": c.notes,
                }
                for c in result.checks
            ],
            "notes": result.notes,
        }
        out_path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote gate_result.json → %s", out_path)

    def _check_session_success_rate(self, manifest: dict[str, Any] | None) -> GateCheck:
        """Gate 1: session success rate ≥ threshold."""
        name = "session_success_rate"
        threshold = self.thresholds.min_session_success_rate

        if manifest is None:
            return GateCheck(
                name=name,
                passed=False,
                current_value="N/A",
                threshold=threshold,
                notes="manifest.json missing",
            )

        sessions = manifest.get("sessions", [])
        if not sessions:
            return GateCheck(
                name=name,
                passed=False,
                current_value="N/A",
                threshold=threshold,
                notes="No sessions recorded in manifest",
            )

        successes = sum(1 for s in sessions if s.get("success", False))
        rate = successes / len(sessions)

        return GateCheck(
            name=name,
            passed=rate >= threshold,
            current_value=round(rate, 4),
            threshold=threshold,
            notes=f"{successes}/{len(sessions)} sessions succeeded",
        )

    def _check_approval_rate(self, scorecard: dict[str, Any] | None) -> GateCheck:
        """Gate 2: approval rate within configured bounds."""
        name = "approval_rate"
        min_t = self.thresholds.min_approval_rate
        max_t = self.thresholds.max_approval_rate

        if min_t is None and max_t is None:
            return GateCheck(
                name=name,
                passed=True,
                current_value="N/A",
                threshold="no bounds set",
                notes="Approval rate bounds not configured; skipped",
            )

        if scorecard is None:
            return GateCheck(
                name=name,
                passed=True,
                current_value="N/A",
                threshold=f"min={min_t}, max={max_t}",
                notes="Scorecard missing; check skipped (degraded)",
            )

        rate = _safe_float(scorecard.get("approval_rate"))
        if rate is None:
            return GateCheck(
                name=name,
                passed=True,
                current_value="N/A",
                threshold=f"min={min_t}, max={max_t}",
                notes="approval_rate not in scorecard; check skipped",
            )

        passed = True
        notes_parts: list[str] = []
        if min_t is not None and rate < min_t:
            passed = False
            notes_parts.append(f"below minimum ({rate:.4f} < {min_t})")
        if max_t is not None and rate > max_t:
            passed = False
            notes_parts.append(f"above maximum ({rate:.4f} > {max_t})")

        return GateCheck(
            name=name,
            passed=passed,
            current_value=round(rate, 4),
            threshold=f"min={min_t}, max={max_t}",
            notes="; ".join(notes_parts) if notes_parts else "within bounds",
        )

    def _check_expectancy(self, scorecard: dict[str, Any] | None) -> GateCheck:
        """Gate 3: expectancy in R ≥ threshold (when trades available)."""
        name = "expectancy_r"
        threshold = self.thresholds.min_expectancy_r

        if threshold is None:
            return GateCheck(
                name=name,
                passed=True,
                current_value="N/A",
                threshold="not set",
                notes="Expectancy threshold not configured; skipped",
            )

        if scorecard is None:
            return GateCheck(
                name=name,
                passed=True,
                current_value="N/A",
                threshold=threshold,
                notes="Scorecard missing; check skipped (degraded)",
            )

        expectancy = _safe_float(scorecard.get("expectancy_r"))
        if expectancy is None:
            return GateCheck(
                name=name,
                passed=True,
                current_value="N/A",
                threshold=threshold,
                notes="expectancy_r not in scorecard; skipped",
            )

        return GateCheck(
            name=name,
            passed=expectancy >= threshold,
            current_value=round(expectancy, 4),
            threshold=threshold,
            notes="" if expectancy >= threshold else f"below threshold ({expectancy:.4f} < {threshold})",
        )

    def _check_mc_drawdown(self, mc_profile: dict[str, Any] | None) -> GateCheck:
        """Gate 4: MC p95 drawdown ≤ threshold."""
        name = "mc_p95_drawdown"
        threshold = self.thresholds.max_p95_drawdown

        if threshold is None:
            return GateCheck(
                name=name,
                passed=True,
                current_value="N/A",
                threshold="not set",
                notes="MC drawdown threshold not configured; skipped",
            )

        if mc_profile is None:
            return GateCheck(
                name=name,
                passed=True,
                current_value="N/A",
                threshold=threshold,
                notes="MC profile missing; check skipped (degraded)",
            )

        dd = _safe_float(mc_profile.get("p95_max_drawdown"))
        if dd is None:
            return GateCheck(
                name=name,
                passed=True,
                current_value="N/A",
                threshold=threshold,
                notes="p95_max_drawdown not in MC profile; skipped",
            )

        return GateCheck(
            name=name,
            passed=dd <= threshold,
            current_value=round(dd, 2),
            threshold=threshold,
            notes="" if dd <= threshold else f"exceeds threshold ({dd:.2f} > {threshold})",
        )

    def _check_min_trade_count(self, mc_profile: dict[str, Any] | None) -> GateCheck:
        """Gate 4.5: MC profile must have at least MIN_TRADE_COUNT trades."""
        import config
        name = "min_trade_count"
        threshold = config.MC_PROFILE_MIN_TRADE_COUNT

        if mc_profile is None:
            return GateCheck(
                name=name,
                passed=False,
                current_value="N/A",
                threshold=threshold,
                notes="MC profile missing; cannot verify trade count",
            )

        count = mc_profile.get("sample_size_trades")
        if count is None:
            return GateCheck(
                name=name,
                passed=False,
                current_value="N/A",
                threshold=threshold,
                notes="sample_size_trades not in MC profile",
            )

        return GateCheck(
            name=name,
            passed=count >= threshold,
            current_value=count,
            threshold=threshold,
            notes="insufficient trades" if count < threshold else "sufficient trades",
        )

    # ── New MC-results-driven checks ────────────────────────────────────

    def _check_mc_drawdown_from_results(self, mc_results: dict[str, Any] | None) -> GateCheck:
        """Gate: dd_p95 from mc_results.json ≤ PROMOTION_MC_MAX_DD_P95."""
        import config
        name = "mc_dd_p95"
        threshold = config.PROMOTION_MC_MAX_DD_P95

        if mc_results is None:
            return GateCheck(
                name=name, passed=False, current_value="N/A",
                threshold=threshold, notes="mc_results.json missing",
            )
        dd = _safe_float(mc_results.get("dd_p95"))
        if dd is None:
            return GateCheck(
                name=name, passed=False, current_value="N/A",
                threshold=threshold, notes="dd_p95 not in mc_results",
            )
        return GateCheck(
            name=name, passed=dd <= threshold,
            current_value=round(dd, 2), threshold=threshold,
            notes="" if dd <= threshold else f"exceeds ({dd:.2f} > {threshold})",
        )

    def _check_mc_target_prob(self, mc_results: dict[str, Any] | None) -> GateCheck:
        """Gate: p_target_before_ruin ≥ MC_TARGET_THRESHOLD."""
        import config
        name = "mc_target_prob"
        threshold = config.MC_TARGET_THRESHOLD

        if mc_results is None:
            return GateCheck(
                name=name, passed=False, current_value="N/A",
                threshold=threshold, notes="mc_results.json missing",
            )
        prob = _safe_float(mc_results.get("p_target_before_ruin"))
        if prob is None:
            return GateCheck(
                name=name, passed=False, current_value="N/A",
                threshold=threshold, notes="p_target_before_ruin not in mc_results",
            )
        return GateCheck(
            name=name, passed=prob >= threshold,
            current_value=round(prob, 4), threshold=threshold,
            notes="" if prob >= threshold else f"below ({prob:.4f} < {threshold})",
        )

    def _check_mc_ruin_prob(self, mc_results: dict[str, Any] | None) -> GateCheck:
        """Gate: p_ruin ≤ MC_RUIN_THRESHOLD."""
        import config
        name = "mc_ruin_prob"
        threshold = config.MC_RUIN_THRESHOLD

        if mc_results is None:
            return GateCheck(
                name=name, passed=False, current_value="N/A",
                threshold=threshold, notes="mc_results.json missing",
            )
        prob = _safe_float(mc_results.get("p_ruin"))
        if prob is None:
            return GateCheck(
                name=name, passed=False, current_value="N/A",
                threshold=threshold, notes="p_ruin not in mc_results",
            )
        return GateCheck(
            name=name, passed=prob <= threshold,
            current_value=round(prob, 4), threshold=threshold,
            notes="" if prob <= threshold else f"exceeds ({prob:.4f} > {threshold})",
        )

    def _check_min_trade_count_agg(self, aggregate_metrics: dict[str, Any] | None) -> GateCheck:
        """Gate: trade_count_total from aggregate_metrics ≥ MC_PROFILE_MIN_TRADE_COUNT."""
        import config
        name = "min_trade_count"
        threshold = config.MC_PROFILE_MIN_TRADE_COUNT

        if aggregate_metrics is None:
            return GateCheck(
                name=name, passed=False, current_value="N/A",
                threshold=threshold, notes="aggregate_metrics.json missing",
            )
        count = aggregate_metrics.get("trade_count_total")
        if count is None:
            return GateCheck(
                name=name, passed=False, current_value="N/A",
                threshold=threshold, notes="trade_count_total not in aggregate_metrics",
            )
        return GateCheck(
            name=name, passed=int(count) >= threshold,
            current_value=int(count), threshold=threshold,
            notes="insufficient trades" if int(count) < threshold else "sufficient trades",
        )

    def _check_mc_streak(self, mc_results: dict[str, Any] | None) -> GateCheck:
        """Gate: losing_streak_p95 < MC_LOSING_STREAK_P95_MAX."""
        import config
        name = "mc_losing_streak_p95"
        threshold = config.MC_LOSING_STREAK_P95_MAX

        if mc_results is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold=threshold, notes="mc_results.json missing; skipped",
            )
        streak = _safe_float(mc_results.get("losing_streak_p95"))
        if streak is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold=threshold, notes="losing_streak_p95 not in mc_results; skipped",
            )
        return GateCheck(
            name=name, passed=streak < threshold,
            current_value=round(streak, 1), threshold=threshold,
            notes="" if streak < threshold else f"exceeds ({streak:.1f} >= {threshold})",
        )

    # ── Stress scenario checks ──────────────────────────────────────────

    def _check_stress_mild_target(
        self, mc_results_mild: dict[str, Any] | None,
    ) -> GateCheck:
        """Gate: p_target under mild stress — two-tier CI gate.

        Tier 1: median(batch p_target) >= threshold
        Tier 2: Wilson CI lower bound >= MC_STRESS_SEVERE_FLOOR
        """
        import config
        name = "stress_mild_target_prob"
        threshold = config.MC_STRESS_MILD_TARGET_THRESHOLD
        floor = config.MC_STRESS_SEVERE_FLOOR

        if mc_results_mild is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold=threshold,
                notes="mc_results_stress_mild.json not found; skipped",
            )
        prob = _safe_float(mc_results_mild.get("p_target_before_ruin"))
        if prob is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold=threshold, notes="p_target_before_ruin not in mild stress results",
            )

        # Try CI-aware check
        ci_lo = _safe_float(
            (mc_results_mild.get("p_target_ci") or {}).get("lower")
        )
        batch_median = _safe_float(
            (mc_results_mild.get("p_target_batch_spread") or {}).get("median")
        )
        batch_min = _safe_float(
            (mc_results_mild.get("p_target_batch_spread") or {}).get("min")
        )
        batch_max = _safe_float(
            (mc_results_mild.get("p_target_batch_spread") or {}).get("max")
        )

        # Use batch_median if available, else point estimate
        effective_p = batch_median if batch_median is not None else prob

        # Tier 1: median >= threshold
        tier1_pass = effective_p >= threshold
        # Tier 2: CI lower >= floor (only checked if CI available)
        tier2_pass = ci_lo >= floor if ci_lo is not None else True
        passed = tier1_pass and tier2_pass

        notes_parts: list[str] = []
        if not tier1_pass:
            notes_parts.append(f"median {effective_p:.4f} < {threshold}")
        if ci_lo is not None and not tier2_pass:
            notes_parts.append(f"CI lower {ci_lo:.4f} < floor {floor}")
        if ci_lo is not None:
            notes_parts.append(f"CI=[{ci_lo:.4f}, {_safe_float((mc_results_mild.get('p_target_ci') or {}).get('upper'), 0):.4f}]")
        if batch_min is not None and batch_max is not None:
            notes_parts.append(f"batch=[{batch_min:.4f}, {batch_max:.4f}]")

        return GateCheck(
            name=name, passed=passed,
            current_value=round(effective_p, 4), threshold=threshold,
            notes="; ".join(notes_parts),
        )

    def _check_stress_severe_target(
        self, mc_results_severe: dict[str, Any] | None,
    ) -> GateCheck:
        """Gate: p_target under severe stress — two-tier CI gate.

        Tier 1: median(batch p_target) >= threshold
        Tier 2: Wilson CI lower bound >= MC_STRESS_SEVERE_FLOOR
        """
        import config
        name = "stress_severe_target_prob"
        threshold = config.MC_STRESS_SEVERE_TARGET_THRESHOLD
        floor = config.MC_STRESS_SEVERE_FLOOR

        if mc_results_severe is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold=threshold,
                notes="mc_results_stress_severe.json not found; skipped",
            )
        prob = _safe_float(mc_results_severe.get("p_target_before_ruin"))
        if prob is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold=threshold, notes="p_target_before_ruin not in severe stress results",
            )

        # Try CI-aware check
        ci_lo = _safe_float(
            (mc_results_severe.get("p_target_ci") or {}).get("lower")
        )
        batch_median = _safe_float(
            (mc_results_severe.get("p_target_batch_spread") or {}).get("median")
        )
        batch_min = _safe_float(
            (mc_results_severe.get("p_target_batch_spread") or {}).get("min")
        )
        batch_max = _safe_float(
            (mc_results_severe.get("p_target_batch_spread") or {}).get("max")
        )

        effective_p = batch_median if batch_median is not None else prob

        # Tier 1: median >= threshold
        tier1_pass = effective_p >= threshold
        # Tier 2: CI lower >= floor
        tier2_pass = ci_lo >= floor if ci_lo is not None else True
        passed = tier1_pass and tier2_pass

        notes_parts: list[str] = []
        if not tier1_pass:
            notes_parts.append(f"median {effective_p:.4f} < {threshold}")
        if ci_lo is not None and not tier2_pass:
            notes_parts.append(f"CI lower {ci_lo:.4f} < floor {floor}")
        if ci_lo is not None:
            notes_parts.append(f"CI=[{ci_lo:.4f}, {_safe_float((mc_results_severe.get('p_target_ci') or {}).get('upper'), 0):.4f}]")
        if batch_min is not None and batch_max is not None:
            notes_parts.append(f"batch=[{batch_min:.4f}, {batch_max:.4f}]")

        return GateCheck(
            name=name, passed=passed,
            current_value=round(effective_p, 4), threshold=threshold,
            notes="; ".join(notes_parts),
        )

    def _check_stress_tilt_target(
        self, mc_results_tilt: dict[str, Any] | None,
    ) -> GateCheck:
        """Advisory gate: tilt_bad_week p_target (always passes, just logs)."""
        name = "stress_tilt_target_prob"

        if mc_results_tilt is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold="advisory",
                notes="mc_results_stress_tilt_bad_week.json not found; skipped",
            )
        prob = _safe_float(mc_results_tilt.get("p_target_before_ruin"))
        if prob is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold="advisory",
                notes="p_target_before_ruin not in tilt results",
            )

        ci_lo = _safe_float(
            (mc_results_tilt.get("p_target_ci") or {}).get("lower")
        )
        ci_hi = _safe_float(
            (mc_results_tilt.get("p_target_ci") or {}).get("upper")
        )
        notes = f"p_target={prob:.4f}"
        if ci_lo is not None and ci_hi is not None:
            notes += f", CI=[{ci_lo:.4f}, {ci_hi:.4f}]"

        return GateCheck(
            name=name, passed=True,  # advisory — never blocks promotion
            current_value=round(prob, 4), threshold="advisory",
            notes=notes,
        )

    def _check_stress_severe_ruin(
        self, mc_results_severe: dict[str, Any] | None,
    ) -> GateCheck:
        """Gate: p_ruin under severe stress <= MC_RUIN_THRESHOLD."""
        import config
        name = "stress_severe_ruin_prob"
        threshold = config.MC_RUIN_THRESHOLD

        if mc_results_severe is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold=threshold,
                notes="mc_results_stress_severe.json not found; skipped",
            )
        prob = _safe_float(mc_results_severe.get("p_ruin"))
        if prob is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold=threshold, notes="p_ruin not in severe stress results",
            )
        return GateCheck(
            name=name, passed=prob <= threshold,
            current_value=round(prob, 4), threshold=threshold,
            notes="" if prob <= threshold else f"high ruin under severe stress ({prob:.4f} > {threshold})",
        )

    def _check_mc_trade_count_warning(
        self, aggregate_metrics: dict[str, Any] | None,
    ) -> GateCheck:
        """Warning gate: trade count below VALIDATION_MIN_TRADES_FOR_MC."""
        import config
        name = "mc_trade_count_warning"
        threshold = config.VALIDATION_MIN_TRADES_FOR_MC

        if aggregate_metrics is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold=f">={threshold} (advisory)",
                notes="aggregate_metrics missing; skipped",
            )
        count = aggregate_metrics.get("trade_count_total", 0)
        # This is a warning — always passes but notes the shortfall
        is_sufficient = int(count) >= threshold
        return GateCheck(
            name=name, passed=True,  # advisory, does not fail gate
            current_value=int(count), threshold=f">={threshold} (advisory)",
            notes="" if is_sufficient else f"WARNING: only {count} trades — MC results may be unreliable (need >={threshold})",
        )

    def _check_artifacts(self) -> GateCheck:
        """Gate 5: expected artifact files exist."""
        name = "artifact_completeness"

        expected_files = [
            self.run_dir / "manifest.json",
        ]

        missing: list[str] = []
        for f in expected_files:
            if not f.exists():
                missing.append(f.name)

        if missing:
            return GateCheck(
                name=name,
                passed=False,
                current_value=f"{len(expected_files) - len(missing)}/{len(expected_files)}",
                threshold="all present",
                notes=f"Missing: {', '.join(missing)}",
            )

        return GateCheck(
            name=name,
            passed=True,
            current_value=f"{len(expected_files)}/{len(expected_files)}",
            threshold="all present",
            notes="All expected artifacts found",
        )

    # ── Day-horizon MC gate ─────────────────────────────────────────────

    def _check_mc_day_horizon(self, mc_results: dict[str, Any] | None) -> GateCheck:
        """Gate: when day-horizon mode is active, verify days_to_target_median ≤ MC_MAX_DAYS.

        This is an informational/soft gate — the hard pass/fail is still
        controlled by p_target_before_ruin.  But we flag if the median
        path needs more days than the budget.
        """
        import config
        name = "mc_day_horizon"

        if not config.MC_DAY_HORIZON_ENABLED:
            return GateCheck(
                name=name, passed=True, current_value="disabled",
                threshold="N/A", notes="Day-horizon mode not enabled",
            )

        if mc_results is None:
            return GateCheck(
                name=name, passed=False, current_value="N/A",
                threshold=config.MC_MAX_DAYS,
                notes="mc_results.json missing",
            )

        dh = mc_results.get("day_horizon", {})
        median_days = _safe_float(dh.get("days_to_target_median"))
        max_days = config.MC_MAX_DAYS

        if median_days is None:
            return GateCheck(
                name=name, passed=True, current_value="N/A",
                threshold=max_days,
                notes="days_to_target_median not present (possibly old results)",
            )

        passed = median_days <= max_days
        return GateCheck(
            name=name, passed=passed,
            current_value=round(median_days, 1), threshold=max_days,
            notes=(
                f"Median path reaches target in {median_days:.1f} days"
                if passed
                else f"Median {median_days:.1f} days exceeds {max_days}-day budget"
            ),
        )

    # ── Regression detection ────────────────────────────────────────────

    def _check_regressions(self) -> list[GateCheck]:
        """Compare current scorecard to baseline and flag regressions."""
        assert self.compare_to_dir is not None
        checks: list[GateCheck] = []
        tol = self.thresholds.regression_tolerance

        prior_scorecard = _load_json(
            self.compare_to_dir / "scorecard" / "aggregate_metrics.json"
        )
        current_scorecard = _load_json(
            self.run_dir / "scorecard" / "aggregate_metrics.json"
        )

        if prior_scorecard is None or current_scorecard is None:
            checks.append(
                GateCheck(
                    name="regression:scorecard_comparison",
                    passed=True,
                    current_value="N/A",
                    threshold=f"tolerance={tol}",
                    notes="One or both scorecards missing; regression check skipped",
                )
            )
            return checks

        # -- Approval rate regression -----------------------------------------
        prior_ar = _safe_float(prior_scorecard.get("approval_rate"))
        current_ar = _safe_float(current_scorecard.get("approval_rate"))

        if prior_ar is not None and current_ar is not None:
            delta = prior_ar - current_ar
            regressed = delta > tol
            checks.append(
                GateCheck(
                    name="regression:approval_rate",
                    passed=not regressed,
                    current_value=round(current_ar, 4),
                    threshold=f"prior={round(prior_ar, 4)}, tol={tol}",
                    notes=(
                        f"decreased by {delta:.4f} (>{tol})"
                        if regressed
                        else f"delta={delta:.4f}, within tolerance"
                    ),
                )
            )

        # -- Expectancy regression --------------------------------------------
        prior_exp = _safe_float(prior_scorecard.get("expectancy_r"))
        current_exp = _safe_float(current_scorecard.get("expectancy_r"))

        if prior_exp is not None and current_exp is not None:
            delta = prior_exp - current_exp
            regressed = delta > tol
            checks.append(
                GateCheck(
                    name="regression:expectancy_r",
                    passed=not regressed,
                    current_value=round(current_exp, 4),
                    threshold=f"prior={round(prior_exp, 4)}, tol={tol}",
                    notes=(
                        f"decreased by {delta:.4f} (>{tol})"
                        if regressed
                        else f"delta={delta:.4f}, within tolerance"
                    ),
                )
            )

        return checks


# ═══════════════════════════════════════════════════════════════════════
#  Standalone convenience function
# ═══════════════════════════════════════════════════════════════════════


def run_promotion_gate(
    run_dir: str,
    compare_to: str | None = None,
    thresholds: GateThresholds | None = None,
) -> PromotionGateResult:
    """Evaluate promotion readiness for a validation run.

    Parameters
    ----------
    run_dir:
        Path to the completed validation run directory.
    compare_to:
        Optional path to a prior run for regression comparison.
    thresholds:
        Custom gate thresholds.  Uses defaults when *None*.

    Returns
    -------
    PromotionGateResult
        Aggregated pass/fail with individual check details.
    """
    gate = PromotionGate(
        run_dir=run_dir,
        thresholds=thresholds,
        compare_to_dir=compare_to,
    )
    result = gate.evaluate()
    gate.write_summary(result)
    return result
