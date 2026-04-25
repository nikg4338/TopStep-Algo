"""
tests/test_orb_selectivity_experiment.py — helper tests for ORB selectivity.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.run_orb_selectivity_experiment import (
    FilterSpec,
    build_filter_specs,
    evaluate_filter,
    max_drawdown,
    rank_results,
    render_report,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "opening_range_width": [10.0, 12.0, 30.0, 32.0],
            "atr": [5.0, 6.0, 15.0, 16.0],
            "atr_regime": ["low", "low", "medium", "high"],
            "opening_impulse": [0.8, 0.6, -0.2, -0.1],
            "one_sidedness_score": [0.8, 0.6, 0.2, 0.1],
            "vwap_relationship": ["above_vwap", "above_vwap", "below_vwap", "below_vwap"],
            "pullback_depth": [0.0, 0.25, 5.0, 6.0],
            "breakout_direction": ["BUY", "BUY", "SELL", "SELL"],
            "max_favorable_excursion": [1.5, 1.2, 0.2, 0.1],
            "max_adverse_excursion": [0.2, 0.3, 1.2, 1.4],
            "final_r": [1.0, 0.5, -1.0, -0.5],
            "label": ["good_orb", "good_orb", "bad_orb", "bad_orb"],
        }
    )


def test_max_drawdown_tracks_peak_to_trough() -> None:
    assert max_drawdown([1.0, -0.5, -1.0, 2.0]) == 1.5


def test_evaluate_filter_computes_metrics_and_low_confidence() -> None:
    df = _sample_df()
    spec = FilterSpec(
        name="strong_impulse",
        description="impulse >= 0.5",
        fn=lambda d: d["opening_impulse"] >= 0.5,
    )

    result = evaluate_filter(
        df,
        spec,
        min_trades=3,
        dollars_per_r=100,
        combine_target_dollars=3000,
    )

    assert result.allowed == 2
    assert result.rejected == 2
    assert result.win_rate == 1.0
    assert result.avg_r == 0.75
    assert result.avg_mfe_r == 1.35
    assert result.target_contribution_pct == 0.05
    assert result.low_confidence is True


def test_build_filter_specs_includes_requested_filter_families() -> None:
    names = {spec.name for spec in build_filter_specs(_sample_df())}

    assert "or_width_le_median" in names
    assert "atr_regime_low" in names
    assert "abs_impulse_ge_median" in names
    assert "one_sidedness_ge_median" in names
    assert "vwap_above_vwap" in names
    assert "pullback_touch_or_none" in names


def test_rank_results_prefers_confident_positive_total_r() -> None:
    df = _sample_df()
    all_spec = FilterSpec("all", "all", lambda d: d["final_r"].notna())
    small_spec = FilterSpec("small", "small", lambda d: d["final_r"] > 0)

    all_result = evaluate_filter(
        df,
        all_spec,
        min_trades=3,
        dollars_per_r=100,
        combine_target_dollars=3000,
    )
    small_result = evaluate_filter(
        df,
        small_spec,
        min_trades=3,
        dollars_per_r=100,
        combine_target_dollars=3000,
    )

    ranked = rank_results([small_result, all_result])

    assert ranked[0].name == "all"


def test_render_report_contains_top3_and_overfitting_warning(tmp_path: Path) -> None:
    df = _sample_df()
    results = [
        evaluate_filter(
            df,
            FilterSpec("all_orb_trades", "all", lambda d: d["final_r"].notna()),
            min_trades=30,
            dollars_per_r=100,
            combine_target_dollars=3000,
        )
    ]

    report = render_report(
        source_csv=tmp_path / "orb_diagnostics.csv",
        df=df,
        results=results,
        min_trades=30,
        dollars_per_r=100,
        combine_target_dollars=3000,
    )

    assert "Top 3 Candidate Filters" in report
    assert "Overfitting Warning" in report
    assert "LOW-confidence" in report
