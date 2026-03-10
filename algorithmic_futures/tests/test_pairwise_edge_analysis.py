"""Tests for pairwise conditional edge analysis helpers."""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from validation.pairwise_edge_analysis import (  # noqa: E402
    analyze_pairwise_slices,
    qbucket,
    rank_pairwise_rows,
)


def _synthetic_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    # ORB slices
    for impulse, persistence, atr_pct, vol_pct, pnl in [
        (0.2, 0, 10, 10, -20),
        (0.3, 0, 20, 20, -10),
        (1.8, 1, 30, 30, 80),
        (1.9, 1, 35, 35, 70),
        (2.2, 2, 80, 80, 90),
        (2.1, 2, 85, 85, 100),
    ]:
        rows.append(
            {
                "trade_type": "ORB",
                "pnl_dollars": pnl,
                "impulse": impulse,
                "persistence": persistence,
                "atr_percentile": atr_pct,
                "vol_percentile": vol_pct,
                "distance_from_vwap_atr": None,
                "one_sidedness": impulse,
                "confidence_score": 0.5 + impulse / 10,
            }
        )
    # MR slices
    for dist, persistence, atr_pct, vol_pct, pnl in [
        (0.4, 0, 15, 10, -5),
        (0.5, 0, 20, 20, 5),
        (1.5, 1, 35, 30, 20),
        (1.6, 1, 40, 40, 30),
        (2.5, 2, 75, 80, 60),
        (2.6, 2, 85, 85, 70),
    ]:
        rows.append(
            {
                "trade_type": "MR",
                "pnl_dollars": pnl,
                "impulse": None,
                "persistence": persistence,
                "atr_percentile": atr_pct,
                "vol_percentile": vol_pct,
                "distance_from_vwap_atr": dist,
                "one_sidedness": None,
                "confidence_score": 0.3 + dist / 10,
            }
        )
    return pd.DataFrame(rows)


def test_qbucket_is_deterministic_for_same_input():
    series = pd.Series([1, 2, 3, 4, 5, 6])
    first = qbucket(series, ["low", "medium", "high"])
    second = qbucket(series, ["low", "medium", "high"])
    assert first.tolist() == second.tolist()


def test_sample_size_threshold_flags_apply_correctly():
    rows, _, _ = analyze_pairwise_slices(_synthetic_dataset(), reporting_min_sample=2, candidate_min_sample=3)
    target = next(
        row
        for row in rows
        if row["trade_type"] == "ORB"
        and row["feature_1"] == "opening_impulse"
        and row["feature_2"] == "persistence"
        and row["feature_1_bucket"] == "high"
        and row["feature_2_bucket"] == "strong"
    )
    assert target["sample_size"] == 2
    assert target["reporting_sample_ok"] is True
    assert target["candidate_sample_ok"] is False


def test_ranking_is_stable_on_small_fixture():
    rows, _, _ = analyze_pairwise_slices(_synthetic_dataset(), reporting_min_sample=2, candidate_min_sample=2)
    ranked = rank_pairwise_rows(rows, "ORB", reporting_min_sample=2)
    assert ranked[0]["feature_1"] == "opening_impulse"
    assert ranked[0]["feature_1_bucket"] == "high"
    assert ranked[0]["positive_expectancy"] is True


def test_missing_optional_columns_are_handled_gracefully():
    dataset = _synthetic_dataset().drop(columns=["one_sidedness"])
    rows, bucket_defs, mr_proxy = analyze_pairwise_slices(dataset, reporting_min_sample=1, candidate_min_sample=2)
    assert rows
    assert mr_proxy == "persistence"
    assert "one_sidedness" not in bucket_defs


def test_output_schema_is_stable():
    rows, _, _ = analyze_pairwise_slices(_synthetic_dataset(), reporting_min_sample=1, candidate_min_sample=2)
    expected = {
        "trade_type",
        "feature_1",
        "feature_1_bucket",
        "feature_2",
        "feature_2_bucket",
        "sample_size",
        "mean_trade_pnl",
        "median_trade_pnl",
        "win_rate",
        "avg_win",
        "avg_loss",
        "std_dev",
        "variance",
        "skewness",
        "positive_expectancy",
        "reporting_sample_ok",
        "candidate_sample_ok",
        "research_score",
    }
    assert rows
    assert set(rows[0].keys()) == expected
