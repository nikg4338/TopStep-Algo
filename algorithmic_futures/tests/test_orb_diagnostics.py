"""
tests/test_orb_diagnostics.py — helper tests for ORB diagnostic labeling.
"""

from __future__ import annotations

import csv
from pathlib import Path

from experiments.run_orb_diagnostics import (
    OrbDiagnosticRow,
    OrbLabelThresholds,
    classify_atr_regime,
    classify_vwap_relationship,
    label_counts,
    label_orb_result,
    rows_from_autopsy_csv,
    rows_to_markdown,
)


def test_label_orb_result_good_from_mfe_before_exit() -> None:
    label = label_orb_result(
        has_valid_setup=True,
        final_r=-0.1,
        max_favorable_r=1.2,
        max_adverse_r=0.4,
    )

    assert label == "good_orb"


def test_label_orb_result_bad_from_stop_or_adverse_excursion() -> None:
    assert (
        label_orb_result(
            has_valid_setup=True,
            final_r=-1.0,
            max_favorable_r=0.2,
            max_adverse_r=0.5,
        )
        == "bad_orb"
    )
    assert (
        label_orb_result(
            has_valid_setup=True,
            final_r=0.1,
            max_favorable_r=0.3,
            max_adverse_r=1.1,
        )
        == "bad_orb"
    )


def test_label_orb_result_neutral_and_no_trade() -> None:
    assert (
        label_orb_result(
            has_valid_setup=True,
            final_r=0.1,
            max_favorable_r=0.4,
            max_adverse_r=0.3,
        )
        == "neutral_orb"
    )
    assert (
        label_orb_result(
            has_valid_setup=False,
            final_r=2.0,
            max_favorable_r=2.0,
            max_adverse_r=0.0,
        )
        == "no-trade"
    )


def test_label_thresholds_are_configurable() -> None:
    label = label_orb_result(
        has_valid_setup=True,
        final_r=0.6,
        max_favorable_r=0.7,
        max_adverse_r=0.2,
        thresholds=OrbLabelThresholds(good_r=0.5),
    )

    assert label == "good_orb"


def test_atr_and_vwap_classifiers() -> None:
    assert classify_atr_regime(atr=20.0, atr_percentile=20.0) == "low"
    assert classify_atr_regime(atr=20.0, atr_percentile=50.0) == "medium"
    assert classify_atr_regime(atr=5.0, atr_percentile=80.0) == "high"
    assert classify_atr_regime(atr=None, atr_percentile=None) == "unknown"

    assert classify_vwap_relationship(101.0, 100.0) == "above_vwap"
    assert classify_vwap_relationship(99.0, 100.0) == "below_vwap"
    assert classify_vwap_relationship(100.1, 100.0, tolerance=0.25) == "at_vwap"
    assert classify_vwap_relationship(None, 100.0) == "unknown"


def test_rows_from_autopsy_csv_normalizes_existing_dataset(tmp_path: Path) -> None:
    csv_path = tmp_path / "orb_autopsy_dataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "date",
                "session_id",
                "source_run_id",
                "route",
                "opening_range_width",
                "atr",
                "impulse",
                "one_sidedness",
                "breakout_direction",
                "session_pnl",
                "label",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "date": "20260223",
                "session_id": "session_20260223",
                "source_run_id": "source",
                "route": "orb",
                "opening_range_width": "26.25",
                "atr": "16.5",
                "impulse": "0.18",
                "one_sidedness": "-0.18",
                "breakout_direction": "DOWN",
                "session_pnl": "812.5",
                "label": "good_orb",
            }
        )

    rows = rows_from_autopsy_csv(csv_path, OrbLabelThresholds())

    assert len(rows) == 1
    assert rows[0].label == "good_orb"
    assert rows[0].opening_range_width == 26.25
    assert rows[0].breakout_direction == "DOWN"


def test_markdown_summary_includes_label_counts() -> None:
    rows = [
        OrbDiagnosticRow(
            date="20260101",
            session_id="session_20260101",
            source_run_id="run",
            opportunity_id="one",
            opening_range_width=10.0,
            atr=8.0,
            atr_regime="medium",
            opening_impulse=0.5,
            one_sidedness_score=0.5,
            vwap_relationship="above_vwap",
            pullback_depth=0.0,
            breakout_direction="BUY",
            max_favorable_excursion=1.1,
            max_adverse_excursion=0.2,
            final_r=0.8,
            label="good_orb",
        )
    ]

    markdown = rows_to_markdown(rows, "ORB Diagnostics")

    assert "# ORB Diagnostics" in markdown
    assert "| good_orb | 1 |" in markdown
    assert label_counts(rows)["good_orb"] == 1
