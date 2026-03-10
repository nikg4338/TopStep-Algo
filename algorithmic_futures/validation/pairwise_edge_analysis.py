"""Pairwise conditional expectancy analysis for validation-run trade datasets.

This module builds a descriptive research dataset from existing validation
artifacts and evaluates transparent pairwise bucket combinations without
changing strategy behavior.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from validation.candidate_openfix import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    ensure_report_dir,
    load_run_dir,
    write_csv,
    write_markdown,
)


FEATURE_LABELS = {
    "impulse": "opening_impulse",
    "persistence": "persistence",
    "atr_percentile": "atr_percentile",
    "vol_percentile": "volatility_percentile",
    "distance_from_vwap_atr": "distance_from_vwap",
    "one_sidedness": "one_sidedness",
    "confidence_score": "confidence_score",
}


@dataclass(frozen=True)
class PairSpec:
    trade_type: str
    feature_1: str
    feature_2: str


def _read_csv_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [frame for frame in frames if not frame.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def _to_timestamp(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")


def qbucket(series: pd.Series, labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    output = pd.Series(pd.NA, index=series.index, dtype="object")
    if valid.empty:
        return output
    ranked = valid.rank(method="first")
    try:
        buckets = pd.qcut(ranked, q=len(labels), labels=labels)
    except ValueError:
        buckets = pd.cut(ranked, bins=len(labels), labels=labels, include_lowest=True)
    output.loc[valid.index] = buckets.astype("object")
    return output


def fixed_percentile_bucket(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-0.001, 25, 50, 75, 100.0001],
        labels=["0-25", "25-50", "50-75", "75-100"],
    ).astype("object")


def persistence_bucket(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-0.1, 0.5, 1.5, math.inf],
        labels=["weak", "moderate", "strong"],
    ).astype("object")


def mean_positive(values: pd.Series) -> float | str:
    wins = values[values > 0]
    return round(float(wins.mean()), 4) if not wins.empty else ""


def mean_nonpositive(values: pd.Series) -> float | str:
    losses = values[values <= 0]
    return round(float(losses.mean()), 4) if not losses.empty else ""


def research_score(mean_pnl: float, win_rate: float, sample_size: int, std_dev: float) -> float:
    if sample_size <= 0:
        return 0.0
    numerator = mean_pnl * max(win_rate, 0.0) * math.log1p(sample_size)
    return round(numerator / (1.0 + max(std_dev, 0.0)), 6)


def skewness_or_blank(values: pd.Series) -> float | str:
    if len(values) <= 2:
        return ""
    skew_value = values.skew()
    coerced = pd.to_numeric(pd.Series([skew_value]), errors="coerce").iloc[0]
    if pd.isna(coerced):
        return ""
    return round(float(coerced), 4)


def build_trade_level_dataset(run_dir: Path | str) -> pd.DataFrame:
    run_path = load_run_dir(str(run_dir), DEFAULT_ARTIFACT_ROOT)
    allocator = _read_csv_frame(run_path / "allocator_debug.csv")
    trades = _read_csv_frame(run_path / "aggregate_trades.csv")

    signal_frames: list[pd.DataFrame] = []
    feature_frames: list[pd.DataFrame] = []
    sessions_dir = run_path / "sessions"
    if sessions_dir.is_dir():
        for session_dir in sorted(path for path in sessions_dir.iterdir() if path.is_dir()):
            session_id = session_dir.name
            signals = _read_csv_frame(session_dir / "signals.csv")
            if not signals.empty:
                signals["session_id"] = session_id
                signal_frames.append(signals)
            features = _read_csv_frame(session_dir / "features_snapshot.csv")
            if not features.empty:
                features["session_id"] = session_id
                feature_frames.append(features)

    signals = _safe_concat(signal_frames)
    features = _safe_concat(feature_frames)

    _to_timestamp(trades, ["signal_timestamp", "entry_timestamp", "exit_timestamp"])
    _to_timestamp(signals, ["timestamp"])
    _to_timestamp(features, ["timestamp"])

    merged = trades.copy()
    if not signals.empty:
        merged = merged.merge(
            signals.rename(columns={"timestamp": "signal_timestamp"}),
            on=["session_id", "signal_timestamp"],
            how="left",
        )

    if not features.empty:
        entry_features = features.rename(columns={"timestamp": "entry_timestamp"})
        entry_cols = [
            column
            for column in [
                "session_id",
                "entry_timestamp",
                "adx",
                "atr",
                "atr_percentile",
                "realized_vol",
                "vol_percentile",
                "regime",
            ]
            if column in entry_features.columns
        ]
        if entry_cols:
            merged = merged.merge(entry_features[entry_cols], on=["session_id", "entry_timestamp"], how="left")

        signal_features = features.rename(
            columns={
                "timestamp": "signal_timestamp",
                "adx": "signal_adx",
                "atr": "signal_atr",
                "atr_percentile": "signal_atr_percentile",
                "realized_vol": "signal_realized_vol",
                "vol_percentile": "signal_vol_percentile",
                "regime": "signal_regime",
            }
        )
        signal_cols = [
            column
            for column in [
                "session_id",
                "signal_timestamp",
                "signal_adx",
                "signal_atr",
                "signal_atr_percentile",
                "signal_realized_vol",
                "signal_vol_percentile",
                "signal_regime",
            ]
            if column in signal_features.columns
        ]
        if signal_cols:
            merged = merged.merge(signal_features[signal_cols], on=["session_id", "signal_timestamp"], how="left")

    for left, right in [
        ("adx", "signal_adx"),
        ("atr", "signal_atr"),
        ("atr_percentile", "signal_atr_percentile"),
        ("realized_vol", "signal_realized_vol"),
        ("vol_percentile", "signal_vol_percentile"),
        ("regime", "signal_regime"),
    ]:
        if left in merged.columns and right in merged.columns:
            merged[left] = merged[left].fillna(merged[right])

    if not allocator.empty:
        allocator_columns = [
            column
            for column in [
                "session_id",
                "date",
                "route",
                "opening_range_width",
                "atr",
                "width_atr",
                "impulse",
                "persistence",
                "close_location",
                "one_sidedness",
                "confidence_score",
                "breakout_direction",
                "trade_count",
                "session_pnl_dollars",
            ]
            if column in allocator.columns
        ]
        merged = merged.merge(allocator[allocator_columns], on="session_id", how="left", suffixes=("", "_session"))

    route_source = merged.get("route", pd.Series(index=merged.index, dtype="object")).fillna("")
    signal_type = merged.get("signal_type", pd.Series(index=merged.index, dtype="object")).fillna("")
    merged["trade_type"] = signal_type.where(signal_type != "", route_source.str.upper())
    merged["trade_type"] = merged["trade_type"].replace({"BOTH": "MR"})
    if "candidate_price" in merged.columns and "vwap" in merged.columns:
        merged["distance_from_vwap_pts"] = (pd.to_numeric(merged["candidate_price"], errors="coerce") - pd.to_numeric(merged["vwap"], errors="coerce")).abs()
    else:
        merged["distance_from_vwap_pts"] = pd.NA
    atr_series = pd.to_numeric(merged.get("atr", pd.Series(index=merged.index)), errors="coerce")
    merged["distance_from_vwap_atr"] = pd.to_numeric(merged["distance_from_vwap_pts"], errors="coerce") / atr_series.replace(0, pd.NA)
    merged["session_return"] = merged.groupby("session_id")["pnl_dollars"].transform("sum")
    merged["source_run_id"] = run_path.name
    return merged


def choose_mr_proxy_feature(dataset: pd.DataFrame) -> str:
    for candidate in ("persistence", "one_sidedness", "confidence_score"):
        if candidate in dataset.columns and dataset[candidate].notna().any():
            return candidate
    return "persistence"


def bucket_feature(series: pd.Series, feature: str) -> pd.Series:
    if feature == "impulse":
        return qbucket(series, ["low", "medium", "high"])
    if feature == "distance_from_vwap_atr":
        return qbucket(series, ["small deviation", "moderate deviation", "large deviation"])
    if feature in {"atr_percentile", "vol_percentile"}:
        return fixed_percentile_bucket(series)
    if feature == "persistence":
        return persistence_bucket(series)
    if feature in {"one_sidedness", "confidence_score"}:
        return qbucket(series, ["low", "medium", "high"])
    raise ValueError(f"Unsupported feature for bucketing: {feature}")


def bucket_definition(feature: str, proxy_feature: str | None = None) -> str:
    if feature == "impulse":
        return "opening_impulse tertiles from current dataset: low / medium / high"
    if feature == "distance_from_vwap_atr":
        return "distance_from_vwap normalized by ATR, bucketed into tertiles: small / moderate / large deviation"
    if feature == "persistence":
        return "persistence buckets: weak=0, moderate=1, strong>=2"
    if feature == "atr_percentile":
        return "ATR percentile quartiles: 0-25 / 25-50 / 50-75 / 75-100"
    if feature == "vol_percentile":
        return "volatility percentile quartiles (repo field `vol_percentile`): 0-25 / 25-50 / 50-75 / 75-100"
    if feature == "one_sidedness":
        return "one_sidedness tertiles from current dataset: low / medium / high"
    if feature == "confidence_score":
        return "confidence_score tertiles from current dataset: low / medium / high"
    if proxy_feature:
        return bucket_definition(proxy_feature)
    return feature


def build_pair_specs(dataset: pd.DataFrame) -> tuple[list[PairSpec], str]:
    mr_proxy = choose_mr_proxy_feature(dataset)
    specs = [
        PairSpec("ORB", "impulse", "persistence"),
        PairSpec("ORB", "impulse", "atr_percentile"),
        PairSpec("ORB", "impulse", "vol_percentile"),
        PairSpec("ORB", "persistence", "atr_percentile"),
        PairSpec("MR", "distance_from_vwap_atr", "vol_percentile"),
        PairSpec("MR", "distance_from_vwap_atr", mr_proxy),
        PairSpec("MR", "distance_from_vwap_atr", "atr_percentile"),
    ]
    if "one_sidedness" in dataset.columns and dataset["one_sidedness"].notna().any():
        specs.append(PairSpec("ORB", "impulse", "one_sidedness"))
    return specs, mr_proxy


def analyze_pairwise_slices(
    dataset: pd.DataFrame,
    *,
    reporting_min_sample: int = 10,
    candidate_min_sample: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    specs, mr_proxy = build_pair_specs(dataset)
    bucket_definitions: dict[str, str] = {}
    rows: list[dict[str, Any]] = []

    for spec in specs:
        subset = dataset.loc[dataset["trade_type"].eq(spec.trade_type)].copy()
        if subset.empty:
            continue
        if spec.feature_1 not in subset.columns or spec.feature_2 not in subset.columns:
            continue

        subset["feature_1_bucket"] = bucket_feature(subset[spec.feature_1], spec.feature_1)
        subset["feature_2_bucket"] = bucket_feature(subset[spec.feature_2], spec.feature_2)
        subset = subset.dropna(subset=["feature_1_bucket", "feature_2_bucket", "pnl_dollars"])
        if subset.empty:
            continue

        bucket_definitions[FEATURE_LABELS.get(spec.feature_1, spec.feature_1)] = bucket_definition(spec.feature_1)
        bucket_definitions[FEATURE_LABELS.get(spec.feature_2, spec.feature_2)] = bucket_definition(spec.feature_2, mr_proxy)

        grouped = subset.groupby(["feature_1_bucket", "feature_2_bucket"], dropna=False)
        for (bucket_1, bucket_2), frame in grouped:
            pnl = pd.to_numeric(frame["pnl_dollars"], errors="coerce").dropna()
            if pnl.empty:
                continue
            mean_pnl = float(pnl.mean())
            median_pnl = float(pnl.median())
            win_rate = float((pnl > 0).mean())
            std_dev = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
            variance = float(pnl.var(ddof=1)) if len(pnl) > 1 else 0.0
            sample_size = int(len(pnl))
            row = {
                "trade_type": spec.trade_type,
                "feature_1": FEATURE_LABELS.get(spec.feature_1, spec.feature_1),
                "feature_1_bucket": str(bucket_1),
                "feature_2": FEATURE_LABELS.get(spec.feature_2, spec.feature_2),
                "feature_2_bucket": str(bucket_2),
                "sample_size": sample_size,
                "mean_trade_pnl": round(mean_pnl, 4),
                "median_trade_pnl": round(median_pnl, 4),
                "win_rate": round(win_rate, 4),
                "avg_win": mean_positive(pnl),
                "avg_loss": mean_nonpositive(pnl),
                "std_dev": round(std_dev, 4),
                "variance": round(variance, 4),
                "skewness": skewness_or_blank(pnl),
                "positive_expectancy": mean_pnl > 0,
                "reporting_sample_ok": sample_size >= reporting_min_sample,
                "candidate_sample_ok": sample_size >= candidate_min_sample,
                "research_score": research_score(mean_pnl, win_rate, sample_size, std_dev),
            }
            rows.append(row)

    rows.sort(
        key=lambda row: (
            row["trade_type"],
            row["feature_1"],
            row["feature_2"],
            row["feature_1_bucket"],
            row["feature_2_bucket"],
        )
    )
    return rows, bucket_definitions, mr_proxy


def rank_pairwise_rows(rows: list[dict[str, Any]], trade_type: str, *, reporting_min_sample: int = 10) -> list[dict[str, Any]]:
    filtered = [row for row in rows if row["trade_type"] == trade_type and row["sample_size"] >= reporting_min_sample]
    return sorted(
        filtered,
        key=lambda row: (
            not row["positive_expectancy"],
            not row["candidate_sample_ok"],
            -float(row["research_score"]),
            -float(row["mean_trade_pnl"]),
            float(row["variance"]),
            -int(row["sample_size"]),
        ),
    )


def build_pairwise_pivot(rows: list[dict[str, Any]], *, trade_type: str, feature_1: str, feature_2: str) -> list[dict[str, Any]]:
    filtered = [
        row
        for row in rows
        if row["trade_type"] == trade_type and row["feature_1"] == feature_1 and row["feature_2"] == feature_2
    ]
    feature_2_buckets = sorted({row["feature_2_bucket"] for row in filtered})
    output: list[dict[str, Any]] = []
    for bucket_1 in sorted({row["feature_1_bucket"] for row in filtered}):
        row_out: dict[str, Any] = {feature_1: bucket_1}
        for bucket_2 in feature_2_buckets:
            match = next(
                (
                    row
                    for row in filtered
                    if row["feature_1_bucket"] == bucket_1 and row["feature_2_bucket"] == bucket_2
                ),
                None,
            )
            row_out[bucket_2] = "" if match is None else f"{match['mean_trade_pnl']:.2f} | n={match['sample_size']}"
        output.append(row_out)
    return output


def summarize_pairwise_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["trade_type"] for row in rows)
    return {
        "rows": len(rows),
        "by_trade_type": dict(counts),
        "positive_rows": sum(1 for row in rows if row["positive_expectancy"]),
        "candidate_rows": sum(1 for row in rows if row["candidate_sample_ok"] and row["positive_expectancy"]),
    }


def candidate_hypotheses(orb_rows: list[dict[str, Any]], mr_rows: list[dict[str, Any]]) -> list[str]:
    items: list[str] = []
    if orb_rows:
        top_orb = orb_rows[0]
        items.append(
            "ORB expectancy appears strongest when "
            f"{top_orb['feature_1']}={top_orb['feature_1_bucket']} and {top_orb['feature_2']}={top_orb['feature_2_bucket']}, "
            f"but sample size is {top_orb['sample_size']}."
        )
    if len(orb_rows) > 1:
        weak_orb = next((row for row in reversed(orb_rows) if not row["positive_expectancy"]), None)
        if weak_orb:
            items.append(
                "ORB looks weaker or deceptive when "
                f"{weak_orb['feature_1']}={weak_orb['feature_1_bucket']} and {weak_orb['feature_2']}={weak_orb['feature_2_bucket']}."
            )
    if mr_rows:
        top_mr = mr_rows[0]
        items.append(
            "MR expectancy appears strongest when "
            f"{top_mr['feature_1']}={top_mr['feature_1_bucket']} and {top_mr['feature_2']}={top_mr['feature_2_bucket']}, "
            f"again with sample size {top_mr['sample_size']}."
        )
    if not items:
        items.append("No pairwise slice met the reporting threshold; gather more trades before elevating any routing hypothesis.")
    return items


def write_pairwise_edge_artifacts(
    dataset: pd.DataFrame,
    rows: list[dict[str, Any]],
    bucket_definitions: dict[str, str],
    *,
    run_id: str,
    reporting_min_sample: int,
    candidate_min_sample: int,
    mr_proxy_feature: str,
    output_root: Path = DEFAULT_REPORT_ROOT,
) -> Path:
    report_dir = ensure_report_dir("pairwise_edge_analysis", output_root)
    orb_ranked = rank_pairwise_rows(rows, "ORB", reporting_min_sample=reporting_min_sample)
    mr_ranked = rank_pairwise_rows(rows, "MR", reporting_min_sample=reporting_min_sample)
    summary = summarize_pairwise_results(rows)

    write_csv(report_dir / "pairwise_results.csv", rows)
    write_csv(report_dir / "top_orb_pairwise.csv", orb_ranked[:20])
    write_csv(report_dir / "top_mr_pairwise.csv", mr_ranked[:20])
    dataset.to_csv(report_dir / "trade_level_dataset.csv", index=False)
    with (report_dir / "bucket_definitions.json").open("w", encoding="utf-8") as fh:
        json.dump(bucket_definitions, fh, indent=2)

    orb_pivot = build_pairwise_pivot(rows, trade_type="ORB", feature_1="opening_impulse", feature_2="persistence")
    mr_pivot = build_pairwise_pivot(rows, trade_type="MR", feature_1="distance_from_vwap", feature_2="volatility_percentile")
    write_csv(report_dir / "orb_impulse_persistence_pivot.csv", orb_pivot)
    write_csv(report_dir / "mr_distance_volatility_pivot.csv", mr_pivot)

    lines = [
        "# Pairwise Conditional Edge Analysis",
        "",
        "## Dataset scope",
        f"- Run id: {run_id}",
        f"- Total trades: {len(dataset)}",
        f"- ORB trades: {int((dataset['trade_type'] == 'ORB').sum())}",
        f"- MR trades: {int((dataset['trade_type'] == 'MR').sum())}",
        f"- Reporting threshold: n >= {reporting_min_sample}",
        f"- Candidate threshold: n >= {candidate_min_sample}",
        f"- MR proxy feature used: {FEATURE_LABELS.get(mr_proxy_feature, mr_proxy_feature)}",
        "- Ranking rule: prioritize positive expectancy, then candidate-threshold support, then research score = mean_pnl × win_rate × log(1+n) / (1+std_dev)",
        "",
        "## Bucket definitions used",
    ]
    for feature, definition in sorted(bucket_definitions.items()):
        lines.append(f"- {feature}: {definition}")
    lines.extend([
        "",
        "## Observed facts",
        f"- Pairwise rows generated: {summary['rows']}",
        f"- Positive-expectancy rows: {summary['positive_rows']}",
        f"- Positive-expectancy rows meeting candidate threshold: {summary['candidate_rows']}",
        f"- ORB rows analyzed: {summary['by_trade_type'].get('ORB', 0)}",
        f"- MR rows analyzed: {summary['by_trade_type'].get('MR', 0)}",
        "",
        "## Top ORB pairwise slices",
    ])
    for row in orb_ranked[:5]:
        lines.append(
            f"- {row['feature_1']}={row['feature_1_bucket']} × {row['feature_2']}={row['feature_2_bucket']}: "
            f"mean={row['mean_trade_pnl']:.2f}, win_rate={row['win_rate']:.2%}, std={row['std_dev']:.2f}, n={row['sample_size']}"
        )
    lines.extend(["", "## Top MR pairwise slices"])
    for row in mr_ranked[:5]:
        lines.append(
            f"- {row['feature_1']}={row['feature_1_bucket']} × {row['feature_2']}={row['feature_2_bucket']}: "
            f"mean={row['mean_trade_pnl']:.2f}, win_rate={row['win_rate']:.2%}, std={row['std_dev']:.2f}, n={row['sample_size']}"
        )

    weak_rows = [row for row in rows if row['sample_size'] >= reporting_min_sample and not row['positive_expectancy']]
    weak_rows = sorted(weak_rows, key=lambda row: (row['trade_type'], row['mean_trade_pnl']))
    lines.extend(["", "## Weak or deceptive combinations"])
    for row in weak_rows[:5]:
        lines.append(
            f"- {row['trade_type']}: {row['feature_1']}={row['feature_1_bucket']} × {row['feature_2']}={row['feature_2_bucket']} "
            f"has mean={row['mean_trade_pnl']:.2f} with n={row['sample_size']}"
        )

    lines.extend(["", "## Sample-size warnings"])
    undersized = [row for row in rows if row['positive_expectancy'] and not row['candidate_sample_ok']]
    if undersized:
        lines.append(f"- {len(undersized)} positive-expectancy slices remain below the candidate threshold of {candidate_min_sample} trades.")
    else:
        lines.append("- No positive-expectancy slice remains below the candidate threshold.")

    lines.extend(["", "## Tentative hypotheses", "- The following are research hypotheses only; they are not strategy rules."])
    for item in candidate_hypotheses(orb_ranked, mr_ranked):
        lines.append(f"- {item}")

    write_markdown(report_dir / "summary.md", lines)
    return report_dir
