#!/usr/bin/env python3
"""Audit ALIKED buoy errors against repeated-position runs in IABP Level-1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.qc_full70_iabp_level1 import position_repeat_context
from experiments.qc_selected_iabp_level1 import load_level1_track


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repeat_context_by_observation(
    observations: pd.DataFrame, level1_dir: Path
) -> pd.DataFrame:
    records = []
    tracks = {}
    for row in observations.itertuples(index=False):
        buoy_id = str(row.buoy_id)
        if buoy_id not in tracks:
            path = level1_dir / f"{buoy_id}.csv"
            tracks[buoy_id] = load_level1_track(path) if path.is_file() else None
        track = tracks[buoy_id]
        context = (
            position_repeat_context(track, pd.Timestamp(row.image_time))
            if track is not None
            else {
                "level1_repeat_fix_count": 0,
                "level1_repeat_span_hours": float("nan"),
                "level1_repeat_previous_speed_m_per_day": float("nan"),
                "level1_repeat_next_speed_m_per_day": float("nan"),
            }
        )
        records.append({"buoy_id": buoy_id, "image_id": row.image_id, **context})
    return pd.DataFrame.from_records(records)


def summarize(frame: pd.DataFrame) -> list[dict]:
    working = frame.copy()
    working["correct_within_2km"] = working["available"] & working["error_m"].le(
        2000.0
    )
    working["maximum_repeat_span_hours"] = working[
        ["source_level1_repeat_span_hours", "target_level1_repeat_span_hours"]
    ].max(axis=1)
    working["repeat_span_bin"] = pd.cut(
        working["maximum_repeat_span_hours"],
        [-0.01, 0.0, 5.99, float("inf")],
        labels=["no_repeat_span", "under_6h", "6h_or_more"],
    )
    records = []
    groups = [("all", "all", working)]
    groups.extend(
        ("repeat_span", str(value), group)
        for value, group in working.groupby("repeat_span_bin", observed=True)
    )
    groups.extend(
        ("existing_track_qc", str(bool(value)), group)
        for value, group in working.groupby("both_existing_track_qc", dropna=False)
    )
    groups.extend(
        ("stale_jump_diagnostic", str(bool(value)), group)
        for value, group in working.groupby("stale_jump_diagnostic", dropna=False)
    )
    groups.extend(
        ("combined_label_qc", str(bool(value)), group)
        for value, group in working.groupby("combined_label_qc", dropna=False)
    )
    for stratum, value, group in groups:
        available_error = group.loc[group["available"], "error_m"]
        records.append(
            {
                "stratum": stratum,
                "value": value,
                "expected": int(len(group)),
                "available": int(group["available"].sum()),
                "correct_within_2km": int(group["correct_within_2km"].sum()),
                "median_error_m": (
                    float(available_error.median()) if len(available_error) else None
                ),
                "p90_error_m": (
                    float(available_error.quantile(0.90))
                    if len(available_error)
                    else None
                ),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--policy-results", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--level1-dir", type=Path, required=True)
    parser.add_argument("--policy", default="consensus_within_2km")
    parser.add_argument(
        "--mode",
        help="Optional result mode, for example truth_reinitialized or propagated.",
    )
    parser.add_argument("--repeat-span-hours", type=float, default=6.0)
    parser.add_argument("--adjacent-speed-km-day", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    transitions = pd.read_csv(
        args.transitions, dtype={"buoy_id": str}, low_memory=False
    )
    policies = pd.read_csv(args.policy_results, low_memory=False)
    if "panel" in policies:
        policies = policies.loc[policies["panel"].eq("representative")].copy()
    if "policy" in policies:
        policies = policies.loc[policies["policy"].eq(args.policy)].copy()
    if args.mode is not None:
        if "mode" not in policies:
            raise ValueError("policy results lack mode")
        policies = policies.loc[policies["mode"].eq(args.mode)].copy()
    if "error_m" not in policies and "endpoint_error_m" in policies:
        policies = policies.rename(columns={"endpoint_error_m": "error_m"})
    required_policy_columns = {"transition_id", "available", "error_m"}
    missing_policy_columns = required_policy_columns.difference(policies.columns)
    if missing_policy_columns:
        raise ValueError(
            f"policy results lack required columns: {sorted(missing_policy_columns)}"
        )
    transitions = transitions.loc[
        transitions["transition_id"].isin(policies["transition_id"])
    ].copy()
    observations = pd.read_csv(
        args.observations, dtype={"buoy_id": str}, low_memory=False
    )
    needed = pd.concat(
        [
            transitions[["buoy_id", "source_image_id"]].rename(
                columns={"source_image_id": "image_id"}
            ),
            transitions[["buoy_id", "target_image_id"]].rename(
                columns={"target_image_id": "image_id"}
            ),
        ]
    ).drop_duplicates()
    linked_observations = needed.merge(
        observations[
            ["buoy_id", "image_id", "image_time", "track_qc_pass", "track_qc_status"]
        ],
        on=["buoy_id", "image_id"],
        how="left",
        validate="one_to_one",
    )
    repeat = repeat_context_by_observation(linked_observations, args.level1_dir)
    observation_qc = linked_observations.merge(
        repeat, on=["buoy_id", "image_id"], validate="one_to_one"
    )
    source_qc = observation_qc.add_prefix("source_").rename(
        columns={
            "source_buoy_id": "buoy_id",
            "source_image_id": "source_image_id",
        }
    )
    target_qc = observation_qc.add_prefix("target_").rename(
        columns={
            "target_buoy_id": "buoy_id",
            "target_image_id": "target_image_id",
        }
    )
    policies = policies.drop(
        columns=["buoy_id", "source_image_id", "target_image_id"],
        errors="ignore",
    )
    frame = policies.merge(
        transitions[
            ["transition_id", "buoy_id", "source_image_id", "target_image_id"]
        ],
        on="transition_id",
        validate="one_to_one",
    )
    frame = frame.merge(
        source_qc, on=["buoy_id", "source_image_id"], validate="many_to_one"
    ).merge(target_qc, on=["buoy_id", "target_image_id"], validate="many_to_one")
    frame["both_existing_track_qc"] = (
        frame["source_track_qc_pass"].fillna(False)
        & frame["target_track_qc_pass"].fillna(False)
    )
    frame["maximum_repeat_span_hours"] = frame[
        ["source_level1_repeat_span_hours", "target_level1_repeat_span_hours"]
    ].max(axis=1)
    adjacent_speed_columns = [
        "source_level1_repeat_previous_speed_m_per_day",
        "source_level1_repeat_next_speed_m_per_day",
        "target_level1_repeat_previous_speed_m_per_day",
        "target_level1_repeat_next_speed_m_per_day",
    ]
    frame["maximum_repeat_adjacent_speed_m_per_day"] = frame[
        adjacent_speed_columns
    ].max(axis=1)
    frame["stale_jump_diagnostic"] = frame["maximum_repeat_span_hours"].ge(
        args.repeat_span_hours
    ) & frame["maximum_repeat_adjacent_speed_m_per_day"].gt(
        args.adjacent_speed_km_day * 1000.0
    )
    frame["combined_label_qc"] = (
        frame["both_existing_track_qc"] & ~frame["stale_jump_diagnostic"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "transition_repeat_qc.csv", index=False)
    summary = summarize(frame)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "transitions_sha256": sha256(args.transitions),
        "policy_results_sha256": sha256(args.policy_results),
        "observations_sha256": sha256(args.observations),
        "policy": args.policy,
        "mode": args.mode,
        "repeat_coordinates": "exact Level-1 latitude/longitude equality",
        "six_hour_bin": "diagnostic sensitivity only; not a selected QC threshold",
        "diagnostic_thresholds": {
            "repeat_span_hours": args.repeat_span_hours,
            "adjacent_speed_km_day": args.adjacent_speed_km_day,
            "provenance": "existing project maximum track-gap and track-speed thresholds",
        },
        "rows": int(len(frame)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
