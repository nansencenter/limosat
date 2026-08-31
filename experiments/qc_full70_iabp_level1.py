#!/usr/bin/env python3
"""Validate every full-70 buoy/image link against official IABP Level-1 tracks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from experiments.qc_selected_iabp_level1 import (
        clean_buoy_id,
        interpolate_level1_frame,
        load_level1_summary,
        load_level1_track,
        platform_evidence_status,
    )
except ModuleNotFoundError:  # Direct execution from the experiments directory.
    from qc_selected_iabp_level1 import (  # type: ignore[no-redef]
        clean_buoy_id,
        interpolate_level1_frame,
        load_level1_summary,
        load_level1_track,
        platform_evidence_status,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results/iabp_s1_stratified_coverage"
DEFAULT_LEVEL1_DIR = Path(
    "/Volumes/KINGSTON/arktalas/experiments/"
    "limosat_descriptor_update_2020/iabp_level1_full70"
)


def position_repeat_context(track: pd.DataFrame, when: pd.Timestamp) -> dict:
    """Describe an exact repeated-coordinate run around the nearest Level-1 fix."""
    if track.empty:
        return {
            "level1_repeat_fix_count": 0,
            "level1_repeat_span_hours": np.nan,
            "level1_repeat_previous_speed_m_per_day": np.nan,
            "level1_repeat_next_speed_m_per_day": np.nan,
        }
    times = pd.DatetimeIndex(track["time"])
    target = pd.Timestamp(when)
    index = int(np.argmin(np.abs(times - target)))
    latitude = track.iloc[index]["Lat"]
    longitude = track.iloc[index]["Lon"]
    start = index
    while start > 0 and (
        track.iloc[start - 1]["Lat"] == latitude
        and track.iloc[start - 1]["Lon"] == longitude
    ):
        start -= 1
    stop = index
    while stop + 1 < len(track) and (
        track.iloc[stop + 1]["Lat"] == latitude
        and track.iloc[stop + 1]["Lon"] == longitude
    ):
        stop += 1

    def adjacent_speed(left: int, right: int) -> float:
        if left < 0 or right >= len(track):
            return np.nan
        elapsed_days = (
            track.iloc[right]["time"] - track.iloc[left]["time"]
        ).total_seconds() / 86400.0
        if elapsed_days <= 0:
            return np.nan
        distance = np.hypot(
            track.iloc[right]["x_3413"] - track.iloc[left]["x_3413"],
            track.iloc[right]["y_3413"] - track.iloc[left]["y_3413"],
        )
        return float(distance / elapsed_days)

    return {
        "level1_repeat_fix_count": stop - start + 1,
        "level1_repeat_span_hours": float(
            (track.iloc[stop]["time"] - track.iloc[start]["time"]).total_seconds()
            / 3600.0
        ),
        "level1_repeat_previous_speed_m_per_day": adjacent_speed(start - 1, start),
        "level1_repeat_next_speed_m_per_day": adjacent_speed(stop, stop + 1),
    }


def final_link_status(
    original_status: str,
    level1_file_available: bool,
    level1_bracketed: bool,
    bracket_gap_hours: float,
    bracket_speed_m_per_day: float,
    position_difference_m: float,
    platform_status: str,
    maximum_track_gap_hours: float,
    maximum_track_speed_m_per_day: float,
    maximum_position_difference_m: float,
) -> str:
    if original_status in {
        "exclude_exact_position_outside_image",
        "hold_128px_image_border",
    }:
        return original_status
    if not level1_file_available:
        return "hold_missing_level1_file"
    if not level1_bracketed:
        return "hold_level1_track_not_bracketed"
    if not np.isfinite(bracket_gap_hours) or bracket_gap_hours > maximum_track_gap_hours:
        return "hold_level1_track_gap_exceeds_threshold"
    if (
        not np.isfinite(bracket_speed_m_per_day)
        or bracket_speed_m_per_day > maximum_track_speed_m_per_day
    ):
        return "reject_level1_speed_exceeds_threshold"
    if (
        not np.isfinite(position_difference_m)
        or position_difference_m > maximum_position_difference_m
    ):
        return "reject_level1_catalog_position_disagreement"
    if platform_status == "reject_aoml_ocean_drifter":
        return "reject_aoml_ocean_drifter"
    if (
        original_status == "hold_level1_on_ice_platform_evidence"
        and platform_status != "accept_on_ice_platform_evidence"
    ):
        return platform_status
    return "ready_level1_validated"


def build_platform_qc(
    frame_qc: pd.DataFrame,
    summary: pd.DataFrame,
    seawater_freezing_temperature_c: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for buoy_id, group in frame_qc.groupby("buoy_id", sort=True):
        valid_temperature = group["surface_temperature_c"].dropna()
        required_temperature_frames = math.ceil(0.75 * len(group))
        fraction_cold = (
            float((valid_temperature < seawater_freezing_temperature_c).mean())
            if len(valid_temperature)
            else np.nan
        )
        metadata_rows = summary[summary["buoy_id"].eq(buoy_id)]
        metadata = metadata_rows.iloc[0] if len(metadata_rows) else pd.Series(dtype=object)
        status = platform_evidence_status(
            metadata.get("buoy_type"),
            metadata.get("owner"),
            len(valid_temperature),
            fraction_cold,
            float(valid_temperature.median()) if len(valid_temperature) else np.nan,
            required_temperature_frames,
            seawater_freezing_temperature_c,
        )
        records.append(
            {
                "buoy_id": buoy_id,
                "buoy_type": metadata.get("buoy_type"),
                "owner": metadata.get("owner"),
                "logistics": metadata.get("logistics"),
                "linked_images": len(group),
                "level1_file_available": bool(group["level1_file_available"].all()),
                "valid_surface_temperature_images": len(valid_temperature),
                "median_surface_temperature_c": (
                    float(valid_temperature.median()) if len(valid_temperature) else np.nan
                ),
                "fraction_surface_temperature_below_minus1_8": fraction_cold,
                "platform_evidence_status": status,
            }
        )
    return pd.DataFrame.from_records(records)


def build_frame_qc(
    links: pd.DataFrame,
    level1_dir: Path,
    maximum_temperature_offset_hours: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for buoy_id, group in links.groupby("buoy_id", sort=True):
        track_path = level1_dir / f"{buoy_id}.csv"
        track = load_level1_track(track_path) if track_path.is_file() else None
        for row in group.itertuples(index=False):
            if track is None:
                result = {
                    "level1_bracketed": False,
                    "level1_bracket_gap_hours": np.nan,
                    "level1_bracket_speed_m_per_day": np.nan,
                    "level1_x_3413": np.nan,
                    "level1_y_3413": np.nan,
                    "temperature_offset_hours": np.nan,
                    "surface_temperature_c": np.nan,
                    "air_temperature_c": np.nan,
                    "hull_temperature_c": np.nan,
                    "level1_repeat_fix_count": 0,
                    "level1_repeat_span_hours": np.nan,
                    "level1_repeat_previous_speed_m_per_day": np.nan,
                    "level1_repeat_next_speed_m_per_day": np.nan,
                }
            else:
                result = interpolate_level1_frame(
                    track,
                    pd.Timestamp(row.image_time),
                    maximum_temperature_offset_hours,
                )
                result.update(
                    position_repeat_context(track, pd.Timestamp(row.image_time))
                )
            position_difference_m = (
                float(
                    np.hypot(
                        result["level1_x_3413"] - row.x,
                        result["level1_y_3413"] - row.y,
                    )
                )
                if result["level1_bracketed"]
                else np.nan
            )
            records.append(
                {
                    **row._asdict(),
                    "level1_file_available": track is not None,
                    **result,
                    "level1_to_catalog_position_difference_m": position_difference_m,
                }
            )
    return pd.DataFrame.from_records(records)


def apply_final_qc(
    frame_qc: pd.DataFrame,
    platform_qc: pd.DataFrame,
    maximum_track_gap_hours: float,
    maximum_track_speed_m_per_day: float,
    maximum_position_difference_m: float,
) -> pd.DataFrame:
    merged = frame_qc.merge(
        platform_qc[
            ["buoy_id", "buoy_type", "owner", "platform_evidence_status"]
        ],
        on="buoy_id",
        how="left",
        validate="many_to_one",
    )
    merged["level1_final_status"] = merged.apply(
        lambda row: final_link_status(
            row["current_experiment_status"],
            bool(row["level1_file_available"]),
            bool(row["level1_bracketed"]),
            row["level1_bracket_gap_hours"],
            row["level1_bracket_speed_m_per_day"],
            row["level1_to_catalog_position_difference_m"],
            row["platform_evidence_status"],
            maximum_track_gap_hours,
            maximum_track_speed_m_per_day,
            maximum_position_difference_m,
        ),
        axis=1,
    )
    merged["truth_ready_after_level1"] = merged["level1_final_status"].eq(
        "ready_level1_validated"
    )
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--level1-dir", type=Path, default=DEFAULT_LEVEL1_DIR)
    parser.add_argument("--maximum-temperature-offset-hours", type=float, default=3.0)
    parser.add_argument("--maximum-track-gap-hours", type=float, default=6.0)
    parser.add_argument("--maximum-track-speed-km-day", type=float, default=100.0)
    parser.add_argument("--maximum-position-difference-m", type=float, default=500.0)
    parser.add_argument("--seawater-freezing-temperature-c", type=float, default=-1.8)
    args = parser.parse_args()

    links = pd.read_csv(
        args.results_dir / "full70_buoy_image_links.csv",
        dtype={"buoy_id": "string"},
        low_memory=False,
    )
    links["buoy_id"] = clean_buoy_id(links["buoy_id"])
    links["image_time"] = pd.to_datetime(links["image_time"], utc=True)
    summary = load_level1_summary(args.level1_dir / "L1_Summary.txt")
    frame_qc = build_frame_qc(
        links, args.level1_dir, args.maximum_temperature_offset_hours
    )
    platform_qc = build_platform_qc(
        frame_qc, summary, args.seawater_freezing_temperature_c
    )
    final = apply_final_qc(
        frame_qc,
        platform_qc,
        args.maximum_track_gap_hours,
        args.maximum_track_speed_km_day * 1000.0,
        args.maximum_position_difference_m,
    )
    buoy_outcomes = final.groupby("buoy_id")["level1_final_status"].value_counts()
    platform_qc = platform_qc.merge(
        buoy_outcomes.unstack(fill_value=0).reset_index(), on="buoy_id", how="left"
    )
    final.to_csv(args.results_dir / "full70_iabp_level1_frame_qc.csv", index=False)
    platform_qc.to_csv(args.results_dir / "full70_iabp_level1_buoy_qc.csv", index=False)
    payload = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "level1_dir": str(args.level1_dir),
        "linked_observations": len(final),
        "buoys": int(final["buoy_id"].nunique()),
        "level1_files_available": int(
            final.loc[final["level1_file_available"], "buoy_id"].nunique()
        ),
        "truth_ready_observations_after_level1": int(
            final["truth_ready_after_level1"].sum()
        ),
        "status_counts": final["level1_final_status"].value_counts().to_dict(),
        "thresholds": {
            "maximum_track_gap_hours": args.maximum_track_gap_hours,
            "maximum_track_speed_km_day": args.maximum_track_speed_km_day,
            "maximum_position_difference_m": args.maximum_position_difference_m,
            "seawater_freezing_temperature_c": args.seawater_freezing_temperature_c,
        },
    }
    (args.results_dir / "full70_iabp_level1_qc_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload["level1_files_available"] == payload["buoys"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
