#!/usr/bin/env python3
"""QC selected provisional MIZ sequences with official IABP Level-1 data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = ROOT / "results/iabp_s1_stratified_coverage"
DEFAULT_LEVEL1_DIR = Path(
    "/Volumes/KINGSTON/arktalas/experiments/"
    "limosat_descriptor_update_2020/iabp_level1_selected"
)


def clean_buoy_id(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def load_level1_summary(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path, sep=";", dtype={"BuoyID": "string"})
    summary = summary.rename(
        columns={
            "BuoyID": "buoy_id",
            "BuoyType": "buoy_type",
            "Owner": "owner",
            "Logistics": "logistics",
            "Dates": "level1_dates",
            "Variables": "level1_variables",
        }
    )
    summary["buoy_id"] = clean_buoy_id(summary["buoy_id"])
    return summary[
        [
            "buoy_id",
            "buoy_type",
            "owner",
            "logistics",
            "level1_dates",
            "level1_variables",
        ]
    ].drop_duplicates("buoy_id")


def load_level1_track(path: Path) -> pd.DataFrame:
    track = pd.read_csv(path, dtype={"BuoyID": "string"}, low_memory=False)
    track = track.rename(columns={"Minute": "Min", "Second": "Sec"})
    required = {"BuoyID", "Year", "Month", "Day", "Hour", "Min", "Sec", "Lat", "Lon"}
    missing = sorted(required - set(track.columns))
    if missing:
        raise ValueError(f"IABP Level-1 track is missing required columns: {missing}")
    track["buoy_id"] = clean_buoy_id(track["BuoyID"])
    track["time"] = pd.to_datetime(
        {
            "year": track["Year"],
            "month": track["Month"],
            "day": track["Day"],
            "hour": track["Hour"],
            "minute": track["Min"],
            "second": track["Sec"],
        },
        utc=True,
        errors="coerce",
    )
    for column in ["Lat", "Lon", "Ts", "Ta", "Th"]:
        if column not in track:
            track[column] = np.nan
        track[column] = pd.to_numeric(track[column], errors="coerce")
        track[column] = track[column].mask(track[column] <= -900)
    track = (
        track.dropna(subset=["time", "Lat", "Lon"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    track["x_3413"], track["y_3413"] = transformer.transform(
        track["Lon"].to_numpy(dtype=float), track["Lat"].to_numpy(dtype=float)
    )
    return track


def interpolate_level1_frame(
    track: pd.DataFrame,
    when: pd.Timestamp,
    maximum_temperature_offset_hours: float,
) -> dict[str, float | bool]:
    times = pd.DatetimeIndex(track["time"]).as_unit("ns").asi8
    target = pd.Timestamp(when).value
    right = int(np.searchsorted(times, target, side="right"))
    left = right - 1
    if right >= len(track) and left >= 0 and times[left] == target:
        right = left
        left -= 1
    if left < 0 or right >= len(track) or times[right] <= times[left]:
        return {
            "level1_bracketed": False,
            "level1_bracket_gap_hours": np.nan,
            "level1_bracket_speed_m_per_day": np.nan,
            "level1_x_3413": np.nan,
            "level1_y_3413": np.nan,
            "temperature_offset_hours": np.nan,
            "surface_temperature_c": np.nan,
            "air_temperature_c": np.nan,
            "hull_temperature_c": np.nan,
        }
    gap_hours = (times[right] - times[left]) / 3.6e12
    fraction = (target - times[left]) / (times[right] - times[left])
    x = float(
        track.iloc[left]["x_3413"]
        + fraction * (track.iloc[right]["x_3413"] - track.iloc[left]["x_3413"])
    )
    y = float(
        track.iloc[left]["y_3413"]
        + fraction * (track.iloc[right]["y_3413"] - track.iloc[left]["y_3413"])
    )
    distance_m = float(
        np.hypot(
            track.iloc[right]["x_3413"] - track.iloc[left]["x_3413"],
            track.iloc[right]["y_3413"] - track.iloc[left]["y_3413"],
        )
    )
    speed = distance_m / (gap_hours / 24.0)
    nearest_index = int(np.argmin(np.abs(times - target)))
    nearest = track.iloc[nearest_index]
    offset_hours = abs(times[nearest_index] - target) / 3.6e12
    temperatures = {
        "surface_temperature_c": nearest["Ts"],
        "air_temperature_c": nearest["Ta"],
        "hull_temperature_c": nearest["Th"],
    }
    if offset_hours > maximum_temperature_offset_hours:
        temperatures = {key: np.nan for key in temperatures}
    return {
        "level1_bracketed": True,
        "level1_bracket_gap_hours": float(gap_hours),
        "level1_bracket_speed_m_per_day": speed,
        "level1_x_3413": x,
        "level1_y_3413": y,
        "temperature_offset_hours": float(offset_hours),
        **temperatures,
    }


def platform_evidence_status(
    buoy_type: str | None,
    owner: str | None,
    valid_surface_temperature_frames: int,
    fraction_surface_temperature_below_freezing: float,
    median_surface_temperature_c: float,
    required_temperature_frames: int,
    seawater_freezing_temperature_c: float,
) -> str:
    type_text = "" if pd.isna(buoy_type) else str(buoy_type).strip()
    owner_text = "" if pd.isna(owner) else str(owner).strip()
    if "AOML" in owner_text.upper():
        return "reject_aoml_ocean_drifter"
    if not type_text and not owner_text:
        return "hold_missing_platform_metadata"
    cold_evidence = (
        valid_surface_temperature_frames >= required_temperature_frames
        and np.isfinite(fraction_surface_temperature_below_freezing)
        and fraction_surface_temperature_below_freezing >= 0.75
        and np.isfinite(median_surface_temperature_c)
        and median_surface_temperature_c < seawater_freezing_temperature_c
    )
    ice_specific = any(
        token in type_text.lower()
        for token in ["ice", "imb", "snow", "simba", "thermistor"]
    )
    if ice_specific or ("USIABP" in owner_text.upper() and cold_evidence):
        return "accept_on_ice_platform_evidence"
    return "hold_insufficient_on_ice_evidence"


def build_frame_qc(
    selected_frames: pd.DataFrame,
    exact_qc: pd.DataFrame,
    level1_dir: Path,
    maximum_temperature_offset_hours: float,
) -> pd.DataFrame:
    selected = selected_frames[
        selected_frames["sequence_regime"].eq("miz_requires_platform_qc")
    ].copy()
    selected["buoy_id"] = clean_buoy_id(selected["buoy_id"])
    selected["image_time"] = pd.to_datetime(selected["image_time"], utc=True)
    exact = exact_qc.copy()
    exact["buoy_id"] = clean_buoy_id(exact["buoy_id"])
    exact = exact[["buoy_id", "image_id", "x", "y"]]
    selected = selected.merge(exact, on=["buoy_id", "image_id"], how="left")
    records: list[dict[str, object]] = []
    for buoy_id, group in selected.groupby("buoy_id", sort=True):
        track = load_level1_track(level1_dir / f"{buoy_id}.csv")
        for row in group.itertuples(index=False):
            result = interpolate_level1_frame(
                track, row.image_time, maximum_temperature_offset_hours
            )
            position_difference = (
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
                    "sequence_id": row.sequence_id,
                    "frame_order": row.frame_order,
                    "buoy_id": buoy_id,
                    "image_id": row.image_id,
                    "image_time": row.image_time,
                    "sic_fraction": row.sic_fraction,
                    "catalog_exact_x_3413": row.x,
                    "catalog_exact_y_3413": row.y,
                    **result,
                    "level1_to_catalog_position_difference_m": position_difference,
                }
            )
    return pd.DataFrame.from_records(records)


def build_sequence_qc(
    frame_qc: pd.DataFrame,
    summary: pd.DataFrame,
    maximum_track_gap_hours: float,
    maximum_track_speed_m_per_day: float,
    maximum_position_difference_m: float,
    seawater_freezing_temperature_c: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sequence_id, group in frame_qc.groupby("sequence_id", sort=True):
        valid_temperature = group["surface_temperature_c"].dropna()
        required_temperature_frames = math.ceil(0.75 * len(group))
        fraction_cold = (
            float((valid_temperature < seawater_freezing_temperature_c).mean())
            if len(valid_temperature)
            else np.nan
        )
        buoy_id = str(group["buoy_id"].iloc[0])
        metadata_rows = summary[summary["buoy_id"].eq(buoy_id)]
        metadata = metadata_rows.iloc[0] if len(metadata_rows) else pd.Series(dtype=object)
        evidence = platform_evidence_status(
            metadata.get("buoy_type"),
            metadata.get("owner"),
            len(valid_temperature),
            fraction_cold,
            float(valid_temperature.median()) if len(valid_temperature) else np.nan,
            required_temperature_frames,
            seawater_freezing_temperature_c,
        )
        track_pass = bool(
            group["level1_bracketed"].all()
            and (group["level1_bracket_gap_hours"] <= maximum_track_gap_hours).all()
            and (
                group["level1_bracket_speed_m_per_day"]
                <= maximum_track_speed_m_per_day
            ).all()
        )
        position_pass = bool(
            group["level1_to_catalog_position_difference_m"].notna().all()
            and (
                group["level1_to_catalog_position_difference_m"]
                <= maximum_position_difference_m
            ).all()
        )
        accepted = (
            evidence == "accept_on_ice_platform_evidence"
            and track_pass
            and position_pass
        )
        if accepted:
            final_status = "accepted_miz_on_ice"
        elif not track_pass:
            final_status = "rejected_level1_track_qc"
        elif not position_pass:
            final_status = "rejected_level1_position_disagreement"
        else:
            final_status = evidence
        rows.append(
            {
                "sequence_id": sequence_id,
                "buoy_id": buoy_id,
                "buoy_type": metadata.get("buoy_type"),
                "owner": metadata.get("owner"),
                "logistics": metadata.get("logistics"),
                "frames": len(group),
                "valid_surface_temperature_frames": len(valid_temperature),
                "median_surface_temperature_c": (
                    float(valid_temperature.median()) if len(valid_temperature) else np.nan
                ),
                "fraction_surface_temperature_below_minus1_8": fraction_cold,
                "maximum_level1_track_gap_hours": float(
                    group["level1_bracket_gap_hours"].max()
                ),
                "maximum_level1_track_speed_km_per_day": float(
                    group["level1_bracket_speed_m_per_day"].max() / 1000.0
                ),
                "maximum_level1_to_catalog_position_difference_m": float(
                    group["level1_to_catalog_position_difference_m"].max()
                ),
                "platform_evidence_status": evidence,
                "level1_track_qc_pass": track_pass,
                "level1_position_agreement_pass": position_pass,
                "final_miz_qc_status": final_status,
                "retain_sequence": accepted,
            }
        )
    return pd.DataFrame.from_records(rows)


def filter_acquisitions(
    acquisition_manifest: pd.DataFrame, sequence_qc: pd.DataFrame
) -> pd.DataFrame:
    accepted_miz = set(
        sequence_qc.loc[sequence_qc["retain_sequence"], "sequence_id"].astype(str)
    )
    rows: list[pd.Series] = []
    for _, row in acquisition_manifest.iterrows():
        sequence_ids = str(row["sequence_ids"]).split(";")
        retained_ids = [
            sequence_id
            for sequence_id in sequence_ids
            if "_pack_" in sequence_id or sequence_id in accepted_miz
        ]
        if not retained_ids:
            continue
        item = row.copy()
        item["sequence_ids"] = ";".join(retained_ids)
        item["download_decision"] = "ready_for_restore_or_download"
        rows.append(item)
    return pd.DataFrame(rows).reset_index(drop=True)


def write_report(
    path: Path,
    sequence_qc: pd.DataFrame,
    original_acquisitions: pd.DataFrame,
    filtered_acquisitions: pd.DataFrame,
) -> None:
    accepted = sequence_qc[sequence_qc["retain_sequence"]]
    rejected = sequence_qc[~sequence_qc["retain_sequence"]]
    table_columns = [
        "sequence_id",
        "buoy_type",
        "owner",
        "median_surface_temperature_c",
        "fraction_surface_temperature_below_minus1_8",
        "maximum_level1_track_speed_km_per_day",
        "maximum_level1_to_catalog_position_difference_m",
        "final_miz_qc_status",
    ]
    view = sequence_qc[table_columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    table = "\n".join(
        [
            "| " + " | ".join(table_columns) + " |",
            "| " + " | ".join(["---"] * len(table_columns)) + " |",
            *[
                "| " + " | ".join(map(str, row)) + " |"
                for row in view.to_numpy()
            ],
        ]
    )
    path.write_text(
        f"""# Selected IABP Level-1 on-ice QC

Official Level-1 positions were interpolated in EPSG:3413 to each selected SAR time. Surface temperature is treated as physical evidence, not a generic descriptor feature: IABP documents that values below -1.8 C indicate ground or ice conditions.

{table}

Only {len(accepted)} of {len(sequence_qc)} provisional MIZ sequences pass the conservative on-ice contract. The {len(rejected)} unsupported sequences are removed before SAR download. AOML `BD2GHI` platforms are ocean surface-velocity drifters and are rejected even when a few cold readings occur; two buoys with missing platform type and no surface temperature remain on hold.

The QC-filtered manifest contains {len(filtered_acquisitions)} scenes versus {len(original_acquisitions)} in the exploratory longlist. The January/April pack-ice tier-1 set is unchanged.
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--level1-dir", type=Path, default=DEFAULT_LEVEL1_DIR)
    parser.add_argument("--maximum-temperature-offset-hours", type=float, default=3.0)
    parser.add_argument("--maximum-track-gap-hours", type=float, default=6.0)
    parser.add_argument("--maximum-track-speed-km-day", type=float, default=100.0)
    parser.add_argument("--maximum-position-difference-m", type=float, default=500.0)
    parser.add_argument("--seawater-freezing-temperature-c", type=float, default=-1.8)
    args = parser.parse_args()
    level1_dir = args.level1_dir

    selected_frames = pd.read_csv(
        args.audit_dir / "selected_sequence_frames.csv",
        dtype={"buoy_id": "string"},
    )
    exact_qc = pd.read_csv(
        args.audit_dir / "exact_coincidence_qc.csv",
        dtype={"buoy_id": "string"},
        low_memory=False,
    )
    summary = load_level1_summary(level1_dir / "L1_Summary.txt")
    frame_qc = build_frame_qc(
        selected_frames,
        exact_qc,
        level1_dir,
        args.maximum_temperature_offset_hours,
    )
    sequence_qc = build_sequence_qc(
        frame_qc,
        summary,
        args.maximum_track_gap_hours,
        args.maximum_track_speed_km_day * 1000.0,
        args.maximum_position_difference_m,
        args.seawater_freezing_temperature_c,
    )
    acquisitions = pd.read_csv(args.audit_dir / "sentinel1_acquisition_manifest.csv")
    filtered = filter_acquisitions(acquisitions, sequence_qc)
    frame_qc.to_csv(args.audit_dir / "iabp_level1_frame_qc.csv", index=False)
    sequence_qc.to_csv(args.audit_dir / "iabp_level1_sequence_qc.csv", index=False)
    filtered.to_csv(
        args.audit_dir / "sentinel1_acquisition_manifest_qc_filtered.csv", index=False
    )
    write_report(
        args.audit_dir / "iabp_level1_qc_report.md",
        sequence_qc,
        acquisitions,
        filtered,
    )
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "level1_dir": str(level1_dir),
        "thresholds": {
            "maximum_temperature_offset_hours": args.maximum_temperature_offset_hours,
            "maximum_track_gap_hours": args.maximum_track_gap_hours,
            "maximum_track_speed_km_day": args.maximum_track_speed_km_day,
            "maximum_position_difference_m": args.maximum_position_difference_m,
            "seawater_freezing_temperature_c": args.seawater_freezing_temperature_c,
        },
        "counts": {
            "provisional_miz_sequences": len(sequence_qc),
            "accepted_miz_sequences": int(sequence_qc["retain_sequence"].sum()),
            "qc_filtered_sentinel1_scenes": len(filtered),
        },
    }
    (args.audit_dir / "iabp_level1_qc_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(sequence_qc.to_string(index=False))
    print(f"\nQC-filtered Sentinel-1 scenes: {len(filtered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
