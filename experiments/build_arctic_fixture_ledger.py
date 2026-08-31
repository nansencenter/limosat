#!/usr/bin/env python3
"""Build reproducible Arctic SAR/buoy fixtures without reading image pixels.

Every retained buoy position is linearly interpolated in EPSG:3413 to the exact
SAR acquisition time. Records outside the observed buoy interval are counted
and excluded; no temporal extrapolation is allowed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FixtureSpec:
    sequence: str
    role: str
    source: str
    catalog: Path
    buoys: Path


def repaired_path(value: str) -> str:
    return str(value).replace(
        "/Users/seachu/arktalas/arktalas_vae",
        "/Users/seachu/results/arktalas_vae",
    )


def _utc_seconds(values) -> np.ndarray:
    timestamps = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    return timestamps.astype("int64").to_numpy(dtype=np.float64) / 1.0e9


def interpolate_projected(track: pd.DataFrame, when) -> tuple[float, float]:
    ordered = track.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True, errors="coerce")
    ordered = (
        ordered.dropna(subset=["timestamp", "x", "y"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
    )
    if len(ordered) < 2:
        raise ValueError("fewer than two unique buoy observations")
    times = _utc_seconds(ordered["timestamp"])
    target = float(_utc_seconds([when])[0])
    if target < times[0] or target > times[-1]:
        raise ValueError("SAR acquisition lies outside the buoy-track interval")
    return (
        float(np.interp(target, times, ordered["x"].to_numpy(dtype=float))),
        float(np.interp(target, times, ordered["y"].to_numpy(dtype=float))),
    )


def load_catalog(path: Path) -> pd.DataFrame:
    catalog = gpd.read_file(path).copy()
    path_column = "filepath" if "filepath" in catalog else "filename"
    catalog["image_filepath"] = catalog[path_column].astype(str).map(repaired_path)
    catalog["image_filename"] = catalog["image_filepath"].map(lambda value: Path(value).name)
    catalog["image_time"] = pd.to_datetime(catalog["datetime"], utc=True)
    catalog["image_exists"] = catalog["image_filepath"].map(lambda value: Path(value).exists())
    return catalog


def _exact_records(
    matches: pd.DataFrame,
    tracks: pd.DataFrame,
    buoy_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    tracks_by_buoy = {str(key): group for key, group in tracks.groupby(buoy_column)}
    records: list[dict] = []
    outside_track = 0
    underdetermined_track = 0
    for row in matches.to_dict("records"):
        buoy_id = str(row[buoy_column])
        try:
            x, y = interpolate_projected(tracks_by_buoy[buoy_id], row["image_time"])
        except ValueError as error:
            if "outside" in str(error):
                outside_track += 1
            else:
                underdetermined_track += 1
            continue
        records.append(
            {
                "buoy_id": buoy_id,
                "image_id": int(row["image_id"]),
                "image_filename": row["image_filename"],
                "image_filepath": row["image_filepath"],
                "image_time": pd.Timestamp(row["image_time"]),
                "nearest_observation_time": pd.Timestamp(row["timestamp"]),
                "nearest_observation_offset_minutes": float(row["abs_time_diff_min"]),
                "x": x,
                "y": y,
                "analysis_crs": "EPSG:3413",
            }
        )
    exact = pd.DataFrame.from_records(records)
    if len(exact):
        exact = exact.sort_values(["buoy_id", "image_time", "image_id"]).reset_index(drop=True)
    return exact, {
        "outside_track_records": outside_track,
        "underdetermined_track_records": underdetermined_track,
    }


def build_iabp_fixture(
    catalog: pd.DataFrame,
    buoy_path: Path,
    max_time_difference_minutes: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = gpd.read_file(buoy_path).copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["abs_time_diff_min"] = rows["time_diff_min"].abs()
    by_name = catalog.set_index("image_filename")
    matches = rows[
        rows["image_filename"].isin(by_name.index)
        & (rows["abs_time_diff_min"] <= max_time_difference_minutes)
    ].copy()
    matches = (
        matches.sort_values("abs_time_diff_min")
        .drop_duplicates(["BuoyID", "image_filename"])
        .reset_index(drop=True)
    )
    matches = matches.join(
        by_name[["image_id", "image_filepath", "image_time"]],
        on="image_filename",
        how="left",
    )
    tracks = (
        rows[["BuoyID", "timestamp", "x", "y"]]
        .dropna()
        .groupby(["BuoyID", "timestamp"], as_index=False)[["x", "y"]]
        .mean()
    )
    exact, diagnostics = _exact_records(matches, tracks, "BuoyID")
    diagnostics["matched_records_before_exact_time_filter"] = len(matches)
    return exact, diagnostics


def build_nice2015_fixture(
    catalog: pd.DataFrame,
    buoy_path: Path,
    max_time_difference_minutes: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = gpd.read_file(buoy_path).copy()
    min_x, min_y, max_x, max_y = rows.total_bounds
    declared_geographic_but_projected = (
        rows.crs is not None
        and rows.crs.to_epsg() == 4326
        and (min_x < -180 or max_x > 180 or min_y < -90 or max_y > 90)
    )
    if declared_geographic_but_projected:
        rows = rows.set_crs(3413, allow_override=True)
    else:
        rows = rows.to_crs(3413)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["x"] = rows.geometry.x
    rows["y"] = rows.geometry.y
    rows["image_filename"] = rows["image_filepath"].astype(str).map(lambda value: Path(value).name)
    rows["abs_time_diff_min"] = rows["time_difference_minutes"].abs()
    by_name = catalog.set_index("image_filename")
    matches = rows[
        rows["image_filename"].isin(by_name.index)
        & (rows["abs_time_diff_min"] <= max_time_difference_minutes)
    ].copy()
    matches = (
        matches.sort_values("abs_time_diff_min")
        .drop_duplicates(["buoy_name", "image_filename"])
        .drop(columns=["image_filepath"])
        .join(
            by_name[["image_id", "image_filepath", "image_time"]],
            on="image_filename",
            how="left",
        )
        .reset_index(drop=True)
    )
    tracks = (
        rows[["buoy_name", "timestamp", "x", "y"]]
        .dropna()
        .groupby(["buoy_name", "timestamp"], as_index=False)[["x", "y"]]
        .mean()
    )
    exact, diagnostics = _exact_records(matches, tracks, "buoy_name")
    diagnostics["matched_records_before_exact_time_filter"] = len(matches)
    diagnostics["source_crs_metadata_overridden"] = int(declared_geographic_but_projected)
    return exact, diagnostics


def summarize_fixture(
    spec: FixtureSpec,
    catalog: pd.DataFrame,
    coincidences: pd.DataFrame,
    diagnostics: dict[str, int],
) -> dict:
    if len(coincidences):
        sizes = coincidences.groupby("buoy_id").size()
        gaps = (
            coincidences.sort_values(["buoy_id", "image_time"])
            .groupby("buoy_id")["image_time"]
            .diff()
            .dt.total_seconds()
            .div(3600.0)
            .dropna()
        )
    else:
        sizes = pd.Series(dtype=int)
        gaps = pd.Series(dtype=float)
    return {
        "sequence": spec.sequence,
        "role": spec.role,
        "buoy_source": spec.source,
        "preprocessing": "balanced_q2q98_clahe25",
        "analysis_crs": "EPSG:3413",
        "catalog_images": len(catalog),
        "existing_catalog_images": int(catalog["image_exists"].sum()),
        "source_crs_metadata_overridden": diagnostics.get(
            "source_crs_metadata_overridden", 0
        ),
        **diagnostics,
        "exact_time_coincidences": len(coincidences),
        "buoys": int(coincidences["buoy_id"].nunique()) if len(coincidences) else 0,
        "images": int(coincidences["image_id"].nunique()) if len(coincidences) else 0,
        "paths_ge_2": int((sizes >= 2).sum()),
        "paths_ge_3": int((sizes >= 3).sum()),
        "transitions": int((sizes - 1).clip(lower=0).sum()),
        "median_gap_hours": float(gaps.median()) if len(gaps) else np.nan,
        "p90_gap_hours": float(gaps.quantile(0.9)) if len(gaps) else np.nan,
        "max_gap_hours": float(gaps.max()) if len(gaps) else np.nan,
        "catalog": str(spec.catalog),
        "buoys_path": str(spec.buoys),
    }


def write_report(path: Path, summary: pd.DataFrame) -> None:
    view = summary[
        [
            "sequence",
            "role",
            "exact_time_coincidences",
            "buoys",
            "images",
            "paths_ge_3",
            "transitions",
            "median_gap_hours",
            "outside_track_records",
        ]
    ].copy()
    view["median_gap_hours"] = view["median_gap_hours"].map(
        lambda value: "" if pd.isna(value) else f"{value:.2f}"
    )
    columns = view.columns.tolist()
    table = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()],
    ]
    path.write_text(
        "# Arctic exact-time SAR/buoy fixture ledger\n\n"
        "All positions are interpolated in EPSG:3413 to the SAR acquisition time. "
        "Temporal extrapolation is forbidden. Image pixels are not read at this stage; "
        "scene containment, mask state, and local texture are measured by the downstream "
        "descriptor/graph experiment.\n\n"
        + "\n".join(table)
        + "\n\nThe split is by sequence: March 2020 is development, February 2020 is "
        "validation, and N-ICE2015 is a different-season holdout. No path is split "
        "between roles.\n"
    )


def default_specs() -> tuple[FixtureSpec, ...]:
    root = Path(
        "/Users/seachu/results/arktalas_vae/reports/"
        "generalist_export_contrast_sweep_20260308b/catalogs"
    )
    iabp = Path("/Users/seachu/data/shared/iapb_buoy_satellite_matches_2020_01_04_new.geojson")
    nice = Path(
        "/Users/seachu/projects/arktalas_ice_drift_experiments/data/"
        "N-ICE2015buoy_arktalas_image_buoy_matches_filtered.geojson"
    )
    return (
        FixtureSpec("2020_03", "development", "IABP", root / "2020_03/balanced_q2q98_clahe25.geojson", iabp),
        FixtureSpec("2020_02", "validation", "IABP", root / "2020_02/balanced_q2q98_clahe25.geojson", iabp),
        FixtureSpec("2015_full15", "holdout", "N-ICE2015", root / "2015_full15/balanced_q2q98_clahe25.geojson", nice),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-time-difference-minutes", type=float, default=60.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in default_specs():
        catalog = load_catalog(spec.catalog)
        if spec.source == "IABP":
            coincidences, diagnostics = build_iabp_fixture(
                catalog, spec.buoys, args.max_time_difference_minutes
            )
        else:
            coincidences, diagnostics = build_nice2015_fixture(
                catalog, spec.buoys, args.max_time_difference_minutes
            )
        coincidences.insert(0, "sequence", spec.sequence)
        coincidences.insert(1, "role", spec.role)
        coincidences.to_csv(args.out_dir / f"coincidences_{spec.sequence}.csv", index=False)
        summaries.append(summarize_fixture(spec, catalog, coincidences, diagnostics))

    summary = pd.DataFrame.from_records(summaries)
    summary.to_csv(args.out_dir / "fixture_summary.csv", index=False)
    write_report(args.out_dir / "report.md", summary)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "max_time_difference_minutes": args.max_time_difference_minutes,
                "exact_time_interpolation": True,
                "temporal_extrapolation": False,
                "analysis_crs": "EPSG:3413",
                "preprocessing": "balanced_q2q98_clahe25",
                "specs": [
                    {
                        **spec.__dict__,
                        "catalog": str(spec.catalog),
                        "buoys": str(spec.buoys),
                    }
                    for spec in default_specs()
                ],
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
