#!/usr/bin/env python3
"""Inventory monthly Arctic buoy/SAR fixtures and local pixel readiness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from build_arctic_fixture_ledger import build_iabp_fixture  # noqa: E402


DEFAULT_CATALOG = Path("/Users/seachu/data/shared/s1_2020_01_04_sie_filtered_new.geojson")
DEFAULT_BUOYS = Path("/Users/seachu/data/shared/iapb_buoy_satellite_matches_2020_01_04_new.geojson")
DEFAULT_VAE_CATALOG_ROOT = Path(
    "/Users/seachu/results/arktalas_vae/reports/"
    "generalist_export_contrast_sweep_20260308b/catalogs"
)
DEFAULT_OUTPUT = ROOT / "results/arctic_sequence_extension_inventory"


def normalize_catalog(path: Path) -> pd.DataFrame:
    catalog = gpd.read_file(path).copy()
    catalog["image_filename"] = catalog["filename"].astype(str).map(
        lambda value: Path(value).name
    )
    catalog["image_filepath"] = catalog["filepath"].astype(str)
    catalog["image_time"] = pd.to_datetime(catalog["timestamp"], utc=True)
    catalog["month"] = catalog.image_time.dt.strftime("%Y_%m")
    return catalog


def local_vae_files(catalog_root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(catalog_root.glob("*/balanced_q2q98_clahe25.geojson")):
        catalog = gpd.read_file(path)
        path_column = "filepath" if "filepath" in catalog else "filename"
        for value in catalog[path_column].astype(str):
            local = Path(
                value.replace(
                    "/Users/seachu/arktalas/arktalas_vae",
                    "/Users/seachu/results/arktalas_vae",
                )
            )
            if local.exists():
                files[local.name] = str(local)
    return files


def summarize_month(
    month: str,
    catalog: pd.DataFrame,
    coincidences: pd.DataFrame,
    diagnostics: dict,
    local_files: dict[str, str],
) -> tuple[dict, pd.DataFrame]:
    sizes = coincidences.groupby("buoy_id").size()
    gaps = (
        coincidences.sort_values(["buoy_id", "image_time"])
        .groupby("buoy_id")["image_time"]
        .diff()
        .dt.total_seconds()
        .div(3600.0)
        .dropna()
    )
    local_mask = catalog.image_filename.isin(local_files)
    pixel_ready_names = set(catalog.loc[local_mask, "image_filename"])
    ready_coincidences = coincidences[
        coincidences.image_filename.isin(pixel_ready_names)
    ]
    summary = {
        "month": month,
        "catalog_images": len(catalog),
        "catalog_paths_exist_as_recorded": int(
            catalog.image_filepath.map(lambda value: Path(value).exists()).sum()
        ),
        "standard_vae_images_local": int(local_mask.sum()),
        "matched_records_before_exact_time_filter": diagnostics[
            "matched_records_before_exact_time_filter"
        ],
        "outside_track_records": diagnostics["outside_track_records"],
        "exact_time_coincidences": len(coincidences),
        "pixel_ready_coincidences": len(ready_coincidences),
        "buoys": int(coincidences.buoy_id.nunique()),
        "images_with_coincidences": int(coincidences.image_id.nunique()),
        "paths_ge_2": int((sizes >= 2).sum()),
        "paths_ge_3": int((sizes >= 3).sum()),
        "transitions": int((sizes - 1).clip(lower=0).sum()),
        "median_gap_hours": float(gaps.median()) if len(gaps) else np.nan,
        "p90_gap_hours": float(gaps.quantile(0.9)) if len(gaps) else np.nan,
        "maximum_gap_hours": float(gaps.max()) if len(gaps) else np.nan,
    }
    path_summary = (
        coincidences.groupby("buoy_id")
        .agg(
            observations=("image_id", "size"),
            first_image_time=("image_time", "min"),
            last_image_time=("image_time", "max"),
            unique_images=("image_id", "nunique"),
        )
        .reset_index()
    )
    path_summary.insert(0, "month", month)
    return summary, path_summary


def markdown_table(data: pd.DataFrame, columns: list[str]) -> str:
    view = data[columns].copy()
    for column in view.select_dtypes(include=["float"]).columns:
        view[column] = view[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.2f}"
        )
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in view.to_numpy()],
        ]
    )


def write_report(path: Path, summary: pd.DataFrame) -> None:
    columns = [
        "month",
        "exact_time_coincidences",
        "buoys",
        "images_with_coincidences",
        "paths_ge_3",
        "transitions",
        "median_gap_hours",
        "standard_vae_images_local",
        "pixel_ready_coincidences",
    ]
    path.write_text(
        "# Arctic sequence extension inventory\n\n"
        "The January-April 2020 IABP/Sentinel-1 catalogue was converted to exact-time "
        "EPSG:3413 buoy positions without temporal extrapolation. These counts describe "
        "potential descriptor fixtures; they do not claim that image pixels were read.\n\n"
        + markdown_table(summary, columns)
        + "\n\nThe connected drive contains the catalogue, buoy tracks, LiMOSAT trajectory "
        "outputs, and sea-ice-concentration products. It does not contain the recorded "
        "`/Data/sat/downloads/.../processed_VAE_2_16_ELU_64` rasters. The only local "
        "standard-VAE image subsets found are the already frozen February and March "
        "catalogues. January and April are therefore the best next regime extensions, "
        "but require restoring or regenerating their standard-VAE rasters first.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--buoys", type=Path, default=DEFAULT_BUOYS)
    parser.add_argument("--vae-catalog-root", type=Path, default=DEFAULT_VAE_CATALOG_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--maximum-time-difference-minutes", type=float, default=60.0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    catalog = normalize_catalog(args.catalog)
    local_files = local_vae_files(args.vae_catalog_root)
    summaries = []
    path_summaries = []
    for month, month_catalog in catalog.groupby("month", sort=True):
        coincidences, diagnostics = build_iabp_fixture(
            month_catalog,
            args.buoys,
            args.maximum_time_difference_minutes,
        )
        summary, paths = summarize_month(
            month,
            month_catalog,
            coincidences,
            diagnostics,
            local_files,
        )
        summaries.append(summary)
        path_summaries.append(paths)
        coincidences.to_csv(
            args.out_dir / f"potential_exact_time_coincidences_{month}.csv",
            index=False,
        )

    summary = pd.DataFrame.from_records(summaries)
    paths = pd.concat(path_summaries, ignore_index=True)
    summary.to_csv(args.out_dir / "monthly_summary.csv", index=False)
    paths.to_csv(args.out_dir / "monthly_buoy_path_summary.csv", index=False)
    write_report(args.out_dir / "report.md", summary)
    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "catalog": str(args.catalog),
        "buoys": str(args.buoys),
        "vae_catalog_root": str(args.vae_catalog_root),
        "maximum_time_difference_minutes": args.maximum_time_difference_minutes,
        "exact_time_interpolation": True,
        "temporal_extrapolation": False,
        "analysis_crs": "EPSG:3413",
        "local_standard_vae_files": len(local_files),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
