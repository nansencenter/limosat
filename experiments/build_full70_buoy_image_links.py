#!/usr/bin/env python3
"""Build explicit all-buoy links and controls for the 70-scene experiment."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results/iabp_s1_stratified_coverage"
DEFAULT_LEVEL1_DIR = Path(
    "/Volumes/KINGSTON/arktalas/experiments/"
    "limosat_descriptor_update_2020/iabp_level1_full70"
)
DEFAULT_CATALOG = Path(
    "/Users/seachu/data/shared/s1_2020_01_04_sie_filtered_new.geojson"
)
DEFAULT_DATA_ROOT = Path(
    "/Volumes/KINGSTON/arktalas/experiments/limosat_descriptor_update_2020"
)


def clean_buoy_id(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def named_buoy_target(row: pd.Series) -> bool:
    targets = {item.strip() for item in str(row["buoy_ids"]).split(";")}
    return str(row["buoy_id"]) in targets


def link_status(row: pd.Series) -> str:
    if not bool(row["exact_position_inside_scene"]):
        return "exclude_exact_position_outside_image"
    if not bool(row["descriptor_border_safe"]):
        return "hold_128px_image_border"
    if row["track_qc_status"] != "pass":
        return "hold_full_level1_track_context"
    if row["buoy_ice_qc_status"] == "provisional_miz_needs_platform_qc":
        return "hold_level1_on_ice_platform_evidence"
    if row["buoy_ice_qc_status"] == "on_ice_high_confidence_from_sic_track":
        return "ready_current_catalog_qc"
    return "hold_other_buoy_qc"


def build_links(exact_qc: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    exact = exact_qc.copy()
    exact["buoy_id"] = clean_buoy_id(exact["buoy_id"])
    inventory = inventory.copy()
    inventory["image_id"] = pd.to_numeric(inventory["image_id"], errors="raise")
    inventory_columns = [
        "image_id",
        "sentinel1_product_name",
        "resolved_product_name",
        "raw_zip_path",
        "standard_vae_output_path",
        "sequence_ids",
        "buoy_ids",
    ]
    links = exact[exact["image_id"].isin(inventory["image_id"])].merge(
        inventory[inventory_columns], on="image_id", how="left", validate="many_to_one"
    )
    links["image_time"] = pd.to_datetime(links["image_time"], utc=True)
    links["buoy_is_named_sequence_target"] = links.apply(named_buoy_target, axis=1)
    links["buoy_link_role"] = np.where(
        links["buoy_is_named_sequence_target"],
        "named_sequence_buoy",
        "additional_buoy_in_selected_image",
    )
    links["current_experiment_status"] = links.apply(link_status, axis=1)
    links["truth_ready_before_level1"] = links["current_experiment_status"].eq(
        "ready_current_catalog_qc"
    )
    links["descriptor_image_border_pixels"] = 128
    return links.sort_values(["buoy_id", "image_time", "image_id"]).reset_index(
        drop=True
    )


def build_level1_targets(links: pd.DataFrame, level1_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for buoy_id, group in links.groupby("buoy_id", sort=True):
        records.append(
            {
                "buoy_id": buoy_id,
                "linked_images": len(group),
                "named_sequence_images": int(group["buoy_is_named_sequence_target"].sum()),
                "truth_ready_before_level1": int(group["truth_ready_before_level1"].sum()),
                "track_context_holds": int(
                    group["current_experiment_status"].eq(
                        "hold_full_level1_track_context"
                    ).sum()
                ),
                "on_ice_platform_holds": int(
                    group["current_experiment_status"].eq(
                        "hold_level1_on_ice_platform_evidence"
                    ).sum()
                ),
                "months": ";".join(sorted(group["month"].astype(str).unique())),
                "spatial_blocks": ";".join(
                    sorted(group["spatial_block"].astype(str).unique())
                ),
                "first_linked_image_time": group["image_time"].min(),
                "last_linked_image_time": group["image_time"].max(),
                "iabp_level1_download_url": (
                    "https://iabp.apl.uw.edu/downloadL1?bid="
                    f"{buoy_id}&requesttype=bybuoy&option=download"
                ),
                "level1_destination_path": str(level1_dir / f"{buoy_id}.csv"),
            }
        )
    return pd.DataFrame.from_records(records)


def build_same_pass_scene_pairs(inventory: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    ordered = inventory.copy()
    ordered["image_time"] = pd.to_datetime(ordered["image_time"], utc=True)
    for pass_id, group in ordered.groupby("acquisition_pass_id", sort=True):
        if len(group) < 2:
            continue
        rows = list(group.sort_values("image_time").itertuples(index=False))
        for first, second in itertools.combinations(rows, 2):
            records.append(
                {
                    "same_pass_pair_id": f"{pass_id}_{first.image_id}_{second.image_id}",
                    "acquisition_pass_id": pass_id,
                    "time_separation_seconds": (
                        second.image_time - first.image_time
                    ).total_seconds(),
                    "first_image_id": first.image_id,
                    "first_product_name": first.resolved_product_name,
                    "first_standard_vae_path": first.standard_vae_output_path,
                    "second_image_id": second.image_id,
                    "second_product_name": second.resolved_product_name,
                    "second_standard_vae_path": second.standard_vae_output_path,
                    "control_purpose": (
                        "measure_adjacent_slice_seam_consistency_without_repeat_pixels"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def logical_product_key(product_name: str) -> str:
    return "_".join(str(product_name).split("_")[:-1])


def build_repeat_publication_controls(
    inventory: pd.DataFrame,
    catalog: pd.DataFrame,
    data_root: Path,
) -> pd.DataFrame:
    selected = inventory.copy()
    selected["logical_product_key"] = selected["resolved_product_name"].map(
        logical_product_key
    )
    candidates = catalog.copy()
    candidates["catalog_product_name"] = candidates["filename"].astype(str).map(
        lambda value: Path(value).stem
    )
    candidates["logical_product_key"] = candidates["catalog_product_name"].map(
        logical_product_key
    )
    records: list[dict[str, object]] = []
    for primary in selected.itertuples(index=False):
        group = candidates[
            candidates["logical_product_key"].eq(primary.logical_product_key)
        ]
        repeats = group[
            ~group["catalog_product_name"].eq(primary.resolved_product_name)
        ]
        primary_geometry_rows = group[
            group["catalog_product_name"].eq(primary.resolved_product_name)
        ]
        if len(primary_geometry_rows) != 1:
            continue
        primary_geometry = primary_geometry_rows.iloc[0]["geometry"]
        for repeat in repeats.itertuples(index=False):
            month = pd.Timestamp(primary.image_time).strftime("%m")
            platform_dir = f"S{repeat.catalog_product_name[2]}"
            repeat_raw_path = (
                data_root
                / "sentinel1"
                / "repeat_publication_controls"
                / "raw"
                / "2020"
                / month
                / f"{repeat.catalog_product_name}.zip"
            )
            repeat_vae_path = (
                data_root
                / "sentinel1"
                / "repeat_publication_controls"
                / "standard_vae"
                / "2020"
                / month
                / f"{repeat.catalog_product_name}.tiff"
            )
            intersection_area = primary_geometry.intersection(repeat.geometry).area
            smaller_area = min(primary_geometry.area, repeat.geometry.area)
            records.append(
                {
                    "repeat_control_id": primary.logical_product_key,
                    "image_time": primary.image_time,
                    "primary_image_id": primary.image_id,
                    "primary_product_name": primary.resolved_product_name,
                    "primary_raw_zip_path": primary.raw_zip_path,
                    "primary_standard_vae_path": primary.standard_vae_output_path,
                    "repeat_catalog_image_id": repeat.image_id,
                    "repeat_product_name": repeat.catalog_product_name,
                    "repeat_asf_url": (
                        "https://datapool.asf.alaska.edu/GRD_MD/"
                        f"{platform_dir}/{repeat.catalog_product_name}.zip"
                    ),
                    "repeat_raw_zip_path": str(repeat_raw_path),
                    "repeat_standard_vae_path": str(repeat_vae_path),
                    "catalog_footprint_overlap_fraction": (
                        float(intersection_area / smaller_area)
                        if smaller_area > 0
                        else np.nan
                    ),
                    "control_purpose": (
                        "audit_catalog_duplicate_before_repeat_consistency_test"
                    ),
                }
            )
    return pd.DataFrame.from_records(records).sort_values(
        ["image_time", "repeat_product_name"]
    ).reset_index(drop=True)


def cadence_band(hours: float) -> str:
    if hours <= 3:
        return "00_to_03h"
    if hours <= 12:
        return "03_to_12h"
    if hours <= 30:
        return "12_to_30h"
    if hours <= 72:
        return "30_to_72h"
    return "over_72h"


def build_buoy_transitions(links: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for buoy_id, group in links.groupby("buoy_id", sort=True):
        rows = list(group.sort_values(["image_time", "image_id"]).itertuples(index=False))
        for source, target in zip(rows[:-1], rows[1:]):
            hours = (target.image_time - source.image_time).total_seconds() / 3600.0
            distance_m = float(np.hypot(target.x - source.x, target.y - source.y))
            ready = bool(
                0 < hours <= 72
                and source.truth_ready_before_level1
                and target.truth_ready_before_level1
            )
            records.append(
                {
                    "transition_id": (
                        f"{buoy_id}_{source.image_id}_{target.image_id}"
                    ),
                    "buoy_id": buoy_id,
                    "source_image_id": source.image_id,
                    "source_image_time": source.image_time,
                    "source_month": source.month,
                    "source_spatial_block": source.spatial_block,
                    "source_sic_regime": source.sic_regime,
                    "source_standard_vae_path": source.standard_vae_output_path,
                    "target_image_id": target.image_id,
                    "target_image_time": target.image_time,
                    "target_month": target.month,
                    "target_spatial_block": target.spatial_block,
                    "target_sic_regime": target.sic_regime,
                    "target_standard_vae_path": target.standard_vae_output_path,
                    "elapsed_hours": hours,
                    "cadence_band": cadence_band(hours),
                    "truth_dx_m": float(target.x - source.x),
                    "truth_dy_m": float(target.y - source.y),
                    "truth_distance_m": distance_m,
                    "truth_speed_km_per_day": (
                        distance_m / 1000.0 / (hours / 24.0) if hours > 0 else np.nan
                    ),
                    "source_status": source.current_experiment_status,
                    "target_status": target.current_experiment_status,
                    "ready_for_tracking_before_level1": ready,
                }
            )
    return pd.DataFrame.from_records(records)


def write_outputs(
    output_dir: Path,
    links: pd.DataFrame,
    level1_targets: pd.DataFrame,
    same_pass_pairs: pd.DataFrame,
    repeat_publication_controls: pd.DataFrame,
    transitions: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    links.to_csv(output_dir / "full70_buoy_image_links.csv", index=False)
    level1_targets.to_csv(output_dir / "full70_iabp_level1_targets.csv", index=False)
    same_pass_pairs.to_csv(output_dir / "full70_same_pass_scene_pairs.csv", index=False)
    repeat_publication_controls.to_csv(
        output_dir / "full70_repeat_publication_controls.csv", index=False
    )
    transitions.to_csv(output_dir / "full70_buoy_transitions.csv", index=False)
    observation_strata = (
        links.groupby(
            ["month", "sic_regime", "current_experiment_status", "buoy_link_role"],
            dropna=False,
        )
        .agg(
            observations=("image_id", "size"),
            buoys=("buoy_id", "nunique"),
            images=("image_id", "nunique"),
            spatial_blocks=("spatial_block", "nunique"),
        )
        .reset_index()
    )
    observation_strata.to_csv(
        output_dir / "full70_buoy_observation_strata.csv", index=False
    )
    transition_strata = (
        transitions.groupby(
            [
                "source_month",
                "cadence_band",
                "source_sic_regime",
                "ready_for_tracking_before_level1",
            ],
            dropna=False,
        )
        .agg(
            transitions=("transition_id", "size"),
            buoys=("buoy_id", "nunique"),
            source_images=("source_image_id", "nunique"),
            spatial_blocks=("source_spatial_block", "nunique"),
        )
        .reset_index()
    )
    transition_strata.to_csv(
        output_dir / "full70_buoy_transition_strata.csv", index=False
    )
    summary = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "linked_observations": len(links),
        "buoys": int(links["buoy_id"].nunique()),
        "images": int(links["image_id"].nunique()),
        "named_sequence_observations": int(
            links["buoy_is_named_sequence_target"].sum()
        ),
        "additional_observations_in_selected_images": int(
            (~links["buoy_is_named_sequence_target"]).sum()
        ),
        "truth_ready_observations_before_level1": int(
            links["truth_ready_before_level1"].sum()
        ),
        "level1_target_buoys": len(level1_targets),
        "same_pass_scene_pairs": len(same_pass_pairs),
        "catalog_repeat_publication_candidates_requiring_official_audit": len(
            repeat_publication_controls
        ),
        "consecutive_buoy_transitions": len(transitions),
        "tracking_ready_transitions_at_most_72h_before_level1": int(
            transitions["ready_for_tracking_before_level1"].sum()
        ),
        "units": {
            "coordinates": "EPSG:3413 metres",
            "time": "UTC",
            "image_border": "128 standard-VAE pixels",
        },
        "truth_policy": (
            "Buoy positions are used for extraction and scoring; future positions must "
            "not enter candidate generation or descriptor-state updates."
        ),
    }
    (output_dir / "full70_buoy_linkage_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--level1-dir", type=Path, default=DEFAULT_LEVEL1_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    inventory = pd.read_csv(args.results_dir / "full70_sentinel1_download_inventory.csv")
    exact_qc = pd.read_csv(
        args.results_dir / "exact_coincidence_qc.csv",
        dtype={"buoy_id": "string"},
        low_memory=False,
    )
    links = build_links(exact_qc, inventory)
    level1_targets = build_level1_targets(links, args.level1_dir)
    same_pass_pairs = build_same_pass_scene_pairs(inventory)
    import geopandas as gpd

    catalog = gpd.read_file(args.catalog)
    repeat_publication_controls = build_repeat_publication_controls(
        inventory, catalog, args.data_root
    )
    transitions = build_buoy_transitions(links)
    write_outputs(
        args.results_dir,
        links,
        level1_targets,
        same_pass_pairs,
        repeat_publication_controls,
        transitions,
    )
    print(
        f"{len(links)} buoy/image links, {len(level1_targets)} buoys, "
        f"{len(transitions)} transitions, {len(same_pass_pairs)} seam pairs, "
        f"{len(repeat_publication_controls)} catalogue repeat candidates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
