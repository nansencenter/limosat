#!/usr/bin/env python3
"""Prepare a chained pan-Arctic EfficientLoFTR run on Olivia (any UTC window).

Writes ``control/catalogue.json`` and ``control/config.json`` under the run root for
``submit_chain_olivia.sh``. The configuration is the production one: next-observation
planning on ice cells (land and AMSR2 ASI open water dropped) with backups for pairs
longer than 30 h, the OSI-SAF tile gate, land (OSM inland distance) plus ASI point
open-water exclusion, composition with gap filling and shear point estimates, new points
seeded 3 km from existing ones and no trajectory merging. Matching ignores the scenes'
band-2 land mask (land comes from the OSM raster). Primary pair products go to a shared
store keyed by matching fingerprint, so runs with the same matching settings reuse them.
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

SOURCE_CATALOGUE_SHA256 = "12ba2654f82a7da78dff3724fd89f880583ea5347345009f2d7a57477c32f58d"
SIC_BASE_URL = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/ice/conc"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, value, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            if compact:
                json.dump(value, stream, separators=(",", ":"))
            else:
                json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def utc_day(value):
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def parse_utc(value):
    value = str(value).strip().replace(" ", "T", 1)
    for suffix in ("Z", "+00:00", "+0000"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    else:
        raise ValueError("timestamp is not explicitly UTC: %s" % value)
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in value else "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)


def feature_timestamp(properties):
    for name in ("datetime", "timestamp", "time_utc", "start_datetime"):
        value = properties.get(name)
        if value:
            return parse_utc(value)
    return None


def feature_filename(feature, properties):
    value = properties.get("filename") or properties.get("filepath") or properties.get("path")
    if value:
        return Path(str(value)).name
    for asset in (feature.get("assets") or {}).values():
        if asset.get("href"):
            return Path(str(asset["href"])).name
    return None


def prepare_catalogue(source_catalogue, scene_root, destination, start_utc, stop_utc,
                      expected_source_sha256=SOURCE_CATALOGUE_SHA256, expected_images=None):
    """Select the window's scenes from the STAC catalogue and point them at ``scene_root``."""
    actual_source_sha256 = file_sha256(source_catalogue)
    if expected_source_sha256 and actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            "source catalogue SHA256 mismatch: expected %s, found %s"
            % (expected_source_sha256, actual_source_sha256)
        )
    with Path(source_catalogue).open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("features"), list):
        raise ValueError("source catalogue is not a GeoJSON/STAC FeatureCollection")
    selected, missing, image_ids = [], [], []
    for original in document["features"]:
        feature = dict(original)
        properties = dict(feature.get("properties") or {})
        timestamp = feature_timestamp(properties)
        if timestamp is None or not start_utc <= timestamp < stop_utc:
            continue
        filename = feature_filename(feature, properties)
        if not filename:
            raise ValueError("selected feature has no raster filename")
        image_id = str(properties.get("scene_id") or feature.get("id") or Path(filename).stem)
        raster = Path(scene_root) / ("%04d" % timestamp.year) / ("%02d" % timestamp.month) / filename
        properties["scene_id"] = image_id
        properties["filepath"] = str(raster)
        feature["properties"] = properties
        selected.append(feature)
        image_ids.append(image_id)
        if not raster.is_file():
            missing.append(str(raster))
    if expected_images is not None and len(selected) != expected_images:
        raise ValueError("expected %d catalogue images, found %d" % (expected_images, len(selected)))
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("selected catalogue image IDs are not globally unique")
    if missing:
        raise FileNotFoundError("%d raster paths are missing; first: %s" % (len(missing), missing[0]))
    write_json_atomic(destination, dict(document, features=selected), compact=True)
    return {
        "catalogue_images": len(selected),
        "catalogue_sha256": file_sha256(destination),
        "source_catalogue_sha256": actual_source_sha256,
    }


def sic_filename(day):
    return "ice_conc_nh_polstere-100_multi_%s1200.nc" % day


def sic_days(start_utc, stop_utc):
    """OSI-SAF days covering the window with the gate's one-day tolerance."""
    day = (start_utc - timedelta(days=1)).date()
    while day <= stop_utc.date():
        yield day.strftime("%Y%m%d")
        day += timedelta(days=1)


def prepare_sic(root, days, download=False, base_url=SIC_BASE_URL):
    """Check (or download) the OSI-SAF daily files and write their SHA256SUMS."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for day in days:
        destination = root / sic_filename(day)
        if destination.is_file() and destination.stat().st_size:
            continue
        if not download:
            raise FileNotFoundError("missing %s; rerun with --download-sic" % destination)
        temporary = destination.with_name(destination.name + ".part")
        url = "%s/%s/%s/%s" % (base_url.rstrip("/"), day[:4], day[4:6], destination.name)
        try:
            subprocess.run(
                ["curl", "--fail", "--location", "--retry", "3", "--output", str(temporary), url],
                check=True,
            )
            os.replace(str(temporary), str(destination))
        finally:
            if temporary.exists():
                temporary.unlink()
    checksums = {sic_filename(day): file_sha256(root / sic_filename(day)) for day in days}
    (root / "SHA256SUMS").write_text(
        "".join("%s  %s\n" % (checksum, name) for name, checksum in sorted(checksums.items())),
        encoding="utf-8",
    )
    return checksums


def asi_coverage(root, grid, start_utc, stop_utc):
    """Count available daily ASI files; missing days keep positions (no exclusion)."""
    present, missing = 0, []
    day = start_utc.date()
    while day < stop_utc.date():
        name = "asi-AMSR2-%s-%s-v5.4" % (grid, day.strftime("%Y%m%d"))
        year = Path(root) / day.strftime("%Y")
        if (year / day.strftime("%b").lower() / (name + ".tif")).is_file() or (year / (name + ".nc")).is_file():
            present += 1
        else:
            missing.append(day.isoformat())
        day += timedelta(days=1)
    return {"present_days": present, "missing_days": missing}


def run_config(arguments, catalogue, run_root):
    """The production configuration. Matching settings must stay as they are: they key the
    shared pair store (tests/unit/test_production_fingerprint.py)."""
    surface = {
        "land_raster": str(arguments.land_raster),
        "land_minimum_inland_km": 1.0,
        "point_sic_root": str(arguments.asi_root),
        "point_sic_grid": arguments.asi_grid,
        "point_open_water_threshold_percent": 15.0,
    }
    if arguments.asi_fallback_root:
        surface["point_sic_fallback_root"] = str(arguments.asi_fallback_root)
        surface["point_sic_fallback_grid"] = arguments.asi_fallback_grid
    config = {
        "run_id": arguments.run_id,
        "catalogue": str(catalogue),
        "database": str(run_root / "outputs" / "limosat.sqlite"),
        "output_directory": str(run_root / "outputs"),
        "pair_product_directory": str(run_root / "work" / "pair-products"),
        # Written by the prepare stage; pair products record and check scene checksums.
        "scene_checksums": str(run_root / "control" / "scene-checksums.json"),
        "analysis_epsg": 3413,
        "pair_workers": 1,
        "retain_pair_matches": True,
        "matcher": {
            "repository": str(arguments.official_repository),
            "checkpoint": str(arguments.checkpoint),
            "model_name": "efficientloftr-official-opt",
            "device": "cuda",
            "pixel_size_m": 80.0,
            "tile_size_px": 512,
            "tile_margin_px": 32,
            "endpoint_support_radius_px": 16,
            "transform_grid_spacing_px": 32,
            "tile_grid_origin_m": 0.0,
            "maximum_speed_m_per_day": arguments.maximum_speed_m_per_day,
            "tile_batch_size": 4,
            "prefix_cuda_graph": True,
            "cuda_graph_warmup_batches": 3,
            "scene_mask_invalid_from": arguments.scene_mask_invalid_from,
        },
        "field": {
            "grid_spacing_m": 4000.0,
            "neighbour_count": 12,
            "minimum_agreeing_matches": 8,
            "maximum_neighbour_distance_m": 6000.0,
            "agreement_distance_m": 1000.0,
            "maximum_triangle_edge_m": 6400.0,
            "gap_fill_rigid_edge_m": 12800.0,
            "gap_fill_rigid_spread_m": 300.0,
            "gap_fill_point_neighbour_count": 12,
            "gap_fill_point_minimum_agreeing_matches": 8,
            "gap_fill_point_neighbour_distance_m": 3000.0,
            "gap_fill_point_agreement_distance_m": 500.0,
            "shear_point_estimates": True,
        },
        "routing": {
            "mode": "sequential_local",
            "initial": "phase_correlation",
            "phase_correlation_failure": "same_center",
            "phase_correlation_minimum_response": 0.05,
            "residual_edge_recovery": True,
            "candidate_minimum_elapsed_hours": 1.0,
            "candidate_maximum_elapsed_hours": 96.0,
            "candidate_minimum_overlap_fraction": 0.05,
            "candidate_minimum_overlap_area_m2": 1024000000.0,
            "exclude_same_acquisition_pass": True,
            "require_orbit_metadata": True,
            "next_observation_minimum_cells": 25,
            "next_observation_backup_after_hours": arguments.backup_after_hours,
            "plan_surface_cells": True,
        },
        "open_water": {
            "enabled": True,
            "sic_root": str(arguments.sic_root),
            "threshold_percent": 15.0,
            "maximum_age_days": 1,
            "samples_per_axis": 5,
        },
        "surface": surface,
        "trajectories": {
            "add_as_coverage_enters": True,
            "new_point_exclusion_radius_m": arguments.new_point_exclusion_radius_m,
            "merge_radius_m": arguments.merge_radius_m,
            "merge_maximum_younger_age_hours": arguments.merge_maximum_younger_age_hours,
        },
    }
    if not arguments.no_pair_store:
        config["pair_store_directory"] = str(arguments.pair_store)
    return config


def parser():
    project_id = os.environ.get("PROJECT_ID", "nn9878k")
    user = os.environ.get("USER", "seachu")
    project_root = Path("/cluster/projects") / project_id / user
    work_root = Path("/cluster/work/projects") / project_id / user
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", required=True)
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--start", type=utc_day, required=True, help="first UTC day (YYYY-MM-DD)")
    result.add_argument("--stop", type=utc_day, required=True, help="UTC day after the window")
    result.add_argument(
        "--source-catalogue",
        type=Path,
        default=project_root / "arktalas-deployment" / "data"
        / "201905_202007_image_catalog_nirdsync_20260824T220354Z.json",
    )
    result.add_argument("--expected-source-sha256", default=SOURCE_CATALOGUE_SHA256)
    result.add_argument(
        "--scene-root", type=Path,
        default=work_root / "limosat_staging" / "201905_202007" / "s1_preprocessed",
    )
    result.add_argument("--official-repository", type=Path, default=project_root / "EfficientLoFTR")
    result.add_argument(
        "--checkpoint", type=Path,
        default=project_root / "checkpoints" / "efficientloftr" / "eloftr_outdoor.ckpt",
    )
    result.add_argument("--sic-root", type=Path, required=True, help="OSI-SAF daily NetCDF directory")
    result.add_argument("--download-sic", action="store_true")
    result.add_argument("--land-raster", type=Path, required=True)
    result.add_argument("--asi-root", type=Path, required=True)
    result.add_argument("--asi-grid", default="n3125")
    result.add_argument("--asi-fallback-root", type=Path)
    result.add_argument("--asi-fallback-grid", default="n6250")
    result.add_argument("--expected-images", type=int, help="fail unless the window has this many scenes")
    result.add_argument("--maximum-speed-m-per-day", type=float, default=43200.0)
    result.add_argument(
        "--scene-mask-invalid-from", type=int, choices=(2, 3), default=3,
        help="2 keeps the scenes' band-2 land mask, which has false land near the date line",
    )
    result.add_argument("--backup-after-hours", type=float, default=30.0,
                        help="next-observation pairs longer than this get a backup pair")
    result.add_argument("--new-point-exclusion-radius-m", type=float, default=3000.0)
    result.add_argument("--merge-radius-m", type=float,
                        help="merge duplicate trajectories closer than this (default: no merging)")
    result.add_argument("--merge-maximum-younger-age-hours", type=float,
                        help="merge only trajectories at most this old (needs --merge-radius-m)")
    result.add_argument(
        "--pair-store", type=Path, default=work_root / "eloftr-pair-store",
        help="shared primary pair-product store (keyed by matching fingerprint); "
        "archived to NIRD with scripts/sync_pair_store_nird.sh",
    )
    result.add_argument("--no-pair-store", action="store_true",
                        help="keep primary pair products inside the run")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    run_root = arguments.run_root
    if (run_root / "control" / "config.json").exists():
        raise SystemExit("run already prepared: %s" % (run_root / "control" / "config.json"))
    for required in (
        arguments.source_catalogue, arguments.scene_root, arguments.official_repository,
        arguments.checkpoint, arguments.land_raster, arguments.asi_root,
    ):
        if not required.exists():
            raise SystemExit("missing required input: %s" % required)
    for directory in ("control", "logs", "outputs", "work"):
        (run_root / directory).mkdir(parents=True, exist_ok=True)

    sic_checksums = prepare_sic(
        arguments.sic_root, tuple(sic_days(arguments.start, arguments.stop)),
        download=arguments.download_sic,
    )
    catalogue = run_root / "control" / "catalogue.json"
    catalogue_summary = prepare_catalogue(
        arguments.source_catalogue, arguments.scene_root, catalogue,
        arguments.start, arguments.stop,
        expected_source_sha256=arguments.expected_source_sha256,
        expected_images=arguments.expected_images,
    )
    config_path = run_root / "control" / "config.json"
    write_json_atomic(config_path, run_config(arguments, catalogue, run_root))
    report = {
        "schema_version": "chain-olivia-preparation-v1",
        "run_id": arguments.run_id,
        "run_root": str(run_root),
        "window_utc": [arguments.start.isoformat(), arguments.stop.isoformat()],
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "sic_checksums": sic_checksums,
        "asi_coverage": asi_coverage(arguments.asi_root, arguments.asi_grid, arguments.start, arguments.stop),
        "data_binds": sorted({
            str(arguments.scene_root), str(arguments.sic_root),
            str(arguments.asi_root), str(arguments.land_raster.parent),
            *([str(arguments.asi_fallback_root)] if arguments.asi_fallback_root else []),
        }),
    }
    report.update(catalogue_summary)
    write_json_atomic(run_root / "control" / "preparation-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
