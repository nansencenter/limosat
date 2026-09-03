#!/usr/bin/env python3
"""Prepare the bounded April 2020 EfficientLoFTR run on Olivia."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SOURCE_CATALOGUE_SHA256 = (
    "12ba2654f82a7da78dff3724fd89f880583ea5347345009f2d7a57477c32f58d"
)
RUN_ID = "april2020-week01-global-5pct-sic-v1"
START_UTC = datetime(2020, 4, 1, tzinfo=timezone.utc)
STOP_UTC = datetime(2020, 4, 8, tzinfo=timezone.utc)
EXPECTED_IMAGES = 781
SIC_DATES = (
    "20200331",
    "20200401",
    "20200402",
    "20200403",
    "20200404",
    "20200405",
    "20200406",
    "20200407",
    "20200408",
)
SIC_BASE_URL = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/ice/conc"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value):
    value = str(value).strip().replace(" ", "T", 1)
    if value.endswith("Z"):
        value = value[:-1]
    elif value.endswith("+00:00"):
        value = value[:-6]
    elif value.endswith("+0000"):
        value = value[:-5]
    else:
        raise ValueError("timestamp is not explicitly UTC: %s" % value)
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in value else "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)


def _feature_timestamp(properties):
    for name in ("datetime", "timestamp", "time_utc", "start_datetime"):
        value = properties.get(name)
        if value:
            return parse_utc(value)
    return None


def _feature_filename(feature, properties):
    value = properties.get("filename")
    if value:
        return Path(str(value)).name
    value = properties.get("filepath") or properties.get("path")
    if value:
        return Path(str(value)).name
    for asset in (feature.get("assets") or {}).values():
        value = asset.get("href")
        if value:
            return Path(str(value)).name
    return None


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


def prepare_catalogue(
    source_catalogue,
    scene_root,
    destination,
    expected_source_sha256=SOURCE_CATALOGUE_SHA256,
    expected_images=EXPECTED_IMAGES,
    start_utc=START_UTC,
    stop_utc=STOP_UTC,
):
    source_catalogue = Path(source_catalogue)
    scene_root = Path(scene_root)
    actual_source_sha256 = file_sha256(source_catalogue)
    if expected_source_sha256 and actual_source_sha256 != expected_source_sha256:
        raise ValueError(
            "source catalogue SHA256 mismatch: expected %s, found %s"
            % (expected_source_sha256, actual_source_sha256)
        )
    with source_catalogue.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("features"), list):
        raise ValueError("source catalogue is not a GeoJSON/STAC FeatureCollection")

    selected = []
    missing = []
    image_ids = []
    for original in document["features"]:
        feature = dict(original)
        properties = dict(feature.get("properties") or {})
        timestamp = _feature_timestamp(properties)
        if timestamp is None or not start_utc <= timestamp < stop_utc:
            continue
        filename = _feature_filename(feature, properties)
        if not filename:
            raise ValueError("selected feature has no raster filename")
        image_id = str(
            properties.get("scene_id") or feature.get("id") or Path(filename).stem
        )
        raster = (
            scene_root
            / ("%04d" % timestamp.year)
            / ("%02d" % timestamp.month)
            / filename
        )
        properties["scene_id"] = image_id
        properties["filepath"] = str(raster)
        feature["properties"] = properties
        selected.append(feature)
        image_ids.append(image_id)
        if not raster.is_file():
            missing.append(str(raster))

    if len(selected) != expected_images:
        raise ValueError(
            "expected %d catalogue images, found %d" % (expected_images, len(selected))
        )
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("selected catalogue image IDs are not globally unique")
    if missing:
        raise FileNotFoundError(
            "%d raster paths are missing; first: %s" % (len(missing), missing[0])
        )

    output = dict(document)
    output["features"] = selected
    write_json_atomic(destination, output, compact=True)
    return {
        "catalogue_images": len(selected),
        "catalogue_sha256": file_sha256(destination),
        "source_catalogue_sha256": actual_source_sha256,
    }


def sic_filename(day):
    return "ice_conc_nh_polstere-100_multi_%s1200.nc" % day


def prepare_sic(root, download=False, base_url=SIC_BASE_URL):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for day in SIC_DATES:
        destination = root / sic_filename(day)
        if destination.is_file() and destination.stat().st_size:
            continue
        if not download:
            raise FileNotFoundError(
                "missing %s; rerun with --download-sic" % destination
            )
        temporary = destination.with_name(destination.name + ".part")
        command = [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "3",
            "--output",
            str(temporary),
            "%s/%s/%s/%s"
            % (base_url.rstrip("/"), day[:4], day[4:6], destination.name),
        ]
        try:
            subprocess.run(command, check=True)
            os.replace(str(temporary), str(destination))
        finally:
            if temporary.exists():
                temporary.unlink()
    checksums = {sic_filename(day): file_sha256(root / sic_filename(day)) for day in SIC_DATES}
    checksum_path = root / "SHA256SUMS"
    checksum_path.write_text(
        "".join("%s  %s\n" % (checksum, name) for name, checksum in sorted(checksums.items())),
        encoding="utf-8",
    )
    return checksums


def run_config(
    run_id,
    catalogue,
    run_root,
    official_repository,
    checkpoint,
    sic_root,
    primary_maximum_pairs_per_target=None,
    targeted_recovery=True,
):
    run_root = Path(run_root)
    return {
        "run_id": run_id,
        "catalogue": str(catalogue),
        "database": str(run_root / "control" / "state.sqlite"),
        "output_directory": str(run_root / "products"),
        "pair_product_directory": str(run_root / "work" / "pair-products"),
        "analysis_epsg": 3413,
        "pair_workers": 1,
        "retain_pair_matches": True,
        "matcher": {
            "repository": str(official_repository),
            "checkpoint": str(checkpoint),
            "model_name": "efficientloftr-official-opt",
            "device": "cuda",
            "pixel_size_m": 80.0,
            "tile_size_px": 512,
            "tile_margin_px": 32,
            "endpoint_support_radius_px": 16,
            "transform_grid_spacing_px": 32,
            "tile_grid_origin_m": 0.0,
            "maximum_speed_m_per_day": 30000.0,
        },
        "field": {
            "grid_spacing_m": 4000.0,
            "neighbour_count": 12,
            "minimum_agreeing_matches": 8,
            "maximum_neighbour_distance_m": 6000.0,
            "agreement_distance_m": 1000.0,
            "maximum_triangle_edge_m": 6400.0,
        },
        "routing": {
            "mode": "sequential_local",
            "initial": "phase_correlation",
            "phase_correlation_failure": "same_center",
            "phase_correlation_minimum_response": 0.05,
            "residual_edge_recovery": True,
            "targeted_recovery": targeted_recovery,
            "maximum_recovery_elapsed_hours": 96.0,
            "targeted_selection_buffer_m": 6400.0,
            "candidate_minimum_elapsed_hours": 1.0,
            "candidate_maximum_elapsed_hours": 96.0,
            "candidate_minimum_overlap_fraction": 0.05,
            "candidate_minimum_overlap_area_m2": 1024000000.0,
            "exclude_same_acquisition_pass": True,
            "require_orbit_metadata": True,
            "primary_maximum_pairs_per_target": (
                primary_maximum_pairs_per_target
            ),
            "candidate_pair_ids": [],
        },
        "open_water": {
            "enabled": True,
            "sic_root": str(sic_root),
            "threshold_percent": 15.0,
            "maximum_age_days": 1,
            "samples_per_axis": 5,
        },
        "trajectories": {
            "add_as_coverage_enters": True,
            "new_point_exclusion_radius_m": 2000.0,
            "convergence_audit_radius_m": None,
        },
    }


def parser():
    project_id = os.environ.get("PROJECT_ID", "nn9878k")
    user = os.environ.get("USER", "seachu")
    project_root = Path("/cluster/projects") / project_id / user
    work_root = Path("/cluster/work/projects") / project_id / user
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-id", default=os.environ.get("RUN_ID", RUN_ID))
    result.add_argument(
        "--run-root",
        type=Path,
        default=Path(os.environ["RUN_ROOT"]) if os.environ.get("RUN_ROOT") else None,
    )
    result.add_argument(
        "--source-catalogue",
        type=Path,
        default=project_root
        / "arktalas-deployment"
        / "data"
        / "201905_202007_image_catalog_nirdsync_20260824T220354Z.json",
    )
    result.add_argument(
        "--scene-root",
        type=Path,
        default=work_root / "limosat_staging" / "201905_202007" / "s1_preprocessed",
    )
    result.add_argument(
        "--official-repository",
        type=Path,
        default=project_root / "EfficientLoFTR",
    )
    result.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "checkpoints" / "efficientloftr" / "eloftr_outdoor.ckpt",
    )
    result.add_argument("--sic-root", type=Path)
    result.add_argument("--download-sic", action="store_true")
    result.add_argument(
        "--primary-maximum-pairs-per-target",
        type=int,
        help=(
            "diagnostic compute bound; keep the strongest spatial-coverage "
            "contributors for each target image"
        ),
    )
    result.add_argument(
        "--disable-recovery",
        action="store_true",
        help="skip non-consecutive recovery work for a bounded diagnostic run",
    )
    result.add_argument("--expected-images", type=int, default=EXPECTED_IMAGES)
    result.add_argument(
        "--expected-source-sha256", default=SOURCE_CATALOGUE_SHA256
    )
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    project_id = os.environ.get("PROJECT_ID", "nn9878k")
    user = os.environ.get("USER", "seachu")
    run_root = arguments.run_root or (
        Path("/cluster/work/projects")
        / project_id
        / user
        / "method-neutral-benchmark"
        / "efficientloftr-production"
        / arguments.run_id
    )
    sic_root = (
        arguments.sic_root
        or (Path(os.environ["SIC_ROOT"]) if os.environ.get("SIC_ROOT") else None)
        or run_root / "inputs" / "osisaf-sic"
    )
    database = run_root / "control" / "state.sqlite"
    if database.exists():
        raise SystemExit(
            "run database already exists; use submit_april_week_olivia.sh to resume"
        )
    for required in (
        arguments.source_catalogue,
        arguments.scene_root,
        arguments.official_repository,
        arguments.checkpoint,
    ):
        if not required.exists():
            raise SystemExit("missing required input: %s" % required)
    for directory in ("control", "inputs", "logs", "products", "work"):
        (run_root / directory).mkdir(parents=True, exist_ok=True)

    sic_checksums = prepare_sic(sic_root, download=arguments.download_sic)
    catalogue = run_root / "control" / "april-week-full-catalog.json"
    catalogue_summary = prepare_catalogue(
        arguments.source_catalogue,
        arguments.scene_root,
        catalogue,
        expected_source_sha256=arguments.expected_source_sha256,
        expected_images=arguments.expected_images,
    )
    config_path = run_root / "control" / "april-week-full.json"
    config = run_config(
        arguments.run_id,
        catalogue,
        run_root,
        arguments.official_repository,
        arguments.checkpoint,
        sic_root,
        primary_maximum_pairs_per_target=(
            arguments.primary_maximum_pairs_per_target
        ),
        targeted_recovery=not arguments.disable_recovery,
    )
    write_json_atomic(config_path, config)
    report = {
        "schema_version": "april-week-olivia-preparation-v1",
        "run_id": arguments.run_id,
        "run_root": str(run_root),
        "catalogue": str(catalogue),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "sic_checksums": sic_checksums,
        "diagnostic_pair_selection": {
            "primary_maximum_pairs_per_target": (
                arguments.primary_maximum_pairs_per_target
            ),
            "targeted_recovery": not arguments.disable_recovery,
        },
    }
    report.update(catalogue_summary)
    write_json_atomic(run_root / "control" / "preparation-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
