#!/usr/bin/env python3
"""Prepare a chained pan-Arctic EfficientLoFTR run on Olivia (any UTC window).

Writes ``control/catalogue.json`` and ``control/config.json`` under the run root for
``submit_chain_olivia.sh``. The configuration is the production one: at most four
primary pairs per target, loss-targeted continuation passes, the OSI-SAF tile gate,
and land (OSM inland distance) plus AMSR2 ASI point open-water exclusion.
"""

import argparse
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "prepare_april_week_olivia", Path(__file__).with_name("prepare_april_week_olivia.py")
)
APRIL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(APRIL)


def utc_day(value):
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def sic_days(start_utc, stop_utc):
    """OSI-SAF days covering the window with the gate's one-day tolerance."""
    day = (start_utc - timedelta(days=1)).date()
    last = stop_utc.date()
    while day <= last:
        yield day.strftime("%Y%m%d")
        day += timedelta(days=1)


def asi_coverage(root, grid, start_utc, stop_utc):
    """Count available daily ASI files; missing days keep positions (no exclusion)."""
    present, missing = 0, []
    day = start_utc.date()
    while day < stop_utc.date():
        path = Path(root) / day.strftime("%Y") / day.strftime("%b").lower() / (
            "asi-AMSR2-%s-%s-v5.4.tif" % (grid, day.strftime("%Y%m%d"))
        )
        if path.is_file():
            present += 1
        else:
            missing.append(day.isoformat())
        day += timedelta(days=1)
    return {"present_days": present, "missing_days": missing}


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
    result.add_argument("--expected-source-sha256", default=APRIL.SOURCE_CATALOGUE_SHA256)
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
    result.add_argument("--primary-maximum-pairs-per-target", type=int, default=4)
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
    for directory in ("control", "logs", "outputs", "passes", "work"):
        (run_root / directory).mkdir(parents=True, exist_ok=True)

    sic_checksums = APRIL.prepare_sic(
        arguments.sic_root, download=arguments.download_sic,
        days=tuple(sic_days(arguments.start, arguments.stop)),
    )
    catalogue = run_root / "control" / "catalogue.json"
    with arguments.source_catalogue.open(encoding="utf-8") as stream:
        times = [
            APRIL._feature_timestamp(feature.get("properties") or {})
            for feature in json.load(stream)["features"]
        ]
    count = sum(1 for t in times if t is not None and arguments.start <= t < arguments.stop)
    if arguments.expected_images is not None and count != arguments.expected_images:
        raise SystemExit("expected %d images, found %d" % (arguments.expected_images, count))
    catalogue_summary = APRIL.prepare_catalogue(
        arguments.source_catalogue, arguments.scene_root, catalogue,
        expected_source_sha256=arguments.expected_source_sha256,
        expected_images=count, start_utc=arguments.start, stop_utc=arguments.stop,
    )
    config = APRIL.run_config(
        arguments.run_id, catalogue, run_root, arguments.official_repository,
        arguments.checkpoint, arguments.sic_root,
        primary_maximum_pairs_per_target=arguments.primary_maximum_pairs_per_target,
    )
    config["database"] = str(run_root / "outputs" / "limosat.sqlite")
    config["output_directory"] = str(run_root / "outputs")
    config["matcher"]["maximum_speed_m_per_day"] = arguments.maximum_speed_m_per_day
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
    config["surface"] = surface
    config_path = run_root / "control" / "config.json"
    APRIL.write_json_atomic(config_path, config)
    report = {
        "schema_version": "chain-olivia-preparation-v1",
        "run_id": arguments.run_id,
        "run_root": str(run_root),
        "window_utc": [arguments.start.isoformat(), arguments.stop.isoformat()],
        "config": str(config_path),
        "config_sha256": APRIL.file_sha256(config_path),
        "sic_checksums": sic_checksums,
        "asi_coverage": asi_coverage(arguments.asi_root, arguments.asi_grid, arguments.start, arguments.stop),
        "data_binds": sorted({
            str(arguments.scene_root), str(arguments.sic_root),
            str(arguments.asi_root), str(arguments.land_raster.parent),
            *([str(arguments.asi_fallback_root)] if arguments.asi_fallback_root else []),
        }),
    }
    report.update(catalogue_summary)
    APRIL.write_json_atomic(run_root / "control" / "preparation-report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
