#!/usr/bin/env python3
"""Run the frozen LiMOSAT ORB baseline on the full-70 image sequence.

Two modes are deliberately separate:

``dense_operational``
    Normal window-based spatial seeding with no buoy injection. Use this for
    runtime, coverage, persistence, and deformation products.

``dense_plus_buoy_probes``
    The same dense seeding plus development-fold buoy observations injected
    through LiMOSAT's existing ``keypoint_from_point`` path. Use this only for
    truth-linked stage attribution because the probes alter the candidate set.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely import wkt
from shapely.geometry import Point
from skimage.transform import AffineTransform
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "experiments/configs/limosat_dense_30000_full70_local.yaml"
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/"
    "operational_baseline/runs"
)
MODES = ("dense_operational", "dense_plus_buoy_probes")
PROBE_SEED_METHODS = ("operational", "nearest_no_response_cutoff", "exact_supplied")

INHERITED_DEFAULTS = {
    "image_processor_params.template_size": 16,
    "keypoint_detector.orb_params.WTA_K": 2,
    "matcher_params.spatial_distance_max": 100000,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return loaded


def resolve_repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def validate_frozen_config(config: dict[str, Any]) -> None:
    """Fail early if the local config drifts from the supplied baseline."""
    expected = {
        "image_processor_params": {
            "persist_updates": True,
            "persist_interval": 50,
            "pruning_interval": 25,
            "temporal_window": 4,
            "convergence_radius_pixels": 7,
            "max_speed_m_per_day": 30000,
            "window_size": 128,
            "window_border": 6,
            "border_size": 128,
            "border_matched": 48,
            "border_interpolated": 48,
            "stride": 15,
            "octave": 5,
            "min_correlation": 0.30,
            "response_threshold": 0.001,
            "use_interpolation": True,
            "max_interpolation_time_gap_hours": 96,
        },
        "keypoint_detector.orb_params": {
            "nfeatures": 200,
            "scaleFactor": 1.20,
            "nlevels": 5,
            "edgeThreshold": 24,
            "firstLevel": 0,
            "patchSize": 88,
            "scoreType": "HARRIS_SCORE",
        },
        "matcher_params": {
            "norm_type": "NORM_HAMMING2",
            "descriptor_distance_max": 120,
            "model_threshold": 15000,
            "lowe_ratio": 0.80,
            "knn_k": 8,
            "plot_matches": False,
        },
        "matcher_params.geometric_model": {
            "type": "AffineTransform",
            "use_model_estimation": True,
            "estimation_method": "USAC_MAGSAC",
            "min_homography_inliers": 3,
        },
    }
    actual_sections = {
        "image_processor_params": config.get("image_processor_params", {}),
        "keypoint_detector.orb_params": config.get("keypoint_detector", {}).get(
            "orb_params", {}
        ),
        "matcher_params": {
            key: value
            for key, value in config.get("matcher_params", {}).items()
            if key != "geometric_model"
        },
        "matcher_params.geometric_model": config.get("matcher_params", {}).get(
            "geometric_model", {}
        ),
    }
    mismatches: list[str] = []
    for section, expected_values in expected.items():
        actual_values = actual_sections[section]
        for key, expected_value in expected_values.items():
            actual_value = actual_values.get(key, "<missing>")
            if actual_value != expected_value:
                mismatches.append(
                    f"{section}.{key}: expected {expected_value!r}, got {actual_value!r}"
                )
    if mismatches:
        raise ValueError("Frozen operational config mismatch:\n" + "\n".join(mismatches))


def make_orb(config: dict[str, Any]) -> cv2.ORB:
    params = dict(config["keypoint_detector"]["orb_params"])
    score_type = params.pop("scoreType")
    score_types = {
        "HARRIS_SCORE": cv2.ORB_HARRIS_SCORE,
        "FAST_SCORE": cv2.ORB_FAST_SCORE,
    }
    if score_type not in score_types:
        raise ValueError(f"Unsupported ORB scoreType: {score_type}")
    return cv2.ORB_create(scoreType=score_types[score_type], **params)


def make_matcher(
    config: dict[str, Any],
    audit_sink=None,
    candidate_selection: str = "global_descriptor_first",
    model_estimator: str = "legacy_homography",
):
    from limosat.matcher import Matcher

    params = config["matcher_params"]
    geometric = params["geometric_model"]
    norm_types = {
        "NORM_HAMMING": cv2.NORM_HAMMING,
        "NORM_HAMMING2": cv2.NORM_HAMMING2,
    }
    if params["norm_type"] not in norm_types:
        raise ValueError(f"Unsupported descriptor norm: {params['norm_type']}")
    if geometric["type"] != "AffineTransform":
        raise ValueError(f"Unsupported geometric model: {geometric['type']}")
    return Matcher(
        norm=norm_types[params["norm_type"]],
        descriptor_distance_max=params["descriptor_distance_max"],
        model=AffineTransform,
        model_threshold=params["model_threshold"],
        use_model_estimation=geometric["use_model_estimation"],
        estimation_method=geometric["estimation_method"],
        min_homography_inliers=geometric["min_homography_inliers"],
        model_coordinate_scale_m=geometric.get("coordinate_scale_m", 1.0),
        model_estimator=model_estimator,
        lowe_ratio=params["lowe_ratio"],
        knn_k=params["knn_k"],
        candidate_selection=candidate_selection,
        plot=params["plot_matches"],
        audit_sink=audit_sink,
    )


def image_table(observations: pd.DataFrame) -> pd.DataFrame:
    required = {"image_id", "image_time", "image_filepath"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Observation table lacks columns: {sorted(missing)}")
    rows = observations[list(required)].copy()
    rows["image_time"] = pd.to_datetime(rows["image_time"], utc=True)
    consistency = rows.groupby("image_id").agg(
        times=("image_time", "nunique"), paths=("image_filepath", "nunique")
    )
    if (consistency[["times", "paths"]] != 1).any(axis=None):
        bad = consistency[(consistency[["times", "paths"]] != 1).any(axis=1)].index
        raise ValueError(f"Image IDs have inconsistent metadata: {bad.tolist()[:10]}")
    rows = rows.drop_duplicates("image_id").sort_values(
        ["image_time", "image_id"], kind="stable"
    )
    rows.insert(0, "run_image_id", np.arange(1, len(rows) + 1, dtype=np.int64))
    missing_files = [path for path in rows["image_filepath"] if not Path(path).is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} images; first: {missing_files[0]}")
    return rows.reset_index(drop=True)


def select_image_split(
    observations: pd.DataFrame, image_split: str | None
) -> pd.DataFrame:
    """Restrict the image catalog to one preregistered dataset split."""
    if image_split is None:
        return observations
    split_column = "within_dataset_split"
    if split_column not in observations.columns:
        raise ValueError(
            f"--image-split requires an observation table with {split_column!r}"
        )
    selected = observations.loc[
        observations[split_column].astype(str).eq(image_split)
    ].copy()
    if selected.empty:
        available = sorted(observations[split_column].astype(str).unique())
        raise ValueError(
            f"No observations for image split {image_split!r}; available: {available}"
        )
    return selected


def parse_catalog_image_ids(value: str) -> tuple[int, ...]:
    try:
        image_ids = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "catalog image IDs must be comma-separated integers"
        ) from error
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise argparse.ArgumentTypeError(
            "catalog image IDs must be a non-empty unique list"
        )
    return image_ids


def select_catalog_images(
    observations: pd.DataFrame, catalog_image_ids: tuple[int, ...] | None
) -> pd.DataFrame:
    """Restrict observations to an explicit, fully present image set."""
    if catalog_image_ids is None:
        return observations
    available = set(observations["image_id"].astype(int))
    missing = sorted(set(catalog_image_ids).difference(available))
    if missing:
        raise ValueError(f"Requested catalog image IDs are absent: {missing}")
    return observations.loc[
        observations["image_id"].astype(int).isin(catalog_image_ids)
    ].copy()


def build_buoy_probes(
    observations: pd.DataFrame,
    selected_images: pd.DataFrame,
    split: str,
) -> gpd.GeoDataFrame:
    required = {"buoy_id", "image_id", "image_filepath", "x", "y", "within_dataset_split"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Cannot build buoy probes; missing columns: {sorted(missing)}")
    probes = observations.loc[
        observations["image_id"].isin(selected_images["image_id"])
        & (observations["within_dataset_split"] == split)
    ].copy()
    if probes.empty:
        raise ValueError(f"No buoy probes for within-dataset split {split!r}")
    if probes.duplicated(["buoy_id", "image_id"]).any():
        raise ValueError("Buoy probes must be unique by buoy_id and image_id")
    probes["probe_id"] = probes["buoy_id"].astype(str) + "|" + probes["image_id"].astype(str)
    probes["image_filepath"] = probes["image_filepath"].map(lambda value: Path(value).name)
    geometry = [Point(float(x), float(y)) for x, y in zip(probes["x"], probes["y"])]
    return gpd.GeoDataFrame(probes, geometry=geometry, crs="EPSG:3413").set_index(
        "probe_id", drop=False
    )


def install_probe_seed_method(processor: Any, method: str) -> None:
    """Install an explicitly labelled validation-only buoy seeding rule."""
    if method == "operational":
        return
    detector = processor.keypoint_detector
    if method == "nearest_no_response_cutoff":
        operational_method = detector.keypoint_from_point

        def nearest_no_response_cutoff(
            points_gdf_for_current_image,
            octave,
            img,
            response_threshold,
        ):
            return operational_method(
                points_gdf_for_current_image,
                octave=octave,
                img=img,
                response_threshold=0.0,
            )

        detector.keypoint_from_point = nearest_no_response_cutoff
        return
    if method == "exact_supplied":

        def exact_supplied(
            points_gdf_for_current_image,
            octave,
            img,
            response_threshold,
        ):
            keypoints = []
            for point_index, point_row in points_gdf_for_current_image.iterrows():
                geometry = point_row.geometry
                if geometry is None:
                    continue
                cols, rows = img.transform_points(
                    [float(geometry.x)],
                    [float(geometry.y)],
                    DstToSrc=1,
                    dst_srs=img.srs,
                )
                col, row = float(cols[0]), float(rows[0])
                if not np.isfinite(col) or not np.isfinite(row):
                    continue
                keypoints.append(
                    (
                        cv2.KeyPoint(
                            col,
                            row,
                            size=31.0,
                            angle=float(img.angle),
                            octave=int(octave),
                        ),
                        point_index,
                    )
                )
            return keypoints

        detector.keypoint_from_point = exact_supplied
        return
    raise ValueError(f"Unknown probe seed method: {method}")


def points_fingerprint(points: Any) -> str:
    """Hash final state, including binary descriptors, without saving a duplicate table."""
    digest = hashlib.sha256()
    if points.empty:
        digest.update(b"empty")
        return digest.hexdigest()
    ordered = points.sort_values(["image_id", "trajectory_id"], kind="stable")
    scalar_columns = (
        "image_id",
        "is_last",
        "trajectory_id",
        "angle",
        "corr",
        "time",
        "interpolated",
        "orbit_num",
        "stopped",
        "converged_to",
    )
    for row in ordered.itertuples(index=False):
        values = row._asdict()
        for column in scalar_columns:
            digest.update(f"{column}={values.get(column)!r}|".encode())
        geometry = values.get("geometry")
        if geometry is not None:
            digest.update(np.asarray([geometry.x, geometry.y], dtype="<f8").tobytes())
        descriptor = values.get("descriptors")
        if isinstance(descriptor, np.ndarray):
            digest.update(str(descriptor.dtype).encode())
            digest.update(np.asarray(descriptor.shape, dtype="<i8").tobytes())
            digest.update(descriptor.tobytes())
    return digest.hexdigest()


def environment_manifest() -> dict[str, Any]:
    try:
        import nansat

        nansat_version = getattr(nansat, "__version__", "unreported")
    except Exception as exc:  # pragma: no cover - the run cannot proceed without it
        nansat_version = f"import-error: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "geopandas": gpd.__version__,
        "nansat": nansat_version,
    }


def format_storage(
    config: dict[str, Any],
    run_name: str,
    zarr_work_root: Path | None = None,
) -> tuple[str, Path]:
    database = config["database"]
    engine_url = database["engine_url"].format(run_name=run_name)
    zarr_path = (
        zarr_work_root.resolve() / f"{run_name}.zarr"
        if zarr_work_root is not None
        else Path(database["zarr_path_template"].format(run_name=run_name))
    )
    if engine_url.startswith("sqlite:////"):
        sqlite_path = Path("/" + engine_url.removeprefix("sqlite:////"))
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    return engine_url, zarr_path


def apply_storage_root(config: dict[str, Any], storage_root: Path) -> None:
    """Redirect persisted state without duplicating the frozen algorithm config."""
    root = storage_root.expanduser().resolve()
    config["database"]["engine_url"] = (
        "sqlite:///" + str(root / "{run_name}.sqlite")
    )
    config["database"]["zarr_path_template"] = str(root / "{run_name}.zarr")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from limosat.database import DriftDatabase
    from limosat.image_processor import ImageProcessor
    import limosat.image_processor as image_processor_module
    from limosat.keypoints import Keypoints
    from limosat.templates import Templates
    from limosat.tracking_audit import TrackingAuditSink
    from limosat.utils import setup_logging

    started = time.perf_counter()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    validate_frozen_config(config)
    effective_config = copy.deepcopy(config)
    if args.storage_root is not None:
        apply_storage_root(effective_config, args.storage_root)
    if args.grid_cache_dir is not None:
        effective_config["keypoint_detector"]["grid_cache_dir"] = str(
            args.grid_cache_dir.expanduser().resolve()
        )
    if args.no_persist:
        effective_config["image_processor_params"]["persist_updates"] = False
    if args.model_threshold_m is not None:
        if args.model_threshold_m <= 0:
            raise ValueError("--model-threshold-m must be positive")
        effective_config["matcher_params"]["model_threshold"] = float(
            args.model_threshold_m
        )
    if args.model_coordinate_scale_m is not None:
        if args.model_coordinate_scale_m <= 0:
            raise ValueError("--model-coordinate-scale-m must be positive")
        effective_config["matcher_params"]["geometric_model"][
            "coordinate_scale_m"
        ] = float(args.model_coordinate_scale_m)
    if args.border_matched is not None:
        if args.border_matched < 1:
            raise ValueError("--border-matched must be positive")
        effective_config["image_processor_params"]["border_matched"] = int(
            args.border_matched
        )
    if args.border_interpolated is not None:
        if args.border_interpolated < 1:
            raise ValueError("--border-interpolated must be positive")
        effective_config["image_processor_params"]["border_interpolated"] = int(
            args.border_interpolated
        )
    if args.pattern_matching_subpixel_method is not None:
        effective_config["image_processor_params"][
            "pattern_matching_subpixel_method"
        ] = args.pattern_matching_subpixel_method
    if args.template_sampling is not None:
        effective_config["image_processor_params"]["template_sampling"] = (
            args.template_sampling
        )

    observations_path = resolve_repo_path(
        args.observations or config["paths"]["tracking_observations"]
    )
    observations = pd.read_csv(observations_path, low_memory=False)
    observations = select_image_split(observations, args.image_split)
    observations = select_catalog_images(observations, args.catalog_image_ids)
    images = image_table(observations)
    if args.max_images is not None:
        if args.max_images < 1:
            raise ValueError("--max-images must be positive")
        images = images.iloc[: args.max_images].copy()

    base_name = config["run_settings"]["run_name"]
    suffix = args.run_suffix or args.mode
    effective_run_name = f"{base_name}__{suffix}"
    output_dir = (args.output_root / effective_run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_sink = (
        TrackingAuditSink(output_dir / "stage_audit") if args.instrument else None
    )

    logging_config = config.get("logging", {})
    log_dir = (
        args.log_dir.expanduser().resolve()
        if args.log_dir is not None
        else Path(logging_config.get("log_directory", output_dir / "logs"))
    )
    if args.no_persist and args.log_dir is None:
        log_dir = output_dir / "logs"
    logger = setup_logging(
        log_dir=str(log_dir),
        filename_prefix=effective_run_name,
        persist_log=bool(logging_config.get("persist_log", True)),
    )

    seed = int(args.seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    max_cores = config["run_settings"].get("max_cpu_cores")
    if max_cores is not None:
        cv2.setNumThreads(int(max_cores))

    probes = None
    if args.mode == "dense_plus_buoy_probes":
        probes = build_buoy_probes(observations, images, args.probe_split)

    persist = effective_config["image_processor_params"]["persist_updates"]
    engine = None
    zarr_path = None
    zarr_archive = None
    if persist:
        engine_url, zarr_path = format_storage(
            effective_config,
            effective_run_name,
            zarr_work_root=args.zarr_work_root,
        )
        engine = create_engine(engine_url)
        initial_db = DriftDatabase(engine=engine, zarr_path=str(zarr_path), run_name=effective_run_name)
        points, templates = initial_db.prepare_run_state(
            clear_existing_data=bool(config["run_settings"]["clear_existing_data"]),
            temporal_window_days=effective_config["image_processor_params"]["temporal_window"],
        )
    else:
        engine_url = None
        points, templates = Keypoints(), Templates()

    processor = ImageProcessor(
        points=points,
        templates=templates,
        model=make_orb(config),
        matcher=make_matcher(
            effective_config,
            audit_sink=audit_sink,
            candidate_selection=args.candidate_selection,
            model_estimator=args.model_estimator,
        ),
        config=effective_config,
        engine=engine,
        zarr_path=str(zarr_path) if zarr_path is not None else None,
        run_name=effective_run_name,
        insitu_points=probes,
        grid_cache_dir=effective_config["keypoint_detector"].get("grid_cache_dir"),
        audit_sink=audit_sink,
        validation_dir=str(output_dir / "validation"),
    )
    setup_seconds = time.perf_counter() - started
    detailed_seconds: defaultdict[str, float] = defaultdict(float)
    if args.detailed_timing:
        def wrap_instance_method(instance, name: str, stage: str) -> None:
            original = getattr(instance, name)

            def measured(*method_args, **method_kwargs):
                stage_started = time.perf_counter()
                try:
                    return original(*method_args, **method_kwargs)
                finally:
                    detailed_seconds[stage] += time.perf_counter() - stage_started

            setattr(instance, name, measured)

        def wrap_module_call(module, name: str, stage: str) -> None:
            original = getattr(module, name)

            def measured(*call_args, **call_kwargs):
                stage_started = time.perf_counter()
                try:
                    return original(*call_args, **call_kwargs)
                finally:
                    detailed_seconds[stage] += time.perf_counter() - stage_started

            setattr(module, name, measured)

        wrap_module_call(image_processor_module, "Image", "image_preparation")
        for method_name in (
            "detect_gridded_points",
            "detect_new_keypoints",
            "compute_descriptors",
        ):
            wrap_instance_method(
                processor.keypoint_detector,
                method_name,
                "detection_and_description",
            )
        wrap_instance_method(processor.matcher, "match_with_grid", "matching")
        wrap_module_call(image_processor_module, "pattern_matching", "pattern_matching")
        if persist:
            wrap_instance_method(processor.db, "save", "persistence")
    if probes is not None:
        install_probe_seed_method(processor, args.probe_seed_method)

    timing_rows: list[dict[str, Any]] = []
    status = "running"
    failure: str | None = None
    try:
        for image in images.itertuples(index=False):
            image_started = time.perf_counter()
            points_before = len(processor.points)
            active_before = int((processor.points["is_last"] == 1).sum()) if points_before else 0
            processor.process_image(int(image.run_image_id), str(image.image_filepath))
            points_after = len(processor.points)
            active_after = int((processor.points["is_last"] == 1).sum()) if points_after else 0
            timing_rows.append(
                {
                    "run_image_id": int(image.run_image_id),
                    "catalog_image_id": int(image.image_id),
                    "image_time": image.image_time.isoformat(),
                    "image_filename": Path(image.image_filepath).name,
                    "seconds": time.perf_counter() - image_started,
                    "points_before": points_before,
                    "points_after": points_after,
                    "active_before": active_before,
                    "active_after": active_after,
                }
            )
        persistence_ok = processor.ensure_final_persistence()
        if not persistence_ok:
            raise RuntimeError("Final SQLite/Zarr persistence failed")
        if persist and args.zarr_work_root is not None:
            archive_base = output_dir / "templates.zarr"
            archive_path = shutil.make_archive(
                str(archive_base),
                "gztar",
                root_dir=zarr_path.parent,
                base_dir=zarr_path.name,
            )
            zarr_archive = Path(archive_path)
        status = "complete"
    except Exception as exc:
        status = "failed"
        failure = f"{type(exc).__name__}: {exc}"
        logger.exception("Operational baseline failed")
        raise
    finally:
        output_writing_started = time.perf_counter()
        timings = pd.DataFrame.from_records(timing_rows)
        timings.to_csv(output_dir / "image_timings.csv", index=False)
        if probes is not None:
            probe_output = processor.insitu_points.copy()
            probe_output["geometry_wkt"] = probe_output.geometry.map(
                lambda geometry: geometry.wkt if geometry is not None else None
            )
            pd.DataFrame(probe_output.drop(columns="geometry")).to_csv(
                output_dir / "buoy_probe_linkage.csv", index=False
            )
            linked = probe_output.loc[
                probe_output["trajectory_id"].notna(),
                ["probe_id", "buoy_id", "trajectory_id", "image_id", "image_time"],
            ].copy()
            if not linked.empty:
                linked["trajectory_id"] = linked["trajectory_id"].astype(int)
                if persist and engine is not None:
                    selected_columns = (
                        "image_id, is_last, trajectory_id, geometry, angle, corr, "
                        "time, interpolated, orbit_num, stopped, converged_to"
                    )
                    probe_trajectories = pd.read_sql_query(
                        f'SELECT {selected_columns} FROM "{effective_run_name}"',
                        engine,
                    )
                    probe_trajectories = probe_trajectories.loc[
                        probe_trajectories["trajectory_id"].isin(
                            linked["trajectory_id"]
                        )
                    ].copy()
                    geometry_values = probe_trajectories["geometry"].map(
                        lambda value: wkt.loads(value) if isinstance(value, str) else value
                    )
                else:
                    probe_trajectories = pd.DataFrame(
                        processor.points.loc[
                            processor.points["trajectory_id"].isin(
                                linked["trajectory_id"]
                            )
                        ].copy()
                    )
                    geometry_values = probe_trajectories["geometry"]
                probe_trajectories["tracked_x"] = geometry_values.map(
                    lambda geometry: geometry.x if geometry is not None else np.nan
                )
                probe_trajectories["tracked_y"] = geometry_values.map(
                    lambda geometry: geometry.y if geometry is not None else np.nan
                )
                probe_trajectories = probe_trajectories.drop(
                    columns=["geometry", "descriptors"], errors="ignore"
                ).merge(
                    linked.rename(
                        columns={
                            "image_id": "seed_catalog_image_id",
                            "image_time": "seed_image_time",
                        }
                    ),
                    on="trajectory_id",
                    how="left",
                    validate="many_to_one",
                )
                probe_trajectories.to_csv(
                    output_dir / "buoy_probe_trajectories.csv", index=False
                )
        detailed_seconds["runner_output_writing"] += (
            time.perf_counter() - output_writing_started
        )
        final_points = len(processor.points)
        active_points = (
            int((processor.points["is_last"] == 1).sum()) if final_points else 0
        )
        audit_counts = dict(audit_sink.counts) if audit_sink is not None else {}
        if audit_sink is not None:
            audit_sink.close()
        stage_timing_path = None
        if args.detailed_timing:
            process_seconds = float(timings["seconds"].sum()) if len(timings) else 0.0
            classified_process = sum(
                detailed_seconds[name]
                for name in (
                    "image_preparation",
                    "detection_and_description",
                    "matching",
                    "pattern_matching",
                )
            )
            detailed_seconds["topology_and_qc_residual"] = max(
                0.0, process_seconds - classified_process
            )
            stage_timing = {
                "setup_seconds": setup_seconds,
                "image_process_seconds": process_seconds,
                **{
                    f"{name}_seconds": float(value)
                    for name, value in sorted(detailed_seconds.items())
                },
                "interpretation": (
                    "Topology/QC is the residual image-processing time after "
                    "directly timed image preparation, detection/description, "
                    "descriptor matching, and pattern matching. Persistence is "
                    "timed directly and may occur after the image loop."
                ),
            }
            stage_timing_path = output_dir / "stage_timings.json"
            stage_timing_path.write_text(json.dumps(stage_timing, indent=2) + "\n")
        manifest = {
            "status": status,
            "failure": failure,
            "mode": args.mode,
            "candidate_selection": args.candidate_selection,
            "model_estimator": args.model_estimator,
            "model_coordinate_scale_m": float(
                effective_config["matcher_params"]["geometric_model"].get(
                    "coordinate_scale_m", 1.0
                )
            ),
            "model_threshold_m": float(
                effective_config["matcher_params"]["model_threshold"]
            ),
            "border_matched": int(
                effective_config["image_processor_params"]["border_matched"]
            ),
            "border_interpolated": int(
                effective_config["image_processor_params"]["border_interpolated"]
            ),
            "pattern_matching_subpixel_method": effective_config[
                "image_processor_params"
            ].get("pattern_matching_subpixel_method", "none"),
            "template_sampling": effective_config["image_processor_params"].get(
                "template_sampling", "integer"
            ),
            "probe_split": args.probe_split if probes is not None else None,
            "image_split": args.image_split,
            "catalog_image_ids_requested": list(args.catalog_image_ids)
            if args.catalog_image_ids is not None
            else None,
            "probe_seed_method": args.probe_seed_method if probes is not None else None,
            "run_name_from_config": base_name,
            "effective_run_name": effective_run_name,
            "seed": seed,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "observations_path": str(observations_path),
            "observations_sha256": sha256_file(observations_path),
            "inherited_defaults": INHERITED_DEFAULTS,
            "persistence_enabled": persist,
            "stage_instrumentation_enabled": bool(args.instrument),
            "detailed_timing_enabled": bool(args.detailed_timing),
            "stage_timings_path": (
                str(stage_timing_path) if stage_timing_path is not None else None
            ),
            "stage_audit_record_counts": audit_counts,
            "engine_url": engine_url,
            "zarr_path": str(zarr_path) if zarr_path is not None else None,
            "zarr_archive": str(zarr_archive) if zarr_archive is not None else None,
            "zarr_archive_sha256": (
                sha256_file(zarr_archive) if zarr_archive is not None else None
            ),
            "grid_cache_dir": effective_config["keypoint_detector"].get(
                "grid_cache_dir"
            ),
            "log_directory": str(log_dir),
            "images_requested": int(len(images)),
            "images_completed": int(len(timing_rows)),
            "catalog_image_ids": [int(value) for value in images["image_id"]],
            "buoy_probes_requested": int(len(probes)) if probes is not None else 0,
            "buoy_probes_linked": (
                int(processor.insitu_points["trajectory_id"].notna().sum())
                if probes is not None
                else 0
            ),
            "final_point_rows_in_memory": final_points,
            "final_active_points_in_memory": active_points,
            "final_points_sha256": points_fingerprint(processor.points),
            "elapsed_seconds": time.perf_counter() - started,
            "environment": environment_manifest(),
            "interpretation": (
                "Production-like dense run; valid for runtime and spatial output."
                if args.mode == "dense_operational"
                else "Dense run with injected buoy probes; valid for truth-linked attribution, not production runtime."
            ),
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str) + "\n"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--mode", choices=MODES, default="dense_operational")
    parser.add_argument(
        "--candidate-selection",
        choices=(
            "global_descriptor_first",
            "global_then_local_physics_fallback",
        ),
        default="global_descriptor_first",
        help="Choose the frozen global-first matcher or the targeted local-physics fallback pilot.",
    )
    parser.add_argument(
        "--model-estimator",
        choices=(
            "legacy_homography",
            "configured_affine",
            "homography_affine_union",
            "homography_kilometre_coordinates",
        ),
        default="legacy_homography",
        help=(
            "Preserve the hardcoded homography baseline, estimate the configured "
            "AffineTransform, retain the union of both inlier sets, or fit the "
            "homography in kilometre coordinates for numerical stability."
        ),
    )
    parser.add_argument(
        "--model-coordinate-scale-m",
        type=float,
        help="Override the homography fitting scale after validating the frozen base config.",
    )
    parser.add_argument(
        "--model-threshold-m",
        type=float,
        help="Override the physical MAGSAC residual threshold after validating the frozen base config.",
    )
    parser.add_argument("--border-matched", type=int)
    parser.add_argument("--border-interpolated", type=int)
    parser.add_argument(
        "--pattern-matching-subpixel-method",
        choices=("none", "quadratic"),
    )
    parser.add_argument(
        "--template-sampling",
        choices=("integer", "bilinear"),
    )
    parser.add_argument("--probe-split", default="development")
    parser.add_argument(
        "--image-split",
        help=(
            "Restrict the image catalog to one within_dataset_split value; "
            "fails closed when the column or value is absent."
        ),
    )
    parser.add_argument(
        "--catalog-image-ids",
        type=parse_catalog_image_ids,
        help="Comma-separated catalog image IDs; every requested ID must exist.",
    )
    parser.add_argument(
        "--probe-seed-method", choices=PROBE_SEED_METHODS, default="operational"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--storage-root",
        type=Path,
        help="Override SQLite/Zarr storage while retaining the frozen config.",
    )
    parser.add_argument(
        "--grid-cache-dir",
        type=Path,
        help="Override only the deterministic gridded-ORB descriptor cache path.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Override the persistent log directory for this dataset.",
    )
    parser.add_argument("--run-suffix")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--instrument", action="store_true")
    parser.add_argument(
        "--detailed-timing",
        action="store_true",
        help="Record non-scientific wall-time attribution for the runtime gate.",
    )
    parser.add_argument(
        "--zarr-work-root",
        type=Path,
        help="Write the live Zarr store on a local filesystem, then archive it in the run output.",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Diagnostic only: keep state in memory and write no SQLite/Zarr data.",
    )
    return parser.parse_args()


def main() -> int:
    manifest = run(parse_args())
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
