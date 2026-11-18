"""What a pair product depends on, so products can be shared between runs.

A pair product (matches plus displacement field for one image pair) depends on the
two scene files, the settings that change matching, the model, a few ancillary
inputs and the matching code. It does not depend on the run that made it: its ID,
its paths, its pair plan or its composition settings. ``matching_sha256`` hashes
exactly the first group, using file contents rather than paths, so a product made
by one run (or on another cluster) is valid for any run whose fingerprint agrees.

Every configuration field is classified below as either matching or excluded, and
a unit test fails when a new field is left unclassified. Bump ``MATCHING_VERSION``
whenever a code change alters matching output; tests/unit/test_pair_identity.py
records a digest of the matching modules to prompt that decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from .store import file_sha256

MATCHING_VERSION = 1

# Settings that change a pair product, by configuration section.
MATCHING_SETTINGS = {
    "matcher": (
        "model_name", "pixel_size_m", "tile_size_px", "tile_margin_px",
        "endpoint_support_radius_px", "transform_grid_spacing_px",
        "tile_grid_origin_m", "maximum_speed_m_per_day", "tile_grid_offsets",
        "masked_rematch_target_valid_fraction", "layout_dedup_m",
        "layout_field_merge", "tile_sampling",
        "layout_gap_minimum_agreeing_matches", "layout_gap_agreement_distance_m",
        "scene_mask_invalid_from",
    ),
    "field": (
        "grid_spacing_m", "neighbour_count", "minimum_agreeing_matches",
        "maximum_neighbour_distance_m", "agreement_distance_m",
        "maximum_triangle_edge_m",
    ),
    "routing": (
        "mode", "initial", "phase_correlation_failure",
        "phase_correlation_minimum_response", "residual_edge_recovery",
        "coarse_matching", "coarse_pixel_size_m", "coarse_minimum_matches",
        "coarse_support_radius_m", "coarse_window_stride",
        "targeted_selection_buffer_m",
    ),
    "open_water": ("enabled", "threshold_percent", "maximum_age_days", "samples_per_axis"),
    "surface": ("land_minimum_inland_km",),
}

# Settings that do not change a pair product: paths (their contents are fingerprinted
# separately), performance knobs, pair selection and composition.
EXCLUDED_SETTINGS = {
    "run": (
        "run_id", "catalogue", "database", "output_directory",
        "pair_product_directory", "pair_store_directory", "scene_checksums",
        "pair_workers", "retain_pair_matches", "trajectories",
    ),
    "matcher": (
        "repository", "checkpoint", "device", "tile_batch_size",
        "prefix_cuda_graph", "cuda_graph_warmup_batches",
    ),
    "field": (
        "missing_node_fallback", "gap_fill_rigid_edge_m", "gap_fill_rigid_spread_m",
        "gap_fill_point_neighbour_count", "gap_fill_point_minimum_agreeing_matches",
        "gap_fill_point_neighbour_distance_m", "gap_fill_point_agreement_distance_m",
        "shear_point_estimates",
    ),
    "routing": (
        "planning_grid_spacing_m", "targeted_recovery",
        "maximum_recovery_elapsed_hours", "candidate_minimum_elapsed_hours",
        "candidate_maximum_elapsed_hours", "candidate_minimum_overlap_fraction",
        "candidate_minimum_overlap_area_m2", "exclude_same_acquisition_pass",
        "require_orbit_metadata", "primary_maximum_pairs_per_target",
        "candidate_pair_ids", "recovery_target_policy",
        "recovery_minimum_target_valid_fraction", "recovery_target_valid_window_m",
        "source_coverage_completion", "source_coverage_minimum_cells",
    ),
    "open_water": ("sic_root",),
    "surface": (
        "land_raster", "point_sic_root", "point_sic_grid", "point_sic_fallback_root",
        "point_sic_fallback_grid", "point_open_water_threshold_percent",
    ),
}


def matching_settings(config) -> dict:
    """The configuration values a pair product depends on."""
    values = asdict(config)
    settings = {"analysis_epsg": values["analysis_epsg"]}
    for section, names in MATCHING_SETTINGS.items():
        settings[section] = {name: values[section][name] for name in names}
    if not config.surface.land_raster:
        # Without a land raster the inland distance threshold has no effect.
        settings["surface"] = {}
    return json.loads(json.dumps(settings))


def matching_inputs(config) -> dict:
    """Content checksums of the run-wide inputs a pair product depends on."""
    matcher = config.matcher
    return {
        "model_checkpoint_sha256": _file_checksum(matcher.checkpoint) if matcher.checkpoint else None,
        "model_source_sha256": _source_tree_sha256(matcher.repository) if matcher.repository else None,
        "land_raster_sha256": (
            _file_checksum(config.surface.land_raster) if config.surface.land_raster else None
        ),
    }


def matching_identity(config) -> dict:
    return {
        "matching_version": MATCHING_VERSION,
        "settings": matching_settings(config),
        "inputs": matching_inputs(config),
    }


def matching_sha256(config) -> str:
    return identity_sha256(matching_identity(config))


def identity_sha256(identity: dict) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_checksum(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _cached_file_sha256(str(resolved), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=None)
def _cached_file_sha256(path: str, _size: int, _mtime_ns: int) -> str:
    return file_sha256(path)


@lru_cache(maxsize=8)
def _source_tree_sha256(repository: str) -> str:
    """Digest of the EfficientLoFTR model code (``src/**/*.py``), independent of paths."""
    root = Path(repository).expanduser().resolve() / "src"
    if not root.is_dir():
        raise FileNotFoundError(f"EfficientLoFTR source directory not found: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class SceneChecksums:
    """SHA256 of each scene file, read from a run's checksum file or computed lazily.

    A checksum file records each file's size and modification time; a scene whose file
    changed after its checksum was taken is refused rather than silently re-hashed.
    """

    SCHEMA = "limosat-scene-checksums-v1"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._records: dict[str, dict] = {}
        if self.path is not None and self.path.is_file():
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if document.get("schema") != self.SCHEMA:
                raise ValueError(f"unsupported scene checksum file: {self.path}")
            self._records = document["images"]

    def sha256(self, image) -> str:
        record = self._records.get(image.image_id)
        stat = Path(image.path).stat()
        if record is None:
            return _cached_file_sha256(str(Path(image.path).resolve()), stat.st_size, stat.st_mtime_ns)
        if record["size_bytes"] != stat.st_size or record["mtime_ns"] != stat.st_mtime_ns:
            raise ValueError(f"scene file changed since its checksum was recorded: {image.path}")
        return record["sha256"]

    def missing(self, images) -> list:
        return [image for image in images if image.image_id not in self._records]


def write_scene_checksums(images, path: str | Path, workers: int | None = None) -> dict:
    """Checksum every catalogue scene not yet in ``path`` and rewrite it atomically."""
    destination = Path(path)
    existing = SceneChecksums(destination)
    pending = existing.missing(images)
    records = dict(existing._records)
    if pending:
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        workers = workers or max(1, min(len(pending), available or 1))
        paths = [str(Path(image.path).resolve()) for image in pending]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for image, record in zip(pending, pool.map(_scene_record, paths, chunksize=4)):
                records[image.image_id] = record
    document = {"schema": SceneChecksums.SCHEMA, "images": dict(sorted(records.items()))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.writing.{os.getpid()}")
    temporary.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return {"scene_checksums": str(destination), "images": len(records), "computed": len(pending)}


def _scene_record(path: str) -> dict:
    stat = Path(path).stat()
    return {"path": path, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": file_sha256(path)}
