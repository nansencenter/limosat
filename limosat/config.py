"""Resolved LiMOSAT configuration with explicit physical units."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field as dataclass_field, fields
from pathlib import Path
from typing import Any, Literal, TypeVar


@dataclass(frozen=True)
class MatcherConfig:
    """EfficientLoFTR inference and north-up tile settings."""

    repository: str = ""
    checkpoint: str = ""
    model_name: str = "efficientloftr-official-opt"
    device: str = "cpu"
    pixel_size_m: float = 80.0
    tile_size_px: int = 512
    tile_margin_px: int = 32
    endpoint_support_radius_px: int = 16
    transform_grid_spacing_px: int = 32
    tile_grid_origin_m: float = 0.0
    maximum_speed_m_per_day: float = 43_200.0
    tile_batch_size: int = 4
    prefix_cuda_graph: bool = True
    cuda_graph_warmup_batches: int = 3
    # Extra tile grids, as (x, y) offsets in fractions of the tile core. The first
    # entry is the primary grid; ((0, 0), (0.5, 0.5)) adds the diagonal half-offset grid.
    tile_grid_offsets: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    # Re-match a fine tile with its source blanked where the aligned target is
    # invalid when the target window is less valid than this fraction (0 disables).
    masked_rematch_target_valid_fraction: float = 0.0
    # Overlapping layouts keep one match per source bin of this size.
    layout_dedup_m: float = 160.0
    # "primary_first" keeps primary-grid field nodes and adds the others only in
    # its gaps; "union" estimates one field from all matches.
    layout_field_merge: str = "primary_first"
    # "pair_cache" resamples each scene once per lattice block and crops tiles.
    tile_sampling: str = "per_tile"
    # Stricter agreement for field nodes that only the layered layout measures
    # (primary_first gap nodes). None keeps the field settings. Buoy evaluation
    # (October 2026, MOSAiC) favoured 10 agreeing matches within 500 m.
    layout_gap_minimum_agreeing_matches: int | None = None
    layout_gap_agreement_distance_m: float | None = None
    # Scene band 2 values at or above this are invalid pixels. The preprocessed
    # scenes mark land and everything within ~5 km of it as 2 (and, near the date
    # line, long strips of sea ice); 3 ignores that land mask, leaving land to
    # surface.land_raster, which must then be set.
    scene_mask_invalid_from: int = 2

    def __post_init__(self) -> None:
        if self.scene_mask_invalid_from not in (2, 3):
            raise ValueError("scene_mask_invalid_from must be 2 or 3")
        _require_positive(
            {
                "pixel_size_m": self.pixel_size_m,
                "tile_size_px": self.tile_size_px,
                "transform_grid_spacing_px": self.transform_grid_spacing_px,
                "maximum_speed_m_per_day": self.maximum_speed_m_per_day,
                "tile_batch_size": self.tile_batch_size,
                "cuda_graph_warmup_batches": self.cuda_graph_warmup_batches,
            }
        )
        if self.tile_margin_px < 0 or self.tile_margin_px * 2 >= self.tile_size_px:
            raise ValueError("tile_margin_px leaves no source-tile core")
        if (
            self.endpoint_support_radius_px < 0
            or self.endpoint_support_radius_px * 2 >= self.tile_size_px
        ):
            raise ValueError("endpoint_support_radius_px leaves no valid tile")
        offsets = tuple(
            (float(item[0]), float(item[1])) for item in self.tile_grid_offsets
        )
        if (
            not offsets
            or any(len(item) != 2 for item in self.tile_grid_offsets)
            or any(not 0.0 <= value < 1.0 for item in offsets for value in item)
            or len(set(offsets)) != len(offsets)
        ):
            raise ValueError(
                "tile_grid_offsets must be distinct (x, y) core fractions in [0, 1)"
            )
        object.__setattr__(self, "tile_grid_offsets", offsets)
        if not 0.0 <= self.masked_rematch_target_valid_fraction <= 1.0:
            raise ValueError("masked_rematch_target_valid_fraction must be in [0, 1]")
        if self.layout_dedup_m <= 0:
            raise ValueError("layout_dedup_m must be positive")
        if self.layout_field_merge not in {"primary_first", "union"}:
            raise ValueError("layout_field_merge must be 'primary_first' or 'union'")
        if self.tile_sampling not in {"per_tile", "pair_cache"}:
            raise ValueError("tile_sampling must be 'per_tile' or 'pair_cache'")
        if self.layout_gap_minimum_agreeing_matches is not None and (
            isinstance(self.layout_gap_minimum_agreeing_matches, bool)
            or not isinstance(self.layout_gap_minimum_agreeing_matches, int)
            or self.layout_gap_minimum_agreeing_matches < 1
        ):
            raise ValueError("layout_gap_minimum_agreeing_matches must be a positive integer")
        if self.layout_gap_agreement_distance_m is not None and not (
            0 < self.layout_gap_agreement_distance_m < float("inf")
        ):
            raise ValueError("layout_gap_agreement_distance_m must be finite and positive")

    @property
    def layered_layout(self) -> bool:
        """Whether a location can be matched by more than one fine tile."""
        return (
            len(self.tile_grid_offsets) > 1
            or self.masked_rematch_target_valid_fraction > 0
        )

    @property
    def tile_core_size_m(self) -> float:
        return (
            self.tile_size_px - 2 * self.tile_margin_px
        ) * self.pixel_size_m

    def maximum_displacement_m(self, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive")
        return self.maximum_speed_m_per_day * elapsed_seconds / 86_400.0


@dataclass(frozen=True)
class FieldConfig:
    """Pair-field consensus settings; every distance is in metres."""

    grid_spacing_m: float = 4_000.0
    neighbour_count: int = 12
    minimum_agreeing_matches: int = 8
    maximum_neighbour_distance_m: float = 6_000.0
    agreement_distance_m: float = 1_000.0
    maximum_triangle_edge_m: float = 6_400.0
    # Off by default since October 2026: at MOSAiC and Beaufort buoys the points it
    # adds were 2-3x less accurate than measured nodes at the same lags.
    missing_node_fallback: bool = False
    # Composition-time gap filling, used only where a point has no usable triangle
    # (off by default; field.py fill_gaps). Rigid: accept a triangle of accepted
    # nodes with edges up to gap_fill_rigid_edge_m when its node displacements
    # differ by at most gap_fill_rigid_spread_m. Point-local: estimate the point's
    # own motion from the raw matches around it (needs retained matches). Neither
    # changes a pair product, so neither is in the matching fingerprint. Tested on
    # the January 2020 week and MOSAiC Jan-Apr 2020 (November 2026) with 12800 m /
    # 300 m and 12 matches within 3 km, 8 agreeing within 500 m.
    gap_fill_rigid_edge_m: float | None = None
    gap_fill_rigid_spread_m: float = 300.0
    gap_fill_point_neighbour_count: int | None = None
    gap_fill_point_minimum_agreeing_matches: int = 8
    gap_fill_point_neighbour_distance_m: float = 3_000.0
    gap_fill_point_agreement_distance_m: float = 500.0
    # Shear-zone point estimates (off by default; field.py refine_shear_samples). A point
    # is normally moved by linear interpolation between three 4 km nodes, which blends the
    # two sides of a lead or shear line running between them. Where those three nodes
    # disagree by more than the point's own point-local estimate (gap_fill_point_*
    # settings) disagrees with its matches (its maximum residual), that estimate is used
    # instead; in coherent ice the interpolation, whose errors neighbouring points share,
    # is kept. No threshold of its own. Points without a sample get their own estimate
    # too, before any rigid-triangle bridge (gap filling then trusts nearby matches
    # first). Needs every raw match of each pair at composition.
    # Tested November 2026 on the Jan-Apr 2020 stored pairs: displacement at MOSAiC and
    # IABP buoys, p90 152 -> 124 m (median 352 -> 88 m where the nodes disagree by more
    # than 1 km); deformation of MOSAiC buoy triangles, median error 2-4 km 0.0122 ->
    # 0.0117 /day, 4-8 km 0.0079 -> 0.0072 /day.
    shear_point_estimates: bool = False

    def __post_init__(self) -> None:
        _require_positive({
            key: value for key, value in asdict(self).items()
            if key not in ("missing_node_fallback", "shear_point_estimates") and value is not None
        })
        if not isinstance(self.missing_node_fallback, bool):
            raise ValueError("missing_node_fallback must be a boolean")
        if not isinstance(self.shear_point_estimates, bool):
            raise ValueError("shear_point_estimates must be a boolean")
        if self.shear_point_estimates and self.gap_fill_point_neighbour_count is None:
            raise ValueError("shear_point_estimates requires the gap_fill_point_* settings")
        if self.minimum_agreeing_matches > self.neighbour_count:
            raise ValueError("minimum_agreeing_matches exceeds neighbour_count")
        if (
            self.gap_fill_rigid_edge_m is not None
            and self.gap_fill_rigid_edge_m <= self.maximum_triangle_edge_m
        ):
            raise ValueError("gap_fill_rigid_edge_m must exceed maximum_triangle_edge_m")
        if self.gap_fill_point_neighbour_count is not None and (
            isinstance(self.gap_fill_point_neighbour_count, bool)
            or not isinstance(self.gap_fill_point_neighbour_count, int)
            or self.gap_fill_point_minimum_agreeing_matches
            > self.gap_fill_point_neighbour_count
        ):
            raise ValueError(
                "gap_fill_point_neighbour_count must be an integer of at least "
                "gap_fill_point_minimum_agreeing_matches"
            )

    @property
    def gap_filling(self) -> bool:
        return (
            self.gap_fill_rigid_edge_m is not None
            or self.gap_fill_point_neighbour_count is not None
        )


@dataclass(frozen=True)
class RoutingConfig:
    """Pair planning and causal pair routing settings."""

    planning_grid_spacing_m: float = 4_000.0
    mode: Literal["same_center", "sequential", "sequential_local"] = (
        "sequential_local"
    )
    initial: Literal["same_center", "phase_correlation"] = "phase_correlation"
    phase_correlation_failure: Literal["same_center", "error"] = "same_center"
    phase_correlation_minimum_response: float = 0.05
    residual_edge_recovery: bool = True
    candidate_minimum_elapsed_hours: float = 1.0
    candidate_maximum_elapsed_hours: float = 96.0
    candidate_minimum_overlap_fraction: float = 0.05
    candidate_minimum_overlap_area_m2: float = 1_024_000_000.0
    exclude_same_acquisition_pass: bool = True
    require_orbit_metadata: bool = False
    candidate_pair_ids: tuple[str, ...] = ()
    coarse_matching: bool = True
    coarse_pixel_size_m: float = 320.0
    coarse_minimum_matches: int = 12
    coarse_support_radius_m: float = 35_840.0
    coarse_window_stride: int = 1
    # Next-observation primary selection (planning.py _next_observation_selection). Each
    # source image's planning cells (4 km, imaged by both images, no drift allowance) go
    # to the earliest later candidate image that images them, so every piece of ice is
    # carried to its next observation and composition chains the steps. A pair is kept
    # when it is the next observation for at least next_observation_minimum_cells cells.
    # Cells whose next observation is more than next_observation_backup_after_hours away
    # (longer pairs fail more often, mostly in fast drift) also get the following
    # observation as a backup (production: 30 h). plan_surface_cells drops cells that the
    # surface gate excludes at the source time (land >= land_minimum_inland_km inland,
    # point open water) before planning (production: on).
    next_observation_minimum_cells: int = 25
    next_observation_backup_after_hours: float | None = None
    plan_surface_cells: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.next_observation_minimum_cells, bool)
            or not isinstance(self.next_observation_minimum_cells, int)
            or self.next_observation_minimum_cells < 1
        ):
            raise ValueError("next_observation_minimum_cells must be a positive integer")
        if self.next_observation_backup_after_hours is not None and not (
            0 < self.next_observation_backup_after_hours < float("inf")
        ):
            raise ValueError("next_observation_backup_after_hours must be positive")
        if not isinstance(self.plan_surface_cells, bool):
            raise ValueError("plan_surface_cells must be a boolean")
        if not 0 < self.planning_grid_spacing_m < float("inf"):
            raise ValueError("planning_grid_spacing_m must be finite and positive")
        if not isinstance(self.coarse_matching, bool):
            raise ValueError("coarse_matching must be a boolean")
        for name in ("coarse_pixel_size_m", "coarse_support_radius_m"):
            value = getattr(self, name)
            if not 0 < value < float("inf"):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.coarse_minimum_matches, bool)
            or not isinstance(self.coarse_minimum_matches, int)
            or self.coarse_minimum_matches < 1
        ):
            raise ValueError("coarse_minimum_matches must be a positive integer")
        if (
            isinstance(self.coarse_window_stride, bool)
            or not isinstance(self.coarse_window_stride, int)
            or self.coarse_window_stride < 1
            or self.coarse_window_stride % 2 == 0
        ):
            raise ValueError("coarse_window_stride must be a positive odd integer")
        if not 0 <= self.phase_correlation_minimum_response <= 1:
            raise ValueError(
                "phase_correlation_minimum_response must be in [0, 1]"
            )
        if self.candidate_minimum_elapsed_hours <= 0:
            raise ValueError("candidate_minimum_elapsed_hours must be positive")
        if (
            self.candidate_maximum_elapsed_hours
            <= self.candidate_minimum_elapsed_hours
        ):
            raise ValueError(
                "candidate_maximum_elapsed_hours must exceed the minimum"
            )
        if not 0 < self.candidate_minimum_overlap_fraction <= 1:
            raise ValueError(
                "candidate_minimum_overlap_fraction must be in (0, 1]"
            )
        if self.candidate_minimum_overlap_area_m2 < 0:
            raise ValueError("candidate_minimum_overlap_area_m2 cannot be negative")
        object.__setattr__(
            self,
            "candidate_pair_ids",
            tuple(sorted(set(self.candidate_pair_ids))),
        )


@dataclass(frozen=True)
class OpenWaterConfig:
    """Conservative ancillary sea-ice-concentration compute gate."""

    enabled: bool = False
    sic_root: str = ""
    threshold_percent: float = 15.0
    maximum_age_days: int = 1
    samples_per_axis: int = 5

    def __post_init__(self) -> None:
        if self.enabled and not self.sic_root:
            raise ValueError("open_water.sic_root is required when enabled")
        if not 0 <= self.threshold_percent <= 100:
            raise ValueError("open-water threshold_percent must be in [0, 100]")
        if self.maximum_age_days < 0:
            raise ValueError("open-water maximum_age_days cannot be negative")
        if self.samples_per_axis < 2:
            raise ValueError("open-water samples_per_axis must be at least two")


@dataclass(frozen=True)
class SurfaceConfig:
    """Land and point open-water exclusion (off unless a source is set).

    ``land_raster``: inland-distance ``.npz`` in EPSG:3413; cells at least
    ``land_minimum_inland_km`` inland are invalid for matching, composed
    endpoints and seeds. ``point_sic_root``: AMSR2 ASI GeoTIFF root (grid
    ``point_sic_grid``), with an optional fallback product; positions below
    ``point_open_water_threshold_percent`` are neither seeded nor accepted as
    endpoints. The open-water tile gate is separate and unchanged.
    """

    land_raster: str = ""
    land_minimum_inland_km: float = 1.0
    point_sic_root: str = ""
    point_sic_grid: str = "n3125"
    point_sic_fallback_root: str = ""
    point_sic_fallback_grid: str = "n6250"
    point_open_water_threshold_percent: float = 15.0

    def __post_init__(self) -> None:
        if self.land_minimum_inland_km < 0:
            raise ValueError("land_minimum_inland_km cannot be negative")
        if not 0 <= self.point_open_water_threshold_percent <= 100:
            raise ValueError("point_open_water_threshold_percent must be in [0, 100]")

    @property
    def enabled(self) -> bool:
        return bool(self.land_raster or self.point_sic_root)


@dataclass(frozen=True)
class TrajectoryConfig:
    """Observed trajectory construction settings in projected metres."""

    add_as_coverage_enters: bool = True
    new_point_exclusion_radius_m: float = 2_000.0
    # Merge duplicate trajectories (off by default). When the positions of two or more
    # trajectories in one image lie within this distance (chained), only the one with the
    # longest history continues (then more selected matches, lower residual, identity, as
    # in the convergence audit); the others end there and are recorded in
    # trajectory-merges-v1.parquet. Duplicates arise when a lost trajectory's reseeded
    # replacement and the old trajectory, continued later, both persist: in Jan-Apr 2020,
    # 91 % of trajectories < 1 km apart were already that close when first in one image
    # (median 0.39 km) and kept their separation (median change 20 m over ~39 days).
    merge_radius_m: float | None = None
    # Merge only young duplicates (None: any). A trajectory is ended by a merge only if it
    # is at most this old (hours since it was created) and was never positioned in the same
    # image as its survivor before. Duplicates are mostly reseeds on ice already tracked
    # (Jan-Apr 2020: 85-88 % of merges ended a trajectory that never shared an image with
    # its survivor, median age 1 day); established trajectories with their own history are
    # kept even when they converge, since they may follow different ice.
    merge_maximum_younger_age_hours: float | None = None

    def __post_init__(self) -> None:
        if self.merge_maximum_younger_age_hours is not None and (
            self.merge_radius_m is None or not self.merge_maximum_younger_age_hours > 0
        ):
            raise ValueError("merge_maximum_younger_age_hours needs merge_radius_m and must be positive")
        if self.new_point_exclusion_radius_m <= 0:
            raise ValueError("new_point_exclusion_radius_m must be positive")
        if self.merge_radius_m is not None and not 0 < self.merge_radius_m < self.new_point_exclusion_radius_m:
            raise ValueError("merge_radius_m must be positive and below new_point_exclusion_radius_m")


@dataclass(frozen=True)
class RunConfig:
    """Complete resolved run configuration."""

    run_id: str
    catalogue: str
    database: str
    output_directory: str
    pair_product_directory: str = ""
    analysis_epsg: int = 3413
    pair_workers: int = 1
    matcher: MatcherConfig = dataclass_field(default_factory=MatcherConfig)
    field: FieldConfig = dataclass_field(default_factory=FieldConfig)
    routing: RoutingConfig = dataclass_field(default_factory=RoutingConfig)
    open_water: OpenWaterConfig = dataclass_field(default_factory=OpenWaterConfig)
    trajectories: TrajectoryConfig = dataclass_field(
        default_factory=TrajectoryConfig
    )
    retain_pair_matches: bool = True
    surface: SurfaceConfig = dataclass_field(default_factory=SurfaceConfig)
    # Shared store for primary pair products, keyed by matching fingerprint
    # (pair_identity.py) so runs with the same matching settings reuse each other's
    # pairs. Empty keeps them in pair_product_directory.
    pair_store_directory: str = ""
    # Scene SHA256 file written by `limosat prepare` (see pair_identity.SceneChecksums);
    # empty computes checksums when first needed.
    scene_checksums: str = ""

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id cannot be empty")
        if self.analysis_epsg != 3413:
            raise ValueError("LiMOSAT coordinates are fixed to EPSG:3413")
        if self.pair_workers < 1:
            raise ValueError("pair_workers must be at least one")
        if self.matcher.device.startswith("cuda") and self.pair_workers != 1:
            raise ValueError(
                "CUDA runs require pair_workers=1; use deterministic "
                "batch workers for multi-GPU execution"
            )
        if not self.catalogue or not self.database or not self.output_directory:
            raise ValueError("catalogue, database, and output_directory are required")
        if self.routing.plan_surface_cells and not self.surface.enabled:
            raise ValueError("routing.plan_surface_cells requires surface.land_raster or point SIC")
        if self.matcher.scene_mask_invalid_from == 3 and not self.surface.land_raster:
            raise ValueError("ignoring the scene land mask requires surface.land_raster")
        if self.field.gap_fill_point_neighbour_count is not None and not self.retain_pair_matches:
            raise ValueError("point-local gap filling requires retain_pair_matches")
        if self.field.missing_node_fallback and not self.retain_pair_matches:
            raise ValueError("missing_node_fallback requires retain_pair_matches")
        if self.field.shear_point_estimates and not self.retain_pair_matches:
            raise ValueError("shear_point_estimates requires retain_pair_matches")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def pair_products(self) -> Path:
        """Intermediate immutable pair products used by staged execution."""
        if self.pair_product_directory:
            return Path(self.pair_product_directory)
        database = Path(self.database)
        return database.with_name(database.name + ".pair-products")

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


T = TypeVar("T")


def load_config(path: str | Path) -> RunConfig:
    """Load JSON or YAML and resolve paths relative to the configuration file."""
    config_path = Path(path).resolve()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        values = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - environment error
            raise RuntimeError("PyYAML is required for YAML configuration") from error
        values = yaml.safe_load(text)
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a mapping")
    base = config_path.parent
    resolved = dict(values)
    for name in (
        "catalogue",
        "database",
        "output_directory",
        "pair_product_directory",
        "pair_store_directory",
        "scene_checksums",
    ):
        if resolved.get(name):
            resolved[name] = str(_resolve_path(base, resolved[name]))
    matcher_values = dict(resolved.get("matcher", {}))
    for name in ("repository", "checkpoint"):
        if matcher_values.get(name):
            matcher_values[name] = str(_resolve_path(base, matcher_values[name]))
    resolved["matcher"] = _construct(MatcherConfig, matcher_values)
    resolved["field"] = _construct(FieldConfig, resolved.get("field", {}))
    resolved["routing"] = _construct(
        RoutingConfig, _without_removed(resolved.get("routing", {}), "routing")
    )
    open_water_values = dict(resolved.get("open_water", {}))
    if open_water_values.get("sic_root"):
        open_water_values["sic_root"] = str(
            _resolve_path(base, open_water_values["sic_root"])
        )
    resolved["open_water"] = _construct(OpenWaterConfig, open_water_values)
    resolved["trajectories"] = _construct(
        TrajectoryConfig, _without_removed(resolved.get("trajectories", {}), "trajectories")
    )
    surface_values = dict(resolved.get("surface", {}))
    for name in ("land_raster", "point_sic_root", "point_sic_fallback_root"):
        if surface_values.get(name):
            surface_values[name] = str(_resolve_path(base, surface_values[name]))
    resolved["surface"] = _construct(SurfaceConfig, surface_values)
    return _construct(RunConfig, resolved)


def _construct(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {unknown}")
    return cls(**values)


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _require_positive(values: dict[str, Any]) -> None:
    invalid = [name for name, value in values.items() if float(value) <= 0]
    if invalid:
        raise ValueError(f"values must be positive: {invalid}")


_ANY = object()
# Settings of removed methods (continuation passes, per-target coverage selection, the
# drift-prior and assigned-tile experiments). Configurations written before their
# removal still load when these describe the kept method: _ANY values are ignored (they
# never changed the kept path), others must equal the kept behaviour.
_REMOVED_SETTINGS = {
    "routing": {
        "primary_selection": "next_observation",
        "primary_maximum_pairs_per_target": None,
        "source_coverage_completion": False,
        "source_coverage_minimum_cells": _ANY,
        "primary_tiles": "all",
        "prior_displacement_file": "",
        "targeted_recovery": _ANY,
        "maximum_recovery_elapsed_hours": _ANY,
        "recovery_target_policy": _ANY,
        "recovery_minimum_target_valid_fraction": _ANY,
        "recovery_target_valid_window_m": _ANY,
        # Part of the matching fingerprint at this value (pair_identity.py).
        "targeted_selection_buffer_m": 6_400.0,
    },
    "trajectories": {"convergence_audit_radius_m": _ANY},
}


def _without_removed(values: dict[str, Any], section: str) -> dict[str, Any]:
    kept = dict(values)
    for name, required in _REMOVED_SETTINGS[section].items():
        if name in kept:
            value = kept.pop(name)
            if required is not _ANY and value != required:
                raise ValueError(
                    f"{section}.{name} = {value!r} selects a removed method; "
                    f"only {required!r} is supported"
                )
    return kept
