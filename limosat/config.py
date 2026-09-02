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
    maximum_speed_m_per_day: float = 30_000.0

    def __post_init__(self) -> None:
        _require_positive(
            {
                "pixel_size_m": self.pixel_size_m,
                "tile_size_px": self.tile_size_px,
                "transform_grid_spacing_px": self.transform_grid_spacing_px,
                "maximum_speed_m_per_day": self.maximum_speed_m_per_day,
            }
        )
        if self.tile_margin_px < 0 or self.tile_margin_px * 2 >= self.tile_size_px:
            raise ValueError("tile_margin_px leaves no source-tile core")
        if (
            self.endpoint_support_radius_px < 0
            or self.endpoint_support_radius_px * 2 >= self.tile_size_px
        ):
            raise ValueError("endpoint_support_radius_px leaves no valid tile")

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

    def __post_init__(self) -> None:
        _require_positive(asdict(self))
        if self.minimum_agreeing_matches > self.neighbour_count:
            raise ValueError("minimum_agreeing_matches exceeds neighbour_count")


@dataclass(frozen=True)
class RoutingConfig:
    """Causal pair routing and measured-loss recovery settings."""

    mode: Literal["same_center", "sequential", "sequential_local"] = (
        "sequential_local"
    )
    initial: Literal["same_center", "phase_correlation"] = "phase_correlation"
    phase_correlation_failure: Literal["same_center", "error"] = "same_center"
    phase_correlation_minimum_response: float = 0.05
    residual_edge_recovery: bool = True
    targeted_recovery: bool = True
    maximum_recovery_elapsed_hours: float = 96.0
    targeted_selection_buffer_m: float = 6_400.0
    candidate_minimum_elapsed_hours: float = 1.0
    candidate_maximum_elapsed_hours: float = 96.0
    candidate_minimum_overlap_fraction: float = 0.05
    candidate_minimum_overlap_area_m2: float = 1_024_000_000.0
    exclude_same_acquisition_pass: bool = True
    require_orbit_metadata: bool = False
    candidate_pair_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= self.phase_correlation_minimum_response <= 1:
            raise ValueError(
                "phase_correlation_minimum_response must be in [0, 1]"
            )
        if self.maximum_recovery_elapsed_hours <= 0:
            raise ValueError("maximum_recovery_elapsed_hours must be positive")
        if self.targeted_selection_buffer_m <= 0:
            raise ValueError("targeted_selection_buffer_m must be positive")
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
class TrajectoryConfig:
    """Observed trajectory construction settings in projected metres."""

    add_as_coverage_enters: bool = True
    new_point_exclusion_radius_m: float = 2_000.0
    convergence_audit_radius_m: float | None = None

    def __post_init__(self) -> None:
        if self.new_point_exclusion_radius_m <= 0:
            raise ValueError("new_point_exclusion_radius_m must be positive")
        if (
            self.convergence_audit_radius_m is not None
            and self.convergence_audit_radius_m <= 0
        ):
            raise ValueError("convergence_audit_radius_m must be positive")


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
    retain_pair_matches: bool = False

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id cannot be empty")
        if self.analysis_epsg != 3413:
            raise ValueError("LiMOSAT coordinates are fixed to EPSG:3413")
        if self.pair_workers < 1:
            raise ValueError("pair_workers must be at least one")
        if not self.catalogue or not self.database or not self.output_directory:
            raise ValueError("catalogue, database, and output_directory are required")

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
    ):
        if resolved.get(name):
            resolved[name] = str(_resolve_path(base, resolved[name]))
    matcher_values = dict(resolved.get("matcher", {}))
    for name in ("repository", "checkpoint"):
        if matcher_values.get(name):
            matcher_values[name] = str(_resolve_path(base, matcher_values[name]))
    resolved["matcher"] = _construct(MatcherConfig, matcher_values)
    resolved["field"] = _construct(FieldConfig, resolved.get("field", {}))
    resolved["routing"] = _construct(RoutingConfig, resolved.get("routing", {}))
    open_water_values = dict(resolved.get("open_water", {}))
    if open_water_values.get("sic_root"):
        open_water_values["sic_root"] = str(
            _resolve_path(base, open_water_values["sic_root"])
        )
    resolved["open_water"] = _construct(OpenWaterConfig, open_water_values)
    resolved["trajectories"] = _construct(
        TrajectoryConfig, resolved.get("trajectories", {})
    )
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
