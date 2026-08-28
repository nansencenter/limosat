"""Configuration for learned sea-ice drift workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ALIKEDConfig:
    """Scientific parameters with explicit pixel and metre units."""

    analysis_epsg: int = 3413
    model_name: str = "aliked-n16"
    random_seed: int = 20260817
    pixel_size_m: float = 80.0
    tile_size_px: int = 512
    tile_margin_px: int = 32
    transform_grid_spacing_px: int = 32
    tile_grid_origin_m: float = 0.0
    features_per_tile: int = 1024
    feature_support_radius_px: int = 16
    detection_threshold: float = 0.2
    minimum_features_per_match: int = 4

    lightglue_layers: int = 5
    lightglue_depth_confidence: float = 0.95
    lightglue_width_confidence: float = 0.99
    lightglue_match_threshold: float = 0.10

    maximum_speed_m_per_day: float = 30_000.0
    target_tile_limit: int | None = None

    grid_spacing_m: float = 4_000.0
    neighbour_count: int = 12
    minimum_agreeing_matches: int = 8
    maximum_neighbour_distance_m: float = 6_000.0
    agreement_distance_m: float = 1_000.0

    def __post_init__(self) -> None:
        positive = {
            "pixel_size_m": self.pixel_size_m,
            "tile_size_px": self.tile_size_px,
            "transform_grid_spacing_px": self.transform_grid_spacing_px,
            "features_per_tile": self.features_per_tile,
            "feature_support_radius_px": self.feature_support_radius_px,
            "minimum_features_per_match": self.minimum_features_per_match,
            "maximum_speed_m_per_day": self.maximum_speed_m_per_day,
            "grid_spacing_m": self.grid_spacing_m,
            "neighbour_count": self.neighbour_count,
            "minimum_agreeing_matches": self.minimum_agreeing_matches,
            "maximum_neighbour_distance_m": self.maximum_neighbour_distance_m,
            "agreement_distance_m": self.agreement_distance_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"ALIKED config values must be positive: {invalid}")
        if self.tile_margin_px * 2 >= self.tile_size_px:
            raise ValueError("tile margins leave no tile core")
        if self.tile_margin_px < 0:
            raise ValueError("tile margin cannot be negative")
        if self.minimum_agreeing_matches > self.neighbour_count:
            raise ValueError("minimum agreeing matches exceed neighbour count")
        if not 1 <= self.lightglue_layers <= 9:
            raise ValueError("LightGlue layers must be between one and nine")
        for name, value in {
            "detection threshold": self.detection_threshold,
            "LightGlue depth confidence": self.lightglue_depth_confidence,
            "LightGlue width confidence": self.lightglue_width_confidence,
        }.items():
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not 0.0 <= self.lightglue_match_threshold < 1.0:
            raise ValueError("LightGlue match threshold must be in [0, 1)")
        if self.target_tile_limit is not None and self.target_tile_limit < 1:
            raise ValueError("target tile limit must be positive")

    @property
    def tile_core_size_m(self) -> float:
        return (
            self.tile_size_px - 2 * self.tile_margin_px
        ) * self.pixel_size_m

    def maximum_displacement_m(self, elapsed_hours: float) -> float:
        return self.maximum_speed_m_per_day * elapsed_hours / 24.0


@dataclass(frozen=True)
class EfficientLoFTRConfig:
    """Parameters used by EfficientLoFTR matching and field construction.

    Matcher-specific settings are deliberately separate from ``ALIKEDConfig``
    so EfficientLoFTR experiments cannot silently inherit detector or
    LightGlue parameters that they do not use.
    """

    analysis_epsg: int = 3413
    model_name: str = "efficientloftr-official-opt"
    pixel_size_m: float = 80.0
    tile_size_px: int = 512
    tile_margin_px: int = 32
    transform_grid_spacing_px: int = 32
    tile_grid_origin_m: float = 0.0
    endpoint_support_radius_px: int = 16

    maximum_speed_m_per_day: float = 30_000.0

    grid_spacing_m: float = 4_000.0
    neighbour_count: int = 12
    minimum_agreeing_matches: int = 8
    maximum_neighbour_distance_m: float = 6_000.0
    agreement_distance_m: float = 1_000.0
    score_weighting: Literal["raw", "uniform"] = "raw"

    # Keep the physical interpolation/topology limit fixed when testing a
    # different output-grid spacing. This is 1.6 * the selected 4 km grid.
    maximum_triangle_edge_m: float = 6_400.0
    new_point_exclusion_radius_m: float = 2_000.0

    def __post_init__(self) -> None:
        positive = {
            "pixel_size_m": self.pixel_size_m,
            "tile_size_px": self.tile_size_px,
            "transform_grid_spacing_px": self.transform_grid_spacing_px,
            "maximum_speed_m_per_day": self.maximum_speed_m_per_day,
            "grid_spacing_m": self.grid_spacing_m,
            "neighbour_count": self.neighbour_count,
            "minimum_agreeing_matches": self.minimum_agreeing_matches,
            "maximum_neighbour_distance_m": self.maximum_neighbour_distance_m,
            "agreement_distance_m": self.agreement_distance_m,
            "maximum_triangle_edge_m": self.maximum_triangle_edge_m,
            "new_point_exclusion_radius_m": self.new_point_exclusion_radius_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                f"EfficientLoFTR config values must be positive: {invalid}"
            )
        if self.tile_margin_px < 0:
            raise ValueError("tile margin cannot be negative")
        if self.tile_margin_px * 2 >= self.tile_size_px:
            raise ValueError("tile margins leave no tile core")
        if self.endpoint_support_radius_px < 0:
            raise ValueError("endpoint support radius cannot be negative")
        if self.endpoint_support_radius_px * 2 >= self.tile_size_px:
            raise ValueError("endpoint support radius leaves no valid pixels")
        if self.minimum_agreeing_matches > self.neighbour_count:
            raise ValueError("minimum agreeing matches exceed neighbour count")
        if self.score_weighting not in {"raw", "uniform"}:
            raise ValueError("score weighting must be 'raw' or 'uniform'")

    @property
    def tile_core_size_m(self) -> float:
        return (
            self.tile_size_px - 2 * self.tile_margin_px
        ) * self.pixel_size_m

    def maximum_displacement_m(self, elapsed_hours: float) -> float:
        return self.maximum_speed_m_per_day * elapsed_hours / 24.0
