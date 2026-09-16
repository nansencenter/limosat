"""Quality-control scoring for composed ELoFTR trajectory extensions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from .trajectory import TrajectoryExtension


QC_PROTOCOL_ID = "eloftr_basic_extension_qc_v1"


@dataclass(frozen=True)
class ExtensionQCConfig:
    """Minimal ELoFTR extension QC with non-destructive local diagnostics."""

    configured_speed_m_per_day: float
    hard_speed_m_per_day: float = 60_000.0
    neighbour_search_radius_m: float = 20_000.0
    minimum_neighbours: int = 8
    maximum_neighbours: int = 24
    local_review_z: float = 8.0
    local_review_floor_m: float = 10_000.0
    residual_noise_floor_m: float = 150.0

    def __post_init__(self) -> None:
        numeric = {
            "configured_speed_m_per_day": self.configured_speed_m_per_day,
            "hard_speed_m_per_day": self.hard_speed_m_per_day,
            "neighbour_search_radius_m": self.neighbour_search_radius_m,
            "local_review_z": self.local_review_z,
            "local_review_floor_m": self.local_review_floor_m,
            "residual_noise_floor_m": self.residual_noise_floor_m,
        }
        invalid = [
            name
            for name, value in numeric.items()
            if not np.isfinite(value) or value <= 0
        ]
        if invalid:
            raise ValueError(f"QC values must be finite and positive: {invalid}")
        if self.hard_speed_m_per_day < self.configured_speed_m_per_day:
            raise ValueError("hard speed must not be below the configured speed")
        if self.minimum_neighbours < 3:
            raise ValueError("minimum_neighbours must be at least three")
        if self.maximum_neighbours < self.minimum_neighbours:
            raise ValueError(
                "maximum_neighbours must be at least minimum_neighbours"
            )


@dataclass(frozen=True)
class ScoredExtension:
    extension: TrajectoryExtension
    elapsed_seconds: float
    dx_m: float
    dy_m: float
    distance_m: float
    speed_m_per_day: float
    local_neighbour_count: int
    local_support_radius_m: float | None
    local_median_dx_m: float | None
    local_median_dy_m: float | None
    local_residual_m: float | None
    local_robust_z: float | None
    exceeds_configured_speed: bool
    hard_speed_reject: bool
    local_neighbour_review: bool
    qc_status: str
    qc_reasons: str


def score_extensions(
    extensions: Sequence[TrajectoryExtension],
    config: ExtensionQCConfig,
) -> tuple[ScoredExtension, ...]:
    """Score exact-pair neighbours without making local rejection decisions."""
    if not extensions:
        return ()
    values: list[ScoredExtension | None] = [None] * len(extensions)
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, extension in enumerate(extensions):
        groups.setdefault(
            (
                extension.pair_id,
                extension.source_image_id,
                extension.target_image_id,
            ),
            [],
        ).append(index)

    for indices in groups.values():
        xy = np.asarray(
            [[extensions[index].x0_m, extensions[index].y0_m] for index in indices],
            dtype=np.float64,
        )
        uv = np.asarray(
            [extensions[index].displacement_m for index in indices],
            dtype=np.float64,
        )
        local = _local_diagnostics(xy, uv, config)
        for local_index, index in enumerate(indices):
            extension = extensions[index]
            dx_m, dy_m = uv[local_index]
            distance_m = float(math.hypot(dx_m, dy_m))
            speed = distance_m / extension.elapsed_seconds * 86_400.0
            exceeds_configured = speed > config.configured_speed_m_per_day
            hard_reject = speed > config.hard_speed_m_per_day
            local_review = bool(local["review"][local_index])
            reasons = []
            if hard_reject:
                reasons.append("hard_speed")
            elif exceeds_configured:
                reasons.append("configured_speed")
            if local_review:
                reasons.append("local_neighbour")
            status = "reject" if hard_reject else "review" if reasons else "accept"
            values[index] = ScoredExtension(
                extension=extension,
                elapsed_seconds=extension.elapsed_seconds,
                dx_m=float(dx_m),
                dy_m=float(dy_m),
                distance_m=distance_m,
                speed_m_per_day=speed,
                local_neighbour_count=int(local["count"][local_index]),
                local_support_radius_m=_optional_float(
                    local["support_radius_m"][local_index]
                ),
                local_median_dx_m=_optional_float(
                    local["median_uv"][local_index, 0]
                ),
                local_median_dy_m=_optional_float(
                    local["median_uv"][local_index, 1]
                ),
                local_residual_m=_optional_float(local["residual_m"][local_index]),
                local_robust_z=_optional_float(local["robust_z"][local_index]),
                exceeds_configured_speed=exceeds_configured,
                hard_speed_reject=hard_reject,
                local_neighbour_review=local_review,
                qc_status=status,
                qc_reasons=",".join(reasons),
            )
    if any(value is None for value in values):  # pragma: no cover - invariant
        raise RuntimeError("extension QC did not score every row")
    return tuple(value for value in values if value is not None)


def _local_diagnostics(
    xy: np.ndarray,
    uv: np.ndarray,
    config: ExtensionQCConfig,
) -> dict[str, np.ndarray]:
    count = len(xy)
    result = {
        "count": np.zeros(count, dtype=np.int32),
        "support_radius_m": np.full(count, np.nan),
        "median_uv": np.full((count, 2), np.nan),
        "residual_m": np.full(count, np.nan),
        "robust_z": np.full(count, np.nan),
        "review": np.zeros(count, dtype=bool),
    }
    if count <= config.minimum_neighbours:
        return result
    k = min(config.maximum_neighbours + 1, count)
    distances, neighbours = cKDTree(xy).query(
        xy,
        k=k,
        distance_upper_bound=config.neighbour_search_radius_m,
    )
    for index in range(count):
        valid = (
            np.isfinite(distances[index])
            & (neighbours[index] < count)
            & (neighbours[index] != index)
        )
        selected = neighbours[index, valid].astype(int)
        selected_distances = distances[index, valid]
        result["count"][index] = len(selected)
        if len(selected) < config.minimum_neighbours:
            continue
        neighbour_uv = uv[selected]
        median_uv = np.median(neighbour_uv, axis=0)
        neighbour_residuals = np.linalg.norm(neighbour_uv - median_uv, axis=1)
        residual = float(np.linalg.norm(uv[index] - median_uv))
        median_residual = float(np.median(neighbour_residuals))
        mad = float(np.median(np.abs(neighbour_residuals - median_residual)))
        scale = max(1.4826 * mad, config.residual_noise_floor_m)
        robust_z = max(0.0, (residual - median_residual) / scale)
        threshold = max(
            config.local_review_floor_m,
            median_residual + config.local_review_z * scale,
        )
        result["support_radius_m"][index] = float(np.max(selected_distances))
        result["median_uv"][index] = median_uv
        result["residual_m"][index] = residual
        result["robust_z"][index] = robust_z
        result["review"][index] = residual > threshold
    return result


def _optional_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
