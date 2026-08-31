"""Validated float64 scientific products used throughout LiMOSAT."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class MotionMatches:
    source_xy_m: np.ndarray
    target_xy_m: np.ndarray
    score: np.ndarray
    source_tile: np.ndarray
    target_tile: np.ndarray

    def __post_init__(self) -> None:
        source = _xy(self.source_xy_m, "source_xy_m")
        target = _xy(self.target_xy_m, "target_xy_m")
        score = np.asarray(self.score, dtype=np.float64)
        source_tile = np.asarray(self.source_tile, dtype=np.int32)
        target_tile = np.asarray(self.target_tile, dtype=np.int32)
        count = len(source)
        if target.shape != source.shape or any(
            len(values) != count for values in (score, source_tile, target_tile)
        ):
            raise ValueError("match arrays must have a common length")
        object.__setattr__(self, "source_xy_m", source)
        object.__setattr__(self, "target_xy_m", target)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "source_tile", source_tile)
        object.__setattr__(self, "target_tile", target_tile)

    @classmethod
    def empty(cls) -> "MotionMatches":
        return cls(
            np.empty((0, 2)),
            np.empty((0, 2)),
            np.empty(0),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    @property
    def displacement_m(self) -> np.ndarray:
        return self.target_xy_m - self.source_xy_m

    def __len__(self) -> int:
        return len(self.score)


@dataclass(frozen=True)
class DisplacementField:
    pair_id: str
    source_image_id: str
    target_image_id: str
    source_time_utc: datetime
    target_time_utc: datetime
    grid_row: np.ndarray
    grid_column: np.ndarray
    source_xy_m: np.ndarray
    displacement_m: np.ndarray
    available: np.ndarray
    selected_matches: np.ndarray
    candidate_matches: np.ndarray
    support_radius_m: np.ndarray
    maximum_residual_m: np.ndarray
    crs_epsg: int = 3413

    def __post_init__(self) -> None:
        if self.crs_epsg != 3413:
            raise ValueError("displacement fields must use EPSG:3413")
        source = _xy(self.source_xy_m, "source_xy_m")
        displacement = _xy(self.displacement_m, "displacement_m")
        available = np.asarray(self.available, dtype=bool)
        count = len(source)
        arrays = {
            "grid_row": np.asarray(self.grid_row, dtype=np.int32),
            "grid_column": np.asarray(self.grid_column, dtype=np.int32),
            "available": available,
            "selected_matches": np.asarray(self.selected_matches, dtype=np.int32),
            "candidate_matches": np.asarray(self.candidate_matches, dtype=np.int32),
            "support_radius_m": np.asarray(self.support_radius_m, dtype=np.float64),
            "maximum_residual_m": np.asarray(
                self.maximum_residual_m, dtype=np.float64
            ),
        }
        if displacement.shape != source.shape or any(
            len(value) != count for value in arrays.values()
        ):
            raise ValueError("field arrays must have a common length")
        if np.isfinite(displacement[~available]).any():
            raise ValueError(
                "unavailable field nodes must contain explicit NaN displacement"
            )
        if not np.isfinite(displacement[available]).all():
            raise ValueError("available field nodes require finite displacement")
        object.__setattr__(self, "source_xy_m", source)
        object.__setattr__(self, "displacement_m", displacement)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    def __len__(self) -> int:
        return len(self.available)

    def with_available(self, available: np.ndarray) -> "DisplacementField":
        keep = np.asarray(available, dtype=bool)
        displacement = self.displacement_m.copy()
        displacement[~keep] = np.nan
        values = dict(self.__dict__)
        values.update(available=keep, displacement_m=displacement)
        return DisplacementField(**values)

    @property
    def checksum(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.grid_row,
            self.grid_column,
            self.source_xy_m,
            self.displacement_m,
            self.available,
            self.selected_matches,
            self.candidate_matches,
            self.support_radius_m,
            self.maximum_residual_m,
        ):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class PairResult:
    matches: MotionMatches
    field: DisplacementField
    fold_rejected_indices: np.ndarray
    runtime_seconds: dict[str, float]
    matcher_calls: int


@dataclass(frozen=True)
class FieldEdge:
    field: DisplacementField
    skipped_images: int = 0

    @property
    def source_image_id(self) -> str:
        return self.field.source_image_id

    @property
    def target_image_id(self) -> str:
        return self.field.target_image_id


def _xy(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    return array
