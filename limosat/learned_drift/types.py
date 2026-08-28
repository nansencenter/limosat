"""Small array containers used by learned sea-ice drift stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class FeatureTile:
    tile_id: int
    row: int
    column: int
    center_xy_m: tuple[float, float]
    core: BaseGeometry
    keypoints_px: torch.Tensor
    descriptors: torch.Tensor
    scores: torch.Tensor
    xy_m: np.ndarray

    def select(self, indices: np.ndarray) -> "FeatureTile":
        tensor_indices = torch.as_tensor(indices, device=self.keypoints_px.device)
        return FeatureTile(
            tile_id=self.tile_id,
            row=self.row,
            column=self.column,
            center_xy_m=self.center_xy_m,
            core=self.core,
            keypoints_px=self.keypoints_px[tensor_indices],
            descriptors=self.descriptors[tensor_indices],
            scores=self.scores[tensor_indices],
            xy_m=self.xy_m[indices],
        )

    def __len__(self) -> int:
        return len(self.keypoints_px)

    def to(self, device: torch.device) -> "FeatureTile":
        return FeatureTile(
            tile_id=self.tile_id,
            row=self.row,
            column=self.column,
            center_xy_m=self.center_xy_m,
            core=self.core,
            keypoints_px=self.keypoints_px.to(device),
            descriptors=self.descriptors.to(device),
            scores=self.scores.to(device),
            xy_m=self.xy_m,
        )


@dataclass(frozen=True)
class ImageFeatures:
    image_path: str
    domain: BaseGeometry
    tiles: tuple[FeatureTile, ...]
    analysis_epsg: int

    @property
    def feature_count(self) -> int:
        return sum(len(tile) for tile in self.tiles)

    def to(self, device: torch.device) -> "ImageFeatures":
        return ImageFeatures(
            image_path=self.image_path,
            domain=self.domain,
            tiles=tuple(tile.to(device) for tile in self.tiles),
            analysis_epsg=self.analysis_epsg,
        )


@dataclass(frozen=True)
class MotionMatches:
    source_feature_id: np.ndarray
    source_tile_id: np.ndarray
    target_tile_id: np.ndarray
    source_xy_m: np.ndarray
    target_xy_m: np.ndarray
    score: np.ndarray

    @classmethod
    def empty(cls) -> "MotionMatches":
        return cls(
            source_feature_id=np.empty(0, dtype=np.int64),
            source_tile_id=np.empty(0, dtype=np.int32),
            target_tile_id=np.empty(0, dtype=np.int32),
            source_xy_m=np.empty((0, 2), dtype=np.float64),
            target_xy_m=np.empty((0, 2), dtype=np.float64),
            score=np.empty(0, dtype=np.float32),
        )

    @classmethod
    def from_frame(cls, rows: pd.DataFrame) -> "MotionMatches":
        if rows.empty:
            return cls.empty()
        score_column = (
            "matcher_score" if "matcher_score" in rows else "lightglue_score"
        )
        return cls(
            source_feature_id=rows["source_feature_id"].to_numpy(np.int64),
            source_tile_id=rows["source_tile_id"].to_numpy(np.int32),
            target_tile_id=rows["target_tile_id"].to_numpy(np.int32),
            source_xy_m=rows[["source_x", "source_y"]].to_numpy(float),
            target_xy_m=rows[["target_x", "target_y"]].to_numpy(float),
            score=rows[score_column].to_numpy(np.float32),
        )

    @property
    def displacement_m(self) -> np.ndarray:
        return self.target_xy_m - self.source_xy_m

    def __len__(self) -> int:
        return len(self.score)

    def select(self, indices: np.ndarray) -> "MotionMatches":
        return MotionMatches(
            source_feature_id=self.source_feature_id[indices],
            source_tile_id=self.source_tile_id[indices],
            target_tile_id=self.target_tile_id[indices],
            source_xy_m=self.source_xy_m[indices],
            target_xy_m=self.target_xy_m[indices],
            score=self.score[indices],
        )

    def to_frame(self) -> pd.DataFrame:
        displacement = self.displacement_m
        return pd.DataFrame(
            {
                "source_feature_id": self.source_feature_id,
                "source_tile_id": self.source_tile_id,
                "target_tile_id": self.target_tile_id,
                "source_x": self.source_xy_m[:, 0],
                "source_y": self.source_xy_m[:, 1],
                "target_x": self.target_xy_m[:, 0],
                "target_y": self.target_xy_m[:, 1],
                "dx_m": displacement[:, 0],
                "dy_m": displacement[:, 1],
                "matcher_score": self.score,
                "lightglue_score": self.score,
                "physics_valid": np.ones(len(self), dtype=bool),
            }
        )


@dataclass(frozen=True)
class DriftField:
    grid_row: np.ndarray
    grid_column: np.ndarray
    source_xy_m: np.ndarray
    displacement_m: np.ndarray
    available: np.ndarray
    selected_matches: np.ndarray
    candidate_matches: np.ndarray
    support_radius_m: np.ndarray
    maximum_residual_m: np.ndarray

    def __len__(self) -> int:
        return len(self.available)

    def with_available(self, available: np.ndarray) -> "DriftField":
        return DriftField(
            grid_row=self.grid_row,
            grid_column=self.grid_column,
            source_xy_m=self.source_xy_m,
            displacement_m=self.displacement_m,
            available=np.asarray(available, dtype=bool),
            selected_matches=self.selected_matches,
            candidate_matches=self.candidate_matches,
            support_radius_m=self.support_radius_m,
            maximum_residual_m=self.maximum_residual_m,
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "grid_row": self.grid_row,
                "grid_column": self.grid_column,
                "source_x": self.source_xy_m[:, 0],
                "source_y": self.source_xy_m[:, 1],
                "available": self.available,
                "selected_vectors": self.selected_matches,
                "candidate_count": self.candidate_matches,
                "support_radius_m": self.support_radius_m,
                "proposal_dx_m": self.displacement_m[:, 0],
                "proposal_dy_m": self.displacement_m[:, 1],
                "maximum_vector_residual_m": self.maximum_residual_m,
            }
        )


@dataclass(frozen=True)
class PairResult:
    matches: MotionMatches
    field: DriftField
    fold_rejected_indices: np.ndarray
    matching_seconds: float
    field_seconds: float
    prior_displacement_m: tuple[float, float] | None = None
