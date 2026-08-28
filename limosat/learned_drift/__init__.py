"""Learned-matcher sea-ice drift components."""

from .config import ALIKEDConfig, EfficientLoFTRConfig
from .efficientloftr import (
    source_core_mask,
    speed_limit_mask,
    valid_endpoints,
    valid_support,
)
from .features import extract_image_features, restrict_features, tile_layout
from .field import estimate_field, reject_folds, regular_grid, topology_summary
from .matching import DirectALIKEDLightGlue, match_features
from .pipeline import ALIKEDDrift
from .routing import CoarseTranslation, coarse_phase_translation, preceding_field_shifts
from .store import ImagePair, LearnedDriftStore
from .trajectory import FieldSamples, advect_trajectories, sample_field
from .trajectory_graph import FieldEdge, advect_trajectory_graph
from .types import DriftField, FeatureTile, ImageFeatures, MotionMatches, PairResult

__all__ = [
    "ALIKEDConfig",
    "ALIKEDDrift",
    "CoarseTranslation",
    "DirectALIKEDLightGlue",
    "DriftField",
    "EfficientLoFTRConfig",
    "FeatureTile",
    "FieldSamples",
    "FieldEdge",
    "ImageFeatures",
    "ImagePair",
    "LearnedDriftStore",
    "MotionMatches",
    "PairResult",
    "advect_trajectories",
    "advect_trajectory_graph",
    "coarse_phase_translation",
    "estimate_field",
    "extract_image_features",
    "match_features",
    "regular_grid",
    "preceding_field_shifts",
    "reject_folds",
    "sample_field",
    "source_core_mask",
    "speed_limit_mask",
    "restrict_features",
    "tile_layout",
    "topology_summary",
    "valid_endpoints",
    "valid_support",
]
