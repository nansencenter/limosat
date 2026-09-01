"""LiMOSAT: EfficientLoFTR sea-ice fields and Lagrangian trajectories."""

from .catalog import ImageCatalogue, ImagePair, ImageRecord, load_catalogue
from .config import (
    FieldConfig,
    MatcherConfig,
    RoutingConfig,
    RunConfig,
    TrajectoryConfig,
    load_config,
)
from .models import DisplacementField, FieldEdge, MotionMatches, PairResult
from .deformation import DeformationCell, deformation_from_field
from .efficientloftr import EfficientLoFTR
from .pairs import PairProcessor
from .planning import PlannedPair, plan_candidate_pairs
from .replay import load_production_field_replay
from .trajectory import (
    TrajectoryPoint,
    build_trajectories,
    compose_global_trajectories,
    iter_global_trajectory_points,
)
from .run import LiMOSATRun
from .store import RunStore

__all__ = [
    "DisplacementField",
    "DeformationCell",
    "EfficientLoFTR",
    "FieldConfig",
    "FieldEdge",
    "ImageCatalogue",
    "ImagePair",
    "ImageRecord",
    "MatcherConfig",
    "LiMOSATRun",
    "MotionMatches",
    "PairResult",
    "PairProcessor",
    "PlannedPair",
    "RoutingConfig",
    "RunConfig",
    "RunStore",
    "TrajectoryConfig",
    "TrajectoryPoint",
    "build_trajectories",
    "compose_global_trajectories",
    "iter_global_trajectory_points",
    "deformation_from_field",
    "load_catalogue",
    "load_config",
    "load_production_field_replay",
    "plan_candidate_pairs",
]
