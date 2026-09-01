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
from .trajectory import (
    TrajectoryPoint,
    build_trajectories,
    compose_global_trajectories,
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
    "RoutingConfig",
    "RunConfig",
    "RunStore",
    "TrajectoryConfig",
    "TrajectoryPoint",
    "build_trajectories",
    "compose_global_trajectories",
    "deformation_from_field",
    "load_catalogue",
    "load_config",
]
