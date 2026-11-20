"""LiMOSAT: EfficientLoFTR sea-ice fields and Lagrangian trajectories."""

from .catalog import ImageCatalogue, ImagePair, ImageRecord, load_catalogue
from .config import (
    FieldConfig,
    MatcherConfig,
    OpenWaterConfig,
    RoutingConfig,
    RunConfig,
    SurfaceConfig,
    TrajectoryConfig,
    load_config,
)
from .models import DisplacementField, FieldEdge, MotionMatches, PairResult
from .efficientloftr import EfficientLoFTR
from .pairs import PairProcessor
from .pair_products import PairProduct, PairProductStore
from .planning import (
    CandidatePlan,
    PlannedPair,
    build_candidate_plan,
    plan_candidate_pairs,
    select_overlap_probe,
)
from .trajectory import TrajectoryPoint, iter_global_trajectory_batches
from .stages import RunStages
from .store import RunStore

__all__ = [
    "CandidatePlan",
    "DisplacementField",
    "EfficientLoFTR",
    "FieldConfig",
    "FieldEdge",
    "ImageCatalogue",
    "ImagePair",
    "ImageRecord",
    "MatcherConfig",
    "MotionMatches",
    "OpenWaterConfig",
    "PairProcessor",
    "PairProduct",
    "PairProductStore",
    "PairResult",
    "PlannedPair",
    "RoutingConfig",
    "RunConfig",
    "RunStages",
    "RunStore",
    "SurfaceConfig",
    "TrajectoryConfig",
    "TrajectoryPoint",
    "build_candidate_plan",
    "iter_global_trajectory_batches",
    "load_catalogue",
    "load_config",
    "plan_candidate_pairs",
    "select_overlap_probe",
]
