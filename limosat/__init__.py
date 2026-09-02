"""LiMOSAT: EfficientLoFTR sea-ice fields and Lagrangian trajectories."""

from .catalog import ImageCatalogue, ImagePair, ImageRecord, load_catalogue
from .config import (
    FieldConfig,
    MatcherConfig,
    OpenWaterConfig,
    RoutingConfig,
    RunConfig,
    TrajectoryConfig,
    load_config,
)
from .models import DisplacementField, FieldEdge, MotionMatches, PairResult
from .deformation import DeformationCell, deformation_from_field
from .efficientloftr import EfficientLoFTR
from .pairs import PairProcessor
from .pair_artifacts import PairProduct, PairProductStore
from .planning import (
    CandidatePlan,
    PlannedPair,
    build_candidate_plan,
    plan_candidate_pairs,
    select_overlap_probe,
)
from .replay import load_production_field_replay
from .trajectory import (
    ConvergenceEvent,
    TrajectoryPoint,
    audit_trajectory_convergence,
    build_trajectories,
    compose_global_trajectories,
    iter_global_trajectory_points,
)
from .run import LiMOSATRun
from .stages import RunStages
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
    "OpenWaterConfig",
    "LiMOSATRun",
    "MotionMatches",
    "PairResult",
    "PairProcessor",
    "PairProduct",
    "PairProductStore",
    "CandidatePlan",
    "ConvergenceEvent",
    "PlannedPair",
    "RoutingConfig",
    "RunConfig",
    "RunStages",
    "RunStore",
    "TrajectoryConfig",
    "TrajectoryPoint",
    "build_trajectories",
    "build_candidate_plan",
    "audit_trajectory_convergence",
    "compose_global_trajectories",
    "iter_global_trajectory_points",
    "deformation_from_field",
    "load_catalogue",
    "load_config",
    "load_production_field_replay",
    "plan_candidate_pairs",
    "select_overlap_probe",
]
