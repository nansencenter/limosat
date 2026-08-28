"""Learned-matcher sea-ice drift components."""

from importlib import import_module

_PUBLIC_MODULES = {
    "ALIKEDConfig": ".config",
    "EfficientLoFTRConfig": ".config",
    "source_core_mask": ".efficientloftr",
    "speed_limit_mask": ".efficientloftr",
    "valid_endpoints": ".efficientloftr",
    "valid_support": ".efficientloftr",
    "extract_image_features": ".features",
    "restrict_features": ".features",
    "tile_layout": ".features",
    "estimate_field": ".field",
    "reject_folds": ".field",
    "regular_grid": ".field",
    "topology_summary": ".field",
    "DirectALIKEDLightGlue": ".matching",
    "match_features": ".matching",
    "ALIKEDDrift": ".pipeline",
    "CoarseTranslation": ".routing",
    "coarse_phase_translation": ".routing",
    "preceding_field_shifts": ".routing",
    "ImagePair": ".store",
    "LearnedDriftStore": ".store",
    "FieldSamples": ".trajectory",
    "advect_trajectories": ".trajectory",
    "sample_field": ".trajectory",
    "FieldEdge": ".trajectory_graph",
    "advect_trajectory_graph": ".trajectory_graph",
    "DriftField": ".types",
    "FeatureTile": ".types",
    "ImageFeatures": ".types",
    "MotionMatches": ".types",
    "PairResult": ".types",
}

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


def __getattr__(name: str):
    """Keep matcher-specific dependencies out of unrelated learned imports."""
    module_name = _PUBLIC_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
