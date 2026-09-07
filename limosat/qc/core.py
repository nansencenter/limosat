"""Trajectory-vector QC: frozen configuration and spatial scoring in metres/days."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError, cKDTree


PROTOCOL_ID = "limosat_trajectory_link_qc_v2_20260907"
PROTOCOL_PATH = Path(__file__).resolve().parent / "protocols" / "trajectory_link_qc_v2.json"
FROZEN_CONFIG_FIELDS = (
    "search_radius_m",
    "max_neighbors",
    "min_neighbors",
    "min_inlier_fraction",
    "fit_clip_z",
    "noise_floor_m",
    "strong_local_z",
    "strong_local_floor_m",
    "ultra_local_floor_m",
    "moderate_local_z",
    "moderate_local_floor_m",
    "hard_speed_m_per_day",
    "topology_max_edge_m",
    "max_prediction_weight_l1",
)


@dataclass(frozen=True)
class QCConfig:
    """Frozen thresholds for conservative, vector QC without external reference data (metres/days)."""

    search_radius_m: float = 30_000.0
    max_neighbors: int = 24
    min_neighbors: int = 8
    min_inlier_fraction: float = 0.60
    fit_clip_z: float = 4.0
    noise_floor_m: float = 150.0
    strong_local_z: float = 8.0
    strong_local_floor_m: float = 10_000.0
    ultra_local_floor_m: float = 20_000.0
    moderate_local_z: float = 6.0
    moderate_local_floor_m: float = 5_000.0
    configured_speed_m_per_day: float = 35_000.0
    hard_speed_m_per_day: float = 60_000.0
    topology_max_edge_m: float = 20_000.0  # Source-triangle side, not drift length.

    max_prediction_weight_l1: float = 2.0  # Bound amplification at the query.

    def validate(self) -> None:
        for field, value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if self.max_prediction_weight_l1 < 2:
            raise ValueError("max_prediction_weight_l1 must be >= 2")
        if self.max_neighbors < self.min_neighbors:
            raise ValueError("max_neighbors must be >= min_neighbors")
        if self.min_neighbors < 3:
            raise ValueError("min_neighbors must be at least 3")
        if not 0 < self.min_inlier_fraction <= 1:
            raise ValueError("min_inlier_fraction must be in (0, 1]")
        if self.hard_speed_m_per_day < self.configured_speed_m_per_day:
            raise ValueError("hard speed must be >= configured speed")
        if self.ultra_local_floor_m < self.strong_local_floor_m:
            raise ValueError("ultra local floor must be >= strong local floor")


def load_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Bundled trajectory-QC protocol ID does not match the code")
    return protocol


def validate_frozen_protocol(config: QCConfig) -> None:
    """Fail if code defaults and the bundled protocol have drifted apart."""

    parameters = load_protocol()["parameters"]
    mismatches = {
        field: (getattr(config, field), parameters.get(field))
        for field in FROZEN_CONFIG_FIELDS
        if getattr(config, field) != parameters.get(field)
    }
    if mismatches:
        raise ValueError(f"QC configuration does not match frozen protocol: {mismatches}")


def robust_scale(values: np.ndarray, floor: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return math.nan, math.nan
    centre = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - centre)))
    return centre, max(scale, floor)


def _weighted_affine(
    query_xy: np.ndarray,
    neighbor_xy: np.ndarray,
    neighbor_uv: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
    max_prediction_weight_l1: float,
) -> tuple[np.ndarray, np.ndarray]:
    centred_km = (neighbor_xy - query_xy) / 1_000.0
    design = np.column_stack([np.ones(len(neighbor_xy)), centred_km])
    spatial_scale = max(float(np.median(distances[mask])), 1.0)
    weights = 1.0 / (1.0 + (distances / spatial_scale) ** 2)
    weighted_design = design[mask] * np.sqrt(weights[mask, None])
    weighted_values = neighbor_uv[mask] * np.sqrt(weights[mask, None])
    # Reuse one SVD for both the fit and its prediction weights. A full-rank
    # fit can still extrapolate wildly from a narrow or one-sided neighbour set.
    left, singular, right = np.linalg.svd(weighted_design, full_matrices=False)
    tolerance = np.finfo(float).eps * max(weighted_design.shape) * singular[0]
    if singular[-1] > tolerance:
        prediction_weights = ((right[:, 0] / singular) @ left.T) * np.sqrt(weights[mask])
        if np.abs(prediction_weights).sum() <= max_prediction_weight_l1:
            coefficients = right.T @ ((left.T @ weighted_values) / singular[:, None])
            return prediction_weights @ neighbor_uv[mask], design @ coefficients
    prediction = np.median(neighbor_uv[mask], axis=0)
    return prediction, np.repeat(prediction[None, :], len(neighbor_uv), axis=0)


def _local_prediction(
    query_xy: np.ndarray,
    neighbor_xy: np.ndarray,
    neighbor_uv: np.ndarray,
    distances: np.ndarray,
    config: QCConfig,
) -> dict[str, float | int | bool]:
    median_uv = np.median(neighbor_uv, axis=0)
    median_residual = np.linalg.norm(neighbor_uv - median_uv, axis=1)
    centre, scale = robust_scale(median_residual, config.noise_floor_m)
    mask = median_residual <= centre + config.fit_clip_z * scale
    if int(mask.sum()) < config.min_neighbors:
        nearest = np.argsort(median_residual)[: config.min_neighbors]
        mask = np.zeros(len(neighbor_uv), dtype=bool)
        mask[nearest] = True

    for _ in range(3):
        prediction, fitted = _weighted_affine(
            query_xy, neighbor_xy, neighbor_uv, distances, mask,
            config.max_prediction_weight_l1,
        )
        fit_residual = np.linalg.norm(neighbor_uv - fitted, axis=1)
        centre, scale = robust_scale(fit_residual[mask], config.noise_floor_m)
        new_mask = fit_residual <= centre + config.fit_clip_z * scale
        if int(new_mask.sum()) < config.min_neighbors:
            break
        if np.array_equal(mask, new_mask):
            mask = new_mask
            break
        mask = new_mask

    else:
        # The last iteration changed the mask. Refit only in this case so
        # prediction and residual scale describe the same final inlier set.
        prediction, fitted = _weighted_affine(
            query_xy, neighbor_xy, neighbor_uv, distances, mask,
            config.max_prediction_weight_l1,
        )
    fit_residual = np.linalg.norm(neighbor_uv - fitted, axis=1)
    centre, scale = robust_scale(fit_residual[mask], config.noise_floor_m)
    inlier_fraction = float(mask.mean())
    supported = bool(
        int(mask.sum()) >= config.min_neighbors
        and inlier_fraction >= config.min_inlier_fraction
    )
    return {
        "local_u_m": float(prediction[0]),
        "local_v_m": float(prediction[1]),
        "local_neighbor_count": int(len(neighbor_uv)),
        "local_inlier_count": int(mask.sum()),
        "local_inlier_fraction": inlier_fraction,
        "local_support_radius_m": float(np.max(distances)),
        "local_typical_residual_m": centre,
        "local_residual_scale_m": scale,
        "local_supported": supported,
    }


def score_vectors(
    vectors: pd.DataFrame,
    config: QCConfig,
    *,
    prescreen_residual_m: float | None = None,
) -> pd.DataFrame:
    """Score drift vectors between consecutive keypoints in each trajectory.

    A vector is the displacement between stored observations, whether directly
    matched or interpolated. Neighbours come from the same source/target image
    pair, with the vector being scored excluded from its own local fit.

    ``prescreen_residual_m`` skips fits only when an upper bound on the
    residual is below both the prescreen and every local decision floor.
    Reject/review decisions therefore agree with unaccelerated scoring.
    """

    config.validate()
    if prescreen_residual_m is not None and (
        not math.isfinite(prescreen_residual_m) or prescreen_residual_m <= 0
    ):
        raise ValueError("prescreen_residual_m must be finite and positive")
    result = vectors.copy().reset_index(drop=True)
    float_defaults = {
        "local_u_m": np.nan,
        "local_v_m": np.nan,
        "local_inlier_fraction": np.nan,
        "local_support_radius_m": np.nan,
        "local_typical_residual_m": np.nan,
        "local_residual_scale_m": np.nan,
    }
    int_defaults = {"local_neighbor_count": 0, "local_inlier_count": 0}
    for column, value in {**float_defaults, **int_defaults}.items():
        result[column] = value
    result["local_supported"] = False
    result["local_evaluated"] = False
    result["coarse_local_residual_m"] = np.nan
    result = annotate_topology_incidence(result, config.topology_max_edge_m)
    diagnostic_columns = [
        *float_defaults, *int_defaults, "local_supported", "local_evaluated",
        "coarse_local_residual_m",
    ]
    diagnostics_buffer = {
        column: result[column].to_numpy(copy=True) for column in diagnostic_columns
    }

    pair_columns = ["source_image_id", "target_image_id"]
    for _, pair_index in result.groupby(pair_columns, sort=True).groups.items():
        pair_index = np.asarray(pair_index, dtype=int)
        if len(pair_index) <= config.min_neighbors:
            continue
        xy = result.loc[pair_index, ["x0_m", "y0_m"]].to_numpy(float)
        uv = result.loc[pair_index, ["u_m", "v_m"]].to_numpy(float)
        tree = cKDTree(xy)
        k = min(config.max_neighbors + 1, len(pair_index))
        distances_all, neighbors_all = tree.query(
            xy, k=k, distance_upper_bound=config.search_radius_m
        )
        valid_all = (
            np.isfinite(distances_all)
            & (neighbors_all < len(pair_index))
            & (neighbors_all != np.arange(len(pair_index))[:, None])
        )
        safe_neighbors = np.where(valid_all, neighbors_all, 0).astype(int)
        neighbor_uv = uv[safe_neighbors].copy()
        neighbor_uv[~valid_all] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            coarse_prediction = np.nanmedian(neighbor_uv, axis=1)
        coarse_residual = np.linalg.norm(uv - coarse_prediction, axis=1)
        coarse_residual[valid_all.sum(axis=1) < config.min_neighbors] = np.nan
        diagnostics_buffer["coarse_local_residual_m"][pair_index] = coarse_residual
        if prescreen_residual_m is None:
            evaluate = np.ones(len(pair_index), dtype=bool)
        else:
            # For any retained affine fit, ||prediction - median|| <= L * R,
            # where L bounds the absolute prediction weights and R is the
            # largest neighbour deviation. This also covers any clipped subset
            # and the coordinate-median fallback (bounded by sqrt(2) * R).
            deviations = np.linalg.norm(neighbor_uv - coarse_prediction[:, None, :], axis=2)
            radius = np.max(np.where(valid_all, deviations, 0.0), axis=1)
            residual_bound = coarse_residual + config.max_prediction_weight_l1 * radius
            safe_floor = min(
                prescreen_residual_m, config.moderate_local_floor_m,
                config.strong_local_floor_m, config.ultra_local_floor_m,
            )
            # A small margin avoids skipping borderline cases through roundoff.
            evaluate = residual_bound >= safe_floor * (1.0 - 1e-10)
        for local_index in np.flatnonzero(evaluate):
            output_index = pair_index[local_index]
            valid = valid_all[local_index]
            neighbors = neighbors_all[local_index, valid].astype(int)
            distances = distances_all[local_index, valid]
            if len(neighbors) < config.min_neighbors:
                continue
            diagnostics_buffer["local_evaluated"][output_index] = True
            diagnostics = _local_prediction(
                xy[local_index], xy[neighbors], uv[neighbors], distances, config
            )
            for column, value in diagnostics.items():
                diagnostics_buffer[column][output_index] = value

    for column, values in diagnostics_buffer.items():
        result[column] = values

    result["local_residual_m"] = np.hypot(
        result["u_m"] - result["local_u_m"],
        result["v_m"] - result["local_v_m"],
    )
    result["strong_local_threshold_m"] = np.maximum(
        config.strong_local_floor_m,
        result["local_typical_residual_m"]
        + config.strong_local_z * result["local_residual_scale_m"],
    )
    result["moderate_local_threshold_m"] = np.maximum(
        config.moderate_local_floor_m,
        result["local_typical_residual_m"]
        + config.moderate_local_z * result["local_residual_scale_m"],
    )
    result["hard_speed_reject"] = (
        result["speed_m_per_day"] > config.hard_speed_m_per_day
    )
    result["strong_local_candidate"] = (
        result["local_supported"]
        & (result["local_residual_m"] > result["strong_local_threshold_m"])
    )
    result["ultra_local_reject"] = (
        result["local_supported"]
        & (
            result["local_residual_m"]
            > np.maximum(
                config.ultra_local_floor_m,
                result["local_typical_residual_m"]
                + config.strong_local_z * result["local_residual_scale_m"],
            )
        )
    )
    result["topology_local_reject"] = (
        result["strong_local_candidate"] & result["topology_flip_incident"]
    )
    result["speed_local_reject"] = (
        result["local_supported"]
        & (result["speed_m_per_day"] > config.configured_speed_m_per_day)
        & (result["local_residual_m"] > result["moderate_local_threshold_m"])
    )
    result["reject"] = result[
        [
            "hard_speed_reject",
            "ultra_local_reject",
            "topology_local_reject",
            "speed_local_reject",
        ]
    ].any(axis=1)
    result["review"] = result["strong_local_candidate"] & ~result["reject"]

    reason = pd.Series("", index=result.index, dtype="string")
    for column, label in (
        ("hard_speed_reject", "hard_speed"),
        ("ultra_local_reject", "ultra_local"),
        ("topology_local_reject", "topology_local"),
        ("speed_local_reject", "speed_local"),
    ):
        reason += np.where(result[column], label + "+", "")
    result["decision_reason"] = reason.str.rstrip("+").replace("", "accept")
    return result


def _orientation_triangles(
    source_xy: np.ndarray, target_xy: np.ndarray, maximum_edge_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return short Delaunay triangles as input-row indices and their flip flags."""

    if len(source_xy) < 3:
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=bool)
    unique, unique_index = np.unique(source_xy, axis=0, return_index=True)
    if len(unique) < 3:
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=bool)
    target_unique = target_xy[unique_index]
    try:
        triangles = Delaunay(unique).simplices
    except QhullError:
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=bool)
    vertices = unique[triangles]
    edge_lengths = np.stack(
        [
            np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
            np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1),
            np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1),
        ],
        axis=1,
    )
    triangles = triangles[np.max(edge_lengths, axis=1) <= maximum_edge_m]
    if not len(triangles):
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=bool)

    def signed_area(xy: np.ndarray) -> np.ndarray:
        tri = xy[triangles]
        return (
            (tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
            - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1])
        )

    source_area = signed_area(unique)
    target_area = signed_area(target_unique)
    valid = (source_area != 0) & (target_area != 0)
    input_triangles = unique_index[triangles[valid]]
    flipped = np.sign(source_area[valid]) != np.sign(target_area[valid])
    return input_triangles, flipped


def annotate_topology_incidence(
    vectors: pd.DataFrame, maximum_edge_m: float
) -> pd.DataFrame:
    """Count short source-mesh orientation flips incident on each drift vector."""

    result = vectors.copy().reset_index(drop=True)
    result["topology_flip_count"] = 0
    pair_columns = ["source_image_id", "target_image_id"]
    for _, pair_index in result.groupby(pair_columns, sort=True).groups.items():
        pair_index = np.asarray(pair_index, dtype=int)
        pair = result.loc[pair_index]
        triangles, flipped = _orientation_triangles(
            pair[["x0_m", "y0_m"]].to_numpy(float),
            pair[["x1_m", "y1_m"]].to_numpy(float),
            maximum_edge_m,
        )
        local_counts = np.zeros(len(pair_index), dtype=int)
        if flipped.any():
            np.add.at(local_counts, triangles[flipped].ravel(), 1)
        result.loc[pair_index, "topology_flip_count"] = local_counts
    result["topology_flip_incident"] = result["topology_flip_count"] > 0
    return result


def score_edges(
    edges: pd.DataFrame,
    config: QCConfig,
    *,
    prescreen_residual_m: float | None = None,
) -> pd.DataFrame:
    """Compatibility entry point; use score_vectors for new callers."""
    return score_vectors(edges, config, prescreen_residual_m=prescreen_residual_m)
