#!/usr/bin/env python3
"""Measure adjacent-slice consistency for real same-pass Sentinel-1 controls.

The selected slices meet at a seam but do not overlap. The analysis therefore
does not claim repeat-pixel agreement. It identifies the nearest geolocated edge
pair, aligns points along that seam, and compares cross-seam intensity change
with equal-offset changes inside each contributing image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from experiments.buoy_descriptor_benchmark import pixels_to_map, read_scene
except ModuleNotFoundError:  # Direct execution from the experiments directory.
    from buoy_descriptor_benchmark import pixels_to_map, read_scene  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results/iabp_s1_stratified_coverage"
EDGES = ("top", "bottom", "left", "right")


def edge_pixels(shape: tuple[int, int], edge: str, samples: int) -> np.ndarray:
    height, width = shape
    if edge in {"top", "bottom"}:
        cols = np.linspace(0, width - 1, samples)
        rows = np.full(samples, 0 if edge == "top" else height - 1)
    elif edge in {"left", "right"}:
        rows = np.linspace(0, height - 1, samples)
        cols = np.full(samples, 0 if edge == "left" else width - 1)
    else:
        raise ValueError(f"Unknown edge: {edge}")
    return np.column_stack((cols, rows)).astype(float)


def move_inward(
    pixels: np.ndarray,
    shape: tuple[int, int],
    edge: str,
    offset_pixels: int,
) -> np.ndarray:
    height, width = shape
    result = pixels.copy()
    if edge == "top":
        result[:, 1] = offset_pixels
    elif edge == "bottom":
        result[:, 1] = height - 1 - offset_pixels
    elif edge == "left":
        result[:, 0] = offset_pixels
    elif edge == "right":
        result[:, 0] = width - 1 - offset_pixels
    else:
        raise ValueError(f"Unknown edge: {edge}")
    return result


def values_at(
    image: np.ndarray,
    mask: np.ndarray | None,
    pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    cols = np.clip(np.rint(pixels[:, 0]).astype(int), 0, image.shape[1] - 1)
    rows = np.clip(np.rint(pixels[:, 1]).astype(int), 0, image.shape[0] - 1)
    valid = np.isfinite(pixels).all(axis=1)
    if mask is not None:
        valid &= mask[rows, cols] < 2
    return image[rows, cols].astype(float), valid


def pair_statistics(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid].astype(float)
    second = second[valid].astype(float)
    if len(first) == 0:
        return {
            "valid_pairs": 0,
            "median_bias_dn": np.nan,
            "mean_absolute_difference_dn": np.nan,
            "p90_absolute_difference_dn": np.nan,
            "correlation": np.nan,
        }
    difference = second - first
    correlation = (
        float(np.corrcoef(first, second)[0, 1])
        if len(first) >= 2 and first.std() > 0 and second.std() > 0
        else np.nan
    )
    return {
        "valid_pairs": len(first),
        "median_bias_dn": float(np.median(difference)),
        "mean_absolute_difference_dn": float(np.mean(np.abs(difference))),
        "p90_absolute_difference_dn": float(np.percentile(np.abs(difference), 90)),
        "correlation": correlation,
    }


def geolocated_edges(
    path: str,
    shape: tuple[int, int],
    samples: int,
    analysis_epsg: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for edge in EDGES:
        pixels = edge_pixels(shape, edge, samples)
        result[edge] = (pixels, pixels_to_map(path, pixels, analysis_epsg))
    return result


def select_seam_edges(
    first: dict[str, tuple[np.ndarray, np.ndarray]],
    second: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    choices = []
    for first_edge, (first_pixels, first_map) in first.items():
        first_valid = np.isfinite(first_map).all(axis=1)
        if not np.any(first_valid):
            continue
        for second_edge, (second_pixels, second_map) in second.items():
            second_valid = np.isfinite(second_map).all(axis=1)
            if not np.any(second_valid):
                continue
            second_indexes = np.flatnonzero(second_valid)
            tree = cKDTree(second_map[second_valid])
            distances, local_indexes = tree.query(first_map[first_valid], k=1)
            choices.append(
                (
                    float(np.median(distances)),
                    first_edge,
                    second_edge,
                    np.flatnonzero(first_valid),
                    second_indexes[np.asarray(local_indexes, dtype=int)],
                    np.asarray(distances, dtype=float),
                )
            )
    if not choices:
        raise ValueError("No finite geolocation is available on either image edge")
    _, first_edge, second_edge, first_indexes, second_indexes, distances = min(
        choices, key=lambda item: item[0]
    )
    return first_edge, second_edge, first_indexes, second_indexes, distances


def analyze_pair(row, offsets: tuple[int, ...], samples: int, analysis_epsg: int) -> list[dict]:
    first_image, first_mask = read_scene(row.first_standard_vae_path)
    second_image, second_mask = read_scene(row.second_standard_vae_path)
    first_edges = geolocated_edges(
        row.first_standard_vae_path, first_image.shape, samples, analysis_epsg
    )
    second_edges = geolocated_edges(
        row.second_standard_vae_path, second_image.shape, samples, analysis_epsg
    )
    first_edge, second_edge, first_indexes, second_indexes, distances = (
        select_seam_edges(first_edges, second_edges)
    )
    first_base = first_edges[first_edge][0][first_indexes]
    second_base = second_edges[second_edge][0][second_indexes]
    records = []
    for offset in offsets:
        first_cross_pixels = move_inward(
            first_base, first_image.shape, first_edge, offset
        )
        second_cross_pixels = move_inward(
            second_base, second_image.shape, second_edge, offset
        )
        first_cross, first_valid = values_at(
            first_image, first_mask, first_cross_pixels
        )
        second_cross, second_valid = values_at(
            second_image, second_mask, second_cross_pixels
        )
        cross_valid = first_valid & second_valid
        cross = pair_statistics(
            np.where(cross_valid, first_cross, np.nan),
            np.where(cross_valid, second_cross, np.nan),
        )

        inside_offset = 3 * offset
        first_inside, first_inside_valid = values_at(
            first_image,
            first_mask,
            move_inward(first_base, first_image.shape, first_edge, inside_offset),
        )
        second_inside, second_inside_valid = values_at(
            second_image,
            second_mask,
            move_inward(second_base, second_image.shape, second_edge, inside_offset),
        )
        first_within = pair_statistics(
            np.where(first_valid & first_inside_valid, first_cross, np.nan),
            np.where(first_valid & first_inside_valid, first_inside, np.nan),
        )
        second_within = pair_statistics(
            np.where(second_valid & second_inside_valid, second_cross, np.nan),
            np.where(second_valid & second_inside_valid, second_inside, np.nan),
        )
        within_mad = np.nanmean(
            [
                first_within["mean_absolute_difference_dn"],
                second_within["mean_absolute_difference_dn"],
            ]
        )
        records.append(
            {
                "same_pass_pair_id": row.same_pass_pair_id,
                "acquisition_pass_id": row.acquisition_pass_id,
                "time_separation_seconds": row.time_separation_seconds,
                "first_image_id": row.first_image_id,
                "second_image_id": row.second_image_id,
                "first_seam_edge": first_edge,
                "second_seam_edge": second_edge,
                "seam_samples": len(distances),
                "unique_second_edge_sample_fraction": len(np.unique(second_indexes))
                / len(second_indexes),
                "median_edge_separation_m": float(np.median(distances)),
                "p90_edge_separation_m": float(np.percentile(distances, 90)),
                "inward_offset_pixels_each_image": offset,
                "cross_seam_valid_pairs": cross["valid_pairs"],
                "cross_seam_median_bias_dn": cross["median_bias_dn"],
                "cross_seam_mean_absolute_difference_dn": cross[
                    "mean_absolute_difference_dn"
                ],
                "cross_seam_p90_absolute_difference_dn": cross[
                    "p90_absolute_difference_dn"
                ],
                "cross_seam_correlation": cross["correlation"],
                "first_within_scene_mean_absolute_difference_dn": first_within[
                    "mean_absolute_difference_dn"
                ],
                "second_within_scene_mean_absolute_difference_dn": second_within[
                    "mean_absolute_difference_dn"
                ],
                "cross_to_mean_within_scene_mad_ratio": (
                    float(cross["mean_absolute_difference_dn"] / within_mad)
                    if np.isfinite(within_mad) and within_mad > 0
                    else np.nan
                ),
                "interpretation": (
                    "adjacent_nonoverlapping_slices; ratio_above_one_can_include_real_ice_change"
                ),
            }
        )
    return records


def parse_offsets(value: str) -> tuple[int, ...]:
    offsets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not offsets or min(offsets) <= 0:
        raise ValueError("Offsets must be positive pixel counts")
    return offsets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--offsets", default="16,32,64")
    parser.add_argument("--edge-samples", type=int, default=512)
    parser.add_argument("--analysis-epsg", type=int, default=3413)
    args = parser.parse_args()
    pairs = pd.read_csv(args.results_dir / "full70_same_pass_scene_pairs.csv")
    records = []
    for row in pairs.itertuples(index=False):
        records.extend(
            analyze_pair(
                row,
                parse_offsets(args.offsets),
                args.edge_samples,
                args.analysis_epsg,
            )
        )
    results = pd.DataFrame.from_records(records)
    results.to_csv(
        args.results_dir / "full70_same_pass_seam_consistency.csv", index=False
    )
    payload = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "same_pass_pairs": int(results["same_pass_pair_id"].nunique()),
        "offsets_pixels_each_image": sorted(
            results["inward_offset_pixels_each_image"].unique().tolist()
        ),
        "median_cross_to_within_scene_mad_ratio": float(
            results["cross_to_mean_within_scene_mad_ratio"].median()
        ),
        "maximum_cross_to_within_scene_mad_ratio": float(
            results["cross_to_mean_within_scene_mad_ratio"].max()
        ),
        "control_limit": (
            "The slices do not overlap; this measures seam-scale distribution and "
            "texture continuity relative to within-scene change, not repeat-pixel error."
        ),
    }
    (args.results_dir / "full70_same_pass_seam_consistency_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
