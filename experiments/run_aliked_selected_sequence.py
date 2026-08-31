#!/usr/bin/env python3
"""Run the historical ALIKED sequence experiment and its detailed audits.

The selected computation is maintained by ``ALIKEDDrift.track_sequence``;
this runner remains for frozen manifests, buoy evaluation, and call auditing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import torch
from kornia.feature import ALIKED

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.refine_aliked_dense_topology import (
    reject_flipped_nodes_until_stable,
)
from experiments.aliked_matchers import build_aliked_matcher
from experiments.run_aliked_dense_pair import (
    extract_tiles,
    match_tiles,
    nearest_consensus_at_queries,
    projected_footprint,
    regular_queries,
    spatially_thin_tiles_for_matching,
    tile_layout,
    topology_summary,
)


def pair_domains(manifest: dict, maximum_speed_m_per_day: float):
    elapsed_days = float(manifest["elapsed_hours"]) / 24.0
    maximum_displacement_m = elapsed_days * maximum_speed_m_per_day
    source_footprint = projected_footprint(manifest["source_image_filepath"])
    target_footprint = projected_footprint(manifest["target_image_filepath"])
    source_domain = source_footprint.intersection(
        target_footprint.buffer(maximum_displacement_m)
    )
    target_domain = target_footprint.intersection(
        source_domain.buffer(maximum_displacement_m)
    )
    if source_domain.is_empty or target_domain.is_empty:
        raise ValueError("source and target have no physics-reachable overlap")
    return source_domain, target_domain, elapsed_days, maximum_displacement_m


def restrict_tiles_to_domain(tiles: list[dict], domain) -> list[dict]:
    """Return pair-specific feature views from one union-domain image load."""
    restricted = []
    for tile in tiles:
        if not domain.intersects(tile["core"]):
            continue
        if len(tile["xy"]):
            keep = np.flatnonzero(
                shapely.intersects_xy(domain, tile["xy"][:, 0], tile["xy"][:, 1])
            )
        else:
            keep = np.empty(0, dtype=int)
        restricted.append(
            {
                **{
                    key: value
                    for key, value in tile.items()
                    if key not in {"keypoints", "descriptors", "scores", "xy"}
                },
                "keypoints": tile["keypoints"][keep],
                "descriptors": tile["descriptors"][keep],
                "scores": tile["scores"][keep],
                "xy": tile["xy"][keep],
            }
        )
    return restricted


def common_parameter(manifests: list[dict], name: str):
    values = [manifest["parameters"].get(name) for manifest in manifests]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"pair manifests disagree on {name}: {values}")
    return values[0]


def validate_chain(manifests: list[dict]) -> None:
    for previous, current in zip(manifests, manifests[1:]):
        if previous["target_image_filepath"] != current["source_image_filepath"]:
            raise ValueError("pair manifests do not form a contiguous image chain")


def summarize_buoys(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            "expected": 0,
            "available": 0,
            "correct_within_2km": 0,
            "median_error_m": None,
            "p90_error_m": None,
            "maximum_error_m": None,
        }
    available = rows["available"].fillna(False)
    errors = rows.loc[available, "error_m"].dropna()
    return {
        "expected": int(len(rows)),
        "available": int(available.sum()),
        "correct_within_2km": int((available & rows["error_m"].le(2000.0)).sum()),
        "median_error_m": float(errors.median()) if len(errors) else None,
        "p90_error_m": float(errors.quantile(0.90)) if len(errors) else None,
        "maximum_error_m": float(errors.max()) if len(errors) else None,
    }


def sequential_matching_prior(
    previous_field: pd.DataFrame | None,
    previous_elapsed_days: float | None,
    current_elapsed_days: float,
    chain_contiguous: bool,
    minimum_available_nodes: int = 8,
) -> tuple[tuple[float, float] | None, dict]:
    """Scale only the preceding accepted field's median velocity to a new gap."""
    audit = {
        "source": "preceding_fold_free_aliked_field",
        "chain_contiguous": bool(chain_contiguous),
        "minimum_available_nodes": int(minimum_available_nodes),
        "fallback": True,
    }
    if previous_field is None or previous_elapsed_days is None:
        audit["reason"] = "prior_absent"
        return None, audit
    if not chain_contiguous:
        audit["reason"] = "prior_stale_noncontiguous_chain"
        return None, audit
    available = previous_field["available"].fillna(False).to_numpy(bool)
    accepted = previous_field.loc[available]
    audit["available_nodes"] = int(len(accepted))
    if len(accepted) < minimum_available_nodes or previous_elapsed_days <= 0:
        audit["reason"] = "prior_inconsistent_insufficient_accepted_field"
        return None, audit
    displacement = accepted[["proposal_dx_m", "proposal_dy_m"]].to_numpy(
        dtype=float
    )
    finite = np.isfinite(displacement).all(axis=1)
    audit["finite_nodes"] = int(finite.sum())
    if finite.sum() < minimum_available_nodes:
        audit["reason"] = "prior_inconsistent_nonfinite_field"
        return None, audit
    velocity = np.median(displacement[finite], axis=0) / previous_elapsed_days
    prior = velocity * current_elapsed_days
    if not np.isfinite(prior).all():
        audit["reason"] = "prior_inconsistent_nonfinite_velocity"
        return None, audit
    audit.update(
        {
            "fallback": False,
            "reason": "preceding_field_velocity_accepted",
            "velocity_dx_m_per_day": float(velocity[0]),
            "velocity_dy_m_per_day": float(velocity[1]),
            "scaled_dx_m": float(prior[0]),
            "scaled_dy_m": float(prior[1]),
        }
    )
    return (float(prior[0]), float(prior[1])), audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path)
    parser.add_argument("--features-per-tile", type=int)
    parser.add_argument("--matching-feature-cap-per-tile", type=int)
    parser.add_argument("--matching-cells-per-axis", type=int, default=4)
    parser.add_argument("--matching-prior-dx-m", type=float)
    parser.add_argument("--matching-prior-dy-m", type=float)
    parser.add_argument("--matching-prior-uncertainty-m", type=float)
    parser.add_argument(
        "--sequential-prior",
        action="store_true",
        help=(
            "Generate each prior only from the immediately preceding accepted "
            "fold-free ALIKED field; the first, stale, or inconsistent case "
            "falls back to the full physics window."
        ),
    )
    parser.add_argument(
        "--sequential-prior-uncertainty-m", type=float, default=15_000.0
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--matcher", choices=("lightglue", "mnn", "smnn"), default="lightglue"
    )
    parser.add_argument("--smnn-ratio", type=float, default=0.95)
    parser.add_argument("--lightglue-layers", type=int, default=9)
    parser.add_argument("--lightglue-depth-confidence", type=float, default=0.95)
    parser.add_argument("--lightglue-width-confidence", type=float, default=0.99)
    parser.add_argument("--lightglue-filter-threshold", type=float, default=0.10)
    parser.add_argument(
        "--lightglue-adapter", choices=("kornia", "direct"), default="kornia"
    )
    parser.add_argument("--lightglue-compile", action="store_true")
    parser.add_argument("--matcher-call-audit", action="store_true")
    parser.add_argument("--audit-mnn-candidates", action="store_true")
    parser.add_argument("--mnn-candidate-limit", type=int)
    parser.add_argument(
        "--lightglue-target-batch-size",
        type=int,
        default=1,
        help="Reachable target tiles matched per direct LightGlue invocation.",
    )
    parser.add_argument(
        "--reuse-device-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Prepare descriptors and local affine frames once per pair. "
            "Defaults on for CUDA/MPS and off for CPU."
        ),
    )
    parser.add_argument("--maximum-radius-m", type=float, default=6000.0)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--minimum-selected-vectors", type=int, default=8)
    parser.add_argument("--consensus-radius-m", type=float, default=1000.0)
    args = parser.parse_args()
    prior_values = (
        args.matching_prior_dx_m,
        args.matching_prior_dy_m,
        args.matching_prior_uncertainty_m,
    )
    if any(value is not None for value in prior_values) and not all(
        value is not None for value in prior_values
    ):
        parser.error("matching prior dx, dy, and uncertainty must be supplied together")
    matching_prior = (
        (args.matching_prior_dx_m, args.matching_prior_dy_m)
        if args.matching_prior_dx_m is not None
        else None
    )
    if len(args.pair_run_dir) < 1:
        parser.error("at least one --pair-run-dir is required")
    if args.sequential_prior and matching_prior is not None:
        parser.error("fixed and sequential matching priors are mutually exclusive")

    sequence_started = time.perf_counter()
    manifests = [
        json.loads((run_dir / "run_manifest.json").read_text())
        for run_dir in args.pair_run_dir
    ]
    validate_chain(manifests)
    parameters = manifests[0]["parameters"]
    pixel_size_m = float(common_parameter(manifests, "pixel_size_m"))
    tile_pixels = int(common_parameter(manifests, "tile_pixels"))
    tile_margin_pixels = int(common_parameter(manifests, "tile_margin_pixels"))
    tile_grid_origin_m = float(common_parameter(manifests, "tile_grid_origin_m"))
    support_radius_pixels = int(common_parameter(manifests, "support_radius_pixels"))
    grid_spacing_m = float(common_parameter(manifests, "grid_spacing_m"))
    maximum_speed_m_per_day = float(
        common_parameter(manifests, "maximum_speed_m_per_day")
    )
    features_per_tile = (
        args.features_per_tile
        if args.features_per_tile is not None
        else int(common_parameter(manifests, "features_per_tile"))
    )
    feature_cache_dir = args.feature_cache_dir or Path(
        common_parameter(manifests, "feature_cache_dir")
    )
    model_cache = Path(common_parameter(manifests, "model_cache"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    reuse_device_features = (
        device.type != "cpu"
        if args.reuse_device_features is None
        else args.reuse_device_features
    )

    pair_contexts = []
    image_domains: dict[str, list] = defaultdict(list)
    image_ids: dict[str, int] = {}
    for run_dir, manifest in zip(args.pair_run_dir, manifests, strict=True):
        source_domain, target_domain, elapsed_days, maximum_displacement_m = (
            pair_domains(manifest, maximum_speed_m_per_day)
        )
        context = {
            "reference_run_dir": run_dir,
            "manifest": manifest,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "elapsed_days": elapsed_days,
            "maximum_displacement_m": maximum_displacement_m,
        }
        pair_contexts.append(context)
        source_path = manifest["source_image_filepath"]
        target_path = manifest["target_image_filepath"]
        image_domains[source_path].append(source_domain)
        image_domains[target_path].append(target_domain)
        image_ids[source_path] = int(manifest["source_image_id"])
        image_ids[target_path] = int(manifest["target_image_id"])

    torch.manual_seed(20260817)
    torch.hub.set_dir(str(model_cache / "hub"))
    model_started = time.perf_counter()
    model = ALIKED.from_pretrained(
        model_name="aliked-n16",
        max_num_keypoints=features_per_tile,
        detection_threshold=0.2,
        device=device,
    ).eval()
    matcher = build_aliked_matcher(
        args.matcher,
        device,
        args.smnn_ratio,
        args.lightglue_layers,
        args.lightglue_depth_confidence,
        args.lightglue_width_confidence,
        args.lightglue_filter_threshold,
        args.lightglue_adapter,
        args.lightglue_compile,
    )
    model_seconds = time.perf_counter() - model_started

    image_tiles: dict[str, list[dict]] = {}
    image_audits = []
    image_timings = []
    ordered_images = [manifests[0]["source_image_filepath"]] + [
        manifest["target_image_filepath"] for manifest in manifests
    ]
    for image_path in ordered_images:
        union_domain = shapely.union_all(image_domains[image_path])
        layout = tile_layout(
            union_domain,
            tile_pixels,
            tile_margin_pixels,
            pixel_size_m,
            tile_grid_origin_m,
        )
        started = time.perf_counter()
        tiles, audit = extract_tiles(
            image_path,
            union_domain,
            layout,
            model,
            device,
            tile_pixels,
            tile_margin_pixels,
            pixel_size_m,
            features_per_tile,
            support_radius_pixels,
            feature_cache_dir,
        )
        elapsed = time.perf_counter() - started
        image_tiles[image_path] = tiles
        image_timings.append(
            {
                "image_id": image_ids[image_path],
                "image_path": image_path,
                "tiles": len(tiles),
                "features": int(sum(len(tile["keypoints"]) for tile in tiles)),
                "cache_hits": int(sum(bool(row["cache_hit"]) for row in audit)),
                "image_preparation_seconds": float(
                    sum(row["image_preparation_seconds"] for row in audit)
                ),
                "detection_description_seconds": float(
                    sum(row["detection_description_seconds"] for row in audit)
                ),
                "cache_read_seconds": float(
                    sum(row["cache_read_seconds"] for row in audit)
                ),
                "cache_write_seconds": float(
                    sum(row["cache_write_seconds"] for row in audit)
                ),
                "seconds": elapsed,
            }
        )
        image_audits.extend(
            {"image_id": image_ids[image_path], "image_path": image_path, **row}
            for row in audit
        )
        print(
            f"image {image_ids[image_path]}: {len(tiles)} tiles, "
            f"{image_timings[-1]['features']} features, {elapsed:.2f} seconds",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(image_timings).to_csv(
        args.output_dir / "image_timings.csv", index=False
    )
    pd.DataFrame(image_audits).to_csv(
        args.output_dir / "image_tiles.csv", index=False
    )

    pair_summaries = []
    prior_audits = []
    previous_field = None
    previous_elapsed_days = None
    previous_target_path = None
    for context in pair_contexts:
        manifest = context["manifest"]
        source_id = int(manifest["source_image_id"])
        target_id = int(manifest["target_image_id"])
        pair_dir = args.output_dir / f"pair_{source_id}_{target_id}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        source_tiles = restrict_tiles_to_domain(
            image_tiles[manifest["source_image_filepath"]], context["source_domain"]
        )
        target_tiles = restrict_tiles_to_domain(
            image_tiles[manifest["target_image_filepath"]], context["target_domain"]
        )
        matching_source_tiles = spatially_thin_tiles_for_matching(
            source_tiles,
            args.matching_feature_cap_per_tile,
            args.matching_cells_per_axis,
            tile_pixels,
            tile_margin_pixels,
        )
        matching_target_tiles = spatially_thin_tiles_for_matching(
            target_tiles,
            args.matching_feature_cap_per_tile,
            args.matching_cells_per_axis,
            tile_pixels,
            tile_margin_pixels,
        )

        pair_matching_prior = matching_prior
        prior_uncertainty = args.matching_prior_uncertainty_m
        if args.sequential_prior:
            pair_matching_prior, prior_audit = sequential_matching_prior(
                previous_field,
                previous_elapsed_days,
                context["elapsed_days"],
                chain_contiguous=(
                    previous_target_path == manifest["source_image_filepath"]
                    if previous_target_path is not None
                    else True
                ),
                minimum_available_nodes=args.minimum_selected_vectors,
            )
            prior_uncertainty = args.sequential_prior_uncertainty_m
            prior_audit.update(
                {
                    "source_image_id": source_id,
                    "target_image_id": target_id,
                    "uncertainty_m": prior_uncertainty,
                }
            )
        else:
            prior_audit = {
                "source_image_id": source_id,
                "target_image_id": target_id,
                "source": "fixed_cli_prior" if matching_prior is not None else None,
                "fallback": matching_prior is None,
                "reason": "fixed_prior" if matching_prior is not None else "prior_absent",
                "uncertainty_m": prior_uncertainty,
            }

        matcher_calls = [] if args.matcher_call_audit else None
        matcher_call_matches = [] if args.matcher_call_audit else None
        matching_started = time.perf_counter()
        matches, matching_audit = match_tiles(
            matching_source_tiles,
            matching_target_tiles,
            matcher,
            device,
            tile_pixels,
            context["maximum_displacement_m"],
            context["elapsed_days"],
            maximum_speed_m_per_day,
            physics_subset_matching=True,
            reuse_device_features=reuse_device_features,
            matching_prior_displacement_m=pair_matching_prior,
            matching_prior_uncertainty_m=(
                prior_uncertainty if pair_matching_prior is not None else None
            ),
            matcher_call_audit=matcher_calls,
            matcher_call_matches=matcher_call_matches,
            audit_mnn_candidates=args.audit_mnn_candidates,
            mnn_candidate_limit=args.mnn_candidate_limit,
            lightglue_target_batch_size=args.lightglue_target_batch_size,
        )
        matching_wall_seconds = time.perf_counter() - matching_started
        candidate_audit_seconds = float(
            sum(row["candidate_audit_seconds"] for row in matching_audit)
        )
        matching_seconds = matching_wall_seconds - candidate_audit_seconds
        if pair_matching_prior is not None and len(matches):
            residual = np.hypot(
                matches["dx_m"].to_numpy(dtype=float) - pair_matching_prior[0],
                matches["dy_m"].to_numpy(dtype=float) - pair_matching_prior[1],
            )
            prior_audit["matched_residual_p90_m"] = float(
                np.quantile(residual, 0.90)
            )
            prior_audit["residual_within_uncertainty"] = bool(
                prior_audit["matched_residual_p90_m"] <= prior_uncertainty
            )
        else:
            prior_audit["matched_residual_p90_m"] = None
            prior_audit["residual_within_uncertainty"] = None
        prior_audits.append(prior_audit)

        field_estimation_started = time.perf_counter()
        queries = regular_queries(context["source_domain"], grid_spacing_m)
        field = nearest_consensus_at_queries(
            matches,
            queries,
            maximum_radius_m=args.maximum_radius_m,
            candidate_count=args.candidate_count,
            minimum_selected_vectors=args.minimum_selected_vectors,
            consensus_radius_m=args.consensus_radius_m,
        )
        field_estimation_seconds = time.perf_counter() - field_estimation_started
        topology_started = time.perf_counter()
        rejected, rejected_indices, rejection_iterations = (
            reject_flipped_nodes_until_stable(field, grid_spacing_m)
        )
        topology_before = topology_summary(field, grid_spacing_m)
        topology_after = topology_summary(rejected, grid_spacing_m)
        topology_qc_seconds = time.perf_counter() - topology_started
        previous_field = rejected
        previous_elapsed_days = context["elapsed_days"]
        previous_target_path = manifest["target_image_filepath"]
        buoy_path = context["reference_run_dir"] / "buoy_results.csv"
        if buoy_path.is_file():
            buoy_queries = pd.read_csv(
                buoy_path,
                dtype={"buoy_id": str},
                low_memory=False,
            )
        else:
            buoy_queries = pd.DataFrame()
        accuracy_evaluation_started = time.perf_counter()
        if buoy_queries.empty:
            buoy = pd.DataFrame()
        else:
            buoy = nearest_consensus_at_queries(
                matches,
                buoy_queries,
                maximum_radius_m=args.maximum_radius_m,
                candidate_count=args.candidate_count,
                minimum_selected_vectors=args.minimum_selected_vectors,
                consensus_radius_m=args.consensus_radius_m,
            )
            truth_columns = {"truth_dx_m", "truth_dy_m"}
            if truth_columns.issubset(buoy.columns):
                buoy["error_m"] = np.hypot(
                    buoy["proposal_dx_m"] - buoy["truth_dx_m"],
                    buoy["proposal_dy_m"] - buoy["truth_dy_m"],
                )
        accuracy_evaluation_seconds = (
            time.perf_counter() - accuracy_evaluation_started
        )
        estimation_seconds = (
            field_estimation_seconds
            + topology_qc_seconds
            + accuracy_evaluation_seconds
        )

        writing_started = time.perf_counter()
        matches.to_csv(pair_dir / "matches.csv", index=False)
        pd.DataFrame(matching_audit).to_csv(
            pair_dir / "matching_tiles.csv", index=False
        )
        if matcher_calls is not None:
            pd.DataFrame(matcher_calls).to_csv(
                pair_dir / "matcher_calls.csv", index=False
            )
            pd.DataFrame(matcher_call_matches).to_csv(
                pair_dir / "matcher_call_matches.csv", index=False
            )
        field.to_csv(pair_dir / "field_nearest12.csv", index=False)
        rejected.to_csv(pair_dir / "field_nearest12_fold_rejected.csv", index=False)
        buoy.to_csv(pair_dir / "buoy_nearest12.csv", index=False)
        writing_seconds = time.perf_counter() - writing_started

        available = field["available"].fillna(False)
        rejected_available = rejected["available"].fillna(False)
        summary = {
            "source_image_id": source_id,
            "target_image_id": target_id,
            "elapsed_hours": float(manifest["elapsed_hours"]),
            "source_tiles": len(source_tiles),
            "target_tiles": len(target_tiles),
            "matching_source_features": int(
                sum(len(tile["keypoints"]) for tile in matching_source_tiles)
            ),
            "matching_target_features": int(
                sum(len(tile["keypoints"]) for tile in matching_target_tiles)
            ),
            "physics_valid_matches": int(len(matches)),
            "grid_nodes": int(len(field)),
            "available_nodes": int(available.sum()),
            "coverage_fraction": float(available.mean()),
            "fold_rejected_available_nodes": int(rejected_available.sum()),
            "fold_rejected_coverage_fraction": float(rejected_available.mean()),
            "rejected_nodes": int(len(rejected_indices)),
            "rejection_iterations": rejection_iterations,
            "matching_prior": prior_audit,
            "matcher_calls": (
                int(
                    sum(
                        bool(row.get("matcher_executed", True))
                        for row in matcher_calls
                    )
                )
                if matcher_calls is not None
                else None
            ),
            "matcher_invocations": int(
                sum(row["matcher_invocations"] for row in matching_audit)
            ),
            "topology_before_rejection": topology_before,
            "topology_after_rejection": topology_after,
            "buoys": summarize_buoys(buoy),
            "timing_seconds": {
                "matching": matching_seconds,
                "matching_wall": matching_wall_seconds,
                "candidate_audit": candidate_audit_seconds,
                "field_estimation": field_estimation_seconds,
                "topology_and_qc": topology_qc_seconds,
                "accuracy_evaluation": accuracy_evaluation_seconds,
                "estimation_and_topology": estimation_seconds,
                "writing": writing_seconds,
                "pair_total": matching_seconds + estimation_seconds + writing_seconds,
            },
        }
        (pair_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        pair_summaries.append(summary)
        buoy_median = summary["buoys"]["median_error_m"]
        buoy_label = (
            f"{buoy_median:.1f}m buoy median"
            if buoy_median is not None
            else "unlabelled pair"
        )
        print(
            f"pair {source_id}->{target_id}: {matching_seconds:.2f}s matching, "
            f"{summary['fold_rejected_coverage_fraction']:.3f} coverage, "
            + buoy_label,
            flush=True,
        )

    total_seconds = time.perf_counter() - sequence_started
    summary = {
        "status": "complete",
        "device": str(device),
        "images": len(ordered_images),
        "pairs": len(pair_summaries),
        "features_per_tile": features_per_tile,
        "matching_feature_cap_per_tile": args.matching_feature_cap_per_tile,
        "matching_cells_per_axis": args.matching_cells_per_axis,
        "matcher": args.matcher,
        "smnn_ratio": args.smnn_ratio if args.matcher == "smnn" else None,
        "lightglue_layers": (
            args.lightglue_layers if args.matcher == "lightglue" else None
        ),
        "lightglue_depth_confidence": (
            args.lightglue_depth_confidence if args.matcher == "lightglue" else None
        ),
        "lightglue_width_confidence": (
            args.lightglue_width_confidence if args.matcher == "lightglue" else None
        ),
        "lightglue_filter_threshold": (
            args.lightglue_filter_threshold if args.matcher == "lightglue" else None
        ),
        "lightglue_adapter": (
            args.lightglue_adapter if args.matcher == "lightglue" else None
        ),
        "lightglue_compile": bool(
            args.lightglue_compile if args.matcher == "lightglue" else False
        ),
        "matcher_call_audit": bool(args.matcher_call_audit),
        "audit_mnn_candidates": bool(args.audit_mnn_candidates),
        "mnn_candidate_limit": args.mnn_candidate_limit,
        "lightglue_target_batch_size": args.lightglue_target_batch_size,
        "matching_prior_dx_m": args.matching_prior_dx_m,
        "matching_prior_dy_m": args.matching_prior_dy_m,
        "matching_prior_uncertainty_m": args.matching_prior_uncertainty_m,
        "sequential_prior": bool(args.sequential_prior),
        "sequential_prior_uncertainty_m": (
            args.sequential_prior_uncertainty_m if args.sequential_prior else None
        ),
        "prior_audits": prior_audits,
        "model_setup_seconds": model_seconds,
        "unique_image_feature_seconds": float(
            sum(row["seconds"] for row in image_timings)
        ),
        "image_preparation_seconds": float(
            sum(row["image_preparation_seconds"] for row in image_timings)
        ),
        "detection_description_seconds": float(
            sum(row["detection_description_seconds"] for row in image_timings)
        ),
        "feature_cache_read_seconds": float(
            sum(row["cache_read_seconds"] for row in image_timings)
        ),
        "feature_cache_write_seconds": float(
            sum(row["cache_write_seconds"] for row in image_timings)
        ),
        "pair_matching_seconds": float(
            sum(row["timing_seconds"]["matching"] for row in pair_summaries)
        ),
        "pair_estimation_topology_seconds": float(
            sum(
                row["timing_seconds"]["estimation_and_topology"]
                for row in pair_summaries
            )
        ),
        "pair_field_estimation_seconds": float(
            sum(
                row["timing_seconds"]["field_estimation"]
                for row in pair_summaries
            )
        ),
        "pair_topology_qc_seconds": float(
            sum(
                row["timing_seconds"]["topology_and_qc"]
                for row in pair_summaries
            )
        ),
        "pair_accuracy_evaluation_seconds": float(
            sum(
                row["timing_seconds"]["accuracy_evaluation"]
                for row in pair_summaries
            )
        ),
        "pair_writing_seconds": float(
            sum(row["timing_seconds"]["writing"] for row in pair_summaries)
        ),
        "elapsed_seconds": total_seconds,
        "pairs_summary": pair_summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    run_manifest = {
        "status": "complete",
        "pair_run_dirs": [str(path) for path in args.pair_run_dir],
        "output_dir": str(args.output_dir),
        "feature_cache_dir": str(feature_cache_dir),
        "model_cache": str(model_cache),
        "parameters": {
            "pixel_size_m": pixel_size_m,
            "tile_pixels": tile_pixels,
            "tile_margin_pixels": tile_margin_pixels,
            "tile_grid_origin_m": tile_grid_origin_m,
            "support_radius_pixels": support_radius_pixels,
            "grid_spacing_m": grid_spacing_m,
            "maximum_speed_m_per_day": maximum_speed_m_per_day,
            "features_per_tile": features_per_tile,
            "matching_feature_cap_per_tile": args.matching_feature_cap_per_tile,
            "matching_cells_per_axis": args.matching_cells_per_axis,
            "matcher": args.matcher,
            "smnn_ratio": args.smnn_ratio if args.matcher == "smnn" else None,
            "lightglue_layers": (
                args.lightglue_layers if args.matcher == "lightglue" else None
            ),
            "lightglue_depth_confidence": (
                args.lightglue_depth_confidence
                if args.matcher == "lightglue"
                else None
            ),
            "lightglue_width_confidence": (
                args.lightglue_width_confidence
                if args.matcher == "lightglue"
                else None
            ),
            "lightglue_filter_threshold": (
                args.lightglue_filter_threshold
                if args.matcher == "lightglue"
                else None
            ),
            "lightglue_adapter": (
                args.lightglue_adapter if args.matcher == "lightglue" else None
            ),
            "lightglue_compile": bool(
                args.lightglue_compile if args.matcher == "lightglue" else False
            ),
            "matcher_call_audit": bool(args.matcher_call_audit),
            "audit_mnn_candidates": bool(args.audit_mnn_candidates),
            "mnn_candidate_limit": args.mnn_candidate_limit,
            "lightglue_target_batch_size": args.lightglue_target_batch_size,
            "matching_prior_dx_m": args.matching_prior_dx_m,
            "matching_prior_dy_m": args.matching_prior_dy_m,
            "matching_prior_uncertainty_m": args.matching_prior_uncertainty_m,
            "sequential_prior": bool(args.sequential_prior),
            "sequential_prior_uncertainty_m": (
                args.sequential_prior_uncertainty_m
                if args.sequential_prior
                else None
            ),
            "maximum_radius_m": args.maximum_radius_m,
            "candidate_count": args.candidate_count,
            "minimum_selected_vectors": args.minimum_selected_vectors,
            "consensus_radius_m": args.consensus_radius_m,
            "physics_subset_matching": True,
            "reuse_device_features": reuse_device_features,
            "device": str(device),
        },
        "reference_parameters": parameters,
        "summary": summary,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
