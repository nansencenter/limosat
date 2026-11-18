#!/usr/bin/env python3
"""Compare direct phase and OSI-405-d-assisted EfficientLoFTR in Weddell Sea."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.osisaf405_drift import advect_with_osi405, required_product_paths
from experiments.run_efficientloftr_sequence import (
    PairSpec,
    file_sha256,
    load_completed_pair,
    pair_domains,
    track_pair,
)
from limosat.learned_drift.config import EfficientLoFTRConfig
from limosat.learned_drift.efficientloftr import load_optimized_model
from limosat.learned_drift.features import tile_layout
from limosat.learned_drift.routing import coarse_phase_translation


DEFAULT_CONFIG = (
    ROOT / "experiments/configs/osisaf405_antarctic_routing_20260831.json"
)
DEFAULT_OUTPUT = ROOT / "results/osisaf_antarctic_routing_20260831"
DEFAULT_REPO = Path("/private/tmp/limosat_efficientloftr_official")
DEFAULT_CHECKPOINT = Path(
    "/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/"
    "model_cache/efficientloftr_official/weights/eloftr_outdoor.ckpt"
)


@dataclass(frozen=True)
class FrozenCase:
    case_id: str
    source_scene_id: str
    target_scene_id: str
    source_time: pd.Timestamp
    target_time: pd.Timestamp
    elapsed_hours: float
    source_path: str
    target_path: str
    source_image_id: int
    target_image_id: int
    truth_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--osi405-cache", type=Path)
    parser.add_argument("--efficientloftr-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--efficientloftr-checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare truth and prior diagnostics without loading EfficientLoFTR.",
    )
    return parser.parse_args()


def prepare_cases(
    experiment: dict, output_dir: Path
) -> tuple[list[FrozenCase], EfficientLoFTRConfig]:
    analysis_epsg = int(experiment["analysis_epsg"])
    config = EfficientLoFTRConfig(
        analysis_epsg=analysis_epsg,
        maximum_speed_m_per_day=float(experiment["maximum_speed_km_per_day"])
        * 1_000.0,
    )
    gaps = pd.read_csv(experiment["gap_cases_csv"])
    inventory = pd.read_csv(experiment["scene_inventory_csv"])
    direct_truth = pd.read_csv(
        experiment["direct_sar_truth_csv"], dtype={"trajectory_id": str}
    )
    selected = gaps.set_index("case_id").loc[experiment["case_ids"]].reset_index()
    if not selected["split"].eq("test").all():
        raise ValueError("frozen Antarctic cohort must contain held-out test rows only")
    path_by_scene = inventory.set_index("scene_id")["tiff_path"].to_dict()
    scene_ids = sorted(
        set(selected["source_scene_id"]) | set(selected["target_scene_id"])
    )
    numeric_id = {scene_id: index + 1 for index, scene_id in enumerate(scene_ids)}
    reference_root = output_dir / "references"
    cases = []
    cohort_rows = []
    for row in selected.itertuples(index=False):
        source_path = str(path_by_scene[row.source_scene_id])
        target_path = str(path_by_scene[row.target_scene_id])
        if not Path(source_path).exists() or not Path(target_path).exists():
            raise FileNotFoundError(f"missing SAR input for {row.case_id}")
        case_dir = reference_root / row.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        reference = direct_truth.loc[direct_truth["case_id"] == row.case_id].copy()
        if reference.empty:
            raise ValueError(f"no direct-SAR reference rows in {row.case_id}")
        reference["buoy_id"] = reference["trajectory_id"].astype(str)
        reference["truth_dx_m"] = reference["displacement_x_m"]
        reference["truth_dy_m"] = reference["displacement_y_m"]
        reference.to_csv(case_dir / "buoy_results.csv", index=False)
        manifest = {
            "case_id": row.case_id,
            "split": row.split,
            "source_image_id": numeric_id[row.source_scene_id],
            "target_image_id": numeric_id[row.target_scene_id],
            "source_scene_id": row.source_scene_id,
            "target_scene_id": row.target_scene_id,
            "source_image_time": pd.Timestamp(row.source_time).isoformat(),
            "target_image_time": pd.Timestamp(row.target_time).isoformat(),
            "source_image_filepath": source_path,
            "target_image_filepath": target_path,
            "elapsed_hours": float(row.gap_hours),
            "analysis_epsg": analysis_epsg,
            "regional_direct_sar_reference_rows": int(len(reference)),
            "independent_truth": False,
        }
        (case_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        cases.append(
            FrozenCase(
                case_id=row.case_id,
                source_scene_id=row.source_scene_id,
                target_scene_id=row.target_scene_id,
                source_time=pd.Timestamp(row.source_time),
                target_time=pd.Timestamp(row.target_time),
                elapsed_hours=float(row.gap_hours),
                source_path=source_path,
                target_path=target_path,
                source_image_id=numeric_id[row.source_scene_id],
                target_image_id=numeric_id[row.target_scene_id],
                truth_path=case_dir / "buoy_results.csv",
            )
        )
        cohort_rows.append(manifest)
    pd.DataFrame(cohort_rows).to_csv(output_dir / "frozen_cohort.csv", index=False)
    return cases, config


def phase_for_case(case: FrozenCase, config: EfficientLoFTRConfig) -> dict:
    spec = pair_spec(case)
    source_domain, _ = pair_domains(spec, config)
    started = time.perf_counter()
    phase = coarse_phase_translation(
        case.source_path,
        case.target_path,
        source_domain,
        config.maximum_displacement_m(case.elapsed_hours),
        config.analysis_epsg,
        config.transform_grid_spacing_px,
    )
    return {
        "case_id": case.case_id,
        "phase_dx_m": float(phase.displacement_m[0]),
        "phase_dy_m": float(phase.displacement_m[1]),
        "phase_magnitude_m": float(np.linalg.norm(phase.displacement_m)),
        "phase_response": float(phase.response),
        "phase_overlap_fraction": float(phase.overlap_fraction),
        "phase_pixel_size_m": float(phase.pixel_size_m),
        "phase_pixels": int(phase.pixels),
        "phase_seconds": time.perf_counter() - started,
    }


def pair_spec(case: FrozenCase) -> PairSpec:
    return PairSpec(
        source_image_id=case.source_image_id,
        target_image_id=case.target_image_id,
        source_path=case.source_path,
        target_path=case.target_path,
        elapsed_hours=case.elapsed_hours,
        buoy_path=case.truth_path,
    )


def prepare_prior_diagnostics(
    cases: list[FrozenCase],
    config: EfficientLoFTRConfig,
    osi_cache: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_rows = [phase_for_case(case, config) for case in cases]
    phase_table = pd.DataFrame(phase_rows)
    phase_table.to_csv(output_dir / "phase_pairs.csv", index=False)
    prior_rows = []
    for case, phase in zip(cases, phase_rows, strict=True):
        missing = [
            path
            for path in required_product_paths(
                case.source_time.to_pydatetime(),
                case.target_time.to_pydatetime(),
                osi_cache,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(missing[0])
        truth = pd.read_csv(case.truth_path, dtype={"buoy_id": str})
        source_xy = truth[["source_x", "source_y"]].to_numpy(float)
        observed = truth[["truth_dx_m", "truth_dy_m"]].to_numpy(float)
        osi = advect_with_osi405(
            source_xy,
            case.source_time.to_pydatetime(),
            case.target_time.to_pydatetime(),
            osi_cache,
            config.analysis_epsg,
        )
        phase_vector = np.array([phase["phase_dx_m"], phase["phase_dy_m"]])
        for index, row in truth.iterrows():
            prior_rows.append(
                {
                    "case_id": case.case_id,
                    "elapsed_hours": case.elapsed_hours,
                    "reference_id": str(row["buoy_id"]),
                    "truth_dx_m": observed[index, 0],
                    "truth_dy_m": observed[index, 1],
                    "phase_dx_m": phase_vector[0],
                    "phase_dy_m": phase_vector[1],
                    "phase_error_m": float(np.linalg.norm(phase_vector - observed[index])),
                    "phase_response": phase["phase_response"],
                    "osi405_available": bool(osi["available"][index]),
                    "osi405_dx_m": osi["displacement_m"][index, 0],
                    "osi405_dy_m": osi["displacement_m"][index, 1],
                    "osi405_error_m": float(
                        np.linalg.norm(osi["displacement_m"][index] - observed[index])
                    ),
                    "osi405_uncertainty_m": osi["uncertainty_m"][index],
                    "osi405_flags": osi["flags"][index],
                    "phase_osi405_distance_m": float(
                        np.linalg.norm(phase_vector - osi["displacement_m"][index])
                    ),
                }
            )
    prior_table = pd.DataFrame(prior_rows)
    prior_table.to_csv(output_dir / "prior_reference_diagnostics.csv", index=False)
    return phase_table, prior_table


def _routing_identity(
    case: FrozenCase,
    arm: str,
    config: EfficientLoFTRConfig,
    checkpoint_sha256: str,
    shifts: np.ndarray,
    sources: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    fixed = {
        "case_id": case.case_id,
        "arm": arm,
        "source_path": case.source_path,
        "target_path": case.target_path,
        "elapsed_hours": case.elapsed_hours,
        "config": asdict(config),
        "checkpoint_sha256": checkpoint_sha256,
        "routing_policy": (
            "global_direct_phase_v1"
            if arm == "phase"
            else "osi405d_trailing_velocity_phase_fallback_v1"
        ),
    }
    digest.update(json.dumps(fixed, sort_keys=True).encode())
    digest.update(np.round(shifts, 6).astype("<f8").tobytes())
    digest.update("\0".join(map(str, sources)).encode())
    return digest.hexdigest()


def _physics_clipped(
    shifts: np.ndarray, maximum_displacement_m: float
) -> tuple[np.ndarray, np.ndarray]:
    shifts = np.asarray(shifts, dtype=np.float64).copy()
    magnitudes = np.linalg.norm(shifts, axis=1)
    clipped = magnitudes > maximum_displacement_m
    shifts[clipped] *= (maximum_displacement_m / magnitudes[clipped])[:, None]
    return shifts, clipped


def routing_for_case(
    case: FrozenCase,
    phase: pd.Series,
    config: EfficientLoFTRConfig,
    osi_cache: Path,
) -> dict[str, dict[str, np.ndarray]]:
    source_domain, _ = pair_domains(pair_spec(case), config)
    regions = tile_layout(source_domain, config)
    centers = np.asarray([region.center_xy_m for region in regions], dtype=np.float64)
    tile_ids = np.asarray([region.tile_id for region in regions], dtype=np.int64)
    phase_vector = phase[["phase_dx_m", "phase_dy_m"]].to_numpy(float)
    phase_shifts = np.repeat(phase_vector[None, :], len(regions), axis=0)
    phase_shifts, phase_clipped = _physics_clipped(
        phase_shifts, config.maximum_displacement_m(case.elapsed_hours)
    )
    osi = advect_with_osi405(
        centers,
        case.source_time.to_pydatetime(),
        case.target_time.to_pydatetime(),
        osi_cache,
        config.analysis_epsg,
    )
    assisted = phase_shifts.copy()
    assisted[osi["available"]] = osi["displacement_m"][osi["available"]]
    assisted, assisted_clipped = _physics_clipped(
        assisted, config.maximum_displacement_m(case.elapsed_hours)
    )
    return {
        "phase": {
            "tile_ids": tile_ids,
            "centers": centers,
            "shifts": phase_shifts,
            "sources": np.full(len(regions), "phase", dtype=object),
            "clipped": phase_clipped,
            "osi_available": osi["available"],
            "osi_displacement_m": osi["displacement_m"],
            "osi_uncertainty_m": osi["uncertainty_m"],
            "osi_flags": osi["flags"],
        },
        "osi405_assisted": {
            "tile_ids": tile_ids,
            "centers": centers,
            "shifts": assisted,
            "sources": np.where(
                osi["available"], "osi405d", "phase_fallback"
            ).astype(object),
            "clipped": assisted_clipped,
            "osi_available": osi["available"],
            "osi_displacement_m": osi["displacement_m"],
            "osi_uncertainty_m": osi["uncertainty_m"],
            "osi_flags": osi["flags"],
        },
    }


def _save_routing(path: Path, routing: dict[str, np.ndarray]) -> None:
    rows = pd.DataFrame(
        {
            "tile_id": routing["tile_ids"],
            "source_center_x_m": routing["centers"][:, 0],
            "source_center_y_m": routing["centers"][:, 1],
            "routing_dx_m": routing["shifts"][:, 0],
            "routing_dy_m": routing["shifts"][:, 1],
            "routing_source": routing["sources"],
            "osi405_available": routing["osi_available"],
            "osi405_dx_m": routing["osi_displacement_m"][:, 0],
            "osi405_dy_m": routing["osi_displacement_m"][:, 1],
            "osi405_uncertainty_m": routing["osi_uncertainty_m"],
            "osi405_flags": routing["osi_flags"],
            "physics_clipped": routing["clipped"],
        }
    )
    rows.to_csv(path, index=False)


def _outcome_table(
    cases: list[FrozenCase], summaries: dict[tuple[str, str], dict]
) -> pd.DataFrame:
    rows = []
    for case in cases:
        phase = summaries[(case.case_id, "phase")]
        assisted = summaries[(case.case_id, "osi405_assisted")]
        rows.append(
            {
                "case_id": case.case_id,
                "elapsed_hours": case.elapsed_hours,
                "phase_coverage": phase["coverage_after_fold_rejection"],
                "assisted_coverage": assisted["coverage_after_fold_rejection"],
                "coverage_difference_pp": 100.0
                * (
                    assisted["coverage_after_fold_rejection"]
                    - phase["coverage_after_fold_rejection"]
                ),
                "phase_matches": phase["physics_valid_matches"],
                "assisted_matches": assisted["physics_valid_matches"],
                "phase_reference_available": phase["buoys"]["available"],
                "assisted_reference_available": assisted["buoys"]["available"],
                "phase_reference_median_error_m": phase["buoys"]["median_error_m"],
                "assisted_reference_median_error_m": assisted["buoys"]["median_error_m"],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "run_manifest.json"
    previous_report = (
        json.loads(report_path.read_text()) if report_path.exists() else None
    )
    osi_cache = args.osi405_cache or args.output_dir / "osi405_cache"
    experiment = json.loads(args.config.read_text())
    cases, config = prepare_cases(experiment, args.output_dir)
    phase_table, prior_table = prepare_prior_diagnostics(
        cases, config, osi_cache, args.output_dir
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "cases": len(cases),
                    "regional_direct_sar_reference_rows": len(prior_table),
                    "phase_seconds": float(phase_table["phase_seconds"].sum()),
                },
                indent=2,
            )
        )
        return 0
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    checkpoint_sha256 = file_sha256(args.efficientloftr_checkpoint)
    model = None
    summaries: dict[tuple[str, str], dict] = {}
    resumed_arms = 0
    started = time.perf_counter()
    for case in cases:
        phase = phase_table.loc[phase_table["case_id"] == case.case_id].iloc[0]
        routes = routing_for_case(case, phase, config, osi_cache)
        for arm in ("phase", "osi405_assisted"):
            routing = routes[arm]
            arm_dir = args.output_dir / "matched" / case.case_id / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            _save_routing(arm_dir / "routing_prior_tiles.csv", routing)
            identity = _routing_identity(
                case,
                arm,
                config,
                checkpoint_sha256,
                routing["shifts"],
                routing["sources"],
            )
            completed = load_completed_pair(arm_dir, identity)
            if completed is None:
                if model is None:
                    model = load_optimized_model(
                        args.efficientloftr_repo,
                        args.efficientloftr_checkpoint,
                        device,
                    )
                _, summary = track_pair(
                    pair_spec(case),
                    model,
                    device,
                    config,
                    "same_center",
                    None,
                    None,
                    None,
                    identity,
                    arm_dir,
                    tile_shifts_override_m=routing["shifts"],
                    tile_routing_sources_override=routing["sources"],
                )
            else:
                _, summary = completed
                resumed_arms += 1
            summaries[(case.case_id, arm)] = summary
            print(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "arm": arm,
                        "coverage": summary["coverage_after_fold_rejection"],
                        "matches": summary["physics_valid_matches"],
                        "buoys": summary["buoys"],
                    }
                ),
                flush=True,
            )
    outcomes = _outcome_table(cases, summaries)
    outcomes.to_csv(args.output_dir / "matched_outcomes.csv", index=False)
    long_gap = outcomes.loc[
        outcomes["case_id"].isin(experiment["primary_long_gap_case_ids"])
    ]
    execution_elapsed_seconds = time.perf_counter() - started
    initial_execution_elapsed_seconds = (
        previous_report.get(
            "initial_execution_elapsed_seconds",
            previous_report.get("elapsed_seconds"),
        )
        if previous_report is not None and resumed_arms == 2 * len(cases)
        else execution_elapsed_seconds
    )
    report = {
        "status": "complete",
        "experiment": experiment,
        "device": device.type,
        "config": asdict(config),
        "cases": len(cases),
        "primary_long_gap_cases": len(long_gap),
        "matched_arms": 2 * len(cases),
        "resumed_arms": resumed_arms,
        "regional_direct_sar_reference_rows": int(len(prior_table)),
        "long_gap_coverage_wins": int((long_gap["coverage_difference_pp"] > 0).sum()),
        "long_gap_coverage_ties": int((long_gap["coverage_difference_pp"] == 0).sum()),
        "long_gap_coverage_losses": int((long_gap["coverage_difference_pp"] < 0).sum()),
        "long_gap_median_coverage_difference_pp": float(
            long_gap["coverage_difference_pp"].median()
        ),
        "long_gap_mean_coverage_difference_pp": float(
            long_gap["coverage_difference_pp"].mean()
        ),
        "pair_compute_seconds": float(
            sum(summary["timing_seconds"]["pair_total"] for summary in summaries.values())
        ),
        "initial_execution_elapsed_seconds": initial_execution_elapsed_seconds,
        "current_execution_elapsed_seconds": execution_elapsed_seconds,
        "checkpoint_sha256": checkpoint_sha256,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
