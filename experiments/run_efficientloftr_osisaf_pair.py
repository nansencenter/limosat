#!/usr/bin/env python3
"""Run one EfficientLoFTR pair with OSI-455 search-window routing on MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.osisaf_routing_prior_audit import (
    advect_with_osi455,
    load_pair_output,
    load_reference_manifest,
)
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


DEFAULT_REPO = Path("/private/tmp/limosat_efficientloftr_official")
DEFAULT_CHECKPOINT = Path(
    "/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/"
    "model_cache/efficientloftr_official/weights/eloftr_outdoor.ckpt"
)
DEFAULT_OSI_CACHE = (
    ROOT / "results/osisaf_routing_prior_audit_20260831/osisaf_cache"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--efficientloftr-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--efficientloftr-checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--osi455-cache", type=Path, default=DEFAULT_OSI_CACHE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument(
        "--fallback",
        choices=("same_center", "phase"),
        default="same_center",
        help="Routing for tiles where OSI-455 is unavailable.",
    )
    parser.add_argument("--phase-pairs", type=Path)
    parser.add_argument("--maximum-speed-km-per-day", type=float, default=30.0)
    return parser.parse_args()


def load_case(pair_dir: Path, cohort: str):
    manifest = pair_dir / "run_manifest.json"
    if manifest.exists():
        return load_reference_manifest(manifest, cohort)
    return load_pair_output(pair_dir, cohort)


def phase_vector_for_case(path: Path, case_id: str) -> np.ndarray:
    rows = pd.read_csv(path)
    selected = rows.loc[rows["case_id"] == case_id]
    if len(selected) != 1:
        raise ValueError(f"phase table has {len(selected)} rows for {case_id}")
    return selected[["phase_dx_m", "phase_dy_m"]].iloc[0].to_numpy(float)


def external_routing_shifts(
    osi_displacement_m: np.ndarray,
    osi_available: np.ndarray,
    fallback_displacement_m: np.ndarray,
    maximum_displacement_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose finite external/fallback routes and enforce the physics limit."""
    osi_displacement_m = np.asarray(osi_displacement_m, dtype=np.float64)
    osi_available = np.asarray(osi_available, dtype=bool)
    fallback_displacement_m = np.asarray(fallback_displacement_m, dtype=np.float64)
    if osi_displacement_m.ndim != 2 or osi_displacement_m.shape[1] != 2:
        raise ValueError("OSI displacements must have shape (tiles, 2)")
    if osi_available.shape != (len(osi_displacement_m),):
        raise ValueError("OSI availability must have shape (tiles,)")
    if fallback_displacement_m.shape != (2,):
        raise ValueError("fallback displacement must have shape (2,)")
    shifts = np.repeat(fallback_displacement_m[None, :], len(osi_displacement_m), axis=0)
    shifts[osi_available] = osi_displacement_m[osi_available]
    sources = np.where(osi_available, "osi455", "same_center_fallback").astype(object)
    if np.linalg.norm(fallback_displacement_m) > 0:
        sources[~osi_available] = "phase_fallback"
    magnitudes = np.linalg.norm(shifts, axis=1)
    clipped = magnitudes > maximum_displacement_m
    shifts[clipped] *= (maximum_displacement_m / magnitudes[clipped])[:, None]
    sources[clipped] = np.char.add(sources[clipped].astype(str), "_clipped")
    if not np.isfinite(shifts).all():
        raise ValueError("routing shifts must be finite after fallback")
    return shifts, sources, clipped


def routing_identity(
    case,
    config: EfficientLoFTRConfig,
    shifts: np.ndarray,
    sources: np.ndarray,
    checkpoint_sha256: str,
    fallback: str,
) -> str:
    digest = hashlib.sha256()
    fixed = {
        "case_id": case.case_id,
        "source_path": case.source_path,
        "target_path": case.target_path,
        "elapsed_hours": case.elapsed_hours,
        "config": asdict(config),
        "checkpoint_sha256": checkpoint_sha256,
        "routing": "osi455_daily_integrated_v1",
        "fallback": fallback,
    }
    digest.update(json.dumps(fixed, sort_keys=True).encode())
    digest.update(np.round(shifts, 6).astype("<f8").tobytes())
    digest.update("\0".join(map(str, sources)).encode())
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    case = load_case(args.pair_dir, args.cohort)
    config = EfficientLoFTRConfig(
        maximum_speed_m_per_day=args.maximum_speed_km_per_day * 1000.0
    )
    spec = PairSpec(
        source_image_id=case.source_image_id,
        target_image_id=case.target_image_id,
        source_path=case.source_path,
        target_path=case.target_path,
        elapsed_hours=case.elapsed_hours,
        buoy_path=case.truth_path,
    )
    source_domain, _ = pair_domains(spec, config)
    regions = tile_layout(source_domain, config)
    centers = np.asarray([region.center_xy_m for region in regions], dtype=np.float64)
    osi = advect_with_osi455(
        centers,
        case.source_time,
        case.target_time,
        args.osi455_cache,
        config.analysis_epsg,
    )
    fallback = np.zeros(2, dtype=np.float64)
    if args.fallback == "phase":
        if args.phase_pairs is None:
            raise ValueError("--phase-pairs is required for phase fallback")
        fallback = phase_vector_for_case(args.phase_pairs, case.case_id)
    shifts, sources, clipped = external_routing_shifts(
        osi["displacement_m"],
        osi["available"],
        fallback,
        config.maximum_displacement_m(case.elapsed_hours),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    routing_rows = pd.DataFrame(
        {
            "tile_id": [region.tile_id for region in regions],
            "source_center_x_m": centers[:, 0],
            "source_center_y_m": centers[:, 1],
            "routing_dx_m": shifts[:, 0],
            "routing_dy_m": shifts[:, 1],
            "routing_source": sources,
            "osi455_available": osi["available"],
            "osi455_dx_m": osi["displacement_m"][:, 0],
            "osi455_dy_m": osi["displacement_m"][:, 1],
            "osi455_uncertainty_m": osi["uncertainty_m"],
            "osi455_wind_fraction": osi["wind_fraction"],
            "osi455_flags": osi["flags"],
            "physics_clipped": clipped,
        }
    )
    routing_rows.to_csv(args.output_dir / "routing_prior_tiles.csv", index=False)
    checkpoint_sha256 = file_sha256(args.efficientloftr_checkpoint)
    identity = routing_identity(
        case, config, shifts, sources, checkpoint_sha256, args.fallback
    )
    pair_output = args.output_dir / f"pair_{case.source_image_id}_{case.target_image_id}"
    completed = load_completed_pair(pair_output, identity)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    started = time.perf_counter()
    model_setup_seconds = 0.0
    if completed is None:
        setup_started = time.perf_counter()
        model = load_optimized_model(
            args.efficientloftr_repo,
            args.efficientloftr_checkpoint,
            device,
        )
        model_setup_seconds = time.perf_counter() - setup_started
        _, summary = track_pair(
            spec,
            model,
            device,
            config,
            "same_center",
            None,
            None,
            None,
            identity,
            pair_output,
            tile_shifts_override_m=shifts,
            tile_routing_sources_override=sources,
        )
        resumed = False
    else:
        _, summary = completed
        resumed = True
    baseline = (
        json.loads(args.baseline_summary.read_text())
        if args.baseline_summary is not None
        else None
    )
    manifest = {
        "status": "complete",
        "role": args.role,
        "case_id": case.case_id,
        "source_image_id": case.source_image_id,
        "target_image_id": case.target_image_id,
        "elapsed_hours": case.elapsed_hours,
        "device": device.type,
        "routing": "OSI-455 per source-tile centre",
        "fallback": args.fallback,
        "osi455_available_tiles": int(osi["available"].sum()),
        "source_tiles": len(regions),
        "osi455_available_tile_fraction": float(osi["available"].mean()),
        "physics_clipped_tiles": int(clipped.sum()),
        "checkpoint_sha256": checkpoint_sha256,
        "pair_identity_sha256": identity,
        "model_setup_seconds": model_setup_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "resumed": resumed,
        "baseline_summary_path": (
            None if args.baseline_summary is None else str(args.baseline_summary)
        ),
        "baseline": baseline,
        "osisaf_assisted": summary,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
