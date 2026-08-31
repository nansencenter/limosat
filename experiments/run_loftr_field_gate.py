#!/usr/bin/env python3
"""Compare LoFTR front ends using the selected learned-drift field stages."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import shapely
import torch
from kornia.feature import LoFTR

from limosat.learned_drift.config import ALIKEDConfig
from limosat.learned_drift.field import estimate_field, reject_folds, topology_summary
from limosat.learned_drift.imagery import north_up_patch
from limosat.learned_drift.types import MotionMatches


@dataclass(frozen=True)
class TilePair:
    tile_id: int
    center_xy_m: tuple[float, float]
    source: np.ndarray
    target: np.ndarray
    source_valid: np.ndarray
    target_valid: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--efficientloftr-repo", type=Path, required=True)
    parser.add_argument("--efficientloftr-checkpoint", type=Path, required=True)
    parser.add_argument("--elapsed-hours", type=float, required=True)
    parser.add_argument("--center-x-m", type=float, default=197_120.0)
    parser.add_argument("--center-y-m", type=float, default=448_000.0)
    parser.add_argument("--tile-radius", type=int, default=1)
    parser.add_argument("--warm-repetitions", type=int, default=5)
    return parser.parse_args()


def sample_tiles(args: argparse.Namespace, config: ALIKEDConfig) -> list[TilePair]:
    spacing_m = config.tile_core_size_m
    offsets = range(-args.tile_radius, args.tile_radius + 1)
    pairs = []
    for row, y_offset in enumerate(offsets):
        for column, x_offset in enumerate(offsets):
            center = (
                args.center_x_m + x_offset * spacing_m,
                args.center_y_m + y_offset * spacing_m,
            )
            source, source_valid = north_up_patch(
                str(args.source),
                center,
                config.tile_size_px,
                config.pixel_size_m,
                config.analysis_epsg,
                config.transform_grid_spacing_px,
            )
            target, target_valid = north_up_patch(
                str(args.target),
                center,
                config.tile_size_px,
                config.pixel_size_m,
                config.analysis_epsg,
                config.transform_grid_spacing_px,
            )
            pairs.append(
                TilePair(
                    tile_id=row * len(offsets) + column,
                    center_xy_m=center,
                    source=source,
                    target=target,
                    source_valid=source_valid,
                    target_valid=target_valid,
                )
            )
    return pairs


def image_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(image.copy()).to(device=device, dtype=torch.float32)[None, None]
        / 255.0
    )


def kornia_inputs(pair: TilePair, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "image0": image_tensor(pair.source, device),
        "image1": image_tensor(pair.target, device),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def run_kornia(model, inputs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = model(inputs)
    return (
        output["keypoints0"].detach().cpu().numpy(),
        output["keypoints1"].detach().cpu().numpy(),
        output["confidence"].detach().cpu().numpy(),
    )


def run_official_efficient(
    model,
    inputs,
    optimized: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model(inputs)
    source_px = inputs["mkpts0_f"].detach().cpu().numpy()
    target_px = inputs["mkpts1_f"].detach().cpu().numpy()
    score = inputs["mconf"].detach().cpu().numpy()
    if optimized and len(score):
        lower = min(20.0, float(score.min()))
        upper = max(30.0, float(score.max()))
        score = (score - lower) / (upper - lower)
    return source_px, target_px, score


def run_matcher(name, model, inputs):
    if name == "loftr":
        return run_kornia(model, inputs)
    return run_official_efficient(
        model,
        inputs,
        optimized=name.endswith("_opt"),
    )


def projected_xy(
    points_px: np.ndarray,
    center_xy_m: tuple[float, float],
    config: ALIKEDConfig,
) -> np.ndarray:
    center_px = (config.tile_size_px - 1) / 2.0
    return np.column_stack(
        (
            center_xy_m[0] + (points_px[:, 0] - center_px) * config.pixel_size_m,
            center_xy_m[1] - (points_px[:, 1] - center_px) * config.pixel_size_m,
        )
    )


def valid_endpoints(points_px: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rounded = np.rint(points_px).astype(int)
    inside = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < valid.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < valid.shape[0])
    )
    accepted = np.zeros(len(points_px), dtype=bool)
    accepted[inside] = valid[rounded[inside, 1], rounded[inside, 0]]
    return accepted


def match_tiles(
    name: str,
    model,
    tiles: list[TilePair],
    device: torch.device,
    config: ALIKEDConfig,
    elapsed_hours: float,
    repetitions: int,
) -> tuple[MotionMatches, list[dict], list[float]]:
    central = tiles[len(tiles) // 2]
    benchmark_input = kornia_inputs(central, device)
    warm_seconds = []
    for repetition in range(repetitions + 1):
        started = time.perf_counter()
        with torch.inference_mode():
            run_matcher(name, model, benchmark_input)
        synchronize(device)
        seconds = time.perf_counter() - started
        if repetition > 0:
            warm_seconds.append(seconds)

    maximum_displacement_m = config.maximum_displacement_m(elapsed_hours)
    source_rows = []
    target_rows = []
    score_rows = []
    tile_rows = []
    metrics = []
    for tile in tiles:
        inputs = kornia_inputs(tile, device)
        started = time.perf_counter()
        with torch.inference_mode():
            source_px, target_px, score = run_matcher(name, model, inputs)
        synchronize(device)
        seconds = time.perf_counter() - started

        margin = config.tile_margin_px
        in_core = (
            (source_px[:, 0] >= margin)
            & (source_px[:, 0] < config.tile_size_px - margin)
            & (source_px[:, 1] >= margin)
            & (source_px[:, 1] < config.tile_size_px - margin)
        )
        valid = (
            in_core
            & valid_endpoints(source_px, tile.source_valid)
            & valid_endpoints(target_px, tile.target_valid)
        )
        source_xy_m = projected_xy(source_px, tile.center_xy_m, config)
        target_xy_m = projected_xy(target_px, tile.center_xy_m, config)
        speed_valid = (
            np.linalg.norm(target_xy_m - source_xy_m, axis=1)
            <= maximum_displacement_m
        )
        accepted = valid & speed_valid
        source_rows.append(source_xy_m[accepted])
        target_rows.append(target_xy_m[accepted])
        score_rows.append(score[accepted])
        tile_rows.append(np.full(accepted.sum(), tile.tile_id, dtype=np.int32))
        metrics.append(
            {
                "tile_id": tile.tile_id,
                "center_xy_m": list(tile.center_xy_m),
                "raw_matches": int(len(source_px)),
                "core_valid_matches": int(valid.sum()),
                "physics_valid_matches": int(accepted.sum()),
                "matching_seconds": seconds,
            }
        )

    source_xy_m = np.concatenate(source_rows)
    target_xy_m = np.concatenate(target_rows)
    scores = np.concatenate(score_rows).astype(np.float32)
    tile_ids = np.concatenate(tile_rows)
    matches = MotionMatches(
        source_feature_id=np.arange(len(scores), dtype=np.int64),
        source_tile_id=tile_ids,
        target_tile_id=tile_ids.copy(),
        source_xy_m=source_xy_m,
        target_xy_m=target_xy_m,
        score=scores,
    )
    return matches, metrics, warm_seconds


def domain_for_tiles(tiles: list[TilePair], config: ALIKEDConfig):
    half = config.tile_core_size_m / 2.0
    xs = [tile.center_xy_m[0] for tile in tiles]
    ys = [tile.center_xy_m[1] for tile in tiles]
    return shapely.box(min(xs) - half, min(ys) - half, max(xs) + half, max(ys) + half)


def save_arm(
    output_dir: Path,
    name: str,
    matches: MotionMatches,
    field,
    tile_metrics: list[dict],
) -> None:
    np.savez_compressed(
        output_dir / f"{name}_matches.npz",
        source_xy_m=matches.source_xy_m,
        target_xy_m=matches.target_xy_m,
        score=matches.score,
    )
    field.to_frame().to_csv(output_dir / f"{name}_field_4km.csv", index=False)
    (output_dir / f"{name}_tiles.json").write_text(
        json.dumps(tile_metrics, indent=2) + "\n"
    )


def plot_fields(output_dir: Path, fields: dict[str, object]) -> None:
    figure, axes = plt.subplots(1, len(fields), figsize=(6.5 * len(fields), 6), constrained_layout=True)
    for axis, (name, field) in zip(axes, fields.items(), strict=True):
        keep = field.available
        axis.quiver(
            field.source_xy_m[keep, 0] / 1000.0,
            field.source_xy_m[keep, 1] / 1000.0,
            field.displacement_m[keep, 0] / 1000.0,
            field.displacement_m[keep, 1] / 1000.0,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.0025,
        )
        axis.set_title(f"{name}: {keep.sum():,}/{len(keep):,} 4 km nodes")
        axis.set_aspect("equal")
        axis.set_xlabel("EPSG:3413 x (km)")
        axis.set_ylabel("EPSG:3413 y (km)")
    figure.savefig(output_dir / "field_comparison.png", dpi=180)
    plt.close(figure)


def field_agreement(fields: dict[str, object]) -> dict[str, dict[str, float | int]]:
    summaries = {}
    for (left_name, left), (right_name, right) in combinations(fields.items(), 2):
        if not np.array_equal(left.source_xy_m, right.source_xy_m):
            raise ValueError("field grids differ")
        common = left.available & right.available
        difference_m = np.linalg.norm(
            left.displacement_m[common] - right.displacement_m[common], axis=1
        )
        summaries[f"{left_name}_vs_{right_name}"] = {
            "common_nodes": int(common.sum()),
            "median_m": float(np.median(difference_m)),
            "p90_m": float(np.percentile(difference_m, 90)),
            "p99_m": float(np.percentile(difference_m, 99)),
            "maximum_m": float(difference_m.max()),
        }
    return summaries


def load_official_efficient_models(
    repo: Path,
    checkpoint: Path,
    device: torch.device,
    variants: tuple[str, ...] = ("full", "opt"),
) -> dict[str, torch.nn.Module]:
    # Compatibility shims for the authors' pinned 2024 inference environment.
    from kornia.geometry import create_meshgrid

    grid_module = ModuleType("kornia.utils.grid")
    grid_module.create_meshgrid = create_meshgrid
    sys.modules.setdefault("kornia.utils.grid", grid_module)
    if "pytorch_lightning.utilities" not in sys.modules:
        lightning_module = ModuleType("pytorch_lightning")
        utilities_module = ModuleType("pytorch_lightning.utilities")

        class RankZeroOnly:
            rank = 0

            def __call__(self, function):
                return function

        utilities_module.rank_zero_only = RankZeroOnly()
        lightning_module.utilities = utilities_module
        sys.modules.setdefault("pytorch_lightning", lightning_module)
        sys.modules.setdefault("pytorch_lightning.utilities", utilities_module)

    sys.path.insert(0, str(repo))
    from src.loftr import LoFTR as EfficientLoFTR
    from src.loftr import full_default_cfg, opt_default_cfg, reparameter

    class ModelCheckpoint:
        pass

    safe_global = (
        ModelCheckpoint,
        "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint",
    )
    with torch.serialization.safe_globals([safe_global]):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)[
            "state_dict"
        ]
    models = {}
    available = {
        "full": ("efficientloftr_official_full", full_default_cfg),
        "opt": ("efficientloftr_official_opt", opt_default_cfg),
    }
    unknown = set(variants).difference(available)
    if unknown:
        raise ValueError(f"unknown EfficientLoFTR variants: {sorted(unknown)}")
    for variant in variants:
        name, model_config = available[variant]
        model = EfficientLoFTR(config=deepcopy(model_config))
        model.load_state_dict(state.copy())
        models[name] = reparameter(model).eval().to(device)
    return models


def main() -> int:
    args = parse_args()
    config = ALIKEDConfig()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tiles = sample_tiles(args, config)
    domain = domain_for_tiles(tiles, config)

    models = {"loftr": LoFTR(pretrained="outdoor").eval().to(device)}
    models.update(
        load_official_efficient_models(
            args.efficientloftr_repo,
            args.efficientloftr_checkpoint,
            device,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    fields = {}
    for name, model in models.items():
        matches, tile_metrics, warm_seconds = match_tiles(
            name,
            model,
            tiles,
            device,
            config,
            args.elapsed_hours,
            args.warm_repetitions,
        )
        raw_field = estimate_field(matches, domain, config)
        field, rejected = reject_folds(raw_field, config.grid_spacing_m)
        fields[name] = field
        save_arm(args.output_dir, name, matches, field, tile_metrics)
        summaries[name] = {
            "physics_valid_matches": len(matches),
            "grid_nodes": len(field),
            "available_before_fold_rejection": int(raw_field.available.sum()),
            "available_after_fold_rejection": int(field.available.sum()),
            "coverage_after_fold_rejection": float(field.available.mean()),
            "fold_rejected_nodes": int(len(rejected)),
            "topology_after_rejection": topology_summary(field, config.grid_spacing_m),
            "warm_matching_seconds": warm_seconds,
            "warm_matching_seconds_median": float(np.median(warm_seconds)),
            "tile_matching_seconds_total": float(
                sum(row["matching_seconds"] for row in tile_metrics)
            ),
        }

    report = {
        "status": "basic_field_architecture_gate",
        "source": str(args.source),
        "target": str(args.target),
        "elapsed_hours": args.elapsed_hours,
        "device": device.type,
        "tile_count": len(tiles),
        "tile_pixels": config.tile_size_px,
        "tile_margin_pixels": config.tile_margin_px,
        "pixel_size_m": config.pixel_size_m,
        "grid_spacing_m": config.grid_spacing_m,
        "confidence_filter": None,
        "shared_downstream_stages": [
            "invalid endpoint exclusion",
            "30 km/day physics limit",
            "nearest-12 regular-grid consensus",
            "fold rejection",
        ],
        "arms": summaries,
        "field_agreement": field_agreement(fields),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_fields(args.output_dir, fields)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
