#!/usr/bin/env python3
"""Minimal LoFTR smoke test on one north-up LiMOSAT VAE tile pair."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.feature import LoFTR

from limosat.learned_drift.imagery import north_up_patch


def occupied_fraction(points_px: np.ndarray, pixels: int, cells: int = 8) -> float:
    if not len(points_px):
        return 0.0
    indices = np.clip((points_px / pixels * cells).astype(int), 0, cells - 1)
    return len({tuple(index) for index in indices}) / float(cells * cells)


def display_indices(
    points_px: np.ndarray,
    confidence: np.ndarray,
    pixels: int,
    cells: int = 16,
) -> np.ndarray:
    """Select one high-confidence match per cell for an uncluttered plot."""
    if not len(points_px):
        return np.empty(0, dtype=int)
    cell_xy = np.clip((points_px / pixels * cells).astype(int), 0, cells - 1)
    selected: dict[tuple[int, int], int] = {}
    for index in np.argsort(confidence)[::-1]:
        cell = tuple(cell_xy[index])
        selected.setdefault(cell, int(index))
    return np.asarray(list(selected.values()), dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--center-x-m", type=float, required=True)
    parser.add_argument("--center-y-m", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--pixel-size-m", type=float, default=80.0)
    parser.add_argument("--analysis-epsg", type=int, default=3413)
    parser.add_argument("--transform-grid-spacing-px", type=int, default=32)
    parser.add_argument("--pretrained", choices=("outdoor", "indoor"), default="outdoor")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    center_xy_m = (args.center_x_m, args.center_y_m)
    source, source_valid = north_up_patch(
        str(args.source),
        center_xy_m,
        args.pixels,
        args.pixel_size_m,
        args.analysis_epsg,
        args.transform_grid_spacing_px,
    )
    target, target_valid = north_up_patch(
        str(args.target),
        center_xy_m,
        args.pixels,
        args.pixel_size_m,
        args.analysis_epsg,
        args.transform_grid_spacing_px,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    matcher = LoFTR(pretrained=args.pretrained).eval().to(device)
    batch = {
        "image0": torch.from_numpy(source.copy()).to(device=device, dtype=torch.float32)[None, None] / 255.0,
        "image1": torch.from_numpy(target.copy()).to(device=device, dtype=torch.float32)[None, None] / 255.0,
    }
    started = time.perf_counter()
    with torch.inference_mode():
        result = matcher(batch)
    matching_seconds = time.perf_counter() - started

    source_px = result["keypoints0"].detach().cpu().numpy()
    target_px = result["keypoints1"].detach().cpu().numpy()
    confidence = result["confidence"].detach().cpu().numpy()
    displacement_px = target_px - source_px
    median_displacement_px = (
        np.median(displacement_px, axis=0) if len(displacement_px) else np.full(2, np.nan)
    )
    residual_px = (
        np.linalg.norm(displacement_px - median_displacement_px, axis=1)
        if len(displacement_px)
        else np.empty(0)
    )
    metrics = {
        "source": str(args.source),
        "target": str(args.target),
        "center_xy_m": list(center_xy_m),
        "analysis_epsg": args.analysis_epsg,
        "tile_shape_px": [args.pixels, args.pixels],
        "pixel_size_m": args.pixel_size_m,
        "device": device.type,
        "pretrained": args.pretrained,
        "source_valid_fraction": float(source_valid.mean()),
        "target_valid_fraction": float(target_valid.mean()),
        "raw_match_count": int(len(source_px)),
        "source_8x8_coverage": occupied_fraction(source_px, args.pixels),
        "target_8x8_coverage": occupied_fraction(target_px, args.pixels),
        "confidence_median": float(np.median(confidence)) if len(confidence) else None,
        "confidence_p10": float(np.percentile(confidence, 10)) if len(confidence) else None,
        "median_displacement_px": median_displacement_px.tolist(),
        "median_displacement_image_axes_m": (
            median_displacement_px * args.pixel_size_m
        ).tolist(),
        "median_displacement_epsg3413_m": [
            float(median_displacement_px[0] * args.pixel_size_m),
            float(-median_displacement_px[1] * args.pixel_size_m),
        ],
        "residual_to_median_px_median": float(np.median(residual_px)) if len(residual_px) else None,
        "residual_to_median_px_p90": float(np.percentile(residual_px, 90)) if len(residual_px) else None,
        "matching_seconds": matching_seconds,
        "interpretation": "Unfiltered matcher output; cell selection is used only to declutter the visual.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.output_dir / "matches.npz",
        source_px=source_px,
        target_px=target_px,
        confidence=confidence,
    )

    shown = display_indices(source_px, confidence, args.pixels)
    canvas = np.concatenate((source, target), axis=1)
    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    axis.imshow(canvas, cmap="gray", vmin=0, vmax=255)
    for index in shown:
        axis.plot(
            [source_px[index, 0], target_px[index, 0] + args.pixels],
            [source_px[index, 1], target_px[index, 1]],
            color=plt.cm.viridis(float(confidence[index])),
            alpha=0.65,
            linewidth=0.55,
        )
    axis.axvline(args.pixels - 0.5, color="white", linewidth=1.0)
    axis.set_title(
        f"LoFTR on standard VAE imagery: {len(source_px):,} raw matches; "
        f"{len(shown):,} spatially balanced matches shown"
    )
    axis.text(8, 20, "source", color="white", fontsize=11, va="top")
    axis.text(args.pixels + 8, 20, "target", color="white", fontsize=11, va="top")
    axis.set_axis_off()
    figure.savefig(args.output_dir / "matches.png", dpi=180)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
