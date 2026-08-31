#!/usr/bin/env python3
"""Compare duplicate Sentinel-1 publications before and after frozen preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal


gdal.UseExceptions()
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results/iabp_s1_stratified_coverage"


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_zip_members(path: Path) -> dict[str, tuple[int, int]]:
    with zipfile.ZipFile(path) as archive:
        result = {}
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = Path(member.filename).parts
            normalized = "/".join(parts[1:] if len(parts) > 1 else parts)
            result[normalized] = (member.file_size, member.CRC)
        return result


def compare_zip_archives(primary: Path, repeat: Path) -> dict[str, object]:
    primary_members = normalized_zip_members(primary)
    repeat_members = normalized_zip_members(repeat)
    all_names = set(primary_members) | set(repeat_members)
    differing = {
        name
        for name in all_names
        if primary_members.get(name) != repeat_members.get(name)
    }
    measurement_differences = {
        name for name in differing if "/measurement/" in f"/{name.lower()}"
    }
    primary_hash = sha256_file(primary)
    repeat_hash = sha256_file(repeat)
    return {
        "primary_raw_bytes": primary.stat().st_size,
        "repeat_raw_bytes": repeat.stat().st_size,
        "primary_raw_sha256": primary_hash,
        "repeat_raw_sha256": repeat_hash,
        "raw_zip_sha256_equal": primary_hash == repeat_hash,
        "normalized_member_names_equal": set(primary_members) == set(repeat_members),
        "normalized_members_with_size_or_crc_difference": len(differing),
        "measurement_members_with_size_or_crc_difference": len(
            measurement_differences
        ),
    }


def gcp_max_difference(left, right) -> float:
    if len(left) != len(right):
        return np.nan
    if not left:
        return 0.0
    differences = []
    for first, second in zip(left, right):
        differences.extend(
            [
                abs(first.GCPPixel - second.GCPPixel),
                abs(first.GCPLine - second.GCPLine),
                abs(first.GCPX - second.GCPX),
                abs(first.GCPY - second.GCPY),
                abs(first.GCPZ - second.GCPZ),
            ]
        )
    return float(max(differences))


def compare_vae_rasters(
    primary: Path, repeat: Path, block_rows: int = 512
) -> dict[str, object]:
    left = gdal.Open(str(primary), gdal.GA_ReadOnly)
    right = gdal.Open(str(repeat), gdal.GA_ReadOnly)
    left_shape = (left.RasterYSize, left.RasterXSize, left.RasterCount)
    right_shape = (right.RasterYSize, right.RasterXSize, right.RasterCount)
    result: dict[str, object] = {
        "primary_vae_bytes": primary.stat().st_size,
        "repeat_vae_bytes": repeat.stat().st_size,
        "raster_shapes_equal": left_shape == right_shape,
        "primary_raster_shape": "x".join(map(str, left_shape)),
        "repeat_raster_shape": "x".join(map(str, right_shape)),
        "gcp_counts_equal": left.GetGCPCount() == right.GetGCPCount(),
        "maximum_gcp_coordinate_difference": gcp_max_difference(
            left.GetGCPs(), right.GetGCPs()
        ),
    }
    if left_shape != right_shape or left.RasterCount < 2:
        return result | {"raster_comparison_complete": False}

    difference_histogram = np.zeros(256, dtype=np.int64)
    valid_count = identical_count = 0
    sum_left = sum_right = sum_left_sq = sum_right_sq = sum_product = 0.0
    mask_count = mask_disagreement_count = 0
    for row_start in range(0, left.RasterYSize, block_rows):
        height = min(block_rows, left.RasterYSize - row_start)
        left_fused = left.GetRasterBand(1).ReadAsArray(
            0, row_start, left.RasterXSize, height
        ).astype(np.float64)
        right_fused = right.GetRasterBand(1).ReadAsArray(
            0, row_start, right.RasterXSize, height
        ).astype(np.float64)
        left_mask = left.GetRasterBand(2).ReadAsArray(
            0, row_start, left.RasterXSize, height
        )
        right_mask = right.GetRasterBand(2).ReadAsArray(
            0, row_start, right.RasterXSize, height
        )
        valid = (left_mask < 2) & (right_mask < 2)
        first = left_fused[valid]
        second = right_fused[valid]
        absolute_difference = np.abs(first - second).astype(np.uint8)
        difference_histogram += np.bincount(
            absolute_difference, minlength=256
        ).astype(np.int64)
        valid_count += len(first)
        identical_count += int(np.count_nonzero(first == second))
        sum_left += float(first.sum())
        sum_right += float(second.sum())
        sum_left_sq += float(np.square(first).sum())
        sum_right_sq += float(np.square(second).sum())
        sum_product += float((first * second).sum())
        mask_count += left_mask.size
        mask_disagreement_count += int(np.count_nonzero(left_mask != right_mask))

    cumulative = np.cumsum(difference_histogram)
    p99_index = int(np.searchsorted(cumulative, 0.99 * max(valid_count, 1)))
    mean_absolute_difference = float(
        np.dot(np.arange(256), difference_histogram) / max(valid_count, 1)
    )
    covariance = sum_product - (sum_left * sum_right / max(valid_count, 1))
    left_variance = sum_left_sq - (sum_left * sum_left / max(valid_count, 1))
    right_variance = sum_right_sq - (sum_right * sum_right / max(valid_count, 1))
    denominator = np.sqrt(max(left_variance * right_variance, 0.0))
    ncc = float(covariance / denominator) if denominator > 0 else np.nan
    return result | {
        "raster_comparison_complete": True,
        "valid_intersection_pixels": valid_count,
        "fused_identical_fraction": identical_count / max(valid_count, 1),
        "fused_mean_absolute_difference": mean_absolute_difference,
        "fused_p99_absolute_difference": p99_index,
        "fused_maximum_absolute_difference": int(
            np.flatnonzero(difference_histogram)[-1]
            if difference_histogram.any()
            else 0
        ),
        "fused_ncc": ncc,
        "mask_disagreement_fraction": (
            mask_disagreement_count / max(mask_count, 1)
        ),
    }


def analyze_controls(controls: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for item in controls.itertuples(index=False):
        primary_raw = Path(item.primary_raw_zip_path)
        repeat_raw = Path(item.repeat_raw_zip_path)
        primary_vae = Path(item.primary_standard_vae_path)
        repeat_vae = Path(item.repeat_standard_vae_path)
        paths_present = all(
            path.is_file()
            for path in [primary_raw, repeat_raw, primary_vae, repeat_vae]
        )
        record: dict[str, object] = {
            "repeat_control_id": item.repeat_control_id,
            "primary_product_name": item.primary_product_name,
            "repeat_product_name": item.repeat_product_name,
            "all_inputs_present": paths_present,
        }
        if paths_present:
            record.update(compare_zip_archives(primary_raw, repeat_raw))
            record.update(compare_vae_rasters(primary_vae, repeat_vae))
        records.append(record)
    return pd.DataFrame.from_records(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    controls = pd.read_csv(args.results_dir / "full70_repeat_publication_controls.csv")
    results = analyze_controls(controls)
    results.to_csv(
        args.results_dir / "full70_repeat_publication_control_results.csv", index=False
    )
    summary = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "controls": len(results),
        "complete_controls": int(results["all_inputs_present"].sum()),
        "raw_zip_sha256_equal_controls": int(
            results.get("raw_zip_sha256_equal", pd.Series(dtype=bool))
            .fillna(False)
            .sum()
        ),
        "pixel_identical_controls": int(
            results.get("fused_identical_fraction", pd.Series(dtype=float))
            .eq(1.0)
            .sum()
        ),
    }
    (args.results_dir / "full70_repeat_publication_control_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    complete = summary["complete_controls"] == summary["controls"]
    return 0 if complete or args.allow_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
