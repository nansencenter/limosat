#!/usr/bin/env python3
"""Verify the standard-VAE GeoTIFF data contract for the full-70 experiment."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal


gdal.UseExceptions()
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results/iabp_s1_stratified_coverage"


def inspect_raster(path: Path, sample_size: int = 512) -> dict[str, object]:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        return {"raster_ok": False, "raster_error": "GDAL could not open raster"}
    band_types = [
        gdal.GetDataTypeName(dataset.GetRasterBand(index).DataType)
        for index in range(1, dataset.RasterCount + 1)
    ]
    result: dict[str, object] = {
        "raster_ok": False,
        "raster_error": None,
        "width_pixels": dataset.RasterXSize,
        "height_pixels": dataset.RasterYSize,
        "bands": dataset.RasterCount,
        "band_types": ";".join(band_types),
        "gcp_count": dataset.GetGCPCount(),
        "has_projection_or_gcps": bool(
            dataset.GetProjection() or dataset.GetGCPCount() > 0
        ),
    }
    if dataset.RasterCount != 2 or band_types != ["Byte", "Byte"]:
        result["raster_error"] = "expected two uint8 bands"
        return result
    fused = dataset.GetRasterBand(1).ReadAsArray(
        buf_xsize=min(sample_size, dataset.RasterXSize),
        buf_ysize=min(sample_size, dataset.RasterYSize),
    )
    mask = dataset.GetRasterBand(2).ReadAsArray(
        buf_xsize=min(sample_size, dataset.RasterXSize),
        buf_ysize=min(sample_size, dataset.RasterYSize),
    )
    mask_values = set(np.unique(mask).astype(int).tolist())
    result.update(
        {
            "sample_fused_min": int(fused.min()),
            "sample_fused_median": float(np.median(fused)),
            "sample_fused_max": int(fused.max()),
            "sample_fused_std": float(fused.std()),
            "sample_mask_values": ";".join(map(str, sorted(mask_values))),
            "sample_excluded_mask_fraction": float(np.mean(mask >= 2)),
            "sample_noncanonical_exclusion_values": ";".join(
                map(str, sorted(value for value in mask_values if value > 2))
            ),
        }
    )
    errors = []
    if fused.min() == fused.max():
        errors.append("fused band is constant in 512px sample")
    if not result["has_projection_or_gcps"]:
        errors.append("missing projection and GCPs")
    result["raster_error"] = "; ".join(errors) if errors else None
    result["raster_ok"] = not errors
    return result


def audit_outputs(inventory: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for item in inventory.itertuples(index=False):
        path = Path(item.standard_vae_output_path)
        row: dict[str, object] = {
            "image_id": item.image_id,
            "resolved_product_name": item.resolved_product_name,
            "standard_vae_output_path": str(path),
            "present": path.is_file(),
            "file_bytes": path.stat().st_size if path.is_file() else None,
            "raster_ok": None,
            "raster_error": None,
        }
        if path.is_file():
            row.update(inspect_raster(path))
        rows.append(row)
    missing = [row["resolved_product_name"] for row in rows if not row["present"]]
    bad = [
        row["resolved_product_name"]
        for row in rows
        if row["present"] and not row["raster_ok"]
    ]
    noncanonical_exclusions = [
        row["resolved_product_name"]
        for row in rows
        if row.get("sample_noncanonical_exclusion_values")
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_rasters": len(rows),
        "present_rasters": sum(bool(row["present"]) for row in rows),
        "valid_present_rasters": sum(row["raster_ok"] is True for row in rows),
        "missing_rasters": missing,
        "bad_rasters": bad,
        "rasters_with_noncanonical_exclusion_values": noncanonical_exclusions,
        "contract": {
            "preprocessing": "balanced_q2q98_clahe25",
            "bands": "band 1 fused uint8; band 2 mask uint8",
            "mask_values": "0 and 1 usable; values >=2 excluded",
            "geolocation": "projection or GCPs required",
        },
        "complete": not missing and not bad,
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    inventory = pd.read_csv(args.results_dir / "full70_sentinel1_download_inventory.csv")
    summary, rows = audit_outputs(inventory)
    (args.results_dir / "full70_standard_vae_verification.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (args.results_dir / "full70_standard_vae_verification.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    acceptable = not summary["bad_rasters"] and (
        summary["complete"] or (args.allow_missing and summary["present_rasters"] > 0)
    )
    return 0 if acceptable else 1


if __name__ == "__main__":
    raise SystemExit(main())
