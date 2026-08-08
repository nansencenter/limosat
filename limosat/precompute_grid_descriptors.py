#!/usr/bin/env python3
"""
Precompute gridded descriptors cache for a catalog of processed Sentinel-1 images.

Usage:
  python3 -m limosat.precompute_grid_descriptors /path/to/config.yaml
"""

import argparse
import os
from pathlib import Path
import sys
import yaml

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import cv2
import geopandas as gpd

# Allow direct execution: ensure package root is on sys.path.
if __package__ is None:
    pkg_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(pkg_root))

from limosat.image import Image
from limosat.keypoint_detector import KeypointDetector
from limosat.utils import extract_date


def _build_orb_model(kp_detector_config):
    orb_params = kp_detector_config.get("orb_params", {})
    return cv2.ORB_create(
        nfeatures=orb_params.get("nfeatures", 100),
        scaleFactor=orb_params.get("scaleFactor", 1.25),
        nlevels=orb_params.get("nlevels", 5),
        edgeThreshold=orb_params.get("edgeThreshold", 16),
        firstLevel=orb_params.get("firstLevel", 0),
        patchSize=orb_params.get("patchSize", 64),
        scoreType=getattr(
            cv2,
            f"ORB_{orb_params.get('scoreType', 'HARRIS_SCORE').upper()}",
            cv2.ORB_HARRIS_SCORE,
        ),
    )


class _CacheProbe:
    def __init__(self, filename):
        self.filename = filename
        self.date = extract_date(filename)


def _resolve_paths(catalog_path, gdf):
    path_col = "filepath" if "filepath" in gdf.columns else "filename"
    if path_col not in gdf.columns:
        raise KeyError("Catalog missing 'filepath' or 'filename' column.")
    base_dir = Path(catalog_path).parent
    paths = []
    for val in gdf[path_col].dropna().astype(str).tolist():
        p = Path(val)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        paths.append(str(p))
    return paths


def _worker(args):
    (
        filepath,
        kp_detector_config,
        stride,
        octave,
        border_size,
        cache_dir,
        skip_existing,
    ) = args

    cv2.setNumThreads(1)
    model = _build_orb_model(kp_detector_config)
    detector = KeypointDetector(model=model, cache_dir=cache_dir)

    if skip_existing and cache_dir:
        probe = _CacheProbe(filepath)
        cache_path = detector._grid_cache_key(probe, stride, border_size, octave)
        if cache_path and cache_path.exists():
            return "cached", filepath

    if not Path(filepath).exists():
        return "missing", filepath

    try:
        img = Image(filepath)
        detector.detect_gridded_points(
            img=img,
            stride=stride,
            octave=octave,
            border_size=border_size,
        )
        return "ok", filepath
    except Exception as e:
        return f"error:{type(e).__name__}", filepath


def main():
    parser = argparse.ArgumentParser(
        description="Precompute gridded descriptor cache for LiMOSAT."
    )
    parser.add_argument("config", type=str, help="Path to config YAML.")
    parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Override catalog path (geojson). Defaults to config.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of worker processes (default: half of CPU cores).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit.")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for slicing the catalog.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute even if cache exists.",
    )
    parser.add_argument(
        "--start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    kp_detector_config = config.get("keypoint_detector", {})
    img_proc_params = config.get("image_processor_params", {})

    cache_dir = kp_detector_config.get("grid_cache_dir")
    if not cache_dir:
        print("grid_cache_dir is not set in config. Exiting.", file=sys.stderr)
        return 2

    catalog_path = args.catalog or config.get("paths", {}).get("image_catalog", {}).get("metadata_path")
    if not catalog_path:
        print("Catalog path not provided and not found in config.", file=sys.stderr)
        return 2

    gdf = gpd.read_file(catalog_path)
    paths = _resolve_paths(catalog_path, gdf)

    if args.start:
        paths = paths[args.start:]
    if args.limit is not None:
        paths = paths[: args.limit]

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    paths = deduped

    stride = img_proc_params.get("stride")
    octave = img_proc_params.get("octave")
    border_size = img_proc_params.get("border_size")
    if stride is None or octave is None or border_size is None:
        print("Missing stride/octave/border_size in image_processor_params.", file=sys.stderr)
        return 2

    skip_existing = not args.no_skip_existing
    def task_iter():
        for p in paths:
            yield (
                p,
                kp_detector_config,
                stride,
                octave,
                border_size,
                cache_dir,
                skip_existing,
            )

    ctx = mp.get_context(args.start_method)
    ok = cached = missing = errors = 0

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(tasks), unit="img")
    except Exception:
        progress = None

    with ProcessPoolExecutor(max_workers=args.max_workers, mp_context=ctx) as ex:
        for status, _ in ex.map(_worker, task_iter(), chunksize=8):
            if status == "ok":
                ok += 1
            elif status == "cached":
                cached += 1
            elif status == "missing":
                missing += 1
            else:
                errors += 1
            if progress:
                progress.update(1)

    if progress:
        progress.close()

    print(f"Done. ok={ok}, cached={cached}, missing={missing}, errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
