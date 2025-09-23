# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

"""
catalog.py

File handling functions for preparing remote sensing data for processing by limosat
"""

import os
import re
from datetime import datetime, timezone
from .image import Image
import geopandas as gpd
import glob
import concurrent.futures
import shapely
import hashlib  # added

def process_single_image(file):
    """
    Process a single image file to extract metadata and bounds.

    Parameters:
    - file: Path to the image file
    - verbose: Whether to print error details

    Returns:
    - Dictionary with image metadata and bounds, or None if processing fails
    """

    base_name = os.path.basename(file)
    pattern = (
        r"S1[AB]_EW_GRDM_1SDH_"  # Sentinel-1 mission identifiers
        r"(\d{8}T\d{6})_"  # Start time
        r"(\d{8}T\d{6})_"  # End time
        r"(\d{6})_"  # Orbit number
        r"(\w{6})_"  # Mission data take ID
        r"(\w{4})"  # Product unique ID
    )
    match = re.match(pattern, base_name)

    if match:
        start_time_str, end_time_str, orbit_num, mission_data_take_id, unique_id = match.groups()
        start_time = datetime.strptime(start_time_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

        # Get image bounds
        img = Image(file)
        polygon = shapely.from_geojson(img.get_border_geojson())
        img_bounds_4326 = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')
        # Reproject to EPSG:3413
        img_bounds_3413 = img_bounds_4326.to_crs(epsg=3413)
        bounds = img_bounds_3413.geometry.bounds.iloc[0]
        minx = bounds['minx']
        miny = bounds['miny']
        maxx = bounds['maxx']
        maxy = bounds['maxy']

        return {
            'filename': file,
            'timestamp': start_time,
            'minx': minx,
            'miny': miny,
            'maxx': maxx,
            'maxy': maxy,
            'orbit_num': orbit_num,
            'geometry': img_bounds_3413.geometry.iloc[0],
        }

    return None

# NEW: stable geometry hash helper
def _geom_sha1(geom):
    try:
        if geom is None:
            return None
        # shapely 2.x: to_wkb is the stable way; returns bytes when hex=False
        wkb = shapely.to_wkb(geom, hex=False)
        return hashlib.sha1(wkb).hexdigest()
    except Exception:
        return None

def create_image_gdf(idir, max_workers=None, deduplicate=True):
    """
    Create a GeoDataFrame with metadata and geometry from one or more sources.

    Parameters:
    - idir: Directory path, glob pattern, or an iterable of directory paths / glob patterns
    - max_workers: Number of workers (None uses system default)
    - deduplicate: If True, drop duplicate scenes by (geom_hash, timestamp) if available, else by (geom_hash)

    Returns:
    - GeoDataFrame with metadata and geometry
    """
    # Get all files and sort
    def extract_s1_date(filename):
        match = re.search(r"(\d{8}T\d{6})", os.path.basename(filename))
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S") if match else datetime.min

    # Collect files from single dir/pattern or list of dirs/patterns; include .tif and .tiff
    if isinstance(idir, (list, tuple, set)):
        search_items = list(idir)
    else:
        search_items = [idir]

    files = []
    for item in search_items:
        if os.path.isdir(item):
            files.extend(glob.glob(os.path.join(item, "*.tif")))
            files.extend(glob.glob(os.path.join(item, "*.tiff")))
        else:
            # Treat as a glob pattern (supports '**' when recursive)
            files.extend(glob.glob(item, recursive=True))

    # Deduplicate and keep only .tif/.tiff
    files = [f for f in set(files) if f.lower().endswith((".tif", ".tiff"))]
    files = sorted(files, key=extract_s1_date)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use map to process files in parallel
        results = list(executor.map(process_single_image, files))

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    polygons = [r['geometry'] for r in valid_results]

    # Create GeoDataFrame
    image_gdf = gpd.GeoDataFrame(valid_results, geometry=polygons, crs="EPSG:3413")

    # Sort by timestamp
    image_gdf = image_gdf.sort_values('timestamp').reset_index(drop=True)

    # NEW: de-duplication by geometry (and timestamp if present)
    if deduplicate and not image_gdf.empty and 'geometry' in image_gdf.columns:
        gdf = image_gdf[image_gdf.geometry.notnull()].copy()
        gdf['geom_hash'] = gdf.geometry.apply(_geom_sha1)

        # Build subset key
        if 'timestamp' in gdf.columns:
            subset_key = ['geom_hash', 'timestamp']
        else:
            subset_key = ['geom_hash']

        # Keep first occurrence
        before = len(gdf)
        gdf = gdf.drop_duplicates(subset=subset_key, keep='first')
        gdf = gdf.drop(columns=['geom_hash'])

        # Reassign
        image_gdf = gdf.reset_index(drop=True)

    return image_gdf

def build_geojson(sources, out_geojson, max_workers=None, deduplicate=True):
    """
    Build a GeoJSON by scanning one or more directories or glob patterns.

    Parameters:
    - sources: Directory path, glob pattern, or iterable of them
    - out_geojson: Output GeoJSON file path
    - max_workers: Process pool size
    - deduplicate: Pass-through to create_image_gdf

    Returns:
    - The GeoDataFrame that was written to GeoJSON
    """
    gdf = create_image_gdf(sources, max_workers=max_workers, deduplicate=deduplicate)
    # Ensure parent dir exists if needed
    os.makedirs(os.path.dirname(out_geojson) or ".", exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    return gdf