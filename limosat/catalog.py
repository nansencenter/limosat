import os
import re
import json
from datetime import datetime, timezone
from typing import Iterable, Optional, Literal, List, Dict, Any
import geopandas as gpd
import pystac
import shapely

# Sentinel-1 filename pattern:
# S1B_EW_GRDM_1SDH_20200101T015602_20200101T015706_019617_025132_32F2.tiff
_S1_PATTERN = re.compile(
    r"^S1[AB]_EW_GRDM_1SDH_"
    r"(?P<start>\d{8}T\d{6})_"
    r"(?P<end>\d{8}T\d{6})_"
    r"(?P<orbit>\d{6})_"
    r"(?P<take>\w{6})_"
    r"(?P<uid>\w{4})"
    r"\.tiff$"
)

def _parse_s1_meta(path: str):
    """Parse S1 filename to extract start datetime (UTC), scene id, product UID, and orbit number."""
    base = os.path.basename(path)
    m = _S1_PATTERN.match(base)
    if not m:
        return None, None, None, None
    dt = datetime.strptime(m.group("start"), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    scene_id = os.path.splitext(base)[0]
    uid = m.group("uid")
    orbit = int(m.group("orbit"))
    return dt, scene_id, uid, orbit


GeometryMode = Literal["none", "bbox"]

def build_stac_item_collection(
    files: Iterable[str],
    out_path: Optional[str] = None,
    check_exists: bool = False,
) -> pystac.ItemCollection:
    """
    Build a STAC ItemCollection. If out_path is provided,
    also write a single JSON file.
    """
    files = list(files)
    if not files:
        raise ValueError("No files provided to build_stac_item_collection.")

    records: List[Dict[str, Any]] = []
    seen_scene_paths: Dict[str, str] = {}

    for p in files:
        if check_exists and not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

        dt, scene_id, uid, orbit = _parse_s1_meta(p)
        if dt is None or scene_id is None:
            # Skip non-matching filenames.
            continue
        previous_path = seen_scene_paths.get(scene_id)
        if previous_path is not None:
            raise ValueError(
                f"Duplicate Sentinel-1 scene id '{scene_id}' in catalog input: "
                f"{previous_path} and {p}"
            )
        seen_scene_paths[scene_id] = p

        records.append(
            {
                "path": p,
                "basename": os.path.basename(p),
                "dt": dt,
                "scene_id": scene_id,
                "uid": uid,
                "orbit": orbit,
            }
        )

    # Deterministic order for integer image_id assignment
    records.sort(key=lambda r: (r["dt"], r["basename"]))

    items: List[pystac.Item] = []
    for idx, rec in enumerate(records, start=1):
        it = pystac.Item(
            id=str(rec["scene_id"]),
            geometry=None,  # minimal: no geometry
            bbox=None,      # minimal: no bbox
            datetime=rec["dt"],
            properties={},
        )
        it.properties["image_id"] = int(idx)
        it.properties["scene_id"] = rec["scene_id"]
        it.properties["product_uid"] = rec["uid"]
        it.properties["filename"] = rec["basename"]
        it.properties["filepath"] = rec["path"]
        if rec.get("orbit") is not None:
            it.properties["orbit_num"] = int(rec["orbit"])
        it.add_asset(
            "image",
            pystac.Asset(href=rec["path"], media_type="image/tiff", roles=["data"]),
        )
        items.append(it)

    coll = pystac.ItemCollection(items)

    if out_path:
        tmp = f"{out_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(coll.to_dict(), f, ensure_ascii=False)
        os.replace(tmp, out_path)
    return coll


def compute_footprint_wgs84(path: str):
    """
    Compute image footprint in WGS84 as a shapely geometry.
    Separated from stac_item_collection_to_gdf so ProcessPoolExecutor can pickle it.
    """
    from limosat.image import Image
    gj = Image(path).get_border_geojson()  # WGS84 GeoJSON string
    return shapely.from_geojson(gj)


def stac_item_collection_to_gdf(
    coll: pystac.ItemCollection,
    target_crs: str = "EPSG:3413",
    max_workers: Optional[int] = None,
    chunksize: int = 64,
) -> gpd.GeoDataFrame:
    """
    Convert an in-memory STAC ItemCollection to a GeoDataFrame.
    - Computes exact footprints (optionally in parallel) to mimic previous geometry setup.
    - Reprojects to target_crs (default EPSG:3413).
    - Adds orbit_num (6-digit string) and per-row bounds minx, miny, maxx, maxy in target_crs.

    Columns:
        image_id, filename (basename), filepath (full path), timestamp,
        orbit_num, minx, miny, maxx, maxy, geometry
    """
    from concurrent.futures import ProcessPoolExecutor

    items = list(getattr(coll, "items", []))
    if not items:
        return gpd.GeoDataFrame(
            columns=[
                "image_id", "filename", "filepath", "timestamp",
                "orbit_num", "minx", "miny", "maxx", "maxy", "geometry"
            ],
            geometry="geometry",
            crs=target_crs,
        )

    props_list = [it.properties or {} for it in items]
    image_ids = [int(p.get("image_id")) for p in props_list]
    basenames = [p.get("filename") for p in props_list]
    filepaths = [p.get("filepath") for p in props_list]
    timestamps = [it.datetime for it in items]

    # Orbit number: keep as zero-padded 6-char string to match prior output style (e.g., "019617")
    raw_orbits = [p.get("orbit_num") for p in props_list]
    orbit_nums = []
    for o in raw_orbits:
        if o is None:
            orbit_nums.append(None)
        else:
            try:
                orbit_nums.append(f"{int(o):06d}")
            except Exception:
                # If not numeric, keep as string
                orbit_nums.append(str(o))

    # Compute footprints (WGS84). Parallelize if max_workers is provided.
    if max_workers and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            geoms = list(ex.map(compute_footprint_wgs84, filepaths, chunksize=chunksize))
    else:
        geoms = [compute_footprint_wgs84(p) for p in filepaths]

    # Build GeoDataFrame in WGS84, then reproject to target_crs
    gdf = gpd.GeoDataFrame(
        {
            "image_id": image_ids,
            "filename": basenames,
            "filepath": filepaths,
            "timestamp": timestamps,
            "orbit_num": orbit_nums,
            "geometry": geoms,
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    if target_crs:
        gdf = gdf.to_crs(target_crs)

    # Per-row bounds in target_crs
    b = gdf.geometry.bounds  # DataFrame with columns: minx, miny, maxx, maxy
    gdf["minx"] = b["minx"]
    gdf["miny"] = b["miny"]
    gdf["maxx"] = b["maxx"]
    gdf["maxy"] = b["maxy"]

    # Stable ordering
    gdf = gdf.sort_values("image_id").reset_index(drop=True)

    # Reorder columns to match your example (keeping filepath as an extra, useful field)
    ordered_cols = [
        "filename", "timestamp", "minx", "miny", "maxx", "maxy", "orbit_num",
        "image_id", "filepath", "geometry"
    ]
    # Only keep columns that exist (in case of schema drift)
    ordered_cols = [c for c in ordered_cols if c in gdf.columns]
    return gdf[ordered_cols]
