"""Land raster for LiMOSAT QC: 1 km EPSG:3413 grid of inland distance (km; 0 at sea).

Build one raster per land-polygon source, then combine them so a cell is land only where every source
says land (QC v1 uses Natural Earth 10 m and OpenStreetMap land polygons: either alone removes landfast
ice, Natural Earth in Greenland fjords and OSM in some estuaries and lagoons)::

    python -m limosat.qc.land build --land ne_10m_land.shp --out land_ne_3413_1km.npz
    python -m limosat.qc.land build --land land_polygons.shp --out land_osm_3413_1km.npz
    python -m limosat.qc.land combine land_ne_3413_1km.npz land_osm_3413_1km.npz --out land_agree_3413_1km.npz

Polygons must be EPSG:4326. Inland distance is recomputed on the combined mask. The ``.npz`` holds
``inland_km``, ``x0`` (west edge), ``y1`` (north edge), ``res`` and the SHA-256 of each source, and is read
by ``limosat.qc.v3.LandRaster``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

RES = 1000.0
X0, Y1, NX, NY = -4_000_000.0, 4_500_000.0, 8000, 9000  # EPSG:3413 extent covering the Arctic archive
MIN_LATITUDE = 50.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def build(land: Path, out: Path) -> int:
    """Rasterise land polygons north of 50 N (cell centre inside a polygon) and write inland distance."""

    import shapefile
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from scipy.ndimage import distance_transform_edt
    from shapely.geometry import box, shape
    from shapely.ops import transform

    to_polar = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True).transform
    clip = box(-180, MIN_LATITUDE, 180, 90)
    geoms = []
    for g in shapefile.Reader(str(land)).iterShapes():
        if g.bbox[3] < MIN_LATITUDE:  # skip polygons wholly south of the clip before building geometry
            continue
        g = shape(g).buffer(0).intersection(clip)
        if not g.is_empty:
            geoms.append(transform(to_polar, g.segmentize(0.05)))  # densify so edges follow the projection
    mask = rasterize(((g, 1) for g in geoms), out_shape=(NY, NX), transform=from_origin(X0, Y1, RES, RES),
                     fill=0, dtype="uint8")
    inland = (distance_transform_edt(mask) * RES / 1000).astype("float32")
    np.savez_compressed(out, inland_km=inland, x0=X0, y1=Y1, res=RES, source_sha256=_sha256(Path(land)))
    return int(mask.sum())


def combine(rasters: list[Path], out: Path) -> int:
    """Land only where every raster says land; inland distance recomputed on the combined mask."""

    from scipy.ndimage import distance_transform_edt

    loaded = [np.load(p) for p in rasters]
    for r in loaded[1:]:
        for k in ("x0", "y1", "res"):
            if float(r[k]) != float(loaded[0][k]):
                raise ValueError(f"rasters differ in {k}")
    mask = np.logical_and.reduce([r["inland_km"] > 0 for r in loaded])
    inland = (distance_transform_edt(mask) * float(loaded[0]["res"]) / 1000).astype("float32")
    np.savez_compressed(out, inland_km=inland, x0=loaded[0]["x0"], y1=loaded[0]["y1"], res=loaded[0]["res"],
                        **{f"source{i}_sha256": _sha256(Path(p)) for i, p in enumerate(rasters)})
    return int(mask.sum())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build or combine LiMOSAT QC land rasters.")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="rasterise one land-polygon shapefile (EPSG:4326)")
    b.add_argument("--land", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    c = sub.add_parser("combine", help="land only where all rasters agree")
    c.add_argument("rasters", nargs="+", type=Path)
    c.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    cells = build(a.land, a.out) if a.command == "build" else combine(a.rasters, a.out)
    print(f"{a.out}: {cells} land cells")


if __name__ == "__main__":
    main()
