#!/usr/bin/env python3
"""Intersect ICESat-2 CMR footprints with a SAR displacement-field footprint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from shapely import make_valid
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import transform, unary_union


DATA_LINK_RELATION = "http://esipfed.org/ns/fedsearch/1.1/data#"


def parse_cmr_polygon(text: str) -> Polygon:
    """Parse CMR's ``lat lon ...`` polygon encoding."""
    values = [float(value) for value in text.split()]
    if len(values) < 6 or len(values) % 2:
        raise ValueError("CMR polygon must contain at least three lat/lon pairs")
    coordinates = [(values[i + 1], values[i]) for i in range(0, len(values), 2)]
    return Polygon(coordinates)


def entry_polygons(entry: dict) -> list[Polygon]:
    polygons = [
        parse_cmr_polygon(encoded)
        for group in entry.get("polygons", [])
        for encoded in group
    ]
    if not polygons:
        raise ValueError(f"No CMR polygons for {entry.get('producer_granule_id')}")
    return polygons


def data_link(entry: dict) -> str:
    links = [
        link["href"]
        for link in entry.get("links", [])
        if DATA_LINK_RELATION in link.get("rel", "")
        and not link.get("inherited", False)
    ]
    if not links:
        raise ValueError(f"No data link for {entry.get('producer_granule_id')}")
    return links[0]


def field_footprint(field: pd.DataFrame, spacing_m: float | None = None) -> Polygon:
    required = {"source_x", "source_y"}
    missing = required.difference(field.columns)
    if missing:
        raise ValueError(f"Field is missing columns: {sorted(missing)}")
    support = field[field["available"].astype(bool)] if "available" in field else field
    if len(support) < 3:
        raise ValueError("At least three available field nodes are required")
    if spacing_m is None:
        x_steps = support["source_x"].drop_duplicates().sort_values().diff().dropna()
        y_steps = support["source_y"].drop_duplicates().sort_values().diff().dropna()
        steps = pd.concat([x_steps[x_steps.gt(0)], y_steps[y_steps.gt(0)]])
        spacing_m = float(steps.min()) if not steps.empty else 0.0
    points = MultiPoint(support[["source_x", "source_y"]].to_numpy())
    return points.convex_hull.buffer(spacing_m / 2, cap_style="square")


def build_manifest(
    entries: list[dict],
    sar_footprint_3413: Polygon,
    pair_start: pd.Timestamp,
    pair_end: pd.Timestamp,
) -> pd.DataFrame:
    if pair_start.tzinfo is None or pair_end.tzinfo is None:
        raise ValueError("Pair times must be timezone-aware")
    if pair_end <= pair_start:
        raise ValueError("Pair end must be after pair start")
    project = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True).transform
    records: list[dict] = []
    for entry in entries:
        start = pd.Timestamp(entry["time_start"])
        end = pd.Timestamp(entry["time_end"])
        midpoint = start + (end - start) / 2
        # Project each CMR polygon before unioning. Some polar footprints cross
        # the antimeridian and are invalid when interpreted in lon/lat space.
        footprint = unary_union(
            [make_valid(transform(project, polygon)) for polygon in entry_polygons(entry)]
        )
        overlap = footprint.intersection(sar_footprint_3413)
        if overlap.is_empty:
            continue
        inside_interval = start <= pair_end and end >= pair_start
        if midpoint < pair_start:
            time_offset_hours = (midpoint - pair_start).total_seconds() / 3600
        elif midpoint > pair_end:
            time_offset_hours = (midpoint - pair_end).total_seconds() / 3600
        else:
            time_offset_hours = 0.0
        records.append(
            {
                "granule_id": entry["producer_granule_id"],
                "time_start_utc": start.isoformat(),
                "time_end_utc": end.isoformat(),
                "midpoint_utc": midpoint.isoformat(),
                "inside_sar_interval": inside_interval,
                "time_offset_from_interval_hours": time_offset_hours,
                "footprint_overlap_km2": overlap.area / 1e6,
                "granule_size_mb": float(entry.get("granule_size", "nan")),
                "download_url": data_link(entry),
            }
        )
    columns = [
        "granule_id",
        "time_start_utc",
        "time_end_utc",
        "midpoint_utc",
        "inside_sar_interval",
        "time_offset_from_interval_hours",
        "footprint_overlap_km2",
        "granule_size_mb",
        "download_url",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["inside_sar_interval", "footprint_overlap_km2"], ascending=[False, False]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmr-json", type=Path, required=True)
    parser.add_argument("--field-csv", type=Path, required=True)
    parser.add_argument("--pair-start", required=True, help="UTC ISO-8601 time")
    parser.add_argument("--pair-end", required=True, help="UTC ISO-8601 time")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.cmr_json.read_text())
    entries = payload.get("feed", {}).get("entry", [])
    field = pd.read_csv(args.field_csv)
    manifest = build_manifest(
        entries,
        field_footprint(field),
        pd.Timestamp(args.pair_start),
        pd.Timestamp(args.pair_end),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_csv, index=False)
    print(manifest.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
