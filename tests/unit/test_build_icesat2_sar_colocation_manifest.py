import pandas as pd
from shapely.geometry import box

from experiments.build_icesat2_sar_colocation_manifest import (
    build_manifest,
    parse_cmr_polygon,
)


def test_cmr_polygon_coordinates_are_latitude_then_longitude():
    polygon = parse_cmr_polygon("80 -10 80 10 82 10 82 -10 80 -10")
    assert polygon.bounds == (-10.0, 80.0, 10.0, 82.0)


def test_manifest_keeps_spatial_and_temporal_intersection():
    entry = {
        "producer_granule_id": "ATL07_TEST.h5",
        "time_start": "2020-03-28T12:15:00Z",
        "time_end": "2020-03-28T12:20:00Z",
        "granule_size": "10.5",
        "polygons": [["85 -1 85 1 86 1 86 -1 85 -1"]],
        "links": [
            {
                "href": "https://example.test/ATL07_TEST.h5",
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
            }
        ],
    }
    # This projected box contains the test polygon near 85 N, 0 E.
    sar_footprint = box(300_000, -500_000, 500_000, -300_000)
    result = build_manifest(
        [entry],
        sar_footprint,
        pd.Timestamp("2020-03-28T12:13:29Z"),
        pd.Timestamp("2020-03-29T11:16:05Z"),
    )
    assert len(result) == 1
    assert bool(result.iloc[0]["inside_sar_interval"])
    assert result.iloc[0]["footprint_overlap_km2"] > 0
