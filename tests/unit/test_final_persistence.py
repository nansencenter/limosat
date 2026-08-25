import importlib
import sys
from pathlib import Path
from unittest.mock import Mock

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def test_final_persistence_excludes_singleton_trajectories():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    for name in list(sys.modules):
        if name == "limosat" or name.startswith("limosat."):
            del sys.modules[name]
    ImageProcessor = importlib.import_module("limosat.image_processor").ImageProcessor
    Keypoints = importlib.import_module("limosat.keypoints").Keypoints

    points = Keypoints._from_gdf(
        gpd.GeoDataFrame(
            {
                "image_id": [1, 3, 3],
                "trajectory_id": [10, 10, 20],
                "is_last": [0, 1, 1],
                "time": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-03"]),
            },
            geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
            crs="EPSG:3413",
        )
    )
    processor = ImageProcessor.__new__(ImageProcessor)
    processor.persist_updates = True
    processor.points = points
    processor.templates = Mock()
    processor.insitu_points = None
    processor._last_persisted_id = 1
    processor.db = Mock()
    processor.db.save.return_value = True

    processor.ensure_final_persistence()

    saved_points = processor.db.save.call_args.kwargs["points"]
    assert saved_points["trajectory_id"].unique().tolist() == [10]
    assert processor._last_persisted_id == 3
