import importlib
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def load_database_class():
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    for name in list(sys.modules):
        if name == "limosat" or name.startswith("limosat."):
            del sys.modules[name]
    return importlib.import_module("limosat.database").DriftDatabase


def test_validation_export_ignores_index_named_like_existing_column(tmp_path):
    DriftDatabase = load_database_class()
    validation = gpd.GeoDataFrame(
        {
            "probe_id": ["a|1", "b|2"],
            "seed_time": [pd.Timestamp("2020-01-01", tz="UTC"), pd.NaT],
            "seed_kp_geometry": [Point(2, 2), None],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:3413",
        index=["a|1", "b|2"],
    )
    validation.index.name = "probe_id"
    database = DriftDatabase(
        run_name="fixture",
        validation_dir=tmp_path / "validation",
    )

    database._save_validation_metadata(validation)

    restored = gpd.read_file(tmp_path / "validation/fixture_validation.geojson")
    assert restored["probe_id"].tolist() == ["a|1", "b|2"]
    assert restored.loc[0, "seed_time"] == "2020-01-01T00:00:00+00:00"
    assert restored.loc[0, "seed_kp_geometry"] == "POINT (2 2)"
    assert len(restored) == 2
