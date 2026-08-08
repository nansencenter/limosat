import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.unit
def test_resume_allocates_above_database_wide_maximum(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    code = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path
        from unittest.mock import patch

        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import xarray as xr
        from shapely.geometry import Point
        from sqlalchemy import create_engine

        from limosat.database import DriftDatabase
        from limosat.image_processor import ImageProcessor
        from limosat.keypoints import Keypoints

        root = Path(sys.argv[1])
        db_path = root / "resume.sqlite"
        zarr_path = root / "resume.zarr"
        zarr_path.mkdir()
        run_name = "resume_test"
        engine = create_engine(f"sqlite:///{db_path}")
        pd.DataFrame(
            {
                "image_id": [1, 5],
                "is_last": [0, 1],
                "trajectory_id": [100, 10],
                "geometry": ["POINT (100 100)", "POINT (10 10)"],
                "descriptors": [None, json.dumps(np.zeros((1, 32), dtype=np.uint8).tolist())],
                "angle": [0.0, 0.0],
                "corr": [0.0, 0.5],
                "time": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-05")],
                "interpolated": [0, 0],
                "orbit_num": [1, 2],
                "stopped": [False, False],
                "converged_to": [None, None],
            }
        ).to_sql(run_name, engine, index=False)
        template_data = xr.DataArray(
            np.zeros((1, 1, 1), dtype=np.uint8),
            dims=("trajectory_id", "height", "width"),
            coords={"trajectory_id": [10], "height": [0], "width": [0]},
        )
        dataset = xr.Dataset({"template_data": template_data})
        db = DriftDatabase(engine=engine, zarr_path=str(zarr_path), run_name=run_name)

        with patch("limosat.database.xr.open_zarr") as open_zarr:
            open_zarr.return_value.__enter__.return_value = dataset
            open_zarr.return_value.__exit__.return_value = False
            points, _templates = db.prepare_run_state(
                clear_existing_data=False,
                temporal_window_days=4,
            )

        pruned = Keypoints._from_gdf(points[points["is_last"] == 1])
        new_points = gpd.GeoDataFrame(
            {
                "image_id": [6, 6],
                "is_last": [1, 1],
                "trajectory_id": [-1, -1],
                "time": [pd.Timestamp("2025-01-06")] * 2,
            },
            geometry=[Point(11, 11), Point(12, 12)],
            crs="EPSG:3413",
        )
        appended = pruned.append(new_points)
        processor = ImageProcessor(
            points=points,
            model=None,
            matcher=None,
            persist_updates=False,
        )

        assert points["trajectory_id"].tolist() == [10]
        assert processor._last_persisted_id == 5
        assert appended.iloc[-2:]["trajectory_id"].tolist() == [101, 102]
        assert appended.attrs["_next_trajectory_id"] == 103
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    proj_data = Path(sys.prefix) / "share" / "proj"
    env["PROJ_LIB"] = str(proj_data)
    env["PROJ_DATA"] = str(proj_data)
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=repo_root,
        env=env,
        check=True,
    )
