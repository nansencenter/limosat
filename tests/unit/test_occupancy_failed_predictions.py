import geopandas as gpd
import pandas as pd
import numpy as np
import pytest
from shapely.geometry import Point


@pytest.mark.unit
def test_occupancy_failed_predictions_substitution():
    # Simulate unmatched and failed_predictions mapping
    unmatched = gpd.GeoDataFrame(
        {
            'trajectory_id': [1, 2, 3],
        },
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs='EPSG:3413',
    )
    failed_predictions = pd.DataFrame(
        {
            'trajectory_id': [2, 99],
            'geometry_pred': [Point(9, 9), Point(8, 8)],
        }
    )

    mapping = dict(zip(failed_predictions['trajectory_id'], failed_predictions['geometry_pred']))
    # Apply substitution similar to ImageProcessor logic
    def sub_geom(row):
        return mapping.get(row['trajectory_id'], row['geometry'])

    unmatched['geometry'] = unmatched.apply(sub_geom, axis=1)

    assert unmatched.loc[unmatched['trajectory_id'] == 2, 'geometry'].iloc[0].equals(Point(9, 9))
    assert unmatched.loc[unmatched['trajectory_id'] == 1, 'geometry'].iloc[0].equals(Point(0, 0))
