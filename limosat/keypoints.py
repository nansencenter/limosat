# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import numpy as np
import pandas as pd
import geopandas as gpd
from nansat import NSR
from .utils import log_execution_time


class Keypoints(gpd.GeoDataFrame):
    srs = NSR(3413)

    @staticmethod
    def _normalize_stopped(series):
        """Normalize stopped flags to non-null bool values."""
        return series.fillna(False).astype(bool)

    @staticmethod
    def _normalize_converged_to(series):
        """Normalize converged_to to nullable integer trajectory ids."""
        numeric = pd.to_numeric(series, errors='coerce')
        numeric = numeric.mask(numeric == -1, pd.NA)
        return numeric.astype('Int64')

    def __init__(self, *args, next_trajectory_id=0, **kwargs):
        if not args and not kwargs:
            # Initialize with empty data if no arguments provided
            empty_data = {
                'image_id': [],
                'is_last': [],
                'trajectory_id': [],
                'geometry': [],
                'descriptors': [],
                'angle': [],
                'corr': [],
                'time': [],
                'interpolated': [],
                'orbit_num': [],
                'stopped': [],
                'converged_to': [],
            }
            super().__init__(empty_data, geometry='geometry')
            if 'image_id' in self.columns:
                self['image_id'] = self['image_id'].astype('int32')
            if 'is_last' in self.columns:
                self['is_last'] = self['is_last'].astype('int32')
            if 'orbit_num' in self.columns:
                self['orbit_num'] = self['orbit_num'].astype('int32')
            if 'time' in self.columns:
                self['time'] = pd.to_datetime(self['time'], errors='coerce')
            if 'stopped' in self.columns:
                self['stopped'] = self._normalize_stopped(self['stopped'])
            if 'interpolated' in self.columns:
                self['interpolated'] = self['interpolated'].astype('int32')
            if 'trajectory_id' in self.columns:
                self['trajectory_id'] = self['trajectory_id'].astype('int64')
            if 'converged_to' in self.columns:
                self['converged_to'] = self._normalize_converged_to(self['converged_to'])
        else:
            # Initialize with provided data
            super().__init__(*args, **kwargs)
            # --- Enforce dtypes for existing data ---
            if 'image_id' in self.columns:
                self['image_id'] = self['image_id'].astype('int64', errors='ignore')
            if 'is_last' in self.columns:
                self['is_last'] = self['is_last'].astype('int64', errors='ignore')
            if 'trajectory_id' in self.columns:
                self['trajectory_id'] = self['trajectory_id'].astype('int64', errors='ignore')
            if 'stopped' in self.columns:
                self['stopped'] = self._normalize_stopped(self['stopped'])
            if 'converged_to' in self.columns:
                self['converged_to'] = self._normalize_converged_to(self['converged_to'])
            if 'interpolated' in self.columns:
                self['interpolated'] = self['interpolated'].astype('int64', errors='ignore')
            if 'orbit_num' in self.columns:
                self['orbit_num'] = self['orbit_num'].astype('int64', errors='ignore')

        self.srs = kwargs.get('srs', self.srs)
        max_id = self['trajectory_id'].max() if len(self) else None
        observed_next_id = 0 if pd.isna(max_id) else int(max_id) + 1
        self.attrs['_next_trajectory_id'] = max(observed_next_id, int(next_trajectory_id))

    @property
    def last_image_id(self):
        """Return the highest image_id or 0 if no valid image_ids exist."""
        return 0 if len(self) == 0 or 'image_id' not in self.columns else int(self['image_id'].max() or 0)

    @classmethod
    def _from_gdf(cls, gdf, *, next_trajectory_id=None):
        """Convert a GeoDataFrame to Keypoints"""
        if next_trajectory_id is None:
            next_trajectory_id = gdf.attrs.get('_next_trajectory_id', 0)
        return cls(gdf, crs=gdf.crs, next_trajectory_id=next_trajectory_id)

    def in_poly(self, img):
        points_poly = self[self.within(img.poly)]
        return self._from_gdf(points_poly)

    def last(self):
        last = self[self['is_last'] == 1]
        return self._from_gdf(last)
    
    @log_execution_time
    def append(self, points):
        """
        Appends new points to the current GeoDataFrame.
        Assigns new trajectory_ids to the appended points.

        Args:
            points: GeoDataFrame of points to append

        Returns:
            Updated Keypoints object
        """
        if len(points) == 0:
            return self._from_gdf(self)

        # Ensure trajectory_ids are integers in self
        if 'trajectory_id' in self.columns:
            self['trajectory_id'] = self['trajectory_id'].astype('int64', errors='ignore')

        next_trajectory_id = self.attrs['_next_trajectory_id']
        points['trajectory_id'] = list(range(next_trajectory_id, next_trajectory_id + len(points)))

        # Ensure columns match before concatenation
        missing_cols = set(self.columns) - set(points.columns)
        for col in missing_cols:
            points[col] = None

        # Concatenate the new points
        # **Exclude columns that are all NA to avoid FutureWarning**
        points_to_concat = points[self.columns].dropna(axis=1, how='all')

        # Concatenate the new points
        result = pd.concat([self, points_to_concat], ignore_index=True)

        return self._from_gdf(
            result,
            next_trajectory_id=next_trajectory_id + len(points),
        )

    @log_execution_time
    def update(self, points):
        """Optimized update method using vectorized operations"""
        if len(points) == 0:
            return self._from_gdf(self)

        # Vectorized update of is_last flag
        mask = self['trajectory_id'].isin(points['trajectory_id'])
        if mask.any():
            self.loc[mask, 'is_last'] = 0
            result = pd.concat([self, points], ignore_index=True)
        else:
            # No matches to update, just concatenate
            result = pd.concat([self, points], ignore_index=True)

        # Return the updated Keypoints object
        return self._from_gdf(result, next_trajectory_id=self.attrs['_next_trajectory_id'])

    @classmethod
    def create(cls, keypoints, descriptors, img, image_id, orbit_num):
        xy = img.transform_points(keypoints[:, 0], keypoints[:, 1], DstToSrc=0, dst_srs=cls.srs)
        N = len(xy[0])
        # Some sensors do not expose an orbit number. Keep the integer schema
        # and give each such image a distinct value so valid cross-image
        # matching is not suppressed by the same-orbit filter.
        orbit_value = int(orbit_num) if orbit_num is not None else -int(image_id)
        new_data = {
            'image_id': np.full(N, image_id, dtype=np.int32),
            'is_last': np.ones(N, dtype=np.int32),
            'trajectory_id': range(N),
            'geometry': gpd.points_from_xy(xy[0], xy[1]),
            'descriptors': list(descriptors),
            'angle': [img.angle] * N,
            'corr': np.zeros(N),
            'time': [img.date] * N,
            'interpolated': np.zeros(N, dtype=np.int32),
            'orbit_num': np.full(N, orbit_value, dtype=np.int32),
            'stopped': np.zeros(N, dtype=bool),
            'converged_to': [None] * N,
        }
        return cls(new_data)
