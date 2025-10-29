# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from .utils import log_execution_time, logger

class Templates:
    """
    Manages the collection of tracking templates.
    
    This class encapsulates the xarray.DataArray holding template image data
    and provides methods to add, update, retrieve, and prune templates.
    """
    def __init__(self):
        """Initializes the Templates collection."""
        self.data = xr.DataArray(
            np.empty((0, 1, 1), dtype=np.uint8), # Placeholder size
            dims=("trajectory_id", "height", "width"),
            coords={
                "trajectory_id": np.array([], dtype=np.int64),
                "time": ("trajectory_id", np.array([], dtype='datetime64[ns]')),
                "height": np.arange(1),
                "width": np.arange(1)
            },
            name="template_data"
        )
        self._initialized = False
        logger.debug("Initialized empty Templates object.")

    @staticmethod
    @log_execution_time
    def _extract_from_img(points, img, hs, band=1):
        """
        Extracts square templates from an image centered at specified keypoints.
        (Internal static method)
        
        Parameters:
            hs (int): Half-size of the square template. Full size is (2*hs+1).
        """
        pc = ccrs.PlateCarree()
        nps = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
        x, y = points.geometry.x.values, points.geometry.y.values
        
        if points.crs == img.srs:
            colsrows = np.array(img.transform_points(x, y, DstToSrc=1)).T
        else:
            lon, lat, _ = pc.transform_points(nps, x, y).T
            colsrows = np.array(img.transform_points(lon, lat, DstToSrc=1)).T
        
        img_band = img[band]
        colsrows = colsrows.astype(int)
        
        templates = []
        ids = []

        for i, (c0, r0) in enumerate(colsrows):
            if r0-hs >= 0 and r0+hs+1 <= img_band.shape[0] and c0-hs >= 0 and c0+hs+1 <= img_band.shape[1]:
                ids.append(i)
                template = img_band[r0-hs:r0+hs+1, c0-hs:c0+hs+1]
                templates.append(template)
            else:
                logger.warning(f"Keypoint at index {i} is out of image bounds for template extraction.")
        
        return np.array(templates), np.array(ids)

    def _initialize_data_array(self, first_template_batch):
        """Initializes the DataArray shape based on the first batch of templates."""
        height, width = first_template_batch.shape[1], first_template_batch.shape[2]
        self.data = self.data.reindex(height=np.arange(height), width=np.arange(width))
        self._initialized = True
        logger.info(f"Templates DataArray initialized with shape ({height}, {width}).")

    @log_execution_time
    def add(self, points, img, template_size, band=1):
        """
        Extracts and adds new templates for points that are not yet tracked.
        """
        templates_new, template_new_idx = self._extract_from_img(points, img, hs=template_size, band=band)
        
        if templates_new.size == 0:
            return

        trajectory_ids = points.trajectory_id.iloc[template_new_idx].values
        existing_trajectory_ids = self.data.trajectory_id.values

        is_new = ~np.isin(trajectory_ids, existing_trajectory_ids)
        if np.any(is_new):
            new_templates_data = templates_new[is_new]
            new_trajectory_ids = trajectory_ids[is_new]
            
            if not self._initialized:
                self._initialize_data_array(new_templates_data)

            height, width = new_templates_data.shape[1], new_templates_data.shape[2]

            new_data_array = xr.DataArray(
                data=new_templates_data,
                dims=("trajectory_id", "height", "width"),
                coords={
                    "trajectory_id": new_trajectory_ids,
                    "time": ("trajectory_id", points['time'].iloc[template_new_idx[is_new]].values),
                    "height": np.arange(height),
                    "width": np.arange(width),
                },
            )
            
            self.data = xr.concat([self.data, new_data_array], dim="trajectory_id")
            logger.debug(f"Added {len(new_trajectory_ids)} new templates.")

    @log_execution_time
    def update(self, points, img, template_size, band=1):
        """
        Extracts and updates templates with a strict "all or nothing" policy.

        If a template cannot be extracted for every point, the entire update
        is skipped to ensure data consistency.
        """
        templates_new, template_new_idx = self._extract_from_img(points, img, hs=template_size, band=band)

        if len(template_new_idx) != len(points):
            logger.warning(
                f"Template extraction mismatch (expected {len(points)}, got {len(template_new_idx)}). "
                "Skipping batch update."
            )
            return

        trajectory_ids = points.trajectory_id.iloc[template_new_idx].values
        current_times = points['time'].iloc[template_new_idx].values

        self.data['time'].loc[dict(trajectory_id=trajectory_ids)] = current_times
        self.data.loc[dict(trajectory_id=trajectory_ids)] = templates_new

        logger.debug(f"Updated {len(trajectory_ids)} templates.")

    def get_by_id(self, trajectory_ids):
        """
        Returns a selection of templates from self.data based on trajectory_ids.
        """
        return self.data.sel(trajectory_id=trajectory_ids)

    def prune(self, active_trajectory_ids, time_threshold):
        """
        Prunes the templates collection based on activity and time.
        """
        if self.data.trajectory_id.size == 0:
            return

        num_before = self.data.trajectory_id.size

        # Time-based Mask
        keep_mask_time = self.data['time'] >= time_threshold

        # Activity-based Mask
        keep_mask_active = np.isin(self.data['trajectory_id'].values, active_trajectory_ids)

        # Combine Masks (Keep if Recent AND Active)
        final_keep_mask = keep_mask_time & keep_mask_active

        if not final_keep_mask.all():
            self.data = self.data.where(final_keep_mask, drop=True)
            num_after = self.data.trajectory_id.size
            logger.debug(f"Pruning complete: Kept {num_after} templates / Removed {num_before - num_after} templates.")
        else:
            logger.debug("Pruning check: All templates are active and recent, none removed.")

    @property
    def trajectory_ids(self):
        """Returns a numpy array of all trajectory IDs in the collection."""
        return self.data.trajectory_id.values

    def __len__(self):
        """Returns the number of templates in the collection."""
        return len(self.data.trajectory_id)
