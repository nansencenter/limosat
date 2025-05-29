# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

"""
limosat Ice Drift Analysis Library

This library provides tools for analyzing ice drift using satellite imagery.
It includes functionality for keypoint detection, pattern matching, and trajectory analysis.
"""

import os
from functools import cached_property
from .utils import log_execution_time, extract_date, logger
from .database import DriftDatabase

import numpy as np
from numpy.linalg import norm
import pandas as pd
import geopandas as gpd
import xarray as xr
import cv2
import cartopy.crs as ccrs
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
from skimage.util import view_as_windows
from skimage.transform import AffineTransform
from nansat import Nansat, NSR

# Constants for commonly used values
MAX_DRIFT_SPEED_M_PER_DAY = 10000
WINDOW_SIZE = 64
BORDER_SIZE = 256
STRIDE = 10
OCTAVE = 8
MIN_CORRELATION = 0.35
TEMPLATE_SIZE = 16
TEMPORAL_WINDOW = 4

class ImageProcessor:
    """
    Image processor for ice drift analysis.
    """
    def __init__(self,
                 points,
                 templates,
                 model,
                 matcher,
                 engine=None,
                 zarr_path=None,
                 run_name=None,
                 insitu_points=None,
                 persist_updates=False,
                 persist_interval=100,
                 pruning_interval = 25,
                 temporal_window=TEMPORAL_WINDOW,
                 max_drift_speed_m_per_day=MAX_DRIFT_SPEED_M_PER_DAY,
                 window_size=WINDOW_SIZE,
                 border_size=BORDER_SIZE,
                 stride=STRIDE,
                 octave=OCTAVE,
                 min_correlation=MIN_CORRELATION,
                 use_interpolation=True):
        self.points = points
        self.templates = templates
        self.model = model
        self.matcher = matcher
        self.persist_updates = persist_updates
        self.persist_interval = persist_interval
        self.pruning_interval = pruning_interval
        self._last_persisted_id = 0
        self.min_correlation = min_correlation
        self.use_interpolation = use_interpolation
        self.run_name = run_name
        self.insitu_points = insitu_points
        self.window_size = window_size
        self.border_size = border_size
        self.temporal_window = temporal_window
        self.max_drift_speed_m_per_day = max_drift_speed_m_per_day
        self.stride = stride
        self.octave = octave
        
        # Initialize the KeypointDetector
        self.keypoint_detector = KeypointDetector(model=model, octave=self.octave)

        # Create DriftDatabase instance if persistence is enabled
        if persist_updates:
            if engine is None or zarr_path is None:
                raise ValueError("engine and zarr_path must be provided when persist_updates=True")
            self.db = DriftDatabase(engine=engine, zarr_path=zarr_path, run_name=run_name)

        logger.info(f"Initialized ImageProcessor" + (f" for: {self.run_name}" if run_name else ""))
        logger.info(f"Interpolation: {'enabled' if self.use_interpolation else 'disabled'}")
        if self.insitu_points is not None:
            logger.info("Validation mode enabled with in-situ points")

    def process_image(self, image_id, filename):
        """
        Process a single image and update internal points/templates.
        
        This method modifies the internal state of the processor by updating
        self.points and self.templates. It does not return these objects.
        
        Parameters:
        -----------
        image_id : int
            Unique identifier for the image being processed
        filename : str
            Path to the image file
        """
        basename = os.path.basename(filename)

        # Skip if already processed
        if image_id <= self.points.last_image_id:
            logger.info(f"Skipping image {image_id}: {basename}")
            return

        logger.info(f"Processing image {image_id}: {basename}")

        img = Image(filename)
        points_last = self.points.last()
        
        # Max drift over the full temporal window
        max_possible_drift = self.max_drift_speed_m_per_day * self.temporal_window 
        
        # Buffer needed is the minimum of max possible drift and the matcher's limit
        buffer_distance = min(max_possible_drift, self.matcher.spatial_distance_max)

        logger.info(f"Using buffer distance: {buffer_distance/1000.0:.2f} km "
                    f"(Min of {max_possible_drift/1000.0:.1f}km theoretical max drift over {self.temporal_window} days "
                    f"and {self.matcher.spatial_distance_max/1000.0:.1f}km spatial match limit)")
    
        # Filter for points within the buffered polygon and time threshold
        buffered_image_poly = img.poly.buffer(buffer_distance)
        time_threshold = img.date - pd.Timedelta(days=self.temporal_window)
        points_poly = points_last[
            points_last.within(buffered_image_poly) &
            (points_last['time'] >= time_threshold)
        ]
 
        if len(points_poly) == 0:
            logger.info("No overlapping points found")
            points_final = Keypoints()
        else:
            logger.info(f"{len(points_poly)} overlapping points found")
            points_final = self._match_existing_points(points_poly, img, image_id)
            if points_final is None:
                logger.info("Insufficient match quality, skipping point matching")
                points_final = Keypoints()
            else:
                if 'interpolated' in points_final.columns:
                    interp_count = points_final['interpolated'].sum()
                    total_count = len(points_final)
                    if interp_count > 0:
                        logger.info(f"Interpolation stats: {interp_count}/{total_count} points ({interp_count/total_count:.1%}) were interpolated")

        # Process new points, including any validation points
        self._process_new_points(points_final, img, image_id, basename)
    
        if image_id > 0 and image_id % self.pruning_interval == 0 and self.templates.trajectory_id.size > 0:
            logger.info(f"Performing periodic template pruning (every {self.pruning_interval} images)...")
            try:
                # Time-based Mask
                current_time_np = img.date.to_numpy()
                time_threshold_prune = current_time_np - np.timedelta64(self.temporal_window, 'D')
                keep_mask_time = (self.templates['time'] >= time_threshold_prune)#.compute()

                # Activity-based Mask
                active_traj_ids = set(self.points.loc[self.points['is_last'] == 1, 'trajectory_id'].unique())
                keep_mask_active = np.isin(self.templates['trajectory_id'].values, list(active_traj_ids))

                # Combine Masks (Keep if Recent AND Active)
                final_keep_mask = keep_mask_time & keep_mask_active

                # Apply Pruning 
                if not final_keep_mask.all():
                    num_before = self.templates.trajectory_id.size
                    # Filter using the computed boolean mask
                    self.templates = self.templates.where(final_keep_mask, drop=True)
                    num_after = self.templates.trajectory_id.size
                    logger.info(f"Pruning complete: Kept {num_after} templates / Removed {num_before - num_after} templates (were old OR inactive).")
                else:
                    logger.info("Pruning check: All templates recent or active.")

            except Exception as e:
                # Basic catch-all for unexpected errors during pruning
                logger.error(f"Error during template pruning: {e}", exc_info=True)
                logger.warning("Skipping template pruning for this interval due to error.")


        if self.persist_updates:
            # Calculate images since last *successful* persistence
            current_points_image_id = 0
            if isinstance(self.points, Keypoints):
                current_points_image_id = self.points.last_image_id
            elif isinstance(self.points, gpd.GeoDataFrame) and 'image_id' in self.points.columns and not self.points.empty:
                current_points_image_id = self.points['image_id'].max()

            images_since_persist = current_points_image_id - self._last_persisted_id

            if images_since_persist >= self.persist_interval:
                logger.info(f"Initiating periodic persistence and point pruning (interval: {self.persist_interval} images)...")
                current_last_image_id = current_points_image_id

                # --- Get Point Memory BEFORE save/prune ---
                mem_points_before_mib = self.points.memory_usage(deep=True).sum() / (1024 * 1024) if not self.points.empty else 0

                try:
                    save_successful = self.db.save(
                        self.points,
                        self.templates,
                        self._last_persisted_id,
                        self.insitu_points
                    )

                    if save_successful:
                        # Log DB save size using the pre-prune count
                        points_delta_count = len(self.points[self.points['image_id'] > self._last_persisted_id])
                        logger.info(f"Database persistence successful for {points_delta_count} new/updated points.")
                        num_before_prune = len(self.points)

                        # --- Prune points using the EXISTING time_threshold ---
                        # Keep only points that are the latest AND within the temporal window
                        keep_mask_points = (self.points['is_last'] == 1) & \
                                           (self.points['time'] >= time_threshold) # Use existing time_threshold

                        points_to_keep = self.points[keep_mask_points].copy()
                        self.points = Keypoints._from_gdf(points_to_keep)

                        # --- Get Point Memory AFTER prune ---
                        num_after_prune = len(self.points)
                        mem_points_after_mib = self.points.memory_usage(deep=True).sum() / (1024 * 1024) if not self.points.empty else 0

                        logger.info(
                            f"In-memory points pruned (active AND recent): {num_before_prune} -> {num_after_prune} "
                            f"({mem_points_after_mib:.1f} MiB). Memory reduced from {mem_points_before_mib:.1f} MiB."
                        )

                        self._last_persisted_id = current_last_image_id
                    else:
                        logger.warning("Database save operation reported failure. Skipping in-memory point pruning.")
                except Exception as e:
                    logger.error(f"Error during persistence or pruning: {e}", exc_info=True)
                    logger.warning("Skipping in-memory point pruning for this interval due to error. Database state may not be current.")

    @log_execution_time
    def _process_new_points(self, points_final, img, image_id, basename):
        """Process new points, including validation data, and update templates."""
        # Use KeypointDetector instead of standalone function
        coords_new = self.keypoint_detector.detect_new_keypoints(
            points=points_final,
            img=img,
            window_size=self.window_size,
            border_size=self.border_size
        )
        logger.info(f"Detected {len(coords_new)} new keypoints from image {image_id}")

        if self.insitu_points is not None:
            matching_insitu_points = self.insitu_points.loc[
                self.insitu_points['image_filepath'].apply(os.path.basename).isin([basename])
            ]

            if len(matching_insitu_points) > 0:
                logger.info(f"Found {len(matching_insitu_points)} matching buoy observations in image {basename}")
                # Use KeypointDetector instead of standalone function
                buoy_kps = self.keypoint_detector.keypoint_from_point(matching_insitu_points, img=img)
                coords_new = coords_new + buoy_kps
                logger.info(f"Added {len(buoy_kps)} buoy keypoints to detection set")

        keypoints, descriptors = compute_descriptors(
            coords_new,
            img,
            self.model,
            polarisation='dual_polarisation',
            normalize=False
        )

        if keypoints is not None:
            points_new = Keypoints.create(keypoints, descriptors, img, image_id)
            self.points = self.points.append(points_new)
            logger.info(f"Added {len(points_new)} new points (total: {len(self.points)})")

            templates_new, template_new_idx = extract_templates(points_new, img, hs=TEMPLATE_SIZE)
            self.templates = append_templates(
                self.templates,
                templates_new,
                template_new_idx,
                points_new
            )

        if len(points_final) > 0:
            self.points = self.points.update(points_final)
            templates_final, template_final_idx = extract_templates(points_final, img, hs=TEMPLATE_SIZE)
            self.templates = update_templates(
                self.templates,
                templates_final,
                template_final_idx,
                self.points
            )
      
    @log_execution_time
    def _match_existing_points(self, points_poly, img, image_id):
        """
        Match points from previous image to current one, with optional interpolation,
        then apply pattern matching and filter by correlation. This version uses the
        empirically correct trajectory_id assignment and robust descriptor filtering.
        """
        # Detect gridded keypoints, compute descriptors and create Keypoints object
        coords_grid = self.keypoint_detector.detect_gridded_points(
            img,
            stride=self.stride,
            border_size=self.border_size
        )

        # Error handling: Check if coords_grid is empty
        if not coords_grid:
            logger.warning("No gridded keypoints detected. Skipping matching for this image.")
            return None

        keypoints, descriptors = compute_descriptors(
            coords_grid,
            img,
            self.model,
            polarisation='dual_polarisation',
            normalize=False
        )

        # Error handling: Check if descriptor computation failed for grid
        if keypoints is None or descriptors is None:
             logger.warning("Failed to compute descriptors for grid points. Skipping matching.")
             return None

        points_grid = Keypoints.create(keypoints, descriptors, img, image_id=image_id)

        # Match and filter points using the matcher
        points_fg1, points_fg2 = self.matcher.match_with_grid(points_poly, points_grid)

        if points_fg1 is None or points_fg2 is None or points_fg1.empty or points_fg2.empty:
            logger.info("Insufficient match quality or no matches found, skipping point matching step.")
            return Keypoints()

        # Assign trajectory IDs to matched points
        points_matched = points_fg2.copy()
        points_matched['trajectory_id'] = points_fg1.trajectory_id.values
        points_matched['interpolated'] = 0

        # Interpolate drift
        if self.use_interpolation and len(points_fg1) < len(points_poly):
            points_combined = interpolate_drift(
                points_poly,
                points_grid,
                points_fg1,
                points_matched,
                img,
                self.model,
                octave=self.octave
            )
            if points_combined is None or points_combined.empty:
                 logger.warning("Interpolation resulted in no valid points. Proceeding with only matched points.")
                 points_combined = points_matched # Fall back to only matched points
        else:
            # Handle logging messages for why interpolation wasn't run
            if not self.use_interpolation:
                logger.info("Interpolation disabled, using only matched points")
            elif len(points_fg1) >= len(points_poly):
                logger.info("All points matched, no interpolation needed")
            points_combined = points_matched

        # Check if points_combined is empty before proceeding
        if points_combined.empty:
             logger.info("No matched or interpolated points remaining before template check.")
             return Keypoints()

        all_traj_ids = points_combined.trajectory_id.values
        # Ensure points_orig correctly references points_poly using trajectory IDs
        points_orig = points_poly[points_poly.trajectory_id.isin(all_traj_ids)]

        # Filter out points that don't have templates
        available_traj_ids = self.templates.trajectory_id.values
        has_template_mask = np.isin(all_traj_ids, available_traj_ids)
        if not np.all(has_template_mask):
            missing_count = np.sum(~has_template_mask)
            logger.info(f"Filtering out {missing_count} points that don't have templates")
            points_combined = points_combined[has_template_mask].copy()
            if points_combined.empty:
                 logger.info("No points remaining after template availability filter.")
                 return Keypoints()
            # Update IDs and points_orig after filtering
            all_traj_ids = points_combined.trajectory_id.values
            points_orig = points_poly[points_poly.trajectory_id.isin(all_traj_ids)]

        # Perform pattern matching
        templates_all = self.templates.sel(trajectory_id=all_traj_ids)
        keypoints_corrected_xy, keypoints_corrected_rc, corr_values = pattern_matching(
            points_combined,
            img,
            templates_all,
            points_orig
        )

        # Remove points below the correlation threshold
        points_combined['corr'] = corr_values
        valid_mask = corr_values >= self.min_correlation
        # Ensure we handle the case where NO points pass the correlation threshold
        if not np.any(valid_mask):
             logger.info(f"No points passed the correlation threshold of {self.min_correlation}. Returning empty.")
             return Keypoints()

        valid_points = points_combined[valid_mask].copy()
        valid_corrected_rc = keypoints_corrected_rc[valid_mask]
        # Assign geometry using the index from valid_points to align correctly
        valid_points = valid_points.assign(
            geometry=gpd.points_from_xy(
                keypoints_corrected_xy[valid_mask, 0], # Use same mask on corrected coords
                keypoints_corrected_xy[valid_mask, 1]
            )
        )

        logger.info(f"Pattern matching kept {len(valid_points)}/{len(points_combined)} points based on correlation threshold of {self.min_correlation}")

        # Prepare KeyPoints for descriptor computation for currently valid points
        if not valid_points.empty:
            keypoints_list = [
                cv2.KeyPoint(px_c, px_r, size=WINDOW_SIZE, angle=img.angle, octave=self.octave)
                # Iterate directly through the corrected pixel coordinates from pattern_matching
                for px_c, px_r in valid_corrected_rc
            ]
            # Recompute descriptors
            raw_kps, new_descriptors = compute_descriptors(
                keypoints_list,
                img,
                self.model,
                polarisation='dual_polarisation',
                normalize=False
            )
        else:
            new_descriptors = None

        # Filter points based on successful descriptor computation
        original_count = len(valid_points)
        removed_count = 0

        if new_descriptors is not None:
            # Ensure descriptor count matches point count
            if len(new_descriptors) != len(valid_points):
                 logger.info(f"Descriptor count mismatch after re-computation! "
                              f"Expected {len(valid_points)}, got {len(new_descriptors)}. Removing points.")
                 removed_count = original_count
                 valid_points = valid_points.iloc[0:0].copy() # Discard points due to ambiguity
            else:
                valid_points['descriptors'] = list(new_descriptors)
                # Define the validity check (isinstance check handles None implicitly if compute_descriptors returned a list with Nones)
                valid_mask_desc = valid_points['descriptors'].apply(lambda d: isinstance(d, np.ndarray))

                # Apply the filter if any descriptors are invalid
                if not valid_mask_desc.all():
                    valid_points = valid_points[valid_mask_desc].copy() # Filter and copy
                    removed_count = original_count - len(valid_points)
        else:
            # Descriptor computation failed entirely (returned None) or was skipped
            if original_count > 0:
                logger.warning("Descriptor re-computation failed or was skipped; removing remaining points.")
                removed_count = original_count
                valid_points = valid_points.iloc[0:0].copy()

        # Log the result of filtering
        if removed_count > 0:
            logger.info(f"Final filtering removed {removed_count} points "
                        f"due to invalid/None descriptors after re-computation attempts. "
                        f"Returning {len(valid_points)} points.")
        elif original_count > 0: # Log success only if we started with points and removed none
             logger.debug("Final descriptor check passed, all points kept.")

        # Update templates ONLY for the points that survived the descriptor check
        if not valid_points.empty:
            new_templates, new_template_idx = extract_templates(valid_points, img, hs=TEMPLATE_SIZE)
            # Check if extract_templates returned valid indices matching remaining points
            if len(new_template_idx) == len(valid_points):
                 self.templates = update_templates(self.templates, new_templates, new_template_idx, valid_points)
                 logger.debug(f"Updated templates for {len(valid_points)} points after descriptor check.")
            else:
                 logger.warning(f"Template extraction index mismatch. Expected {len(valid_points)} indices, "
                                f"got {len(new_template_idx)}. Skipping template update.")
        else:
            logger.info("No valid points remain after descriptor check to update templates.")

        logger.debug(f"Points returned after final descriptor check: {len(valid_points)}")

        return valid_points

    def ensure_final_persistence(self):
        """Ensure final persistence of any remaining unprocessed data."""
        if self.persist_updates:
            images_since_persist = self.points.last_image_id - self._last_persisted_id
            if images_since_persist > 0:
                logger.info(f"Performing final persistence for remaining {images_since_persist} images")
                save_successful = self.db.save(
                    points=self.points,
                    templates=self.templates,
                    last_persisted_id=self._last_persisted_id,
                    insitu_points=self.insitu_points
                )
                if save_successful:
                    self._last_persisted_id = self.points.last_image_id
                    logger.info(f"Final persistence completed. Last persisted ID set to {self._last_persisted_id}")
                else:
                    logger.error("Final persistence FAILED. _last_persisted_id not updated.")

class Image(Nansat):
    """
    Extension of Nansat class with extra functionality and transform caching.

    Attributes:
    -----------
    file_path : str
        Path to the image file.
    srs : object
        Spatial reference system for the image.
    date : datetime
        Date extracted from the file path.
    _transform_cache : dict
        Internal cache for transform_points results.
    _max_cache_entries : int
        Maximum number of entries allowed in the cache.

    Methods:
    --------
    __init__(file_path, srs=NSR(3413)):
        Initializes the Image object with the given file path and SRS.
    angle:
        Returns the angle of rotation of the image in degrees.
    poly:
        Performs coordinate transformations and sets up a polygon for the image.
    transform_points(x, y, DstToSrc=0, dst_srs=None):
        Transforms points with caching for large coordinate arrays.
    """

    def __init__(self, file_path, srs=NSR(3413)):
        """
        Initialize the Image object with the given file path and spatial reference system (SRS).

        Parameters:
        file_path (str): Path to the image file.
        srs (object): Spatial reference system for the image. Default is NSR(3413).
        """
        super().__init__(file_path)
        self.vrt.tps = True # Assuming this is desired behaviour
        self.srs = srs
        self.date = extract_date(file_path)
        # Initialize cache attributes
        self._transform_cache = {}
        self._max_cache_entries = 50  # Limit cache size to prevent excessive memory use

    @cached_property
    def angle(self):
        ''' Returns angle <alpha> of rotation of n'''
        nps_proj = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
        pc_proj = ccrs.PlateCarree()
        # get corners
        corners_n_lons, corners_n_lats = self.get_corners()
        # transform corners into lat, lon of other image
        cx, cy = nps_proj.transform_points(pc_proj, corners_n_lons, corners_n_lats).T[:2]
        # Calculate angle using arctan2 for robustness
        alpha = np.degrees(np.arctan2(cx[1] - cx[0], cy[1] - cy[0]))
        return alpha

    @cached_property
    def poly(self):
        """
        Perform coordinate transformations and polygon setup for the image extent.

        Returns:
        Polygon: A shapely Polygon object representing the image footprint in the target SRS.
        """
        # Get corners in longitude and latitude
        corners_n_lons, corners_n_lats = self.get_corners()

        # Transform points from lon/lat to image pixel coordinates (source)
        # DstToSrc=1 means LonLat (source) to Pixel (destination)
        cols, rows = super().transform_points(corners_n_lons, corners_n_lats, DstToSrc=1)

        # Transform points from image pixel coordinates (source) to meters in the specified SRS (destination)
        # DstToSrc=0 means Pixel (source) to Target SRS (destination)
        corners_nx_meters, corners_ny_meters = super().transform_points(cols, rows, DstToSrc=0, dst_srs=self.srs)

        # Construct coordinates in the required order to form a polygon (e.g., clockwise or counter-clockwise)
        # Nansat corners are typically UL, UR, LR, LL. Polygon needs ordered vertices.
        # Assuming standard corner order: 0:UL, 1:UR, 2:LR, 3:LL
        # A common ordering for Polygon is clockwise: UL, UR, LR, LL
        coords = np.vstack([
            corners_nx_meters[[0, 1, 2, 3]],
            corners_ny_meters[[0, 1, 2, 3]]
        ]).T

        # Create and return the polygon
        poly = Polygon(coords)

        return poly
        
class Keypoints(gpd.GeoDataFrame):
    # _metadata = ['srs'] + gpd.GeoDataFrame._metadata
    srs = NSR(3413)

    def __init__(self, *args, **kwargs):
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
                'interpolated': []
            }
            super().__init__(empty_data)
        else:
            # Initialize with provided data
            super().__init__(*args, **kwargs)
        self.srs = kwargs.get('srs', self.srs)

    @property
    def last_image_id(self):
        """Return the highest image_id or 0 if no valid image_ids exist."""
        return 0 if len(self) == 0 or 'image_id' not in self.columns else (self['image_id'].max() or 0)

    @classmethod
    def _from_gdf(cls, gdf):
        """Convert a GeoDataFrame to Keypoints"""
        return cls(gdf, crs=gdf.crs)

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
            self['trajectory_id'] = self['trajectory_id'].astype(float).astype(int)

        existing_trajectory_ids = set(self['trajectory_id']) if len(self) > 0 else set()

        # Assign new trajectory_ids starting from max existing + 1
        next_trajectory_id = int(max(existing_trajectory_ids)) + 1 if existing_trajectory_ids else 0
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

        return self._from_gdf(result)

    @log_execution_time
    def update(self, points):
        """Optimized update method using vectorized operations"""
        if len(points) == 0:
            return self._from_gdf(self)

        # Vectorized update of is_last flag
        mask = self['trajectory_id'].isin(points['trajectory_id'])
        if mask.any():
            # Create a copy to avoid SettingWithCopyWarning
            temp_df = self.copy()
            temp_df.loc[mask, 'is_last'] = 0
            result = pd.concat([temp_df, points], ignore_index=True)
        else:
            # No matches to update, just concatenate
            result = pd.concat([self, points], ignore_index=True)

        # Return the updated Keypoints object
        return self._from_gdf(result)

    @classmethod
    def create(cls, keypoints, descriptors, img, image_id):
        xy = img.transform_points(keypoints[:, 0], keypoints[:, 1], DstToSrc=0, dst_srs=cls.srs)
        N = len(xy[0])
        new_data = {
            'image_id': np.zeros(N) + image_id,
            'is_last': np.ones(N),
            'trajectory_id': range(N),
            'geometry': gpd.points_from_xy(xy[0], xy[1]),
            'descriptors': list(descriptors),
            'angle': [img.angle] * N,
            'corr': np.zeros(N),
            'time': [img.date] * N,
            'interpolated': np.zeros(N),

        }
        return cls(new_data)

class KeypointDetector:
    """
    Handles detection of keypoints in images.

    This class encapsulates different strategies for keypoint detection,
    including new keypoint detection and gridded detection.
    """

    def __init__(self, model, octave=OCTAVE):
        """
        Initialize the keypoint detector.

        Parameters:
            model: Feature detection model (e.g., SIFT, ORB)
            octave (int): The octave value for keypoints
        """
        self.model = model
        self.octave = octave

    @log_execution_time
    def detect_new_keypoints(
        self,
        points,
        img,
        window_size=WINDOW_SIZE,
        border_size=BORDER_SIZE,
        step=None,
        adjust_keypoint_angle=True,
        band_name="s0_HV",
    ):
        """
        Detect new keypoints in an image avoiding areas with existing keypoints.

        Parameters:
            points (Keypoints): Existing keypoints
            img (Image): Image to detect keypoints in
            window_size (int): Size of detection windows
            border_size (int): Border to exclude
            step (int): Step size between windows
            adjust_keypoint_angle (bool): Whether to adjust keypoint angles
            band_name (str): Image band to use

        Returns:
            list: Detected keypoints
        """
        img0 = img[band_name]
        img0[np.isnan(img0)] = 0
        if step is None:
            step = window_size

        # Extract the coordinates of these points and transform them into image coordinates
        if not points.empty:
            points_poly_kp = np.array([(geom.x, geom.y) for geom in points.geometry])
            cols, rows = img.transform_points(
                points_poly_kp[:, 0],
                points_poly_kp[:, 1],
                DstToSrc=1,
                dst_srs=NSR(3413),
            )
            existing_keypoints_coords = np.vstack((cols.flatten(), rows.flatten())).T
        else:
            existing_keypoints_coords = np.empty((0, 2))

        # Divide the image into windows
        windows = view_as_windows(
            img0, window_shape=(window_size, window_size), step=step
        )

        # Get the number of windows
        n_windows_row, n_windows_col, _, _ = windows.shape

        keypoints = []

        for i in range(n_windows_row):
            for j in range(n_windows_col):
                # Define window coordinates
                x_start = j * (step)
                x_end = x_start + window_size
                y_start = i * (step)
                y_end = y_start + window_size

                # Check existing keypoint density in this window
                num_existing_kp = np.sum(
                    (existing_keypoints_coords[:, 0] >= x_start)
                    & (existing_keypoints_coords[:, 0] < x_end)
                    & (existing_keypoints_coords[:, 1] >= y_start)
                    & (existing_keypoints_coords[:, 1] < y_end)
                )

                max_kp_per_window = 1  # Maximum allowed keypoints per window
                if num_existing_kp >= max_kp_per_window:
                    continue  # Skip detection in this window

                window = windows[i, j]
                # Detect keypoints in the window
                kps = self.model.detect(window, None)
                if len(kps) != 0:
                    kp_best = max(kps, key=lambda kp: kp.response)
                    # Adjust keypoint coordinates to the original image coordinates
                    x_offset = x_start
                    y_offset = y_start
                    kp_best.pt = (kp_best.pt[0] + x_offset, kp_best.pt[1] + y_offset)
                    if adjust_keypoint_angle:
                        kp_best.angle = img.angle
                    kp_best.octave = self.octave
                    keypoints.append(kp_best)

        # Filter keypoints to remove ones too close to the image edges
        filtered_keypoints = [
            kp
            for kp in keypoints
            if (
                border_size <= kp.pt[0] <= img0.shape[1] - border_size
                and border_size <= kp.pt[1] <= img0.shape[0] - border_size
            )
        ]

        return filtered_keypoints

    @log_execution_time
    def detect_gridded_points(
        self,
        img,
        stride=STRIDE,
        size=WINDOW_SIZE,
        border_size=BORDER_SIZE,
        band_name="s0_HV",
    ):
        """
        Generate gridded points over the input image.

        Parameters:
            img (Image): Image to generate points for
            stride (int): Distance between grid points
            size (int): Size of keypoints
            border_size (int): Border to exclude
            band_name (str): Image band to use

        Returns:
            list: Grid of keypoints
        """
        # Get image data and replace NaNs with zero
        img0 = img[band_name]
        img0[np.isnan(img0)] = 0

        # Generate grid keypoints
        keypoints = []
        for r in range(0, img0.shape[0], stride):
            for c in range(0, img0.shape[1], stride):
                keypoints.append(
                    cv2.KeyPoint(r, c, size=size, octave=self.octave, angle=img.angle)
                )

        # Filter keypoints within image borders
        filtered_keypoints = [
            kp
            for kp in keypoints
            if (
                border_size <= kp.pt[0] <= img0.shape[1] - border_size
                and border_size <= kp.pt[1] <= img0.shape[0] - border_size
            )
        ]

        return filtered_keypoints

    def keypoint_from_point(self, points, img):
        """
        Generate keypoints from specific geographic points.

        Parameters:
            points (GeoDataFrame): Points with geometry
            img (Image): Image to place keypoints in

        Returns:
            list: Keypoints
        """
        keypoints = []
        pc = ccrs.PlateCarree()
        nps = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
        for _, point in points.iterrows():
            x, y = np.atleast_1d(point.geometry.x), np.atleast_1d(point.geometry.y)
            lon, lat, _ = pc.transform_points(nps, x, y).T
            cols, rows = img.transform_points(lon, lat, DstToSrc=1)
            kp = cv2.KeyPoint(
                round(cols[0]),
                round(rows[0]),
                size=WINDOW_SIZE,
                octave=self.octave,
                angle=img.angle,
            )
            keypoints.append(kp)
        return keypoints

class Matcher:
    def __init__(self,
                 # General matching parameters
                 norm=cv2.NORM_HAMMING2,
                 descriptor_distance_max=130,
                 spatial_distance_max=100000,

                 # Homography estimation parameters
                 model=AffineTransform,
                 model_threshold=13000,
                 use_model_estimation=True,
                 estimation_method="USAC_MAGSAC",

                 # Lowe's ratio test parameter
                 lowe_ratio=0.9,

                 # Visualization
                 plot=False):

        # General matching parameters
        self.norm = norm
        self.descriptor_distance_max = descriptor_distance_max
        self.spatial_distance_max = spatial_distance_max

        # Homography estimation parameters
        self.model = model
        self.model_threshold = model_threshold
        self.use_model_estimation = use_model_estimation

        # Store the method name and get its value
        if estimation_method.upper() == "DEGENSAC":
            self.estimation_method_name = "DEGENSAC"
            self.estimation_method = None  # pydegensac will be used
        else:
            self.estimation_method_name = estimation_method
            self.estimation_method = getattr(cv2, estimation_method)

        # Additional parameters
        self.lowe_ratio = lowe_ratio
        self.plot = plot

    def plot_quiver(self, pos0, pos1, dist):
        u = pos1[:, 0] - pos0[:, 0]
        v = pos1[:, 1] - pos0[:, 1]
        spd = np.hypot(u,v)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(dist, spd, '.', alpha=0.5)
        qui = axs[1].quiver(pos0[:, 0], pos0[:, 1], pos1[:, 0] - pos0[:, 0], pos1[:, 1] - pos0[:, 1], dist, cmap='jet', clim=np.percentile(dist, [1, 99]), angles='xy', scale_units='xy', scale=1)
        plt.colorbar(qui, ax=axs[0], shrink=0.5)
        plt.show()

    @log_execution_time
    def match_with_grid(self, points_poly, points_grid):
        """
        Match points between polygon and grid representations.

        Parameters:
        points_poly: Points from polygon representation (GeoDataFrame)
        points_grid: Points from grid representation (GeoDataFrame)

        Returns:
        tuple: (points_fg1, points_fg2) matched points, or (None, None) if matching fails
        """
        # Extract positions from geometry columns
        pos0 = np.column_stack((points_poly.geometry.x, points_poly.geometry.y))
        pos1 = np.column_stack((points_grid.geometry.x, points_grid.geometry.y))

        # Extract descriptors
        x0 = np.vstack(points_poly['descriptors'].values)
        x1 = np.vstack(points_grid['descriptors'].values)

        # Try crosscheck matcher first
        matches = self.match_with_crosscheck(x0, x1)
        rc_idx0, rc_idx1, residuals = self.filter(matches, pos0, pos1)

        # If filter returned None or we don't have enough matches, try Lowe's ratio test
        if rc_idx0 is None or (rc_idx0.size / pos0.shape[0] < 0.1):
            matches = self.match_with_lowe_ratio(matches, x0, x1, pos0, pos1)
            rc_idx0, rc_idx1, residuals = self.filter(matches, pos0, pos1)

            # If we still don't have valid matches, return None
            if rc_idx0 is None:
                return None, None

        return points_poly.iloc[rc_idx0], points_grid.iloc[rc_idx1]

    @log_execution_time
    def match_with_crosscheck(self, x0, x1):
        """
        Matches descriptors using cross-checking.

        Parameters:
            x0 (ndarray): Descriptors from the first image.
            x1 (ndarray): Descriptors from the second image.

        Returns:
            list: Matches that pass the cross-check.
        """
        bf = cv2.BFMatcher(self.norm, crossCheck=True)
        matches = bf.match(x0, x1)
        return list(matches)

    @log_execution_time
    def match_with_lowe_ratio(self, matches_bf, x0, x1, pos0, pos1):
        """
        Applies Lowe's ratio test to filter matches and extend them.

        Parameters:
            matches_bf (list): Matches from brute force matching.
            x0 (ndarray): Descriptors from the first image.
            x1 (ndarray): Descriptors from the second image.
            pos0 (ndarray): Keypoint positions in the first image.
            pos1 (ndarray): Keypoint positions in the second image.

        Returns:
            list: Extended list of matches after Lowe's ratio test.
        """
        # Initialize the FLANN-based matcher
        index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                            table_number=12,  # 12 hash tables
                            key_size=20,  # 20-bit keys
                            multi_probe_level=2)  # 2 levels of multi-probing
        search_params = {}  # or pass an empty dictionary
        knn_matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Match descriptors using KNN
        all_matches = knn_matcher.knnMatch(x0, x1, k=4)

        # Apply ratio test as per Lowe's paper
        # Extend with max drift test and max descriptor distance test
        matches = []
        # this compare multiple matches to each other and only keeps the best one by using several filters in geographic and descriptor distance
        for mmm in all_matches:
            for midx, m in enumerate(mmm[:-1]):
                d = norm(pos0[m.queryIdx] - pos1[m.trainIdx])
                if d < self.spatial_distance_max and m.distance < self.descriptor_distance_max and m.distance < self.lowe_ratio * mmm[midx+1].distance:
                    matches.append(m)
                    break
        matches_bf_queryIdx = [m.queryIdx for m in matches_bf]
        for m in matches:
            # loop over new matches and adds new matches only if they are not found in the BF matching
            if m.queryIdx not in matches_bf_queryIdx:
                matches_bf.append(m)
        logger.info(f"Number of matches after Lowe's ratio test: {len(matches_bf)}")
        return matches_bf

    @log_execution_time
    def filter(self, matches, pos0, pos1):
        """
        Filters matches based on descriptor distance, spatial distance, and model estimation.

        Parameters:
        matches (list): Raw matches from a matching algorithm.
        pos0 (ndarray): Keypoint positions in the first image.
        pos1 (ndarray): Keypoint positions in the second image.

        Returns:
        tuple: Indices of filtered matches and residuals if model estimation is used.
                Returns (None, None, None) if estimation fails due to insufficient matches.
        """
        descriptor_distance = np.array([m.distance for m in matches])
        bf_idx0 = np.array([m.queryIdx for m in matches])
        bf_idx1 = np.array([m.trainIdx for m in matches])
        logger.info(f'Matching: Number of matches: {bf_idx0.size}')
        if self.plot: self.plot_quiver(pos0[bf_idx0], pos1[bf_idx1], descriptor_distance)

        # Filter by descriptor distance
        gpi0 = np.nonzero(descriptor_distance < self.descriptor_distance_max)
        dd_idx0 = bf_idx0[gpi0]
        dd_idx1 = bf_idx1[gpi0]
        logger.info(f'Distance: Number of matches: {dd_idx0.size}')
        if self.plot: self.plot_quiver(pos0[dd_idx0], pos1[dd_idx1], descriptor_distance[gpi0])

        # Filter by spatial distance
        gpi1 = np.hypot(pos1[dd_idx1, 0] - pos0[dd_idx0, 0], pos1[dd_idx1, 1] - pos0[dd_idx0, 1]) < self.spatial_distance_max
        md_idx0 = dd_idx0[gpi1]
        md_idx1 = dd_idx1[gpi1]
        logger.info(f'Spatial: Number of matches: {md_idx0.size}')
        if self.plot: self.plot_quiver(pos0[md_idx0], pos1[md_idx1], descriptor_distance[gpi0][gpi1])

        if not self.use_model_estimation:
            return md_idx0, md_idx1, None

        # Check if we have enough points for homography
        if md_idx0.size < 4:
            logger.warning("Warning: Insufficient matches for model estimation (minimum 4 required)")
            return None, None, None

        try:
            # Log the estimation method name

            if self.estimation_method_name.upper() == "DEGENSAC":
                import pydegensac
                H, inliers = pydegensac.findHomography(pos0[md_idx0], pos1[md_idx1], self.model_threshold)
            else:
                H, inliers = cv2.findHomography(pos0[md_idx0], pos1[md_idx1], self.estimation_method, self.model_threshold)

            if H is None:
                logger.warning("Warning: Model estimation failed")
                return None, None, None

            model = self.model(H)
            gpi2 = np.nonzero(inliers)[0]
            rc_idx0 = md_idx0[gpi2]
            rc_idx1 = md_idx1[gpi2]
            residuals = model.residuals(pos0[rc_idx0], pos1[rc_idx1])
            logger.info(f'{self.estimation_method_name}: Number of matches: {rc_idx0.size}')
            if self.plot:
                self.plot_quiver(pos0[rc_idx0], pos1[rc_idx1], residuals)
            return rc_idx0, rc_idx1, residuals

        except cv2.error as e:
            logger.error(f"Warning: OpenCV error during model estimation: {str(e)}")
            return None, None, None
        except Exception as e:
            logger.error(f"Warning: Unexpected error during model estimation: {str(e)}")
            return None, None, None

@log_execution_time
def interpolate_drift(points_poly, points_grid, points_fg1, points_fg2, img, model, octave=8, model_type=AffineTransform):
    """
    Interpolate positions for unmatched points using vectorized filtering.
    Descriptors are NOT computed here; they are recomputed after pattern matching.

    Parameters:
    -----------
    points_poly : GeoDataFrame
        Original points from the previous image.
    points_grid : GeoDataFrame
        Grid points detected in the current image.
    points_fg1 : GeoDataFrame
        Matched points subset from points_poly.
    points_fg2 : GeoDataFrame
        Matched points subset from points_grid.
    img : Image
        The current image object.
    model : cv2.Feature2D
        Passed but not used for descriptor computation here.
    octave : int
        Octave for potential keypoint creation (used by Keypoints.create indirectly).
    model_type : class
        Transformation model class (e.g., AffineTransform).

    Returns:
    --------
    GeoDataFrame
        Combined GeoDataFrame with matched and interpolated points.
        Interpolated points have None in 'descriptors' column initially.
    """
    points_result = points_fg2.copy()
    if 'interpolated' not in points_result.columns:
        points_result['interpolated'] = 0

    # Early exit if no interpolation needed
    if len(points_fg1) >= len(points_poly):
        logger.info("All points matched, no interpolation needed")
        return points_result

    # Estimate Transformation
    pos0 = np.column_stack((points_poly.geometry.x, points_poly.geometry.y))
    pos1 = np.column_stack((points_grid.geometry.x, points_grid.geometry.y))
    rc_idx0_from_fg1 = points_poly.index.get_indexer(points_fg1.index)
    rc_idx1_from_fg2 = points_grid.index.get_indexer(points_fg2.index)
    atm = model_type()
    atm.estimate(pos0[rc_idx0_from_fg1], pos1[rc_idx1_from_fg2])

    # Identify Unmatched Points Valid for Interpolation
    matched_ids = set(points_fg1.index)
    unmatched_mask = ~points_poly.index.isin(matched_ids)
    unmatched_points = points_poly[unmatched_mask]

    # Filter unmatched points to those within the target image bounds (no buffer)
    img_poly_no_buffer = img.poly
    unmatched_within_image_mask = unmatched_points.within(img_poly_no_buffer)
    unmatched_points_for_interp = unmatched_points[unmatched_within_image_mask]

    # Early exit if no suitable points found
    if unmatched_points_for_interp.empty:
        logger.info("No suitable unmatched points found within image area for interpolation.")
        return points_result

    logger.info(f"Found {len(unmatched_points_for_interp)} unmatched points within image area for interpolation")

    # Apply Transformation and Convert to Pixel Coords
    pos0_to_interp = np.column_stack((unmatched_points_for_interp.geometry.x,
                                      unmatched_points_for_interp.geometry.y))
    interpolated_pos_geo = atm(pos0_to_interp)
    cols, rows = img.transform_points(interpolated_pos_geo[:, 0], interpolated_pos_geo[:, 1], DstToSrc=1, dst_srs=NSR(3413))

    # Vectorized Filtering of Pixel Coordinates
    height, width = img['s0_HV'].shape
    border = 16

    valid_col_mask = (cols >= border) & (cols < width - border)
    valid_row_mask = (rows >= border) & (rows < height - border)
    valid_pixel_mask = valid_col_mask & valid_row_mask

    # Select valid pixel coordinates and corresponding original trajectory IDs
    valid_kp_coords_pixel_np = np.column_stack((cols[valid_pixel_mask], rows[valid_pixel_mask]))
    final_interpolated_points = unmatched_points_for_interp[valid_pixel_mask]
    interpolated_trajectory_ids = final_interpolated_points.trajectory_id.values

    # Check for potential mismatch and handle by truncation
    num_interpolated = len(valid_kp_coords_pixel_np)
    if len(interpolated_trajectory_ids) != num_interpolated:
        # This hasn't occured in practice yet
        logger.warning(f"Interpolated coordinate/ID count mismatch after filtering: {num_interpolated} coords vs {len(interpolated_trajectory_ids)} IDs. Truncating.")
        count = min(len(interpolated_trajectory_ids), num_interpolated)
        interpolated_trajectory_ids = interpolated_trajectory_ids[:count]
        valid_kp_coords_pixel_np = valid_kp_coords_pixel_np[:count]
        num_interpolated = count # Update count after truncation

    # Create Keypoints Object for Interpolated Points
    if num_interpolated > 0:
        interpolated_keypoints = Keypoints.create(
            valid_kp_coords_pixel_np,    # Nx2 pixel coordinate array
            [None] * num_interpolated,   # Placeholder for descriptors
            img,
            points_result['image_id'].iloc[0] # Use image_id from matched points
        )

        # Update trajectory IDs and interpolation flag
        interpolated_keypoints['trajectory_id'] = interpolated_trajectory_ids
        interpolated_keypoints['interpolated'] = 1

        # Combine and Return
        points_final = pd.concat([points_result, interpolated_keypoints], ignore_index=True)
        logger.info(f"Successfully interpolated {len(interpolated_keypoints)} points")
    else:
        # If truncation resulted in zero points, just return the original matched points
        logger.warning("Interpolation resulted in zero valid points after mismatch handling.")
        points_final = points_result

    return points_final

@log_execution_time
def compute_descriptors(keypoints, img, model, polarisation='s0_HV', normalize=True):
    """
    Compute descriptors for given keypoints in an image using a specified model.

    Parameters:
        keypoints (list): List of keypoints to compute descriptors for.
        img (dict): Dictionary containing image bands. Must include 's0_HH' and 's0_HV' if dual polarisation is used.
        model (object): Descriptor model with a `compute` method (e.g., OpenCV's SIFT or ORB).
        band_name (str, optional):
            - 's0_HH' or 's0_HV' for single polarisation.
            - 'dual_polarisation' to compute descriptors for both 's0_HH' and 's0_HV'.
            Defaults to 's0_HV'.

    Returns:
        tuple: A tuple containing:
            - rawkeypoints (numpy.ndarray or None):
                - Array of keypoint coordinates from 's0_HH' if dual polarisation is used.
                - Array of keypoint coordinates from the specified band otherwise.
                - Returns None if descriptor computation fails.
            - normdesc (numpy.ndarray or None):
                - Array of normalized descriptors for single polarisation.
                - If dual polarisation is used, descriptors from both bands are horizontally stacked.
                - Returns None if descriptor computation fails for any band.
    """

    if polarisation == 'dual_polarisation':
        # Compute descriptors for 's0_HH'
        img_hh = img['s0_HH'].copy()
        img_hh[np.isnan(img_hh)] = 0
        keypoints_output_hh, descriptors_hh = model.compute(img_hh, keypoints)
        if descriptors_hh is None:
            return None, None
        rawkeypoints = np.array([kp.pt for kp in keypoints_output_hh])
        if normalize:
            descriptors_hh = descriptors_hh - descriptors_hh.mean(axis=1, keepdims=True)

        # Compute descriptors for 's0_HV'
        img_hv = img['s0_HV'].copy()
        img_hv[np.isnan(img_hv)] = 0
        _, descriptors_hv = model.compute(img_hv, keypoints)
        if descriptors_hv is None:
            return None, None
        if normalize:
            descriptors_hv = descriptors_hv - descriptors_hv.mean(axis=1, keepdims=True)

        # Stack descriptors horizontally
        descriptors = np.hstack([descriptors_hh, descriptors_hv])

        return rawkeypoints, descriptors

    else:
        # Single Polarisation Processing
        img_band = img[polarisation].copy()
        img_band[np.isnan(img_band)] = 0
        keypoints_output, descriptors = model.compute(img_band, keypoints)
        if descriptors is None:
            return None, None
        rawkeypoints = np.array([kp.pt for kp in keypoints_output])
        if normalize:
            descriptors = descriptors - descriptors.mean(axis=1, keepdims=True)
        return rawkeypoints, descriptors

@log_execution_time
def extract_templates(points, img, band='s0_HV', hs=TEMPLATE_SIZE):
    """
    Extracts square templates from an image centered at specified keypoints.
    Parameters:
    points (GeoDataFrame): A GeoDataFrame containing the keypoints with 'geometry' column.
    img (xarray.DataArray): An xarray DataArray representing the image from which templates are extracted.
    hs (int, optional): Half-size of the square template. The full size will be (2*hs+1) x (2*hs+1). Default is 16.
    Returns:
    tuple: A tuple containing:
        - templates (np.ndarray): An array of extracted templates.
        - ids (np.ndarray): An array of indices of keypoints for which templates were successfully extracted.
    """
    pc = ccrs.PlateCarree()
    nps = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
    x, y = points.geometry.x.values, points.geometry.y.values
    
    # Use the cached transformation if the points are already in the right CRS
    if points.crs == img.srs:
        colsrows = np.array(img.transform_points(x, y, DstToSrc=1)).T
    else:
        lon, lat, _ = pc.transform_points(nps, x, y).T
        colsrows = np.array(img.transform_points(lon, lat, DstToSrc=1)).T
    img = img[band]
    # Convert keypoints to integers
    colsrows = colsrows.astype(int)
    # Preallocate the templates array with NaNs
    templates = []
    ids = []

    for i, (c0, r0) in enumerate(colsrows):
        # Ensure keypoints are within image boundaries
        if r0-hs >= 0 and r0+hs+1 <= img.shape[0] and c0-hs >= 0 and c0+hs+1 <= img.shape[1]:
            # Extract the template from the first image
            ids.append(i)
            c0 = c0.astype(int)
            r0 = r0.astype(int)
            template = img[r0-hs:r0+hs+1, c0-hs:c0+hs+1]
            templates.append(template)
        else:
            logger.error(f"Keypoint {i} is out of bounds")
    return np.array(templates), np.array(ids)

@log_execution_time
def append_templates(templates, templates_new, template_new_idx, points, template_shape=33):
    """
    Append new templates to the templates DataArray.

    Parameters:
        templates (xr.DataArray): Existing templates DataArray.
        templates_new (np.ndarray): Array of templates to add.
        template_new_idx (array-like): Indices of points corresponding to templates_new.
        points (pd.DataFrame): DataFrame containing trajectory IDs.
        template_shape (int or tuple): Shape of the templates (height, width).

    Returns:
        xr.DataArray: Updated templates DataArray with new templates appended.
    """
    # Extract trajectory_ids for the new templates
    trajectory_ids = points.trajectory_id.iloc[template_new_idx].values
    existing_trajectory_ids = templates.trajectory_id.values

    # Identify new trajectory_ids not already in templates
    is_new = ~np.isin(trajectory_ids, existing_trajectory_ids)
    if np.any(is_new):
        new_templates = templates_new[is_new]
        new_trajectory_ids = trajectory_ids[is_new]
        height, width = (
            (template_shape, template_shape)
            if isinstance(template_shape, int)
            else template_shape
        )
        # Create DataArray for new templates
        new_data = xr.DataArray(
            data=new_templates,
            dims=("trajectory_id", "height", "width"),
            coords={
                "trajectory_id": new_trajectory_ids,
                "height": np.arange(height),
                "width": np.arange(width),
            },
        )
        # Append new templates
        logger.info(f"Adding {new_data.shape[0]} new templates")
        templates = xr.concat([templates, new_data], dim="trajectory_id")
    return templates

@log_execution_time
def update_templates(templates, templates_new, template_new_idx, points):
    """
    Update existing templates in the templates DataArray.

    Parameters:
        templates (xr.DataArray): Existing templates DataArray to update.
        templates_new (np.ndarray): Array of templates to update.
        template_new_idx (array-like): Indices of points corresponding to templates_new.
        points (pd.DataFrame): DataFrame containing trajectory IDs.

    Returns:
        xr.DataArray: Updated templates DataArray with existing templates updated.
    """
    # Extract trajectory_ids for the templates to update
    trajectory_ids = points.trajectory_id.iloc[template_new_idx].values
    existing_trajectory_ids = templates.trajectory_id.values

    # Identify existing trajectory_ids
    is_existing = np.isin(trajectory_ids, existing_trajectory_ids)
    if np.any(is_existing):
        update_templates_array = templates_new[is_existing]
        update_trajectory_ids = trajectory_ids[is_existing]
        logger.info(f"Updating {update_trajectory_ids.shape[0]} templates")
        # Update existing templates
        templates.loc[dict(trajectory_id=update_trajectory_ids)] = update_templates_array
    return templates

@log_execution_time
def pattern_matching(points, img, templates, points_fg1, band='s0_HV', hs=16, border=16):
    """
    Perform pattern matching on a single image band with template rotation search

    Parameters:
        points: GeoDataFrame containing point geometries.
        img: Image object with band data and transformation methods.
        templates: Collection of templates corresponding to each point.
        points_fg1: DataFrame with angle information for each point.
        band (str): The image band to use for matching (default 's0_HV').
        hs (int): Half-size of the template.
        border (int): Border size added to the search window.

    Returns:
        tuple: A tuple containing:
            - corrected_positions_xy (numpy.ndarray): Array of corrected (x, y) positions.
            - corrected_positions_colsrows (numpy.ndarray): Array of corrected pixel coordinates.
            - correlation_values (numpy.ndarray): Array of correlation scores for each match.
    """
    mtype = cv2.TM_CCOEFF_NORMED

    pc = ccrs.PlateCarree()
    nps = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
    x, y = points.geometry.x.values, points.geometry.y.values
    
    if points.crs == img.srs:
        colsrows = np.array(img.transform_points(x, y, DstToSrc=1)).T
    else:
        lon, lat, _ = pc.transform_points(nps, x, y).T
        colsrows = np.array(img.transform_points(lon, lat, DstToSrc=1)).T
    image_1 = img[band]

    # Define rotation angles to test as a simple list
    rotation_angles = [0, -15, 15, -30, 30]
    
    # Correlation threshold for early exit
    correlation_threshold = 0.65

    # Prepare all arguments for processing keypoints
    all_args = [
        (i, (colsrows[i][0], colsrows[i][1]),
         points_fg1['angle'].iloc[i] - img.angle,
         templates[i])
        for i in range(len(colsrows))
    ]

    results = []

    for args in all_args:
        i, (c1, r1), angle_diff, template = args

        # Check dimensions first (quick failure)
        c1_int, r1_int = int(c1), int(r1)
        sub_img = image_1[r1_int-hs-border : r1_int+hs+1+border, c1_int-hs-border : c1_int+hs+1+border]

        if template.shape != (hs * 2 + 1, hs * 2 + 1) or \
           sub_img.shape != (hs * 2 + 1 + border * 2, hs * 2 + 1 + border * 2):
            results.append(([0, 0], 0.0, 0.0))
            continue

        template_np = template.to_numpy().astype(np.uint8)

        # Variables to track best match
        best_corr = -1
        best_dc = 0
        best_dr = 0
        best_rotation = 0.0

        # Process each rotation angle in sequence
        for angle_offset in rotation_angles:
            # Calculate the actual rotation angle
            rotation_angle = angle_diff + angle_offset
            
            # Early exit if we already found a good enough match
            if best_corr >= correlation_threshold:
                break
                
            # Special case for no rotation (angle_offset = 0)
            if angle_offset == 0:
                # Use the original template without rotation
                result = cv2.matchTemplate(sub_img, template_np, mtype)
            else:
                # Rotate the template
                h, w = template_np.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                rotated_template = cv2.warpAffine(template_np, M, (w, h))
                mask = (rotated_template > 0).astype(np.uint8)
                result = cv2.matchTemplate(sub_img, rotated_template, mtype, mask=mask)
                
                        # Find the maximum correlation and its position
            mccr, mccc = np.unravel_index(result.argmax(), result.shape)
            mcc = result[mccr, mccc]

            # Update best match if this is better
            if mcc > best_corr:
                best_corr = mcc
                best_dc = mccc - border
                best_dr = mccr - border
                best_rotation = angle_offset
                
            # Fast exit for high correlation during first (no rotation) check
            if angle_offset == 0 and mcc > correlation_threshold:
                break

        results.append(([best_dc, best_dr], best_corr, best_rotation))

    # Unpack results
    corrections, correlation_values, rotation_found = zip(*results)
    corrections = np.array(corrections)
    correlation_values = np.array(correlation_values)

    # Calculate corrected positions
    corrected_colsrows = corrections + colsrows
    x_corr, y_corr = img.transform_points(
        corrected_colsrows[:, 0], corrected_colsrows[:, 1],
        DstToSrc=0, dst_srs=NSR(3413)
    )

    corrected_positions_xy = np.column_stack((x_corr, y_corr))

    # Log statistics
    rotation_array = np.array(rotation_found)
    logger.info(f"Rotation stats - Mean: {np.mean(rotation_array):.2f}°, "
                f"Std: {np.std(rotation_array):.2f}°, "
                f"Max: {np.max(np.abs(rotation_array)):.2f}°")
    logger.info(f"Correlation stats - Mean: {np.mean(correlation_values):.4f}, "
                f"Min: {np.min(correlation_values):.4f}")

    return corrected_positions_xy, corrected_colsrows, correlation_values