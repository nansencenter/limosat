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
from.deformation import filter_and_interpolate_flipped_triangles

import numpy as np
from numpy.linalg import norm
import pandas as pd
import geopandas as gpd
import xarray as xr
import cv2
import cartopy.crs as ccrs
from shapely.geometry import Polygon, MultiPoint
from matplotlib import pyplot as plt
from skimage.util import view_as_windows
from skimage.transform import AffineTransform
from nansat import Nansat, NSR
from matplotlib.tri import Triangulation

# Constants for commonly used values
CANDIDATE_SEARCH_MAX_DAILY_DRIFT_M_PER_DAY = 10000
WINDOW_SIZE = 64
BORDER_SIZE = 128
STRIDE = 10
OCTAVE = 8
MIN_CORRELATION = 0.35
TEMPLATE_SIZE = 16  # Half size
TEMPORAL_WINDOW = 4
DESCRIPTOR_SIZE = 32
RESPONSE_THRESHOLD = 0
MAX_INTERPOLATION_TIME_GAP_HOURS = 24
BORDER_MATCHED = 16
BORDER_INTERPOLATED = 32

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
                 candidate_search_max_daily_drift_m=CANDIDATE_SEARCH_MAX_DAILY_DRIFT_M_PER_DAY,
                 descriptor_size=DESCRIPTOR_SIZE,
                 window_size=WINDOW_SIZE,
                 border_size=BORDER_SIZE,
                 border_matched=BORDER_MATCHED,
                 border_interpolated=BORDER_INTERPOLATED,
                 stride=STRIDE,
                 octave=OCTAVE,
                 min_correlation=MIN_CORRELATION,
                 response_threshold=RESPONSE_THRESHOLD,
                 template_size=TEMPLATE_SIZE,
                 use_interpolation=True,
                 max_interpolation_time_gap_hours= MAX_INTERPOLATION_TIME_GAP_HOURS,
                 return_insitu_points_on_completion=False
                ):
        self.points = points
        self.templates = templates
        self.model = model
        self.matcher = matcher
        self.persist_updates = persist_updates
        self.persist_interval = persist_interval
        self.pruning_interval = pruning_interval
        self._last_persisted_id = 0
        self.min_correlation = min_correlation
        self.response_threshold = response_threshold
        self.use_interpolation = use_interpolation
        self.run_name = run_name
        self.insitu_points = insitu_points
        self.window_size = window_size
        self.border_size = border_size
        self.border_matched = border_matched
        self.border_interpolated = border_interpolated
        self.descriptor_size = descriptor_size
        self.temporal_window = temporal_window
        self.candidate_search_max_daily_drift_m = candidate_search_max_daily_drift_m
        self.stride = stride
        self.octave = octave
        self.max_interpolation_time_gap_hours = max_interpolation_time_gap_hours
        self.return_insitu_points_on_completion = return_insitu_points_on_completion
        self.template_size = template_size
        
        # Initialize the KeypointDetector
        self.keypoint_detector = KeypointDetector(model=model)

        # Initialize trajectory_id column in insitu_points if in validation mode
        if self.insitu_points is not None:
            if 'trajectory_id' not in self.insitu_points.columns:
                self.insitu_points['trajectory_id'] = pd.NA
                self.insitu_points['trajectory_id'] = self.insitu_points['trajectory_id'].astype(pd.Int64Dtype())
            logger.info("Validation mode: 'trajectory_id' column ensured in self.insitu_points.")

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
        self.points and self.templates.
        
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

        # Create Nansat Image object from file
        img = Image(filename)
        
        # Calculate buffer to capture points drifting INTO image
        max_possible_drift = self.candidate_search_max_daily_drift_m * self.temporal_window 
        buffer_distance = min(max_possible_drift, self.matcher.spatial_distance_max)
        logger.debug(f"Using buffer distance: {buffer_distance/1000.0:.2f} km "
                    f"(Min of {max_possible_drift/1000.0:.1f}km theoretical max drift over {self.temporal_window} days "
                    f"and {self.matcher.spatial_distance_max/1000.0:.1f}km spatial match limit)")
    
        # Filter for active points within the buffered polygon and time threshold
        buffered_image_poly = img.poly.buffer(buffer_distance)
        time_threshold = img.date - pd.Timedelta(days=self.temporal_window)
        points_last = self.points.last()
        points_poly = points_last[
            points_last.within(buffered_image_poly) &
            (points_last['time'] >= time_threshold)
        ]

        # this is to prevent memory errors for rare instances with many overlapping active points
        CANDIDATE_POINT_LIMIT = 40000
        if len(points_poly) > CANDIDATE_POINT_LIMIT:
            logger.info(
                f"Candidate points ({len(points_poly)}) exceed limit. "
                f"Taking a random sample of {CANDIDATE_POINT_LIMIT} points"
            )
            points_poly = points_poly.sample(n=CANDIDATE_POINT_LIMIT, random_state=42)

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
                        logger.debug(f"Interpolation stats: {interp_count}/{total_count} points ({interp_count/total_count:.1%}) were interpolated")

        # Process new points
        self._process_new_points(points_final, img, image_id, basename)

        # Template pruning
        if image_id > 0 and image_id % self.pruning_interval == 0 and self.templates.trajectory_id.size > 0:
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
                    logger.debug(f"Pruning complete: Kept {num_after} templates / Removed {num_before - num_after} templates (were old OR inactive).")
                else:
                    logger.debug("Pruning check: All templates recent or active.")

            except Exception as e:
                # Basic catch-all for unexpected errors during pruning
                logger.error(f"Error during template pruning: {e}", exc_info=True)
                logger.warning("Skipping template pruning for this interval due to error.")

        # Persist database every X images
        if self.persist_updates:
            current_points_image_id = self.points.last_image_id
            images_since_persist = current_points_image_id - self._last_persisted_id
            if images_since_persist >= self.persist_interval:
                current_last_image_id = current_points_image_id
                try:
                    save_successful = self.db.save(
                        self.points,
                        self.templates,
                        self._last_persisted_id,
                        self.insitu_points
                    )

                    if save_successful:
                        # Prune points
                        points_delta_count = len(self.points[self.points['image_id'] > self._last_persisted_id])
                        logger.info(f"Database persistence successful for {points_delta_count} new/updated points.")
                        num_before_prune = len(self.points)
                        keep_mask_points = (self.points['is_last'] == 1) & \
                                           (self.points['time'] >= time_threshold) # Use existing time_threshold
                        points_to_keep = self.points[keep_mask_points].copy()
                        self.points = Keypoints._from_gdf(points_to_keep)
                        num_after_prune = len(self.points)
                        logger.debug(
                            f"In-memory points pruned: {num_before_prune} -> {num_after_prune} "
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
        coords_new = self.keypoint_detector.detect_new_keypoints(
            points=points_final,
            img=img,
            window_size=self.window_size,
            border_size=self.border_size,
            response_threshold=self.response_threshold,
            octave=self.octave
        )
        logger.debug(f"Detected {len(coords_new)} new keypoints from image {image_id}")

        if self.insitu_points is not None:
            matching_insitu_points = self.insitu_points.loc[
                self.insitu_points['image_filepath'].isin([basename])
            ]

            if len(matching_insitu_points) > 0:
                logger.info(f"Found {len(matching_insitu_points)} matching buoy observations in image {basename}")
                buoy_kps = self.keypoint_detector.keypoint_from_point(matching_insitu_points, self.descriptor_size, octave=self.octave,img=img, response_threshold=self.response_threshold)
                if buoy_kps is None:
                    logger.error("keypoint_from_point returned None unexpectedly!")
                    buoy_kps = []
                coords_new = coords_new + buoy_kps
                logger.info(f"Added {len(buoy_kps)} buoy keypoints to detection set")

        keypoints_coords_arr, descriptors_arr, surviving_tags = compute_descriptors(
            coords_new,
            img,
            self.model,
            polarisation=1, 
            normalize=False 
        )

        if keypoints_coords_arr is not None and descriptors_arr is not None:
            points_new = Keypoints.create(keypoints_coords_arr, descriptors_arr, img, image_id)
            
            current_self_points_len = len(self.points)
            self.points = self.points.append(points_new) # This assigns final trajectory_ids to the points_new portion
            
            appended_points_gdf = self.points.iloc[current_self_points_len:]
            logger.info(f"Added {len(appended_points_gdf)} new points (total: {len(self.points)})")

            # Link insitu_points to final trajectory_ids using surviving_tags
            if self.insitu_points is not None and surviving_tags is not None and not appended_points_gdf.empty:
                appended_points_gdf_reset = appended_points_gdf.reset_index(drop=True)
                
                if len(surviving_tags) == len(appended_points_gdf_reset):
                    for i, original_df_idx_tag in enumerate(surviving_tags):
                        if original_df_idx_tag is not None: # Check if the tag is an actual index
                            final_tid = appended_points_gdf_reset.iloc[i]['trajectory_id']
                            self.insitu_points.loc[original_df_idx_tag, 'trajectory_id'] = final_tid
                            logger.debug(f"Linked insitu_point (original index {original_df_idx_tag}) to trajectory_id {final_tid}")
                else:
                    logger.warning(f"Mismatch between surviving_tags ({len(surviving_tags)}) and "
                                   f"appended_points_gdf_reset ({len(appended_points_gdf_reset)}). Skipping insitu linking for this batch.")

            templates_new, template_new_idx = extract_templates(appended_points_gdf, img, band=1, hs=self.template_size)
            self.templates = append_templates(
                self.templates,
                templates_new,
                template_new_idx,
                points_new,
                template_shape=self.template_size*2+1
            )

        if len(points_final) > 0:
            self.points = self.points.update(points_final)
            templates_final, template_final_idx = extract_templates(points_final, img, band=1, hs=self.template_size)
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
            border_size=self.border_size,
            size=self.descriptor_size,
            octave=self.octave,
        )

        if not coords_grid:
            logger.warning("No gridded keypoints detected. Skipping matching for this image.")
            return None

        keypoints_coords_arr_grid, descriptors_grid, _surviving_tags_grid = compute_descriptors(
            coords_grid,
            img,
            self.model,
            polarisation=1,
            normalize=False
        )

        if keypoints_coords_arr_grid is None or descriptors_grid is None:
            logger.warning("Failed to compute descriptors for grid points. Skipping matching.")
            return None

        points_grid = Keypoints.create(keypoints_coords_arr_grid, descriptors_grid, img, image_id=image_id)

        # Match and filter points
        points_fg1, points_fg2 = self.matcher.match_with_grid(points_poly, points_grid)

        if points_fg1 is None or points_fg2 is None or points_fg1.empty or points_fg2.empty:
            logger.info("Insufficient match quality or no matches found, skipping point matching step.")
            return Keypoints()

        # Assign trajectory IDs to matched points
        points_fg2['trajectory_id'] = points_fg1.trajectory_id.values
        points_fg2['interpolated'] = 0
        points_matched = points_fg2

        # Interpolate drift if needed
        if self.use_interpolation and len(points_fg1) < len(points_poly):
            points_matched = interpolate_drift(
                points_poly=points_poly,
                points_fg1=points_fg1,
                points_fg2=points_matched,
                img=img,
                max_interpolation_time_gap_hours=self.max_interpolation_time_gap_hours,
                model_type=AffineTransform,
                border_size=self.border_size
            )
            if points_matched is None or points_matched.empty:
                logger.warning("Interpolation resulted in no valid points. Proceeding with only matched points.")
                points_matched = points_fg2  # Fall back to only matched points
        else:
            if not self.use_interpolation:
                logger.info("Interpolation disabled, using only matched points")
            elif len(points_fg1) >= len(points_poly):
                logger.info("All points matched, no interpolation needed")

        # Filter out points that don't have templates
        all_traj_ids = points_matched.trajectory_id.values
        available_traj_ids = self.templates.trajectory_id.values
        has_template_mask = np.isin(all_traj_ids, available_traj_ids)

        if not np.all(has_template_mask):
            points_matched = points_matched[has_template_mask]
            all_traj_ids = points_matched.trajectory_id.values
            
        if points_matched.empty:
            logger.debug("No points remaining after template filtering.")
            return Keypoints() 

        # Get original points and templates for pattern matching
        points_orig = points_poly[points_poly.trajectory_id.isin(all_traj_ids)]
        templates_all = self.templates.sel(trajectory_id=all_traj_ids) 
        
        if not (len(points_matched) == len(points_orig) == len(templates_all.trajectory_id)):
            logger.error(f"Length mismatch before pattern_matching! "
                        f"points_matched: {len(points_matched)}, "
                        f"points_orig: {len(points_orig)}, "
                        f"templates_all: {len(templates_all.trajectory_id)}")
            return Keypoints() 

        # Perform pattern matching
        keypoints_corrected_xy, keypoints_corrected_rc, corr_values = pattern_matching(
            points_matched,
            img,
            templates_all,
            points_orig,
            hs=self.template_size,
            border_matched=self.border_matched,
            border_interpolated=self.border_interpolated,
            band=1
        )

        # Initial correlation filter
        points_matched['corr'] = corr_values
        correlation_mask = corr_values >= self.min_correlation
        
        if not np.any(correlation_mask):
            logger.debug("No points passed correlation filter.")
            return Keypoints()

        # Apply correlation filter
        points_matched = points_matched[correlation_mask].copy()
        keypoints_corrected_xy = keypoints_corrected_xy[correlation_mask]
        
        # Prepare data for triangulation filter
        points_matched = pd.merge(
            points_matched,
            points_orig[['trajectory_id', 'geometry']].rename(columns={'geometry': 'geometry_orig'}),
            on='trajectory_id', how='left'
        )
        
        # Call the filter to mask bad vectors and propose interpolated positions
        x1_interp, y1_interp, was_interpolated_mask = filter_and_interpolate_flipped_triangles(
            points_matched.geometry_orig.x.to_numpy(),
            points_matched.geometry_orig.y.to_numpy(),
            keypoints_corrected_xy[:, 0],
            keypoints_corrected_xy[:, 1],
        )

        # Separate points into "good" (not interpolated) and "re-check" (interpolated)
        good_mask = ~was_interpolated_mask & ~np.isnan(x1_interp)
        
        # Handle good points
        points_good = points_matched[good_mask].copy()
        xy_good = np.column_stack((x1_interp[good_mask], y1_interp[good_mask]))
        rc_good = keypoints_corrected_rc[correlation_mask][good_mask]

        # Handle points that need re-checking
        if np.any(was_interpolated_mask):
            logger.info(f"Re-validating {np.sum(was_interpolated_mask)} interpolated vectors...")
            
            points_to_recheck = points_matched[was_interpolated_mask].copy()
            points_to_recheck['geometry'] = gpd.points_from_xy(x1_interp[was_interpolated_mask], y1_interp[was_interpolated_mask])
            templates_recheck = self.templates.sel(trajectory_id=points_to_recheck.trajectory_id.values)
            
            points_fg1_recheck = points_matched[was_interpolated_mask][['trajectory_id', 'geometry_orig', 'angle']]
            points_fg1_recheck.rename(columns={'geometry_orig': 'geometry'}, inplace=True)

            xy_rechecked, rc_rechecked, corr_rechecked = pattern_matching(
                points_to_recheck,
                img, 
                templates_recheck, 
                points_fg1_recheck,
                hs=self.template_size,
                border_matched=self.border_matched,
                border_interpolated=self.border_interpolated,
                band=1
            )
            
            recheck_passed_mask = corr_rechecked >= self.min_correlation

            num_passed = np.sum(recheck_passed_mask)
            logger.debug(f"Re-validation: {num_passed}/{len(points_to_recheck)} interpolated vectors passed.")
            
            # Combine survivors from both groups
            points_rechecked = points_to_recheck[recheck_passed_mask].copy()
            points_rechecked['corr'] = corr_rechecked[recheck_passed_mask]

            # Combine final results
            points_matched = pd.concat([points_good, points_rechecked], ignore_index=True)
            corrected_xy = np.vstack([xy_good, xy_rechecked[recheck_passed_mask]])
            corrected_rc = np.vstack([rc_good, rc_rechecked[recheck_passed_mask]])
        else:
            # No re-checking needed - use good points directly
            points_matched = points_good
            corrected_xy = xy_good
            corrected_rc = rc_good

        # Final checks
        if points_matched.empty:
            logger.debug("No points survived filtering.")
            return Keypoints()
                        
        # Update geometry with corrected positions
        points_matched = points_matched.drop(columns=['geometry_orig'])
        points_matched['geometry'] = gpd.points_from_xy(corrected_xy[:, 0], corrected_xy[:, 1])
        
        logger.info(f"Pattern matching kept {len(points_matched)} points (correlation >= {self.min_correlation})")

        # Recompute descriptors for final points
        if not points_matched.empty:
            keypoints_list_with_tags_recompute = [
                (cv2.KeyPoint(px_c, px_r, size=self.descriptor_size, angle=img.angle, octave=self.octave), None)
                for px_c, px_r in corrected_rc
            ]
            
            _raw_kps_recomputed, new_descriptors, _surviving_tags_recomputed = compute_descriptors(
                keypoints_list_with_tags_recompute,
                img,
                self.model,
                polarisation=1, 
                normalize=False 
            )
        else:
            new_descriptors = None

        # Filter points based on successful descriptor computation

        original_count = len(points_matched)

        if new_descriptors is not None and len(new_descriptors) == len(points_matched):
            points_matched['descriptors'] = list(new_descriptors)
            
            # Filter out points with invalid descriptors
            valid_mask_desc = points_matched['descriptors'].apply(lambda d: isinstance(d, np.ndarray))
            points_matched = points_matched[valid_mask_desc].copy()
            
            if len(points_matched) < original_count:
                logger.debug(f"Removed {original_count - len(points_matched)} points with invalid descriptors")
        elif original_count > 0:
            logger.warning("Descriptor computation failed; removing all remaining points.")
            points_matched = points_matched.iloc[0:0]  # Empty DataFrame

        # Update templates for surviving points
        if not points_matched.empty:
            new_templates, new_template_idx = extract_templates(points_matched, img, hs=self.template_size, band=1)
            
            if len(new_template_idx) == len(points_matched):
                self.templates = update_templates(self.templates, new_templates, new_template_idx, points_matched)
            else:
                logger.warning("Template extraction index mismatch. Skipping template update.")

        logger.debug(f"Returning {len(points_matched)} final points")
        return points_matched

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
        coords = np.vstack([
            corners_nx_meters[[0, 2, 3, 1]],
            corners_ny_meters[[0, 2, 3, 1]]
        ]).T

        # Create and return the polygon
        poly = Polygon(coords)

        return poly
        
class Keypoints(gpd.GeoDataFrame):
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
            if 'time' in self.columns:
                self['time'] = pd.to_datetime(self['time'], errors='coerce')
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

    def __init__(self, model):
        """
        Initialize the keypoint detector.

        Parameters:
            model: Feature detection model (e.g., SIFT, ORB)
            octave (int): The octave value for keypoints
        """
        self.model = model
        self.pc = ccrs.PlateCarree()
        central_longitude = -45
        true_scale_latitude = 70
        self.nps = ccrs.NorthPolarStereo(
            central_longitude=central_longitude,
            true_scale_latitude=true_scale_latitude
        )
        
    @log_execution_time
    def detect_new_keypoints(
        self,
        points,
        img,
        octave,
        window_size,
        border_size,
        response_threshold,
        step=None,
        adjust_keypoint_angle=True,
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

        Returns:
            list: List of tuples (cv2.KeyPoint, None)
        """
        img0 = img[1]
        img0[np.isnan(img0)] = 0
        
        # TODO: generalise preprocessing to create boolean landmask
        if img.bands().get(2, {'name': 'none'}).get('name') == 'mask':
            landmask = img[2]
            img0[landmask == 2] = 0

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
                # Filter keypoints based on response
                kps = [kp for kp in kps if kp.response > response_threshold]
                if len(kps) != 0:
                    kp_best = max(kps, key=lambda kp: kp.response)
                    # Adjust keypoint coordinates to the original image coordinates
                    x_offset = x_start
                    y_offset = y_start
                    kp_best.pt = (kp_best.pt[0] + x_offset, kp_best.pt[1] + y_offset)
                    if adjust_keypoint_angle:
                        # combine img and feature angle
                        kp_best.angle = img.angle
                    kp_best.octave = octave
                    keypoints.append((kp_best, None)) # Append tuple with None as tag

        # Filter keypoints to remove ones too close to the image edges
        filtered_keypoints_with_tags = [
            item
            for item in keypoints
            if (
                border_size <= item[0].pt[0] <= img0.shape[1] - border_size
                and border_size <= item[0].pt[1] <= img0.shape[0] - border_size
            )
        ]

        return filtered_keypoints_with_tags

    @log_execution_time
    def detect_gridded_points(
        self,
        img,
        stride,
        size,
        octave,
        border_size,
    ):
        """
        Generate gridded points over the input image.

        Parameters:
            img (Image): Image to generate points for
            stride (int): Distance between grid points
            size (int): Size of keypoints
            border_size (int): Border to exclude

        Returns:
            list: List of tuples (cv2.KeyPoint, None)
        """
        # Get image data and replace NaNs with zero
        img0 = img[1]
        img0[np.isnan(img0)] = 0
        
        # TODO: generalise preprocessing to create boolean landmask
        if img.bands().get(2, {'name': 'none'}).get('name') == 'mask':
            landmask = img[2]
            img0[landmask == 2] = 0

        # Generate grid keypoints
        keypoints = []
        for r in range(0, img0.shape[0], stride):
            for c in range(0, img0.shape[1], stride):
                kp = cv2.KeyPoint(r, c, size=size, octave=octave, angle=img.angle)
                keypoints.append((kp, None)) # Append tuple with None as tag

        # Filter keypoints within image borders
        filtered_keypoints_with_tags = [
            item
            for item in keypoints
            if (
                border_size <= item[0].pt[0] <= img0.shape[1] - border_size
                and border_size <= item[0].pt[1] <= img0.shape[0] - border_size
            )
        ]

        return filtered_keypoints_with_tags

    def get_pixel_coords(self, nansat_obj, geom_x, geom_y):
        # helper function for keypoint_from_point
        x_arr, y_arr = np.atleast_1d(geom_x), np.atleast_1d(geom_y)
        source_crs_cartopy = ccrs.CRS(nansat_obj.srs.ExportToProj4())
        geographic_crs_cartopy = ccrs.PlateCarree()
        lon_lat_alt_array = geographic_crs_cartopy.transform_points(source_crs_cartopy, x_arr, y_arr)
        cols, rows = nansat_obj.transform_points(lon_lat_alt_array[:, 0], lon_lat_alt_array[:, 1], DstToSrc=1)
        if cols.size > 0 and np.isfinite(cols[0]) and np.isfinite(rows[0]):
            return int(round(cols[0])), int(round(rows[0])) # Return rounded integer coords
    
    def keypoint_from_point(
            self, # KeypointDetector instance
            points_gdf_for_current_image, # GDF of buoy points for *this* image
            descriptor_size, # Passed by ImageProcessor, for cv2.KeyPoint.size & descriptor area check
            octave,          # Passed by ImageProcessor, for cv2.KeyPoint.octave
            img,              # Nansat image object for the current image
            response_threshold
        ):
        """
        Dynamically finds the best ORB feature near each buoy point using a method similar to detect_new_keypoints
        Returns list of tuples: (cv2.KeyPoint, original_df_index)
        """
        # this is the window it searches
        patch_size_for_detection = 32
        keypoints_with_indices = []
        orb_model = self.model
        img_band_data = img[1]
        img_height, img_width = img_band_data.shape

        for original_idx, point_row in points_gdf_for_current_image.iterrows():
            buoy_geom = point_row.geometry
            if buoy_geom is None:
                continue

            try:
                buoy_col_px_float, buoy_row_px_float = self.get_pixel_coords(img, buoy_geom.x, buoy_geom.y)
                if buoy_col_px_float is None:
                    continue

                patch_half = patch_size_for_detection // 2
                r0 = max(0, int(round(buoy_row_px_float)) - patch_half)
                r1 = min(img_height, int(round(buoy_row_px_float)) + patch_half + (patch_size_for_detection % 2))
                c0 = max(0, int(round(buoy_col_px_float)) - patch_half)
                c1 = min(img_width, int(round(buoy_col_px_float)) + patch_half + (patch_size_for_detection % 2))

                if not ((r1 - r0) >= patch_size_for_detection * 0.8 and \
                        (c1 - c0) >= patch_size_for_detection * 0.8):
                    continue 

                patch = img_band_data[r0:r1, c0:c1]
                if patch.size == 0:
                    continue
                
                kps_in_patch = orb_model.detect(patch, None)
                
                best_kp_in_patch = None
                max_response_found = -1.0 

                if kps_in_patch:
                    for kp_candidate in kps_in_patch:
                        if kp_candidate.response >= response_threshold:
                            if kp_candidate.response > max_response_found:
                                max_response_found = kp_candidate.response
                                best_kp_in_patch = kp_candidate
                
                if best_kp_in_patch is not None:
                    kp_final_c = c0 + best_kp_in_patch.pt[0]
                    kp_final_r = r0 + best_kp_in_patch.pt[1]

                    # Boundary check for the keypoint for subsequent descriptor computation
                    orb_desc_computation_half_patch = int(descriptor_size / 2) 

                    if not (orb_desc_computation_half_patch <= kp_final_c < img_width - orb_desc_computation_half_patch and \
                            orb_desc_computation_half_patch <= kp_final_r < img_height - orb_desc_computation_half_patch):
                        continue 

                    final_kp_to_add = cv2.KeyPoint(
                        x=float(kp_final_c),
                        y=float(kp_final_r),
                        size=float(descriptor_size), 
                        octave=int(octave),
                        angle=float(img.angle),
                        response=float(best_kp_in_patch.response)
                    )
                    keypoints_with_indices.append((final_kp_to_add, point_row.name))
                
            except Exception as e:
                logger.info(f"Error processing point original_idx {original_idx} (input index {point_row.name}): {e}")
                pass 
                
        return keypoints_with_indices

class Matcher:
    def __init__(self,
                 # General matching parameters
                 norm=cv2.NORM_HAMMING2,
                 descriptor_distance_max=120,
                 spatial_distance_max=100000,

                 # Homography estimation parameters
                 model=AffineTransform,
                 model_threshold=10000,
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
            self.estimation_method = None
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

        # Match
        matches = self.match_with_crosscheck(x0, x1)

        # Filter
        rc_idx0, rc_idx1, residuals = self.filter(matches, pos0, pos1)

        # If filter returned None or we don't have enough matches, try Lowe's ratio test
        if rc_idx0 is None or (rc_idx0.size / pos0.shape[0] < 0.1):
            matches = self.match_with_lowe_ratio(matches, x0, x1, pos0, pos1)
            rc_idx0, rc_idx1, residuals = self.filter(matches, pos0, pos1)
            if rc_idx0 is not None:
                logger.debug(f"After Lowe's ratio test, number of matches: {rc_idx0.size}")
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
    def match_with_lowe_ratio(self, matches_bf_initial, x0, x1, pos0, pos1):
        """
        Applies a Lowe's ratio-like test using k=4 neighbors to find additional matches.
        Relies on the subsequent self.filter call for descriptor and spatial distance filtering.
        """
        index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        search_params = {}
        knn_matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Get k=4 nearest neighbors for each descriptor in x0 from x1
        all_knn_matches = knn_matcher.knnMatch(x0, x1, k=4)

        new_matches_from_lowe_variant = []

        for mmm in all_knn_matches: 
            if len(mmm) < 2: # Need at least two matches to perform a ratio test
                continue

            # Iterate through the first k-1 candidates (m_candidate)
            # and compare each to its immediate successor (m_next)
            for midx, m_candidate in enumerate(mmm[:-1]):
                m_next = mmm[midx+1] # The DMatch object to compare against

                # Apply the Lowe's ratio test
                if m_candidate.distance < self.lowe_ratio * m_next.distance:
                    new_matches_from_lowe_variant.append(m_candidate)
                    break
        
        # Combine initial crosscheck matches with the new ones from the Lowe's variant
        combined_matches = list(matches_bf_initial) # Start with a copy
        existing_query_indices_from_bf = {m.queryIdx for m in matches_bf_initial}

        for m_lowe in new_matches_from_lowe_variant:
            if m_lowe.queryIdx not in existing_query_indices_from_bf:
                combined_matches.append(m_lowe)
        
        return combined_matches

    @log_execution_time
    def filter(self, matches, pos0, pos1):
        bf_idx0 = np.array([m.queryIdx for m in matches]) # Indices into pos0 for ALL 'matches'
        bf_idx1 = np.array([m.trainIdx for m in matches]) # Indices into pos1 for ALL 'matches'
        logger.debug(f'Matching: Number of matches: {bf_idx0.size}')

        # Filter by descriptor distance
        # descriptor_distance is already calculated for ALL 'matches' (initial crosscheck)
        descriptor_distance = np.array([m.distance for m in matches])
        gpi0_desc_filter_mask = descriptor_distance < self.descriptor_distance_max 
        dd_idx0 = bf_idx0[gpi0_desc_filter_mask] # Indices of points passing descriptor distance filter
        dd_idx1 = bf_idx1[gpi0_desc_filter_mask]
        logger.debug(f'Distance: Number of matches: {dd_idx0.size}')
        if dd_idx0.size == 0: # No points passed descriptor distance filter
            if not self.use_model_estimation: return dd_idx0, dd_idx1, None
            return None, None, None
        
        # Filter by spatial distance
        # Calculate spatial distances ONLY for those that passed the descriptor distance filter
        current_spatial_distances = np.hypot(pos1[dd_idx1, 0] - pos0[dd_idx0, 0], 
                                            pos1[dd_idx1, 1] - pos0[dd_idx0, 1])
        gpi1_spatial_filter_mask = current_spatial_distances < self.spatial_distance_max # Mask relative to dd_idx arrays
        md_idx0 = dd_idx0[gpi1_spatial_filter_mask] # Indices of points passing spatial (and descriptor) filter
        md_idx1 = dd_idx1[gpi1_spatial_filter_mask]

        if not self.use_model_estimation:
            return md_idx0, md_idx1, None # md_idx0, md_idx1 are the final indices if no model estimation

        # Filter by homography
        if md_idx0.size < 4: # Check if enough points for homography
            logger.warning("Warning: Insufficient matches for model estimation (minimum 4 required)")
            return None, None, None
        try:
            # H, inliers is your gpi2, applied to md_idx0/md_idx1
            # The `inliers` mask returned by findHomography is relative to the points fed into it (pos0[md_idx0], pos1[md_idx1])
            if self.estimation_method_name.upper() == "DEGENSAC":
                try:
                    import pydegensac
                    H, inliers_mask_homography_relative = pydegensac.findHomography(pos0[md_idx0], pos1[md_idx1], self.model_threshold)
                except ImportError:
                    logger.warning("pydegensac not found, falling back to cv2.USAC_MAGSAC")
                    self.estimation_method_name = "USAC_MAGSAC"
                    self.estimation_method = cv2.USAC_MAGSAC
                    H, inliers_mask_homography_relative = cv2.findHomography(pos0[md_idx0], pos1[md_idx1], 
                                                                            self.estimation_method, self.model_threshold)
            else:
                H, inliers_mask_homography_relative = cv2.findHomography(pos0[md_idx0], pos1[md_idx1], 
                                                                        self.estimation_method, self.model_threshold)

            if H is None:
                logger.warning("Warning: Model estimation failed")
                return None, None, None
            
            inliers_mask_homography_relative = inliers_mask_homography_relative.ravel().astype(bool)

            # Points KEPT by homography
            # these are indices into the original pos0/pos1 arrays
            rc_idx0 = md_idx0[inliers_mask_homography_relative]
            rc_idx1 = md_idx1[inliers_mask_homography_relative]
                
            model = self.model(H) # Assuming self.model is AffineTransform class
            residuals = model.residuals(pos0[rc_idx0], pos1[rc_idx1])
            logger.info(f'{self.estimation_method_name}: Number of matches: {rc_idx0.size}')
            
            if self.plot: # Your existing plot call
                self.plot_quiver(pos0[rc_idx0], pos1[rc_idx1], residuals)
                
            return rc_idx0, rc_idx1, residuals

        except cv2.error as e:
            logger.error(f"Warning: OpenCV error during model estimation: {str(e)}")
            return None, None, None
        except Exception as e:
            logger.error(f"Warning: Unexpected error during model estimation: {str(e)}")
            return None, None, None
        
@log_execution_time
def interpolate_drift(points_poly, points_fg1, points_fg2, img, 
                                        max_interpolation_time_gap_hours,
                                        model_type=AffineTransform,
                                        max_interpolation_speed_m_per_day=50000,
                                        max_anchor_distance_km=20.0,
                                        border_size=BORDER_SIZE
                                        ):
    points_result = points_fg2.copy()
    points_result['interpolated'] = 0 

    MIN_SAMPLES_FOR_AFFINE = 10
    
    if len(points_fg1) < MIN_SAMPLES_FOR_AFFINE: 
        if len(points_fg1) == 0 and (points_poly.empty or len(points_fg1) == len(points_poly)):
             logger.info("interpolate_drift: No valid anchor points for ATM. Skipping interpolation.")
        else:
             logger.info(f"interpolate_drift: Not enough matched points ({len(points_fg1)}) for ATM. Min: {MIN_SAMPLES_FOR_AFFINE}. Skipping.")
        return points_result

    try:
        src_pts_for_atm_geodf = points_poly.loc[points_fg1.index]
        if src_pts_for_atm_geodf.geometry.isna().any() or points_fg2.geometry.isna().any():
            logger.error("interpolate_drift: NaN geometry in ATM anchor points. Skipping.")
            return points_result

        pos0_atm = np.array(list(src_pts_for_atm_geodf.geometry.apply(lambda p: p.coords[0])))
        pos1_atm = np.array(list(points_fg2.geometry.apply(lambda p: p.coords[0])))
        
        atm = model_type()
        if not atm.estimate(pos0_atm, pos1_atm):
            logger.warning("interpolate_drift: Affine transformation estimation failed.")
            return points_result
    except Exception as e:
        logger.error(f"interpolate_drift: Error during ATM estimation: {e}", exc_info=True)
        return points_result

    unmatched_mask = ~points_poly.index.isin(points_fg1.index)
    unmatched_points_current_stage = points_poly[unmatched_mask].copy() 

    if unmatched_points_current_stage.empty:
        logger.debug("interpolate_drift: No unmatched points to consider for interpolation.")
        return points_result

    try:
        unmatched_points_current_stage = unmatched_points_current_stage[unmatched_points_current_stage.within(img.poly)]
    except Exception as e_bounds:
        logger.warning(f"interpolate_drift: Error checking image bounds: {e_bounds}. Proceeding unfiltered.")

    if unmatched_points_current_stage.empty:
        logger.debug("interpolate_drift: No unmatched points within image bounds."); return points_result

    current_img_time_naive = img.date.tz_localize(None) if img.date.tzinfo else img.date
    unmatched_times_naive = unmatched_points_current_stage['time'].dt.tz_localize(None) 
    time_gap_mask = (current_img_time_naive - unmatched_times_naive).dt.total_seconds() <= (max_interpolation_time_gap_hours * 3600)
    unmatched_points_current_stage = unmatched_points_current_stage[time_gap_mask]
    
    logger.debug(f"interpolate_drift: {len(unmatched_points_current_stage)} unmatched points after bounds & time gap filters.")
    if unmatched_points_current_stage.empty: return points_result

    max_anchor_distance_meters = max_anchor_distance_km * 1000.0
    if not src_pts_for_atm_geodf.geometry.empty and len(src_pts_for_atm_geodf.geometry) >= 3:
        try:
            anchor_cloud_multipoint = MultiPoint(src_pts_for_atm_geodf.geometry.tolist())
            convex_hull_of_anchors = anchor_cloud_multipoint.convex_hull
            distances_to_hull_edge = unmatched_points_current_stage.geometry.distance(convex_hull_of_anchors)
            near_anchor_hull_mask = distances_to_hull_edge <= max_anchor_distance_meters
            
            num_discarded = len(unmatched_points_current_stage) - np.sum(near_anchor_hull_mask)
            unmatched_points_current_stage = unmatched_points_current_stage[near_anchor_hull_mask]
            if num_discarded > 0:
                 logger.debug(f"interpolate_drift: Discarded {num_discarded} points far from anchor hull.")
        except Exception as e_hull: # Catches errors from MultiPoint, convex_hull, or distance
            logger.warning(f"interpolate_drift: Error during hull processing or distance calc: {e_hull}. Skipping filter.")
    else:
        logger.debug(f"interpolate_drift: Not enough anchors ({len(src_pts_for_atm_geodf.geometry)}) for hull filter.")

    if unmatched_points_current_stage.empty:
        logger.debug("interpolate_drift: No points remaining after anchor distance filter."); return points_result
        
    pos0_to_interp = np.column_stack((unmatched_points_current_stage.geometry.x, unmatched_points_current_stage.geometry.y))
    interpolated_pos_geo_coords_np = atm(pos0_to_interp)

    prev_times_series = unmatched_points_current_stage['time']
    prev_geoms_series = unmatched_points_current_stage.geometry
    prev_times_for_calc_naive = prev_times_series.dt.tz_localize(None)
    time_diff_seconds_series = (current_img_time_naive - prev_times_for_calc_naive).dt.total_seconds()
    time_diff_days_series = pd.Series(time_diff_seconds_series / (24.0 * 60.0 * 60.0), index=unmatched_points_current_stage.index)
    
    new_interpolated_geoms_series = gpd.GeoSeries(
        gpd.points_from_xy(interpolated_pos_geo_coords_np[:,0], interpolated_pos_geo_coords_np[:,1]),
        crs=prev_geoms_series.crs, index=unmatched_points_current_stage.index)
    
    displacements_m_series = prev_geoms_series.distance(new_interpolated_geoms_series)
    speeds_m_per_day_series = pd.Series(np.nan, index=unmatched_points_current_stage.index)
    valid_time_diff_mask = time_diff_days_series > 1e-9
    if valid_time_diff_mask.any():
        speeds_m_per_day_series.loc[valid_time_diff_mask] = \
            displacements_m_series[valid_time_diff_mask] / time_diff_days_series[valid_time_diff_mask]

    speed_filter_mask_series = (speeds_m_per_day_series <= max_interpolation_speed_m_per_day) & speeds_m_per_day_series.notna()
    unmatched_points_after_velocity_filter = unmatched_points_current_stage[speed_filter_mask_series]
    interpolated_pos_geo_coords_after_velocity_filter_np = interpolated_pos_geo_coords_np[speed_filter_mask_series.values] 

    num_discarded_speed = len(unmatched_points_current_stage) - len(unmatched_points_after_velocity_filter)
    if num_discarded_speed > 0:
        logger.debug(f"Post-interp velocity check: Discarded {num_discarded_speed} points.")
    if unmatched_points_after_velocity_filter.empty:
        logger.debug("interpolate_drift: No points remaining after velocity check."); return points_result
        
    points_final_interpolated = Keypoints()
    if not unmatched_points_after_velocity_filter.empty:
        cols_final_pixel, rows_final_pixel = img.transform_points(
            interpolated_pos_geo_coords_after_velocity_filter_np[:, 0], 
            interpolated_pos_geo_coords_after_velocity_filter_np[:, 1], 
            DstToSrc=1, dst_srs=img.srs
        )
        
        height, width = img.shape()

        pixel_valid_mask_np = (cols_final_pixel >= border_size) & (cols_final_pixel < width - border_size) & \
                              (rows_final_pixel >= border_size) & (rows_final_pixel < height - border_size) & \
                              np.isfinite(cols_final_pixel) & np.isfinite(rows_final_pixel)

        final_valid_metadata_df = unmatched_points_after_velocity_filter[pixel_valid_mask_np]
        final_valid_pixel_coords_np = np.column_stack((cols_final_pixel[pixel_valid_mask_np], rows_final_pixel[pixel_valid_mask_np]))
        final_valid_geo_coords_np = interpolated_pos_geo_coords_after_velocity_filter_np[pixel_valid_mask_np]
        
        if not final_valid_metadata_df.empty:
            base_image_id_val = points_result['image_id'].iloc[0]
            points_final_interpolated = Keypoints.create(
                final_valid_pixel_coords_np, 
                [None] * len(final_valid_metadata_df), img, base_image_id_val 
            )
            points_final_interpolated['trajectory_id'] = final_valid_metadata_df.trajectory_id.values
            points_final_interpolated['interpolated'] = 1
            points_final_interpolated.geometry = gpd.points_from_xy(
                 final_valid_geo_coords_np[:,0], final_valid_geo_coords_np[:,1], crs=points_poly.crs
            )
            logger.info(f"Successfully interpolated and kept {len(points_final_interpolated)} points.")
        else:
            logger.info("interpolate_drift: No points passed final pixel boundary checks.")
    
    if not points_final_interpolated.empty:
        points_final_combined_df = pd.concat(
            [points_result, points_final_interpolated], ignore_index=True, sort=False 
        )
        current_crs = points_poly.crs
        if isinstance(points_final_combined_df, gpd.GeoDataFrame):
            combined_gdf = points_final_combined_df.set_crs(current_crs, allow_override=True)
        else:
            combined_gdf = gpd.GeoDataFrame(points_final_combined_df, geometry='geometry', crs=current_crs)
        return Keypoints._from_gdf(combined_gdf)
    else:
        return points_result

@log_execution_time
def compute_descriptors(keypoints_with_tags, img, model, polarisation='s0_HV', normalize=True):
    """
    Compute descriptors for given keypoints in an image using a specified model.

    Parameters:
        keypoints_with_tags (list): List of tuples (cv2.KeyPoint, Optional[tag]) to compute descriptors for.
        img (dict): Dictionary containing image bands. Must include 's0_HH' and 's0_HV' if dual polarisation is used.
        model (object): Descriptor model with a `compute` method (e.g., OpenCV's SIFT or ORB).
        polarisation (str, optional):
            - 's0_HH' or 's0_HV' or INT for band number for single polarisation.
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
            - surviving_tags (list or None):
                - List of tags corresponding to the keypoints that survived descriptor computation.
                - Returns None if descriptor computation fails.
    """
    if not keypoints_with_tags:
        return None, None, None

    # Extract raw cv2.KeyPoint objects for model.compute
    raw_cv2_kps = [item[0] for item in keypoints_with_tags]

    # Helper to find original tags for surviving keypoints
    def get_tags_for_survivors(original_kps_with_tags_list, surviving_cv2_kps_list):
        ordered_survivor_tags = []
        # Create a temporary list of original (kp, tag) tuples to manage matches
        # This helps if surviving_cv2_kps_list is not in the same order or is a subset
        temp_originals = list(original_kps_with_tags_list)
        
        for surv_kp in surviving_cv2_kps_list:
            found_match = False
            for i, (orig_kp, orig_tag) in enumerate(temp_originals):
                # Compare points using a small tolerance
                if np.allclose(surv_kp.pt, orig_kp.pt, atol=0.5):
                    ordered_survivor_tags.append(orig_tag)
                    temp_originals.pop(i) # Remove found match to avoid re-matching
                    found_match = True
                    break
            if not found_match:
                # This case implies a surviving keypoint from model.compute() did not match any original input keypoint
                # based on .pt comparison, or its .pt was significantly altered.
                logger.warning(f"compute_descriptors: Could not find original tag for a surviving keypoint at {surv_kp.pt}. Appending None as tag.")
                ordered_survivor_tags.append(None)
        
        if len(ordered_survivor_tags) != len(surviving_cv2_kps_list):
             logger.error(f"compute_descriptors: Tag mapping length mismatch. Survivors: {len(surviving_cv2_kps_list)}, Tags found: {len(ordered_survivor_tags)}")
             # This is a critical issue, potentially return all Nones or raise
             return [None] * len(surviving_cv2_kps_list) # Fallback
        return ordered_survivor_tags

    if polarisation == 'dual_polarisation':
        img_hh = img['s0_HH'].copy()
        img_hh[np.isnan(img_hh)] = 0
        keypoints_output_hh, descriptors_hh = model.compute(img_hh, raw_cv2_kps) # Pass raw kps
        if descriptors_hh is None or not keypoints_output_hh: # Check if keypoints_output_hh is empty
            return None, None, None
        
        rawkeypoints_coords = np.array([kp.pt for kp in keypoints_output_hh])
        if normalize:
            descriptors_hh = descriptors_hh - descriptors_hh.mean(axis=1, keepdims=True)

        # Compute descriptors for 's0_HV'
        # For HV, we need to compute descriptors for the SAME keypoints that survived HH.
        # So, we pass keypoints_output_hh to the second compute call.
        img_hv = img['s0_HV'].copy()
        img_hv[np.isnan(img_hv)] = 0
        # keypoints_output_hv should ideally be the same as keypoints_output_hh if all succeed for HV.
        # If model.compute filters keypoints, this ensures consistency.
        keypoints_output_hv, descriptors_hv = model.compute(img_hv, keypoints_output_hh)
        if descriptors_hv is None:
            return None, None, None # Return three values
        
        # The keypoints_output_hh are the survivors from the HH pass.
        # Now compute HV descriptors for *these specific survivors*.
        img_hv = img['s0_HV'].copy()
        img_hv[np.isnan(img_hv)] = 0
        # Pass keypoints_output_hh (survivors from HH) to HV computation
        # _ (ignored returned keypoints for HV), descriptors_hv
        # model.compute should ideally not filter keypoints_output_hh further if it's just computing descriptors.
        # If it does, it means some of keypoints_output_hh didn't get an HV descriptor.
        keypoints_output_hv_check, descriptors_hv = model.compute(img_hv, keypoints_output_hh) 

        if descriptors_hv is None:
            return None, None, None # Failed to get HV descriptors for the HH survivors

        # Critical check: Ensure descriptors_hv corresponds to keypoints_output_hh
        # If model.compute for HV somehow returned fewer keypoints than keypoints_output_hh,
        # we have a mismatch. For now, assume it returns descriptors for all inputs if successful.
        if len(descriptors_hh) != len(descriptors_hv):
            logger.error(f"Dual-pol descriptor length mismatch: HH ({len(descriptors_hh)}), HV ({len(descriptors_hv)}) "
                         f"for {len(keypoints_output_hh)} input HH survivors. This indicates an issue.")
            return None, None, None


        if normalize:
            descriptors_hh = descriptors_hh - descriptors_hh.mean(axis=1, keepdims=True)
            descriptors_hv = descriptors_hv - descriptors_hv.mean(axis=1, keepdims=True)

        descriptors = np.hstack([descriptors_hh, descriptors_hv])
        
        # Get tags for the keypoints that survived the HH pass (keypoints_output_hh)
        surviving_tags = get_tags_for_survivors(keypoints_with_tags, keypoints_output_hh)
        
        return rawkeypoints_coords, descriptors, surviving_tags

    else: # Single band processing
        img_band = img[polarisation].copy()
        img_band[np.isnan(img_band)] = 0
        keypoints_output, descriptors = model.compute(img_band, raw_cv2_kps)
        if descriptors is None or not keypoints_output:
            return None, None, None
            
        rawkeypoints_coords = np.array([kp.pt for kp in keypoints_output])
        if normalize:
            descriptors = descriptors - descriptors.mean(axis=1, keepdims=True)
            
        surviving_tags = get_tags_for_survivors(keypoints_with_tags, keypoints_output)
        
        return rawkeypoints_coords, descriptors, surviving_tags

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
def append_templates(templates, templates_new, template_new_idx, points, template_shape=TEMPLATE_SIZE):
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
                "time": ("trajectory_id", points['time'].iloc[template_new_idx[is_new]].values),
                "height": np.arange(height),
                "width": np.arange(width),
            },
        )
        # Append new templates
        logger.debug(f"Adding {new_data.shape[0]} new templates")
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
        logger.debug(f"Updating {update_trajectory_ids.shape[0]} templates")
        current_time = points['time'].iloc[template_new_idx[is_existing]].iloc[0] # Get time from points df
        templates['time'].loc[dict(trajectory_id=update_trajectory_ids)] = current_time
        # Then update the template data itself
        templates.loc[dict(trajectory_id=update_trajectory_ids)] = update_templates_array
    return templates

@log_execution_time
def pattern_matching(
    points: gpd.GeoDataFrame,
    img,                                 
    templates: xr.DataArray,             
    points_fg1: gpd.GeoDataFrame,         
    band: str = 's0_HV',
    hs: int = TEMPLATE_SIZE,
    border_matched: int = 16,
    border_interpolated: int = 32
):
    """
    Perform pattern matching on a single image band with template rotation search.
    Uses 'points_fg1' to get the angle of the point in the PREVIOUS frame.
    Uses a larger search border for points marked as 'interpolated' (flag in 'points' GDF).

    Parameters:
        points (gpd.GeoDataFrame): Current estimated point locations. Must contain 'trajectory_id' and 'interpolated' flag.
        img (Image): Current image object.
        templates (xr.DataArray): Templates indexed by trajectory_id.
        points_fg1 (gpd.GeoDataFrame): Points from the previous frame, containing 'trajectory_id' and 'angle'.
        band (str): Image band for matching.
        hs (int): Template half-size.
        border_matched (int): Default search border padding.
        border_interpolated (int): Larger search border padding for interpolated points.

    Returns:
        tuple: (corrected_positions_xy, corrected_positions_colsrows, correlation_values)
    """
    mtype = cv2.TM_CCOEFF_NORMED
    image_band_data = img[band]

    pc_crs_pm = ccrs.PlateCarree()
    nps_crs_pm = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)

    try:
        # Get pixel coordinates for the *current* estimated positions in 'points'
        x_proj_all = points.geometry.x.values
        y_proj_all = points.geometry.y.values
        transformed_to_geo_all = pc_crs_pm.transform_points(nps_crs_pm, x_proj_all, y_proj_all)
        lons_all = transformed_to_geo_all[:, 0]
        lats_all = transformed_to_geo_all[:, 1]
        cols_all, rows_all = img.transform_points(lons_all, lats_all, DstToSrc=1)
        valid_transform_mask = np.isfinite(cols_all) & np.isfinite(rows_all)
        if not np.all(valid_transform_mask):
            logger.warning(f"Pattern Matching: {np.sum(~valid_transform_mask)} points had non-finite pixel coords.")
    except Exception as e_transform_all:
        logger.error(f"Pattern Matching: Error during bulk coordinate transformation for current points: {e_transform_all}")
        n_pts = len(points)
        return np.full((n_pts, 2), np.nan), np.full((n_pts, 2), np.nan), np.full(n_pts, -1.0)

    colsrows = np.column_stack((cols_all, rows_all)) # Pixel coords for 'points'

    rotation_angles = [0, -15, 15, -30, 30]
    early_exit_correlation_threshold = 0.65 # If a rotation gives this, stop searching more rotations

    results_list = []

    prev_angle_map = {}
    if points_fg1 is not None and not points_fg1.empty and 'trajectory_id' in points_fg1.columns and 'angle' in points_fg1.columns:
        unique_prev_states = points_fg1.drop_duplicates(subset=['trajectory_id'], keep='first')
        prev_angle_map = pd.Series(unique_prev_states.angle.values, index=unique_prev_states.trajectory_id).to_dict()


    for i in range(len(points)):
        point_row = points.iloc[i] # This is from 'points' (current estimates)
        current_tid = point_row.trajectory_id

        c1_initial, r1_initial = colsrows[i, 0], colsrows[i, 1]

        if not valid_transform_mask[i]:
            results_list.append((([0, 0], -1.0, 0.0)))
            continue

        # Determine the effective border size for this point
        current_border = border_matched
        # Use a larger border if the point is interpolated
        if 'interpolated' in point_row and point_row['interpolated'] == 1:
            current_border = border_interpolated

        # Get the previous angle for this trajectory_id from the map created from points_fg1
        angle_prev_for_tid = prev_angle_map.get(current_tid, None)

        angle_diff = angle_prev_for_tid - img.angle

        try:
            # Select template based on current_tid

            template_data_array = templates.sel(trajectory_id=current_tid)
            # Ensure it's not an empty selection and convert to NumPy
            if template_data_array.trajectory_id.size == 0: # Check if selection is empty
                 raise KeyError(f"Template for TID {current_tid} resulted in empty selection.")
            template_np = template_data_array.data.astype(np.uint8)
        except KeyError:
            # logger.warning(f"Pattern Matching: Template not found for TID {current_tid}. Skipping.")
            results_list.append((([0, 0], -1.0, 0.0)))
            continue
        
        if template_np.shape != (hs * 2 + 1, hs * 2 + 1):
            # logger.warning(f"Pattern Matching: Template for TID {current_tid} has unexpected shape {template_np.shape}. Expected ({hs*2+1}, {hs*2+1}). Skipping.")
            results_list.append( ([0, 0], -1.0, 0.0) )
            continue

        r1_int, c1_int = int(r1_initial), int(c1_initial)
        # Use current_border to define search sub-image
        r_start_search = r1_int - hs - current_border
        r_end_search = r1_int + hs + 1 + current_border
        c_start_search = c1_int - hs - current_border
        c_end_search = c1_int + hs + 1 + current_border

        # Check if search window is within image bounds
        if not (r_start_search >= 0 and r_end_search <= image_band_data.shape[0] and \
                c_start_search >= 0 and c_end_search <= image_band_data.shape[1]):
            results_list.append((([0, 0], -1.0, 0.0)))
            continue

        sub_img = image_band_data[r_start_search:r_end_search, c_start_search:c_end_search]

        # Expected shape of sub_img uses current_border
        expected_sub_img_shape = (hs * 2 + 1 + current_border * 2, hs * 2 + 1 + current_border * 2)
        if sub_img.shape != expected_sub_img_shape:
            logger.warning(f"Pattern Matching: Extracted sub_img for TID {current_tid} has shape {sub_img.shape}, expected {expected_sub_img_shape}. Skipping.")
            results_list.append((([0, 0], -1.0, 0.0)))
            continue

        best_corr_for_point = -1.0 # Initialize with a value lower than any possible correlation
        best_dc_for_point = 0
        best_dr_for_point = 0
        best_rotation_offset_for_point = 0.0

        for angle_offset in rotation_angles:
            current_rotation_to_apply = angle_diff + angle_offset

            if best_corr_for_point >= early_exit_correlation_threshold and angle_offset != 0 : # check 0 rot first
                break 
                
            if abs(current_rotation_to_apply) < 1e-3:
                rotated_template = template_np
                result_match = cv2.matchTemplate(sub_img, rotated_template, mtype)
            else:
                h_t, w_t = template_np.shape
                center_t = (w_t // 2, h_t // 2)
                M = cv2.getRotationMatrix2D(center_t, current_rotation_to_apply, 1.0)
                rotated_template = cv2.warpAffine(template_np, M, (w_t, h_t))
                # Create mask for rotated template to handle empty areas after rotation
                mask = (rotated_template > 0).astype(np.uint8) # Simple mask, assumes 0 is background
                if np.sum(mask) < (0.5 * h_t * w_t): # If less than 50% of template is valid after rotation
                    continue # Skip this rotation if too much of template is lost
                result_match = cv2.matchTemplate(sub_img, rotated_template, mtype, mask=mask)

            _, current_max_corr, _, max_loc_in_result = cv2.minMaxLoc(result_match)

            if current_max_corr > best_corr_for_point:
                best_corr_for_point = current_max_corr
                # Correction is from center of search window to center of matched template
                best_dc_for_point = (max_loc_in_result[0] + hs) - (hs + current_border)
                best_dr_for_point = (max_loc_in_result[1] + hs) - (hs + current_border)
                best_rotation_offset_for_point = angle_offset # Store the offset that gave the best result
            
            if angle_offset == 0 and best_corr_for_point >= early_exit_correlation_threshold:
                break # Early exit if non-rotated version is already good enough

        results_list.append( (([best_dc_for_point, best_dr_for_point], best_corr_for_point, best_rotation_offset_for_point)) )

    # Unpack results
    if not results_list: # Should not happen if len(points) > 0
        n_pts = len(points)
        return np.full((n_pts, 2), np.nan), np.full((n_pts, 2), np.nan), np.full(n_pts, -1.0)

    corrections_rc_list, correlation_values_list, rotation_found_list = zip(*results_list)
    
    corrections_rc_np = np.array(corrections_rc_list) # Nx2 array of (dc, dr) pixel corrections
    correlation_values_np = np.array(correlation_values_list)
    rotation_offset_found_np = np.array(rotation_found_list)

    # Initial pixel coordinates (before correction)
    initial_colsrows_np = colsrows # This is from all points passed to the function

    # Corrected pixel positions (col, row)
    corrected_colsrows_np = initial_colsrows_np + corrections_rc_np

    # Convert corrected pixel positions back to projected XY coordinates
    try:
        x_corr_proj, y_corr_proj = img.transform_points(
            corrected_colsrows_np[:, 0], corrected_colsrows_np[:, 1],
            DstToSrc=0, dst_srs=img.srs
        )
        corrected_positions_xy_np = np.column_stack((x_corr_proj, y_corr_proj))
    except Exception as e_final_transform:
        logger.error(f"Pattern Matching: Error transforming corrected pixel coords to projected: {e_final_transform}")
        # Fill with NaNs if final transform fails
        corrected_positions_xy_np = np.full((len(points), 2), np.nan)

    # Log statistics
    valid_correlations_for_stats = correlation_values_np[np.isfinite(correlation_values_np) & (correlation_values_np > -1.0)]
    if len(valid_correlations_for_stats) > 0:
        matched_mask_for_stats = np.isfinite(correlation_values_np) & (correlation_values_np > -1.0)
        rotation_stats = rotation_offset_found_np[matched_mask_for_stats] # Use only for matched points
        logger.debug(f"Rotation Offset Stats (for matched) - Mean: {np.mean(rotation_stats):.2f}°, "
                    f"Std: {np.std(rotation_stats):.2f}°")
        logger.debug(f"Correlation Stats (for matched) - Mean: {np.mean(valid_correlations_for_stats):.4f}, "
                    f"Min: {np.min(valid_correlations_for_stats):.4f}, Max: {np.max(valid_correlations_for_stats):.4f}")
    else:
        logger.info("Pattern Matching: No valid correlations found to report stats.")

    return corrected_positions_xy_np, corrected_colsrows_np, correlation_values_np