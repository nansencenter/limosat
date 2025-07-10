# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import cv2
from skimage.transform import AffineTransform
from .utils import log_execution_time, logger
from .database import DriftDatabase
from .deformation import filter_and_interpolate_flipped_triangles
from .image import Image
from .keypoints import Keypoints
from .keypoint_detector import KeypointDetector
from .templates import Templates
from .processing import interpolate_drift, pattern_matching

class ImageProcessor:
    """
    Image processor for ice drift analysis.
    """
    def __init__(self,
                 points,
                 model,
                 matcher,
                 config=None,
                 engine=None,
                 zarr_path=None,
                 run_name=None,
                 insitu_points=None,
                 return_insitu_points_on_completion=False,
                 **kwargs
                ):
        self.points = points
        self.templates = Templates()
        self.model = model
        self.matcher = matcher
        self.run_name = run_name
        self.insitu_points = insitu_points
        self.return_insitu_points_on_completion = return_insitu_points_on_completion

        # Define default parameters
        default_params = {
            'persist_updates': True,
            'persist_interval': 10,
            'pruning_interval': 10,
            'temporal_window': 3,
            'candidate_search_max_daily_drift_m': 10000,
            'descriptor_size': 32,
            'window_size': 64,
            'border_size': 128,
            'border_matched': 16,
            'border_interpolated': 64,
            'stride': 15,
            'octave': 8,
            'min_correlation': 0.4,
            'response_threshold': 0.0001,
            'template_size': 16,
            'use_interpolation': True,
            'max_interpolation_time_gap_hours': 25,
        }

        # Start with defaults, update from config, then from kwargs
        proc_params = default_params.copy()
        if config and 'image_processor_params' in config:
            proc_params.update(config['image_processor_params'])
        proc_params.update(kwargs)

        # Set attributes from the final parameters
        for key, value in proc_params.items():
            setattr(self, key, value)

        self._last_persisted_id = 0
        
        # Initialize the KeypointDetector
        self.keypoint_detector = KeypointDetector(model=model)

        # Initialize trajectory_id column in insitu_points if in validation mode
        if self.insitu_points is not None:
            if 'trajectory_id' not in self.insitu_points.columns:
                self.insitu_points['trajectory_id'] = pd.NA
                self.insitu_points['trajectory_id'] = self.insitu_points['trajectory_id'].astype(pd.Int64Dtype())
            logger.info("Validation mode: 'trajectory_id' column ensured in self.insitu_points.")

        # Create DriftDatabase instance if persistence is enabled
        if self.persist_updates:
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
        if image_id > 0 and image_id % self.pruning_interval == 0 and len(self.templates) > 0:
            try:
                time_threshold_prune = img.date.to_numpy() - np.timedelta64(self.temporal_window, 'D')
                active_traj_ids = self.points.loc[self.points['is_last'] == 1, 'trajectory_id'].unique()
                self.templates.prune(active_traj_ids, time_threshold_prune)
            except Exception as e:
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
        # Detect new keypoints (raw)
        raw_kps_new = self.keypoint_detector.detect_new_keypoints(
            points=points_final,
            img=img,
            window_size=self.window_size,
            border_size=self.border_size,
            response_threshold=self.response_threshold,
            octave=self.octave,
            # Set to False because we will compute descriptors in a batch later
            compute_descriptors=False 
        )
        logger.debug(f"Detected {len(raw_kps_new)} new raw keypoints from image {image_id}")

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
                raw_kps_new = raw_kps_new + buoy_kps
                logger.info(f"Added {len(buoy_kps)} buoy keypoints to detection set")

        # Now, compute descriptors for the combined list of new and buoy keypoints
        keypoints_coords_arr, descriptors_arr, surviving_tags = self.keypoint_detector.compute_descriptors(
            raw_kps_new,
            img,
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

            # Add templates for the newly created points
            self.templates.add(appended_points_gdf, img, self.template_size, band=1)

        if len(points_final) > 0:
            self.points = self.points.update(points_final)
            # Update templates for the points that were successfully matched and updated
            self.templates.update(points_final, img, self.template_size, band=1)
            
    @log_execution_time
    def _match_existing_points(self, points_poly, img, image_id):
        """
        Match points from previous image to current one, with optional interpolation,
        then apply pattern matching and filter by correlation. This version uses the
        empirically correct trajectory_id assignment and robust descriptor filtering.
        """
        # Detect gridded keypoints and compute their descriptors in one step
        keypoints_coords_arr_grid, descriptors_grid, _ = self.keypoint_detector.detect_gridded_points(
            img,
            stride=self.stride,
            border_size=self.border_size,
            size=self.descriptor_size,
            octave=self.octave,
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
                border_size=self.border_size,
                model_type=AffineTransform
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
        available_traj_ids = self.templates.trajectory_ids
        has_template_mask = np.isin(all_traj_ids, available_traj_ids)

        if not np.all(has_template_mask):
            points_matched = points_matched[has_template_mask]
            all_traj_ids = points_matched.trajectory_id.values
            
        if points_matched.empty:
            logger.debug("No points remaining after template filtering.")
            return Keypoints() 

        # Get original points and templates for pattern matching
        points_orig = points_poly[points_poly.trajectory_id.isin(all_traj_ids)]
        templates_all = self.templates.get_by_id(all_traj_ids)
        
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
            templates_recheck = self.templates.get_by_id(points_to_recheck.trajectory_id.values)
            
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
            
            _raw_kps_recomputed, new_descriptors, _surviving_tags_recomputed = self.keypoint_detector.compute_descriptors(
                keypoints_list_with_tags_recompute,
                img,
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
            self.templates.update(points_matched, img, self.template_size, band=1)

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
