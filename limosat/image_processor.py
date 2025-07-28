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
            'max_valid_speed_m_per_day': 50000.0,
        }

        # Start with defaults, update from config, then from kwargs
        proc_params = default_params.copy()
        if config and 'image_processor_params' in config:
            proc_params.update(config['image_processor_params'])
        proc_params.update(kwargs)

        # Add new parameters for detection method
        self.top_n = proc_params.pop('top_n', 1) # Default to 1 for new points

        # Set attributes from the final parameters for clarity and static analysis
        self.persist_updates = proc_params['persist_updates']
        self.persist_interval = proc_params['persist_interval']
        self.pruning_interval = proc_params['pruning_interval']
        self.temporal_window = proc_params['temporal_window']
        self.candidate_search_max_daily_drift_m = proc_params['candidate_search_max_daily_drift_m']
        self.window_size = proc_params['window_size']
        self.border_size = proc_params['border_size']
        self.border_matched = proc_params['border_matched']
        self.border_interpolated = proc_params['border_interpolated']
        self.stride = proc_params['stride']
        self.octave = proc_params['octave']
        self.min_correlation = proc_params['min_correlation']
        self.response_threshold = proc_params['response_threshold']
        self.template_size = proc_params['template_size']
        self.use_interpolation = proc_params['use_interpolation']
        self.max_interpolation_time_gap_hours = proc_params['max_interpolation_time_gap_hours']
        self.max_valid_speed_m_per_day = proc_params['max_valid_speed_m_per_day']

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
        basename = os.path.basename(filename)
        if image_id <= self.points.last_image_id:
            logger.info(f"Skipping image {image_id}: {basename}")
            return
        logger.info(f"Processing image {image_id}: {basename}")
        img = Image(filename)
        max_possible_drift = self.candidate_search_max_daily_drift_m * self.temporal_window 
        buffer_distance = min(max_possible_drift, self.matcher.spatial_distance_max)
        buffered_image_poly = img.poly.buffer(buffer_distance)
        time_threshold = img.date - pd.Timedelta(days=self.temporal_window)
        points_last = self.points.last()
        points_poly = points_last[
            points_last.within(buffered_image_poly) &
            (points_last['time'] >= time_threshold)
        ]
        CANDIDATE_POINT_LIMIT = 40000
        if len(points_poly) > CANDIDATE_POINT_LIMIT:
            points_poly = points_poly.sample(n=CANDIDATE_POINT_LIMIT, random_state=42)
        if len(points_poly) == 0:
            points_final = Keypoints()
        else:
            points_final = self._match_existing_points(points_poly, img, image_id, img.orbit_num)
            if points_final is None:
                points_final = Keypoints()
        self._process_new_points(points_final, img, image_id, basename)
        if image_id > 0 and image_id % self.pruning_interval == 0 and len(self.templates) > 0:
            try:
                time_threshold_prune = img.date.to_numpy() - np.timedelta64(self.temporal_window, 'D')
                active_traj_ids = self.points.loc[self.points['is_last'] == 1, 'trajectory_id'].unique()
                self.templates.prune(active_traj_ids, time_threshold_prune)
            except Exception as e:
                logger.error(f"Error during template pruning: {e}", exc_info=True)
        if self.persist_updates:
            current_points_image_id = self.points.last_image_id
            images_since_persist = current_points_image_id - self._last_persisted_id
            if images_since_persist >= self.persist_interval:
                current_last_image_id = current_points_image_id
                traj_id_counts = self.points['trajectory_id'].value_counts()
                matched_traj_ids = traj_id_counts[traj_id_counts > 1].index
                points_to_persist = self.points[self.points['trajectory_id'].isin(matched_traj_ids)]
                save_successful = self.db.save(
                    points_to_persist, self.templates, self._last_persisted_id, self.insitu_points
                )
                if save_successful:
                    try:
                        keep_mask = (self.points['is_last'] == 1) & (self.points['time'] >= time_threshold)
                        self.points = Keypoints._from_gdf(self.points[keep_mask].copy())
                        self._last_persisted_id = current_last_image_id
                    except Exception as e:
                        logger.error(f"Error during in-memory point pruning: {e}", exc_info=True)

    @log_execution_time
    def _process_new_points(self, points_final, img, image_id, basename):
        raw_kps_new = self.keypoint_detector.detect_keypoints(
            img=img,
            points=points_final,
            window_size=self.window_size,
            stride=self.window_size,
            border_size=self.border_size,
            response_threshold=self.response_threshold,
            octave=self.octave,
            top_n=1,
            check_existing=True,
            compute_descriptors=False
        )
        if self.insitu_points is not None:
            matching_insitu_points = self.insitu_points.loc[
                self.insitu_points['image_filepath'].isin([basename])
            ]
            if len(matching_insitu_points) > 0:
                buoy_kps = self.keypoint_detector.keypoint_from_point(
                    matching_insitu_points, octave=self.octave, img=img, response_threshold=self.response_threshold
                )
                if buoy_kps:
                    raw_kps_new.extend(buoy_kps)
        keypoints_coords_arr, descriptors_arr, surviving_tags = self.keypoint_detector.compute_descriptors(
            raw_kps_new, img, polarisation=1, normalize=False
        )
        if keypoints_coords_arr is not None and descriptors_arr is not None:
            points_new = Keypoints.create(keypoints_coords_arr, descriptors_arr, img, image_id, img.orbit_num)
            current_self_points_len = len(self.points)
            self.points = self.points.append(points_new)
            appended_points_gdf = self.points.iloc[current_self_points_len:]
            if self.insitu_points is not None and surviving_tags is not None and not appended_points_gdf.empty:
                appended_points_gdf_reset = appended_points_gdf.reset_index(drop=True)
                if len(surviving_tags) == len(appended_points_gdf_reset):
                    for i, original_df_idx_tag in enumerate(surviving_tags):
                        if original_df_idx_tag is not None:
                            final_tid = appended_points_gdf_reset.iloc[i]['trajectory_id']
                            self.insitu_points.loc[original_df_idx_tag, 'trajectory_id'] = final_tid
            self.templates.add(appended_points_gdf, img, self.template_size, band=1)
        if len(points_final) > 0:
            self.points = self.points.update(points_final)
            self.templates.update(points_final, img, self.template_size, band=1)
            
    @log_execution_time
    def _match_existing_points(self, points_poly, img, image_id, current_orbit_num):
        if 'orbit_num' not in points_poly.columns:
            return Keypoints()
        previous_orbit_nums = points_poly['orbit_num']
        orbit_filter_mask = previous_orbit_nums != current_orbit_num
        points_poly_filtered = points_poly[orbit_filter_mask]
        if points_poly_filtered.empty:
            return Keypoints()
        
        keypoints_coords_arr_grid, descriptors_grid, _ = self.keypoint_detector.detect_keypoints(
            img=img,
            window_size=self.window_size,
            stride=self.stride,
            border_size=self.border_size,
            response_threshold=self.response_threshold,
            octave=self.octave,
            top_n=self.top_n,
            check_existing=False,
            compute_descriptors=True
        )
        if keypoints_coords_arr_grid is None or descriptors_grid is None:
            return Keypoints()

        points_grid = Keypoints.create(keypoints_coords_arr_grid, descriptors_grid, img, image_id=image_id, orbit_num=img.orbit_num)
        points_fg1, points_fg2, _ = self.matcher.match_with_grid(points_poly_filtered, points_grid)
        if points_fg1 is None or points_fg2 is None or points_fg1.empty or points_fg2.empty:
            return Keypoints()

        points_fg2['trajectory_id'] = points_fg1.trajectory_id.values
        points_fg2['interpolated'] = 0
        points_matched = points_fg2

        if self.use_interpolation and len(points_fg1) < len(points_poly_filtered):
            points_matched = interpolate_drift(
                points_poly=points_poly_filtered, points_fg1=points_fg1, points_fg2=points_matched,
                img=img, max_interpolation_time_gap_hours=self.max_interpolation_time_gap_hours,
                border_size=self.border_size, model_type=AffineTransform
            )
            if points_matched is None or points_matched.empty:
                points_matched = points_fg2
        
        if not points_matched.empty:
            candidates_with_history = pd.merge(
                points_matched, points_poly_filtered[['trajectory_id', 'geometry', 'time']],
                on='trajectory_id', suffixes=('', '_prev')
            )
            time_diff_days = (img.date - candidates_with_history['time_prev']).dt.total_seconds() / 86400.0
            distance_m = candidates_with_history.geometry.distance(candidates_with_history.geometry_prev)
            speed_m_per_day = np.divide(
                distance_m, time_diff_days, out=np.full_like(time_diff_days, np.inf), where=time_diff_days > 1e-9
            )
            valid_trajectory_ids = candidates_with_history.loc[speed_m_per_day <= self.max_valid_speed_m_per_day, 'trajectory_id']
            points_matched = points_matched[points_matched['trajectory_id'].isin(valid_trajectory_ids)]
        
        all_traj_ids = points_matched.trajectory_id.values
        available_traj_ids = self.templates.trajectory_ids
        has_template_mask = np.isin(all_traj_ids, available_traj_ids)
        if not np.all(has_template_mask):
            points_matched = points_matched[has_template_mask]
            all_traj_ids = points_matched.trajectory_id.values
        if points_matched.empty:
            return Keypoints() 

        points_orig = points_poly_filtered[points_poly_filtered.trajectory_id.isin(all_traj_ids)]
        templates_all = self.templates.get_by_id(all_traj_ids)
        
        if not (len(points_matched) == len(points_orig) == len(templates_all.trajectory_id)):
            return Keypoints() 

        keypoints_corrected_xy, keypoints_corrected_rc, corr_values = pattern_matching(
            points_matched, img, templates_all, points_orig, hs=self.template_size,
            border_matched=self.border_matched, border_interpolated=self.border_interpolated, band=1
        )
        points_matched['corr'] = corr_values
        correlation_mask = corr_values >= self.min_correlation
        if not np.any(correlation_mask):
            return Keypoints()

        points_matched = points_matched[correlation_mask].copy()
        keypoints_corrected_xy = keypoints_corrected_xy[correlation_mask]
        points_matched = pd.merge(
            points_matched, points_orig[['trajectory_id', 'geometry']].rename(columns={'geometry': 'geometry_orig'}),
            on='trajectory_id', how='left'
        )
        x1_interp, y1_interp, was_interpolated_mask = filter_and_interpolate_flipped_triangles(
            points_matched.geometry_orig.x.to_numpy(), points_matched.geometry_orig.y.to_numpy(),
            keypoints_corrected_xy[:, 0], keypoints_corrected_xy[:, 1],
        )
        good_mask = ~was_interpolated_mask & ~np.isnan(x1_interp)
        points_good = points_matched[good_mask].copy()
        xy_good = np.column_stack((x1_interp[good_mask], y1_interp[good_mask]))
        rc_good = keypoints_corrected_rc[correlation_mask][good_mask]

        if np.any(was_interpolated_mask):
            points_to_recheck = points_matched[was_interpolated_mask].copy()
            points_to_recheck['geometry'] = gpd.points_from_xy(x1_interp[was_interpolated_mask], y1_interp[was_interpolated_mask])
            templates_recheck = self.templates.get_by_id(points_to_recheck.trajectory_id.values)
            points_fg1_recheck = points_matched[was_interpolated_mask][['trajectory_id', 'geometry_orig', 'angle']]
            points_fg1_recheck.rename(columns={'geometry_orig': 'geometry'}, inplace=True)
            xy_rechecked, rc_rechecked, corr_rechecked = pattern_matching(
                points_to_recheck, img, templates_recheck, points_fg1_recheck, hs=self.template_size,
                border_matched=self.border_matched, border_interpolated=self.border_interpolated, band=1
            )
            recheck_passed_mask = corr_rechecked >= self.min_correlation
            points_rechecked = points_to_recheck[recheck_passed_mask].copy()
            points_rechecked['corr'] = corr_rechecked[recheck_passed_mask]
            points_matched = pd.concat([points_good, points_rechecked], ignore_index=True)
            corrected_xy = np.vstack([xy_good, xy_rechecked[recheck_passed_mask]])
            corrected_rc = np.vstack([rc_good, rc_rechecked[recheck_passed_mask]])
        else:
            points_matched = points_good
            corrected_xy = xy_good
            corrected_rc = rc_good

        if points_matched.empty:
            return Keypoints()
                        
        points_matched = points_matched.drop(columns=['geometry_orig'])
        points_matched['geometry'] = gpd.points_from_xy(corrected_xy[:, 0], corrected_xy[:, 1])
        
        if not points_matched.empty:
            keypoints_list_with_tags_recompute = [
                (cv2.KeyPoint(px_c, px_r, size=31, angle=img.angle, octave=self.octave), None)
                for px_c, px_r in corrected_rc
            ]
            _raw_kps_recomputed, new_descriptors, _ = self.keypoint_detector.compute_descriptors(
                keypoints_list_with_tags_recompute, img, polarisation=1, normalize=False
            )
        else:
            new_descriptors = None

        if new_descriptors is not None and len(new_descriptors) == len(points_matched):
            points_matched['descriptors'] = list(new_descriptors)
            valid_mask_desc = points_matched['descriptors'].apply(lambda d: isinstance(d, np.ndarray))
            points_matched = points_matched[valid_mask_desc].copy()
        elif not points_matched.empty:
            points_matched = points_matched.iloc[0:0]

        if not points_matched.empty:
            self.templates.update(points_matched, img, self.template_size, band=1)

        return points_matched

    def ensure_final_persistence(self):
        if self.persist_updates:
            images_since_persist = self.points.last_image_id - self._last_persisted_id
            if images_since_persist > 0:
                save_successful = self.db.save(
                    points=self.points, templates=self.templates,
                    last_persisted_id=self._last_persisted_id, insitu_points=self.insitu_points
                )
                if save_successful:
                    self._last_persisted_id = self.points.last_image_id
