# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import numpy as np
import cv2
import cartopy.crs as ccrs
from skimage.util import view_as_windows
from nansat import NSR
from .utils import log_execution_time, logger

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
    def compute_descriptors(self, keypoints_with_tags, img, polarisation='s0_HV', normalize=True):
        """
        Compute descriptors for given keypoints in an image using a specified model.
        (self refers to the KeypointDetector instance, which has self.model)
        """
        if not keypoints_with_tags:
            return None, None, None

        raw_cv2_kps = [item[0] for item in keypoints_with_tags]

        def get_tags_for_survivors(original_kps_with_tags_list, surviving_cv2_kps_list):
            ordered_survivor_tags = []
            temp_originals = list(original_kps_with_tags_list)
            
            for surv_kp in surviving_cv2_kps_list:
                found_match = False
                for i, (orig_kp, orig_tag) in enumerate(temp_originals):
                    if np.allclose(surv_kp.pt, orig_kp.pt, atol=0.5):
                        ordered_survivor_tags.append(orig_tag)
                        temp_originals.pop(i)
                        found_match = True
                        break
                if not found_match:
                    logger.warning(f"compute_descriptors: Could not find original tag for a surviving keypoint at {surv_kp.pt}. Appending None as tag.")
                    ordered_survivor_tags.append(None)
            
            if len(ordered_survivor_tags) != len(surviving_cv2_kps_list):
                 logger.error(f"compute_descriptors: Tag mapping length mismatch. Survivors: {len(surviving_cv2_kps_list)}, Tags found: {len(ordered_survivor_tags)}")
                 return [None] * len(surviving_cv2_kps_list)
            return ordered_survivor_tags

        if polarisation == 'dual_polarisation':
            img_hh = img['s0_HH'].copy()
            img_hh[np.isnan(img_hh)] = 0
            keypoints_output_hh, descriptors_hh = self.model.compute(img_hh, raw_cv2_kps)
            if descriptors_hh is None or not keypoints_output_hh:
                return None, None, None
            
            rawkeypoints_coords = np.array([kp.pt for kp in keypoints_output_hh])
            if normalize:
                descriptors_hh = descriptors_hh - descriptors_hh.mean(axis=1, keepdims=True)

            img_hv = img['s0_HV'].copy()
            img_hv[np.isnan(img_hv)] = 0
            keypoints_output_hv_check, descriptors_hv = self.model.compute(img_hv, keypoints_output_hh)

            if descriptors_hv is None:
                return None, None, None

            if len(descriptors_hh) != len(descriptors_hv):
                logger.error(f"Dual-pol descriptor length mismatch: HH ({len(descriptors_hh)}), HV ({len(descriptors_hv)}) for {len(keypoints_output_hh)} input HH survivors.")
                return None, None, None

            if normalize:
                descriptors_hh = descriptors_hh - descriptors_hh.mean(axis=1, keepdims=True)
                descriptors_hv = descriptors_hv - descriptors_hv.mean(axis=1, keepdims=True)

            descriptors = np.hstack([descriptors_hh, descriptors_hv])
            surviving_tags = get_tags_for_survivors(keypoints_with_tags, keypoints_output_hh)
            return rawkeypoints_coords, descriptors, surviving_tags

        else:
            img_band = img[polarisation].copy()
            img_band[np.isnan(img_band)] = 0
            keypoints_output, descriptors = self.model.compute(img_band, raw_cv2_kps)
            if descriptors is None or not keypoints_output:
                return None, None, None
                
            rawkeypoints_coords = np.array([kp.pt for kp in keypoints_output])
            if normalize:
                descriptors = descriptors - descriptors.mean(axis=1, keepdims=True)
                
            surviving_tags = get_tags_for_survivors(keypoints_with_tags, keypoints_output)
            return rawkeypoints_coords, descriptors, surviving_tags

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
        compute_descriptors=True,
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
            tuple: (keypoint_coords, descriptors, surviving_tags)
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
        
        if not compute_descriptors:
            return filtered_keypoints_with_tags

        logger.debug(f"Detected {len(filtered_keypoints_with_tags)} raw keypoints. Now computing descriptors.")
        return self.compute_descriptors(
            filtered_keypoints_with_tags,
            img,
            polarisation=1,
            normalize=False
        )

    @log_execution_time
    def detect_gridded_points(
        self,
        img,
        stride,
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
            tuple: (keypoint_coords, descriptors, surviving_tags)
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
                # The size parameter is required, but will be overwritten by the detector's patch size
                # Corrected order: cv2.KeyPoint expects (x, y) which is (c, r)
                kp = cv2.KeyPoint(float(c), float(r), size=31, octave=octave, angle=img.angle)
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

        logger.debug(f"Detected {len(filtered_keypoints_with_tags)} raw gridded points. Now computing descriptors.")
        return self.compute_descriptors(
            filtered_keypoints_with_tags,
            img,
            polarisation=1,
            normalize=False
        )

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
            octave,          # Passed by ImageProcessor, for cv2.KeyPoint.octave
            img,              # Nansat image object for the current image
            response_threshold
        ):
        """
        Dynamically finds the best ORB feature near each buoy point using a method similar to detect_new_keypoints
        Returns list of tuples: (cv2.KeyPoint, original_df_index)
        """
        # This method detects keypoints but does NOT compute their descriptors,
        # as it's part of a specific workflow in ImageProcessor where descriptors
        # are computed later in a batch.
        # this is the window it searches
        window_size_for_detection = 32
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

                patch_half = window_size_for_detection // 2
                r0 = max(0, int(round(buoy_row_px_float)) - patch_half)
                r1 = min(img_height, int(round(buoy_row_px_float)) + patch_half + (window_size_for_detection % 2))
                c0 = max(0, int(round(buoy_col_px_float)) - patch_half)
                c1 = min(img_width, int(round(buoy_col_px_float)) + patch_half + (window_size_for_detection % 2))

                if not ((r1 - r0) >= window_size_for_detection * 0.8 and \
                        (c1 - c0) >= window_size_for_detection * 0.8):
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
                    orb_desc_computation_half_patch = int(best_kp_in_patch.size / 2)

                    if not (orb_desc_computation_half_patch <= kp_final_c < img_width - orb_desc_computation_half_patch and \
                            orb_desc_computation_half_patch <= kp_final_r < img_height - orb_desc_computation_half_patch):
                        continue

                    final_kp_to_add = cv2.KeyPoint(
                        x=float(kp_final_c),
                        y=float(kp_final_r),
                        size=float(best_kp_in_patch.size),
                        octave=int(octave),
                        angle=float(img.angle),
                        response=float(best_kp_in_patch.response)
                    )
                    keypoints_with_indices.append((final_kp_to_add, point_row.name))
                
            except Exception as e:
                logger.info(f"Error processing point original_idx {original_idx} (input index {point_row.name}): {e}")
                pass 
                
        return keypoints_with_indices
