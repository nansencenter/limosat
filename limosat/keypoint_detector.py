# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import numpy as np  # used for Gaussian weighting (np.exp)
import cv2
import cartopy.crs as ccrs
from skimage.util import view_as_windows
from nansat import NSR
import os
from pathlib import Path
from .utils import log_execution_time, logger, extract_date

class KeypointDetector:
    """
    Handles detection of keypoints in images.

    This class encapsulates different strategies for keypoint detection,
    including new keypoint detection and gridded detection.
    """

    def __init__(self, model, cache_dir=None, debug_recorder=None):
        """
        Initialize the keypoint detector.

        Parameters:
            model: Feature detection model (e.g., SIFT, ORB)
            cache_dir: Optional directory to cache deterministic grid descriptors
            debug_recorder: Optional debug recorder for structured event output
        """
        self.model = model
        self.debug_recorder = debug_recorder
        self.pc = ccrs.PlateCarree()
        central_longitude = -45
        true_scale_latitude = 70
        self.nps = ccrs.NorthPolarStereo(
            central_longitude=central_longitude,
            true_scale_latitude=true_scale_latitude
        )
        # Optional on-disk cache for gridded descriptors
        if cache_dir is None:
            cache_dir = os.environ.get("LIMOSAT_GRID_CACHE")
        self.grid_cache_dir = cache_dir
        if self.grid_cache_dir:
            os.makedirs(self.grid_cache_dir, exist_ok=True)

    @staticmethod
    def _read_band(img, band_id):
        """Read a Nansat band, including integer bands that cannot hold NaN."""
        try:
            return img[band_id]
        except ValueError as exc:
            if "cannot convert float NaN to integer" not in str(exc):
                raise
            array = img.vrt.dataset.GetRasterBand(band_id).ReadAsArray()
            if array is None:
                raise RuntimeError(f"Could not read raster band {band_id}") from exc
            return array

    def _model_cache_tag(self):
        # Best-effort tag to avoid collisions across detector configs.
        parts = [self.model.__class__.__name__]
        for getter, label in (
            ("getPatchSize", "ps"),
            ("getNLevels", "nl"),
            ("getMaxFeatures", "nf"),
            ("getNFeatures", "nf"),
            ("getScaleFactor", "sf"),
            ("getEdgeThreshold", "et"),
            ("getFirstLevel", "fl"),
            ("getScoreType", "st"),
        ):
            if hasattr(self.model, getter):
                try:
                    parts.append(f"{label}{getattr(self.model, getter)()}")
                except Exception:
                    pass
        return "-".join(parts)

    def _cache_subdir(self, img, source_path):
        date = getattr(img, "date", None)
        if date is None:
            date = extract_date(str(source_path))
        if date is None:
            return Path("unknown") / "unknown"
        try:
            return Path(f"{int(date.year):04d}") / f"{int(date.month):02d}"
        except Exception:
            return Path("unknown") / "unknown"

    def _grid_cache_key(self, img, stride, border_size, octave):
        if not self.grid_cache_dir:
            return None
        # Use basename and detector params; attempt to use image filename if available
        source_path = (
            getattr(img, "filename", None)
            or getattr(img, "file_path", None)
            or getattr(img, "filepath", None)
            or getattr(img, "name", None)
            or "unknown"
        )
        basename = Path(source_path).stem
        subdir = self._cache_subdir(img, source_path)
        model_tag = self._model_cache_tag()
        return Path(self.grid_cache_dir) / subdir / f"{basename}_s{stride}_b{border_size}_o{octave}_{model_tag}.npz"

    def _load_grid_cache(self, img, stride, border_size, octave):
        cache_path = self._grid_cache_key(img, stride, border_size, octave)
        if not cache_path or not cache_path.exists():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                coords = data['coords']
                descriptors = data['descriptors']
            tags = [None] * len(coords)
            return coords, descriptors, tags
        except Exception as e:
            logger.warning(f"Failed to load grid cache {cache_path}: {e}")
            return None

    def _save_grid_cache(self, img, stride, border_size, octave, compute_result):
        cache_path = self._grid_cache_key(img, stride, border_size, octave)
        if not cache_path or compute_result is None:
            return
        coords, descriptors, tags = compute_result
        if coords is None or descriptors is None:
            return
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, coords=coords, descriptors=descriptors)
        except Exception as e:
            logger.warning(f"Failed to save grid cache {cache_path}: {e}")
        
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
            img_band = self._read_band(img, polarisation).copy()
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
        window_border: int = 0,
    ):
        """
        Detect new keypoints in an image avoiding areas with existing keypoints.

        Parameters:
            points (Keypoints): Existing keypoints
            img (Image): Image to detect keypoints in
            octave (int): Octave value for cv2.KeyPoint.octave
            window_size (int): Size of detection windows
            border_size (int): Border (pixels) to exclude from final keypoints
            response_threshold (float): Minimum raw detector response
            step (int): Step size between windows (defaults to window_size)
            adjust_keypoint_angle (bool): Whether to set keypoint angle to img.angle
            compute_descriptors (bool): Whether to compute descriptors at the end
            window_border (int): If > 0, apply flat-center + tapered-edge weighting mask.
                                 If <= 0, select keypoint with highest raw response.
        Returns:
            tuple or list: (keypoint_coords, descriptors, surviving_tags) or list of (cv2.KeyPoint, tag)
        """
        if not hasattr(self, "_cache"):
            self._cache = {}

        img0 = self._read_band(img, 1)
        img0[np.isnan(img0)] = 0

        # land mask
        landmask = None
        if img.bands().get(2, {'name': 'none'}).get('name') == 'mask':
            landmask = self._read_band(img, 2)
            img0[landmask == 2] = 0

        if step is None:
            step = window_size

        # Prepare existing keypoints (to limit density)
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

        # Create sliding windows
        windows = view_as_windows(
            img0, window_shape=(window_size, window_size), step=step
        )
        n_windows_row, n_windows_col, _, _ = windows.shape

        keypoints = []
        max_kp_per_window = 1

        for i in range(n_windows_row):
            for j in range(n_windows_col):
                x_start = j * step
                x_end = x_start + window_size
                y_start = i * step
                y_end = y_start + window_size

                # Skip window if already filled with enough existing keypoints
                num_existing_kp = np.sum(
                    (existing_keypoints_coords[:, 0] >= x_start)
                    & (existing_keypoints_coords[:, 0] < x_end)
                    & (existing_keypoints_coords[:, 1] >= y_start)
                    & (existing_keypoints_coords[:, 1] < y_end)
                )
                if num_existing_kp >= max_kp_per_window:
                    continue

                window = windows[i, j]

                # Detect candidates
                kps = self.model.detect(window, None)
                if not kps:
                    continue

                # Filter by response
                kps = [kp for kp in kps if kp.response > response_threshold]
                if not kps:
                    continue

                # Filter before selecting a winner so land responses do not
                # displace valid water candidates in the same window.
                if landmask is not None:
                    kps = [
                        kp for kp in kps
                        if landmask[
                            y_start + int(round(kp.pt[1])),
                            x_start + int(round(kp.pt[0])),
                        ] != 2
                    ]
                    if not kps:
                        continue

                # Choose best keypoint
                if window_border > 0:
                    cache_key = (window_size, int(window_border))
                    weighting_mask = self._cache.get(cache_key)

                    if weighting_mask is None:
                        center = (window_size - 1) / 2.0
                        coords = np.arange(window_size)
                        xx, yy = np.meshgrid(coords, coords)
                        radial_distance = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)

                        # Safe (flat) radius
                        safe_radius = (window_size / 2.0) - window_border
                        if safe_radius < 0:
                            safe_radius = 0.0

                        weighting_mask = np.ones((window_size, window_size), dtype=np.float32)
                        taper_zone = radial_distance > safe_radius

                        if np.any(taper_zone):
                            norm_dist = (radial_distance[taper_zone] - safe_radius) / max(window_border, 1e-6)
                            taper_values = 0.5 * (1 + np.cos(np.pi * np.clip(norm_dist, 0, 1)))
                            weighting_mask[taper_zone] = taper_values

                        self._cache[cache_key] = weighting_mask

                    responses = np.array([kp.response for kp in kps], dtype=np.float32)
                    iy = np.array([int(round(kp.pt[1])) for kp in kps], dtype=np.int32)
                    ix = np.array([int(round(kp.pt[0])) for kp in kps], dtype=np.int32)
                    np.clip(iy, 0, window_size - 1, out=iy)
                    np.clip(ix, 0, window_size - 1, out=ix)
                    weights = weighting_mask[iy, ix]
                    composite_scores = responses * weights
                    best_idx = int(np.argmax(composite_scores))
                    best_kp_in_window = kps[best_idx]
                    response_tag = {
                        'response': float(best_kp_in_window.response),
                        'composite_score': float(composite_scores[best_idx]),
                    }
                else:
                    # Raw response selection
                    best_kp_in_window = max(kps, key=lambda kp: kp.response)
                    response_tag = {
                        'response': float(best_kp_in_window.response),
                        'composite_score': None,
                    }

                # Offset to image coords
                best_kp_in_window.pt = (
                    best_kp_in_window.pt[0] + x_start,
                    best_kp_in_window.pt[1] + y_start,
                )
                if adjust_keypoint_angle:
                    best_kp_in_window.angle = img.angle
                best_kp_in_window.octave = octave
                keypoints.append((best_kp_in_window, response_tag))

        # Enforce outer border exclusion
        filtered_keypoints_with_tags = [
            item
            for item in keypoints
            if (
                border_size <= item[0].pt[0] <= img0.shape[1] - border_size
                and border_size <= item[0].pt[1] <= img0.shape[0] - border_size
            )
        ]

        if getattr(self.debug_recorder, "enabled", False) and filtered_keypoints_with_tags:
            responses = [
                tag.get('response')
                for _, tag in filtered_keypoints_with_tags
                if isinstance(tag, dict) and tag.get('response') is not None
            ]
            if responses:
                self.debug_recorder.record(
                    stage="keypoint_detection",
                    event_type="info",
                    message="response summary",
                    min_response=float(np.min(responses)),
                    mean_response=float(np.mean(responses)),
                    max_response=float(np.max(responses)),
                    count=len(responses),
                )

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
        # Attempt to load cached grid descriptors
        cached = self._load_grid_cache(img, stride, border_size, octave)
        if cached is not None:
            return cached

        # Get image data and replace NaNs with zero
        img0 = self._read_band(img, 1)
        img0[np.isnan(img0)] = 0

        # Optional land mask
        landmask = None
        if img.bands().get(2, {'name': 'none'}).get('name') == 'mask':
            landmask = self._read_band(img, 2)
            img0[landmask == 2] = 0

        # Vectorized grid generation to avoid Python double loop
        h, w = img0.shape
        rows, cols = np.mgrid[0:h:stride, 0:w:stride]
        rows_flat = rows.ravel()
        cols_flat = cols.ravel()

        if landmask is not None:
            landmask_bool = (landmask == 2)
            valid_mask = ~landmask_bool[rows_flat, cols_flat]
            rows_flat = rows_flat[valid_mask]
            cols_flat = cols_flat[valid_mask]

        # Border filtering in one pass
        border_mask = (
            (cols_flat >= border_size)
            & (cols_flat <= w - border_size)
            & (rows_flat >= border_size)
            & (rows_flat <= h - border_size)
        )
        rows_flat = rows_flat[border_mask]
        cols_flat = cols_flat[border_mask]

        keypoints_with_tags = [
            (cv2.KeyPoint(float(c), float(r), size=31, octave=octave, angle=img.angle), None)
            for r, c in zip(rows_flat, cols_flat)
        ]

        result = self.compute_descriptors(
            keypoints_with_tags,
            img,
            polarisation=1,
            normalize=False
        )

        self._save_grid_cache(img, stride, border_size, octave, result)
        return result

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
            response_threshold,
        ):
        """
        Dynamically finds the best ORB feature near each buoy point using a method similar to detect_new_keypoints
        Returns list of tuples: (cv2.KeyPoint, original_df_index)
        """
        # This method detects keypoints but does NOT compute their descriptors,
        # as it's part of a specific workflow in ImageProcessor where descriptors
        # are computed later in a batch.
        orb_model = self.model
        patch_size = int(orb_model.getPatchSize())
        window_size_for_detection = max(32, patch_size + 16)

        keypoints_with_indices = []
        img_band_data = self._read_band(img, 1)
        img_height, img_width = img_band_data.shape

        max_center_distance_m = 300.0

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
                min_distance_to_center = float('inf')
                patch_center_x = patch.shape[1] / 2.0
                patch_center_y = patch.shape[0] / 2.0

                if kps_in_patch:
                    # Filter out keypoints below the response threshold first
                    valid_kps = [kp for kp in kps_in_patch if kp.response >= response_threshold]

                    if valid_kps:
                        for kp_candidate in valid_kps:
                            # Calculate squared distance from the candidate to the patch center
                            dist_sq = (kp_candidate.pt[0] - patch_center_x)**2 + (kp_candidate.pt[1] - patch_center_y)**2
                            
                            if dist_sq < min_distance_to_center:
                                min_distance_to_center = dist_sq
                                best_kp_in_patch = kp_candidate
                
                if best_kp_in_patch is not None:
                    kp_final_c = c0 + best_kp_in_patch.pt[0]
                    kp_final_r = r0 + best_kp_in_patch.pt[1]

                    # Boundary check for the keypoint for subsequent descriptor computation
                    orb_desc_computation_half_patch = int(best_kp_in_patch.size / 2)

                    if not (orb_desc_computation_half_patch <= kp_final_c < img_width - orb_desc_computation_half_patch and \
                            orb_desc_computation_half_patch <= kp_final_r < img_height - orb_desc_computation_half_patch):
                        continue

                    center_c = c0 + patch_center_x
                    center_r = r0 + patch_center_y
                    try:
                        cand_x, cand_y = img.transform_points(
                            [kp_final_c],
                            [kp_final_r],
                            DstToSrc=0,
                            dst_srs=img.srs,
                        )
                        center_x, center_y = img.transform_points(
                            [center_c],
                            [center_r],
                            DstToSrc=0,
                            dst_srs=img.srs,
                        )
                        dx = cand_x[0] - center_x[0]
                        dy = cand_y[0] - center_y[0]
                        center_distance = float(np.hypot(dx, dy))
                    except Exception:
                        center_distance = None

                    if center_distance is not None and center_distance > max_center_distance_m:
                        continue

                    final_kp_to_add = cv2.KeyPoint(
                        x=float(kp_final_c),
                        y=float(kp_final_r),
                        size=float(best_kp_in_patch.size),
                        octave=int(octave),
                        angle=float(img.angle),
                        response=float(best_kp_in_patch.response)
                    )
                    keypoints_with_indices.append((
                        final_kp_to_add,
                        {
                            'original_index': point_row.name,
                            'response': float(best_kp_in_patch.response),
                            'composite_score': None,
                        },
                    ))
                
            except Exception as e:
                logger.info(f"Error processing point original_idx {original_idx} (input index {point_row.name}): {e}")
                pass 
        
        return keypoints_with_indices
