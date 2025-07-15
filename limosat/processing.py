# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import cv2
import cartopy.crs as ccrs
from shapely.geometry import MultiPoint
from skimage.transform import AffineTransform
from .utils import log_execution_time, logger
from .keypoints import Keypoints

@log_execution_time
def interpolate_drift(points_poly, points_fg1, points_fg2, img,
                                         max_interpolation_time_gap_hours,
                                         border_size,
                                         model_type=AffineTransform,
                                         max_interpolation_speed_m_per_day=50000,
                                         max_anchor_distance_km=20.0
                                        ):
    """
    Interpolates drift for unmatched points using per-source-image affine transforms.
    (Corrected minimalist style)
    """
    points_result = points_fg2.copy()
    points_result['interpolated'] = 0
    
    MIN_SAMPLES_FOR_AFFINE = 10
    if len(points_fg1) < MIN_SAMPLES_FOR_AFFINE:
        logger.info(f"Not enough matched points ({len(points_fg1)}) for interpolation. Min: {MIN_SAMPLES_FOR_AFFINE}.")
        return points_result

    # 1. Identify unmatched points and group all points by source image_id
    unmatched_points = points_poly[~points_poly['trajectory_id'].isin(points_fg1['trajectory_id'])].copy()
    if unmatched_points.empty:
        logger.debug("No unmatched points to consider for interpolation.")
        return points_result
        
    anchor_groups = points_fg1.groupby('image_id')
    unmatched_groups = unmatched_points.groupby('image_id')
    
    all_interpolated_points = []

    # 2. Loop through each source image group (this is the safer, original structure)
    for source_id, anchor_group in anchor_groups:
        if len(anchor_group) < MIN_SAMPLES_FOR_AFFINE:
            logger.debug(f"Skipping group {source_id}: not enough anchor points ({len(anchor_group)}).")
            continue
        
        # a. Find corresponding destination points and estimate local transform
        tids_group = anchor_group['trajectory_id']
        anchor_dest_group = points_fg2[points_fg2['trajectory_id'].isin(tids_group)]
        
        # Ensure perfect alignment for transform estimation
        # This merge is local to the loop and thus safe and correct.
        local_anchors = pd.merge(
            anchor_group[['trajectory_id', 'geometry']],
            anchor_dest_group[['trajectory_id', 'geometry']],
            on='trajectory_id',
            suffixes=('_src', '_dst')
        )

        pos0_atm = np.column_stack((local_anchors.geometry_src.x, local_anchors.geometry_src.y))
        pos1_atm = np.column_stack((local_anchors.geometry_dst.x, local_anchors.geometry_dst.y))
        
        atm_local = model_type()
        if not atm_local.estimate(pos0_atm, pos1_atm):
            logger.warning(f"Affine transform estimation failed for group {source_id}.")
            continue

        # b. Select and filter unmatched points for this group
        try:
            unmatched_group = unmatched_groups.get_group(source_id)
        except KeyError:
            logger.debug(f"No unmatched points for source_id {source_id}.")
            continue
        
        # Chain all pre-transform filters together for a single pass
        try:
            convex_hull_of_anchors = MultiPoint(local_anchors.geometry_src.tolist()).convex_hull
            
            is_within_bounds = unmatched_group.within(img.poly)
            time_gap_ok = (img.date.tz_localize(None) - unmatched_group['time'].dt.tz_localize(None)).dt.total_seconds() <= (max_interpolation_time_gap_hours * 3600)
            is_near_hull = unmatched_group.geometry.distance(convex_hull_of_anchors) <= (max_anchor_distance_km * 1000.0)
            
            filtered_unmatched = unmatched_group[is_within_bounds & time_gap_ok & is_near_hull].copy()
        except Exception as e:
            logger.warning(f"Error during pre-filtering for group {source_id}: {e}")
            continue

        if filtered_unmatched.empty: continue

        # c. Apply transform and perform post-interpolation checks
        pos0_to_interp = np.column_stack((filtered_unmatched.geometry.x, filtered_unmatched.geometry.y))
        interpolated_coords_np = atm_local(pos0_to_interp)
        
        filtered_unmatched['interpolated_geom'] = gpd.points_from_xy(
            interpolated_coords_np[:, 0], interpolated_coords_np[:, 1], crs=filtered_unmatched.crs
        )
        time_diff_days = (img.date.tz_localize(None) - filtered_unmatched['time'].dt.tz_localize(None)).dt.total_seconds() / 86400.0
        
        filtered_unmatched['speed_m_per_day'] = np.divide(
            filtered_unmatched.geometry.distance(filtered_unmatched['interpolated_geom']),
            time_diff_days,
            out=np.full_like(time_diff_days, np.nan), where=time_diff_days > 1e-9
        )

        # d. Final speed and boundary checks
        speed_ok = filtered_unmatched['speed_m_per_day'] <= max_interpolation_speed_m_per_day
        cols, rows = img.transform_points(interpolated_coords_np[:, 0], interpolated_coords_np[:, 1], DstToSrc=1, dst_srs=img.srs)
        pixel_bounds_ok = (cols >= border_size) & (cols < img.shape()[1] - border_size) & \
                          (rows >= border_size) & (rows < img.shape()[0] - border_size) & \
                          np.isfinite(cols) & np.isfinite(rows)

        final_valid_mask = speed_ok.values & pixel_bounds_ok
        final_points_to_add = filtered_unmatched[final_valid_mask]

        if final_points_to_add.empty: continue
        
        # e. Collect valid interpolated points
        points_interpolated_group = Keypoints.create(
            np.column_stack((cols[final_valid_mask], rows[final_valid_mask])),
            [None] * len(final_points_to_add), img, points_result['image_id'].iloc[0]
        )
        points_interpolated_group['trajectory_id'] = final_points_to_add['trajectory_id'].values
        points_interpolated_group['interpolated'] = 1
        points_interpolated_group.geometry = final_points_to_add['interpolated_geom'].values
        
        all_interpolated_points.append(points_interpolated_group)
        logger.info(f"Group {source_id}: Successfully interpolated and kept {len(points_interpolated_group)} points.")

    # 3. Aggregate results
    if not all_interpolated_points:
        logger.info("No points were successfully interpolated across any group.")
        return points_result

    return Keypoints._from_gdf(pd.concat([points_result] + all_interpolated_points, ignore_index=True))

@log_execution_time
def pattern_matching(
    points: gpd.GeoDataFrame,
    img,                                 
    templates: xr.DataArray,             
    points_fg1: gpd.GeoDataFrame,
    hs: int,
    band: str = 's0_HV',
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
