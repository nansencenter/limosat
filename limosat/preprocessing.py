# MIT License
#
# Copyright (c) 2025 Sean Minhui Tashi Chua, and Anton Korosov
#
# Licensed under the MIT License. See the LICENSE file in the project root for full details.

"""
preprocessing.py

Preprocessing utilities for Sentinel-1 data for limosat
"""

import os
import glob
import time
from datetime import datetime
import numpy as np
import cv2
from nansat import Nansat
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(process)d - %(message)s')

def get_n(filename, pols=('HH', 'HV'), factor_hh=2, factor_hv=5):
    """Open S1 file with Nansat and add calibrated, downsampled, uint8 HH and HV bands."""
    factor = {'HH': factor_hh, 'HV': factor_hv}
    n = Nansat(filename)
    bb = []
    for pol in pols:
        # Read digital numbers
        b = n[f'DN_{pol}']
        # Read and average calibration
        cal_avg = n.vrt.band_vrts[f'sigmaNought_{pol}'].vrt.dataset.ReadAsArray().mean()
        # Create filter for subsampling and calibration
        filter = np.ones((2, 2)) / 4. / cal_avg * 255 * factor[pol]
        # Subsample, calibrate, clip, cast to uint8
        bf = np.clip(ndi.convolve(b, filter)[::2, ::2], 0, 255).astype(np.uint8)
        bb.append(bf)
    # Resize Nansat so that reprojection of coordinates is correct
    n.resize(0.5)
    # Adjust bands shapes
    bb = [b[:n.shape()[0], :n.shape()[1]] for b in bb]
    # Add bands to Nansat with name s0_HH or s0_HV
    parameters = [{'name': f's0_{pol}'} for pol in pols]
    d = Nansat.from_domain(n)
    d.set_metadata(n.get_metadata())
    d.add_bands(bb, parameters=parameters)
    # Improve geolocation accuracy
    d.reproject_gcps()
    d.vrt.tps = True
    return d

def get_n_clahe(filename, pols=('HH', 'HV'),
                factor_hh=2.2, factor_hv=5.5, clip_limit=2.0, grid_size=8):
    """
    Preprocessing using CLAHE for adaptive contrast enhancement
    """
    factor_dict = {'HH': factor_hh, 'HV': factor_hv}
    n = Nansat(filename)
    bb = []
    
    for pol in pols:
        # 1) Read raw DNs
        b = n[f'DN_{pol}']
        
        # 2) Load average calibration
        cal_avg = n.vrt.band_vrts[f'sigmaNought_{pol}'].vrt.dataset.ReadAsArray().mean()
        
        # 3) Downsample & calibrate
        filt = np.ones((2, 2)) / (4. * cal_avg) * 255 * factor_dict[pol]
        bf = ndi.convolve(b, filt)[::2, ::2]
        
        # 4) Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        enhanced = clahe.apply(bf.astype(np.uint8))
        
        # 5) Mild Gaussian smoothing
        enhanced = ndi.gaussian_filter(enhanced, sigma=0.5)
        
        bb.append(enhanced)

    # Preserve original domain but at half size
    n.resize(0.5)
    
    # Trim each band to match the new shape
    bb = [b[:n.shape()[0], :n.shape()[1]] for b in bb]
    
    # Add bands to a new Nansat object
    params = [{'name': f's0_{pol}'} for pol in pols]
    d = Nansat.from_domain(n)
    d.set_metadata(n.get_metadata())
    d.add_bands(bb, parameters=params)
    
    # Update reprojection
    d.reproject_gcps()
    d.vrt.tps = True
    
    return d

def extract_date(file_path):
    """Extract date from file path."""
    base_name = os.path.basename(file_path)
    parts = base_name.split('_')
    date_str = parts[4][:8]  # First 8 characters of the 5th part
    return datetime.strptime(date_str, '%Y%m%d')

def process_file(ifile, odir):
    """Process a single file with optional cropping."""
    # Define the output file path
    output_file = os.path.join(odir, f"{os.path.splitext(os.path.basename(ifile))[0]}.tiff")
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        logging.info(f"Skipping {ifile}; {output_file} already exists.")
        return  # Skip processing
    
    try:
        logging.info(f"Processing {ifile}")
        t0 = time.time()
        n = get_n_clahe(ifile, clip_limit=1.25, grid_size=16)

        n.export(output_file, driver='GTiff', options='')
        elapsed = time.time() - t0
        logging.info(f"Finished processing {ifile} in {elapsed:.2f} seconds")
    except Exception as e:
        logging.error(f"Error processing {ifile}: {e}")

# Define the directory path
idir = '/Data/sat/downloads/sentinel1/2015/'
odir = '/Data/sat/downloads/sentinel1/2015/processed_CLAHE/'

# Find all .SAFE files in the directory
safe_files = glob.glob(os.path.join(idir, '*.SAFE'))
# Sort the file paths by date
sorted_safe_files = sorted(safe_files, key=extract_date)

print(len(sorted_safe_files), 'SAFE files found')

# Use ProcessPoolExecutor to parallelize the processing with a progress bar
with ProcessPoolExecutor(max_workers=16) as executor:
    futures = []
    with tqdm(total=len(sorted_safe_files)) as pbar:
        for ifile in sorted_safe_files:
            future = executor.submit(process_file, ifile, odir)
            futures.append(future)
            future.add_done_callback(lambda p: pbar.update())

    # Ensure all futures are completed
    for future in futures:
        try:
            future.result()
        except Exception as e:
            logging.error(f"Error in future: {e}")