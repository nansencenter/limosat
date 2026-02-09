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
import argparse
from datetime import datetime
import numpy as np
import cv2
from nansat import Nansat
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback # Import traceback for detailed error printing
import sys
os.environ.setdefault('MOD44WPATH', '/Data/sat/auxdata/mod44w/')
os.environ.setdefault('GDAL_VRT_RAWRASTERBAND_ALLOWED_SOURCE', 'ALL')
os.environ.setdefault('VRT_ALLOW_MEM_DRIVER', 'YES')
sys.path.insert(0, os.path.abspath('..'))

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
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))

    for pol in pols:
        # 1) Read raw DNs
        b = n[f'DN_{pol}']
        
        # 2) Load average calibration
        cal_avg = n.vrt.band_vrts[f'sigmaNought_{pol}'].vrt.dataset.ReadAsArray().mean()
        
        # 3) Downsample & calibrate
        filt = np.ones((2, 2)) / (4. * cal_avg) * 255 * factor_dict[pol]
        bf = ndi.convolve(b, filt)[::2, ::2]
        
        # 4) Apply CLAHE
        enhanced = clahe.apply(np.clip(bf, 0, 255).astype(np.uint8))
        
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
    # Add error handling for unexpected filename formats
    if len(parts) > 4 and len(parts[4]) >= 8:
        date_str = parts[4][:8]  # First 8 characters of the 5th part
        try:
            return datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            print(f"WARNING: Could not parse date from filename: {base_name}")
            # Return a default date or raise an error if preferred
            return datetime.min
    else:
        print(f"WARNING: Unexpected filename format for date extraction: {base_name}")
        # Return a default date or raise an error if preferred
        return datetime.min


def process_file(ifile, odir):
    """Process a single file with optional cropping."""
    # Define the output file path
    base_name = os.path.basename(ifile)
    output_file = os.path.join(odir, f"{os.path.splitext(base_name)[0]}.tiff")

    # Check if the output file already exists
    if os.path.exists(output_file):
        # Use process ID to differentiate messages in parallel execution
        pid = os.getpid()
        print(f"{datetime.now()} - {pid} - Skipping {ifile}; {output_file} already exists.")
        return True  # Skip processing

    try:
        pid = os.getpid()
        print(f"{datetime.now()} - {pid} - Processing {ifile}")
        t0 = time.time()
        n = get_n_clahe(ifile, clip_limit=1.25, grid_size=16) # Pass your desired CLAHE params

        # Use an empty list [] for no options, or add valid options like ['COMPRESS=LZW']
        n.export(output_file, driver='GTiff', options=[])
        elapsed = time.time() - t0
        print(f"{datetime.now()} - {pid} - Finished processing {ifile} in {elapsed:.2f} seconds")
        return True
    except Exception as e:
        # Print detailed error information including traceback
        pid = os.getpid()
        print(f"ERROR processing {ifile} in process {pid}:")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print("Traceback:")
        traceback.print_exc() # Prints the full traceback to standard error
        return False


def _is_supported_input(path):
    return (os.path.isdir(path) and path.endswith('.SAFE')) or (os.path.isfile(path) and path.endswith('.zip'))


def _collect_inputs(idir, files_list=None, recursive=False, max_files=0):
    if files_list:
        with open(files_list, 'r', encoding='utf-8') as f:
            entries = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        paths = [p if os.path.isabs(p) else os.path.join(idir, p) for p in entries]
    else:
        if recursive:
            safe_paths = glob.glob(os.path.join(idir, '**', '*.SAFE'), recursive=True)
            zip_paths = glob.glob(os.path.join(idir, '**', '*.zip'), recursive=True)
        else:
            safe_paths = glob.glob(os.path.join(idir, '*.SAFE'))
            zip_paths = glob.glob(os.path.join(idir, '*.zip'))
        paths = safe_paths + zip_paths

    valid = [p for p in paths if _is_supported_input(p)]
    if max_files > 0:
        valid = valid[:max_files]

    try:
        return sorted(valid, key=extract_date)
    except Exception as e:
        print(f"WARNING: Failed to sort by date ({e}), using input order.")
        return valid


def _print_progress(done, total, start_time):
    elapsed = max(1e-9, time.time() - start_time)
    pct = (done / total) * 100 if total else 100.0
    rate = done / elapsed
    print(f"[preprocessing] {done}/{total} ({pct:.1f}%) elapsed={elapsed:.1f}s rate={rate:.2f}/s", flush=True)


def _warmup_nansat(path):
    try:
        Nansat(path)
    except Exception as e:
        print(f"WARNING: Warmup failed for {path}: {e}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Preprocess Sentinel-1 SAFE/ZIP to TIFF')
    parser.add_argument('--idir', default='/Data/sat/downloads/sentinel1/2015/', help='Input directory')
    parser.add_argument('--odir', default='/Data/sat/downloads/sentinel1/2015/processed_CLAHE_update/', help='Output TIFF directory')
    parser.add_argument('--files-list', default=None, help='Optional text file: one SAFE/ZIP path per line')
    parser.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 1), help='Parallel workers')
    parser.add_argument('--recursive', action='store_true', help='Recursively scan input directory')
    parser.add_argument('--max-files', type=int, default=0, help='Limit number of inputs (0 = all)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        os.makedirs(args.odir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Could not create output directory '{args.odir}': {e}")
        return 1

    input_files = _collect_inputs(args.idir, args.files_list, args.recursive, args.max_files)
    if not input_files:
        print("No valid input files found (.SAFE directories or .zip files).")
        return 1

    workers = max(1, args.workers)
    print(f"{len(input_files)} input files found")
    print(f"Starting processing with {workers} workers.")

    total = len(input_files)
    report_every = 1 if total <= 20 else 10
    start_time = time.time()

    if workers > 1 and total > 0:
        _warmup_nansat(input_files[0])

    completed_count = 0
    error_count = 0
    if workers == 1:
        for idx, ifile in enumerate(input_files, 1):
            try:
                ok = process_file(ifile, args.odir)
                if ok:
                    completed_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1
            if idx % report_every == 0 or idx == total:
                _print_progress(idx, total, start_time)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_file, ifile, args.odir) for ifile in input_files]
            print("Waiting for all tasks to complete...")
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    ok = future.result()
                    if ok:
                        completed_count += 1
                    else:
                        error_count += 1
                except Exception:
                    error_count += 1
                if idx % report_every == 0 or idx == total:
                    _print_progress(idx, total, start_time)

    print(f"Processing complete. {completed_count} tasks finished successfully, {error_count} tasks failed.")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
