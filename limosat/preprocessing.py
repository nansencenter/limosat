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
import traceback # Import traceback for detailed error printing
import sys
os.environ['MOD44WPATH'] = '/Data/sat/auxdata/mod44w/'
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
        return  # Skip processing

    try:
        pid = os.getpid()
        print(f"{datetime.now()} - {pid} - Processing {ifile}")
        t0 = time.time()
        n = get_n_clahe(ifile, clip_limit=1.25, grid_size=16) # Pass your desired CLAHE params

        # Use an empty list [] for no options, or add valid options like ['COMPRESS=LZW']
        n.export(output_file, driver='GTiff', options=[])
        elapsed = time.time() - t0
        print(f"{datetime.now()} - {pid} - Finished processing {ifile} in {elapsed:.2f} seconds")
    except Exception as e:
        # Print detailed error information including traceback
        pid = os.getpid()
        print(f"ERROR processing {ifile} in process {pid}:")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print("Traceback:")
        traceback.print_exc() # Prints the full traceback to standard error


# --- Main Execution Block ---

# Define the directory path

files_to_process = ['S1A_EW_GRDM_1SDH_20150203T055708_20150203T055808_004455_005752_CE59.tiff',
 'S1A_EW_GRDM_1SDH_20150203T055808_20150203T055908_004455_005752_ED14.SAFE',
 'S1A_EW_GRDM_1SDH_20150203T073534_20150203T073634_004456_005755_E72C.SAFE',
 'S1A_EW_GRDM_1SDH_20150203T073634_20150203T073734_004456_005755_9F35.SAFE',
 'S1A_EW_GRDM_1SDH_20150203T091337_20150203T091446_004457_00575A_31F9.SAFE',
 'S1A_EW_GRDM_1SDH_20150203T122856_20150203T122956_004459_005766_2953.SAFE',
 'S1A_EW_GRDM_1SDH_20150203T122956_20150203T123056_004459_005766_8AC3.SAFE',
 'S1A_EW_GRDM_1SDH_20150203T140711_20150203T140811_004460_00576B_39A7.SAFE',
 'S1A_EW_GRDM_1SDH_20150203T140811_20150203T140911_004460_00576B_854D.SAFE',
 'S1A_EW_GRDM_1SDH_20150204T063755_20150204T063855_004470_0057AF_6AD9.SAFE',
 'S1A_EW_GRDM_1SDH_20150204T063855_20150204T063955_004470_0057AF_040B.SAFE',
 'S1A_EW_GRDM_1SDH_20150204T130958_20150204T131058_004474_0057C6_B166.SAFE',
 'S1A_EW_GRDM_1SDH_20150204T144734_20150204T144834_004475_0057CC_E07E.SAFE',
 'S1A_EW_GRDM_1SDH_20150204T144834_20150204T144934_004475_0057CC_259E.SAFE',
 'S1A_EW_GRDM_1SDH_20150205T054059_20150205T054159_004484_0057FE_D992.SAFE',
 'S1A_EW_GRDM_1SDH_20150205T071913_20150205T072013_004485_005802_2CD9.SAFE',
 'S1A_EW_GRDM_1SDH_20150205T072013_20150205T072113_004485_005802_EDAD.SAFE',
 'S1A_EW_GRDM_1SDH_20150205T135056_20150205T135156_004489_005817_9EE3.SAFE',
 'S1A_EW_GRDM_1SDH_20150205T152849_20150205T152949_004490_00581C_0419.SAFE',
 'S1A_EW_GRDM_1SDH_20150205T152949_20150205T153049_004490_00581C_538A.SAFE',
 'S1A_EW_GRDM_1SDH_20150206T062144_20150206T062244_004499_00584C_9CE5.SAFE',
 'S1A_EW_GRDM_1SDH_20150206T062244_20150206T062344_004499_00584C_0C71.SAFE',
 'S1A_EW_GRDM_1SDH_20150206T075952_20150206T080052_004500_005854_D0B6.SAFE',
 'S1A_EW_GRDM_1SDH_20150206T080052_20150206T080152_004500_005854_287C.SAFE',
 'S1A_EW_GRDM_1SDH_20150206T093804_20150206T093910_004501_00585C_A5B6.SAFE',
 'S1A_EW_GRDM_1SDH_20150206T125327_20150206T125427_004503_00586D_6CA0.SAFE',
 'S1A_EW_GRDM_1SDH_20150206T125427_20150206T125521_004503_00586D_14AC.SAFE',
 'S1A_EW_GRDM_1SDH_20150206T143152_20150206T143252_004504_005871_880A.SAFE',
 'S1A_EW_GRDM_1SDH_20150206T160943_20150206T161043_004505_005878_CE99.SAFE',
 'S1A_EW_GRDM_1SDH_20150207T052421_20150207T052521_004513_0058A3_C2EB.SAFE',
 'S1A_EW_GRDM_1SDH_20150207T052521_20150207T052621_004513_0058A3_7448.SAFE',
 'S1A_EW_GRDM_1SDH_20150207T070245_20150207T070345_004514_0058A8_D193.SAFE',
 'S1A_EW_GRDM_1SDH_20150207T070345_20150207T070445_004514_0058A8_FAA1.SAFE',
 'S1A_EW_GRDM_1SDH_20150207T084056_20150207T084156_004515_0058AE_AFBB.SAFE',
 'S1A_EW_GRDM_1SDH_20150207T133434_20150207T133534_004518_0058C0_E8F8.SAFE',
 'S1A_EW_GRDM_1SDH_20150207T151223_20150207T151323_004519_0058C6_9A73.SAFE',
 'S1A_EW_GRDM_1SDH_20150207T151323_20150207T151423_004519_0058C6_924B.SAFE',
 'S1A_EW_GRDM_1SDH_20150208T060534_20150208T060634_004528_0058F9_E6B4.SAFE',
 'S1A_EW_GRDM_1SDH_20150208T060634_20150208T060734_004528_0058F9_D5A6.SAFE',
 'S1A_EW_GRDM_1SDH_20150208T074332_20150208T074432_004529_005901_BB82.SAFE',
 'S1A_EW_GRDM_1SDH_20150208T074432_20150208T074532_004529_005901_4D88.SAFE',
 'S1A_EW_GRDM_1SDH_20150208T092205_20150208T092305_004530_005907_4BA2.SAFE',
 'S1A_EW_GRDM_1SDH_20150208T141526_20150208T141626_004533_005914_8794.SAFE',
 'S1A_EW_GRDM_1SDH_20150208T155314_20150208T155414_004534_00591A_A901.SAFE',
 'S1A_EW_GRDM_1SDH_20150208T155414_20150208T155514_004534_00591A_8FDA.SAFE',
 'S1A_EW_GRDM_1SDH_20150209T064622_20150209T064722_004543_00594C_4F01.SAFE',
 'S1A_EW_GRDM_1SDH_20150209T064722_20150209T064822_004543_00594C_2EAC.SAFE',
 'S1A_EW_GRDM_1SDH_20150209T082436_20150209T082536_004544_005951_B2D9.SAFE',
 'S1A_EW_GRDM_1SDH_20150209T131804_20150209T131904_004547_00595F_6AE2.SAFE',
 'S1A_EW_GRDM_1SDH_20150209T131904_20150209T132004_004547_00595F_E650.SAFE',
 'S1A_EW_GRDM_1SDH_20150209T145548_20150209T145648_004548_005966_8B0E.SAFE',
 'S1A_EW_GRDM_1SDH_20150209T145648_20150209T145748_004548_005966_3EF0.SAFE',
 'S1A_EW_GRDM_1SDH_20150210T054823_20150210T054923_004557_0059A2_D44D.SAFE',
 'S1A_EW_GRDM_1SDH_20150210T054923_20150210T055023_004557_0059A2_DD6E.SAFE',
 'S1A_EW_GRDM_1SDH_20150210T072718_20150210T072818_004558_0059A7_7BCB.SAFE',
 'S1A_EW_GRDM_1SDH_20150210T072818_20150210T072918_004558_0059A7_C6CE.SAFE',
 'S1A_EW_GRDM_1SDH_20150210T153645_20150210T153745_004563_0059C0_C563.SAFE',
 'S1A_EW_GRDM_1SDH_20150210T153745_20150210T153845_004563_0059C0_5272.SAFE',
 'S1A_EW_GRDM_1SDH_20150211T130146_20150211T130246_004576_005A12_C892.SAFE',
 'S1A_EW_GRDM_1SDH_20150212T134245_20150212T134345_004591_005A66_6090.SAFE',
 'S1A_EW_GRDM_1SDH_20150212T152016_20150212T152116_004592_005A6D_CA46.SAFE',
 'S1A_EW_GRDM_1SDH_20150212T152116_20150212T152216_004592_005A6D_E2A5.SAFE',
 'S1A_EW_GRDM_1SDH_20150213T075156_20150213T075256_004602_005AAD_A438.SAFE',
 'S1A_EW_GRDM_1SDH_20150213T075256_20150213T075354_004602_005AAD_00B5.SAFE',
 'S1A_EW_GRDM_1SDH_20150213T093014_20150213T093114_004603_005AB3_0349.SAFE',
 'S1A_EW_GRDM_1SDH_20150213T124509_20150213T124609_004605_005ABF_86A9.SAFE',
 'S1A_EW_GRDM_1SDH_20150213T124609_20150213T124709_004605_005ABF_C08D.SAFE',
 'S1A_EW_GRDM_1SDH_20150213T142338_20150213T142438_004606_005AC3_5B54.SAFE',
 'S1A_EW_GRDM_1SDH_20150213T160150_20150213T160250_004607_005ACC_B9C1.SAFE',
 'S1A_EW_GRDM_1SDH_20150214T065403_20150214T065503_004616_005B02_6820.SAFE',
 'S1A_EW_GRDM_1SDH_20150214T065503_20150214T065603_004616_005B02_37B6.SAFE',
 'S1A_EW_GRDM_1SDH_20150214T083245_20150214T083345_004617_005B08_A7CA.SAFE',
 'S1A_EW_GRDM_1SDH_20150214T114828_20150214T114932_004619_005B13_A530.SAFE',
 'S1A_EW_GRDM_1SDH_20150214T132622_20150214T132722_004620_005B15_79F0.SAFE',
 'S1A_EW_GRDM_1SDH_20150215T055707_20150215T055807_004630_005B4F_8EA2.SAFE',
 'S1A_EW_GRDM_1SDH_20150215T055807_20150215T055907_004630_005B4F_3CAA.SAFE',
 'S1A_EW_GRDM_1SDH_20150215T073534_20150215T073634_004631_005B59_5123.SAFE',
 'S1A_EW_GRDM_1SDH_20150215T073634_20150215T073734_004631_005B59_5F60.SAFE',
 'S1A_EW_GRDM_1SDH_20150215T091336_20150215T091445_004632_005B5F_A563.SAFE',
 'S1A_EW_GRDM_1SDH_20150215T122856_20150215T122956_004634_005B68_778F.SAFE',
 'S1A_EW_GRDM_1SDH_20150215T122956_20150215T123056_004634_005B68_E7AE.SAFE',
 'S1A_EW_GRDM_1SDH_20150215T140710_20150215T140810_004635_005B6B_51C8.SAFE',
 'S1A_EW_GRDM_1SDH_20150215T140810_20150215T140910_004635_005B6B_E95D.SAFE',
 'S1A_EW_GRDM_1SDH_20150216T063755_20150216T063855_004645_005BAE_691C.SAFE',
 'S1A_EW_GRDM_1SDH_20150216T063855_20150216T063955_004645_005BAE_AE76.SAFE',
 'S1A_EW_GRDM_1SDH_20150216T130958_20150216T131058_004649_005BC2_AD17.SAFE',
 'S1A_EW_GRDM_1SDH_20150216T144734_20150216T144834_004650_005BC9_359D.SAFE',
 'S1A_EW_GRDM_1SDH_20150216T144834_20150216T144934_004650_005BC9_6918.SAFE',
 'S1A_EW_GRDM_1SDH_20150217T054059_20150217T054159_004659_005C01_5743.SAFE',
 'S1A_EW_GRDM_1SDH_20150217T071913_20150217T072013_004660_005C05_A822.SAFE',
 'S1A_EW_GRDM_1SDH_20150217T072013_20150217T072113_004660_005C05_CA1F.SAFE',
 'S1A_EW_GRDM_1SDH_20150217T135057_20150217T135157_004664_005C19_519D.SAFE',
 'S1A_EW_GRDM_1SDH_20150217T152849_20150217T152949_004665_005C20_385B.SAFE',
 'S1A_EW_GRDM_1SDH_20150217T152949_20150217T153049_004665_005C20_8C78.SAFE',
 'S1A_EW_GRDM_1SDH_20150218T062145_20150218T062245_004674_005C54_9DD6.SAFE',
 'S1A_EW_GRDM_1SDH_20150218T062245_20150218T062345_004674_005C54_EB16.SAFE',
 'S1A_EW_GRDM_1SDH_20150218T075952_20150218T080052_004675_005C5C_60D6.SAFE',
 'S1A_EW_GRDM_1SDH_20150218T080052_20150218T080152_004675_005C5C_83FE.SAFE',
 'S1A_EW_GRDM_1SDH_20150218T093804_20150218T093911_004676_005C63_29B2.SAFE',
 'S1A_EW_GRDM_1SDH_20150218T125327_20150218T125427_004678_005C70_EA98.SAFE',
 'S1A_EW_GRDM_1SDH_20150218T125427_20150218T125521_004678_005C70_419F.SAFE',
 'S1A_EW_GRDM_1SDH_20150218T143152_20150218T143252_004679_005C74_A90A.SAFE',
 'S1A_EW_GRDM_1SDH_20150218T160943_20150218T161043_004680_005C7D_5308.SAFE',
 'S1A_EW_GRDM_1SDH_20150219T070245_20150219T070345_004689_005CB6_82A3.SAFE',
 'S1A_EW_GRDM_1SDH_20150219T070345_20150219T070445_004689_005CB6_F81A.SAFE',
 'S1A_EW_GRDM_1SDH_20150219T084056_20150219T084156_004690_005CBF_867A.SAFE',
 'S1A_EW_GRDM_1SDH_20150220T141526_20150220T141626_004708_005D1C_CB27.SAFE',
 'S1A_EW_GRDM_1SDH_20150220T155314_20150220T155414_004709_005D22_DFFD.SAFE',
 'S1A_EW_GRDM_1SDH_20150220T155414_20150220T155514_004709_005D22_1846.SAFE',
 'S1A_EW_GRDM_1SDH_20150221T064622_20150221T064722_004718_005D54_19C0.SAFE',
 'S1A_EW_GRDM_1SDH_20150221T064722_20150221T064822_004718_005D54_100E.SAFE',
 'S1A_EW_GRDM_1SDH_20150221T082436_20150221T082536_004719_005D5B_32B0.SAFE',
 'S1A_EW_GRDM_1SDH_20150221T131805_20150221T131905_004722_005D6F_EDA0.SAFE',
 'S1A_EW_GRDM_1SDH_20150221T131905_20150221T132005_004722_005D6F_7FD0.SAFE',
 'S1A_EW_GRDM_1SDH_20150221T145548_20150221T145648_004723_005D74_F895.SAFE',
 'S1A_EW_GRDM_1SDH_20150221T145648_20150221T145748_004723_005D74_34C3.SAFE',
 'S1A_EW_GRDM_1SDH_20150222T054823_20150222T054923_004732_005DAE_262F.SAFE',
 'S1A_EW_GRDM_1SDH_20150222T054923_20150222T055023_004732_005DAE_783E.SAFE',
 'S1A_EW_GRDM_1SDH_20150222T072718_20150222T072818_004733_005DB3_7039.SAFE',
 'S1A_EW_GRDM_1SDH_20150222T072818_20150222T072918_004733_005DB3_009C.SAFE',
 'S1A_EW_GRDM_1SDH_20150222T090539_20150222T090644_004734_005DB9_FD64.SAFE',
 'S1A_EW_GRDM_1SDH_20150222T135902_20150222T140002_004737_005DCB_C31F.SAFE',
 'S1A_EW_GRDM_1SDH_20150222T153645_20150222T153745_004738_005DD2_85CA.SAFE',
 'S1A_EW_GRDM_1SDH_20150222T153745_20150222T153845_004738_005DD2_C733.SAFE',
 'S1A_EW_GRDM_1SDH_20150223T062950_20150223T063050_004747_005E11_A5C0.SAFE',
 'S1A_EW_GRDM_1SDH_20150223T063050_20150223T063150_004747_005E11_A98A.SAFE',
 'S1A_EW_GRDM_1SDH_20150223T080743_20150223T080843_004748_005E1D_E61E.SAFE',
 'S1A_EW_GRDM_1SDH_20150223T080843_20150223T080943_004748_005E1D_6F95.SAFE',
 'S1A_EW_GRDM_1SDH_20150223T130146_20150223T130246_004751_005E31_51B1.SAFE',
 'S1A_EW_GRDM_1SDH_20150223T143929_20150223T144029_004752_005E35_F679.SAFE',
 'S1A_EW_GRDM_1SDH_20150223T144029_20150223T144129_004752_005E35_6245.SAFE',
 'S1A_EW_GRDM_1SDH_20150224T053248_20150224T053348_004761_005E6E_06B0.SAFE',
 'S1A_EW_GRDM_1SDH_20150224T071059_20150224T071159_004762_005E75_5564.SAFE',
 'S1A_EW_GRDM_1SDH_20150224T071159_20150224T071259_004762_005E75_B1D8.SAFE',
 'S1A_EW_GRDM_1SDH_20150224T134245_20150224T134345_004766_005E8F_2CA0.SAFE',
 'S1A_EW_GRDM_1SDH_20150224T152016_20150224T152116_004767_005E95_A96A.SAFE',
 'S1A_EW_GRDM_1SDH_20150224T152116_20150224T152216_004767_005E95_7014.SAFE',
 'S1A_EW_GRDM_1SDH_20150225T061339_20150225T061439_004776_005ED2_F26B.SAFE',
 'S1A_EW_GRDM_1SDH_20150225T061439_20150225T061539_004776_005ED2_CB0A.SAFE',
 'S1A_EW_GRDM_1SDH_20150225T075156_20150225T075256_004777_005EDD_059F.SAFE',
 'S1A_EW_GRDM_1SDH_20150225T075256_20150225T075354_004777_005EDD_6D3E.SAFE',
 'S1A_EW_GRDM_1SDH_20150225T124509_20150225T124609_004780_005EF3_A119.SAFE',
 'S1A_EW_GRDM_1SDH_20150225T124609_20150225T124709_004780_005EF3_9C87.SAFE',
 'S1A_EW_GRDM_1SDH_20150225T142338_20150225T142438_004781_005EF6_C2FC.SAFE',
 'S1A_EW_GRDM_1SDH_20150225T160156_20150225T160256_004782_005EFF_797E.SAFE',
 'S1A_EW_GRDM_1SDH_20150226T065403_20150226T065503_004791_005F3C_5FBE.SAFE',
 'S1A_EW_GRDM_1SDH_20150226T065503_20150226T065603_004791_005F3C_663F.SAFE',
 'S1A_EW_GRDM_1SDH_20150226T083245_20150226T083345_004792_005F43_6FEE.SAFE',
 'S1A_EW_GRDM_1SDH_20150226T114828_20150226T114932_004794_005F53_D74A.SAFE',
 'S1A_EW_GRDM_1SDH_20150226T132622_20150226T132722_004795_005F57_3E01.SAFE',
 'S1A_EW_GRDM_1SDH_20150226T150344_20150226T150444_004796_005F5C_FECE.SAFE',
 'S1A_EW_GRDM_1SDH_20150226T150444_20150226T150544_004796_005F5C_07F7.SAFE',
 'S1A_EW_GRDM_1SDH_20150227T055707_20150227T055807_004805_005F90_10F8.SAFE',
 'S1A_EW_GRDM_1SDH_20150227T055807_20150227T055907_004805_005F90_B9D3.SAFE',
 'S1A_EW_GRDM_1SDH_20150227T073533_20150227T073633_004806_005F97_43D2.SAFE',
 'S1A_EW_GRDM_1SDH_20150227T073633_20150227T073733_004806_005F97_BFF0.SAFE',
 'S1A_EW_GRDM_1SDH_20150227T091336_20150227T091445_004807_005F9E_2892.SAFE',
 'S1A_EW_GRDM_1SDH_20150227T122856_20150227T122956_004809_005FA6_0B8E.SAFE',
 'S1A_EW_GRDM_1SDH_20150227T122956_20150227T123056_004809_005FA6_5694.SAFE',
 'S1A_EW_GRDM_1SDH_20150227T140710_20150227T140810_004810_005FAA_29F4.SAFE',
 'S1A_EW_GRDM_1SDH_20150227T140810_20150227T140910_004810_005FAA_0D2D.SAFE',
 'S1A_EW_GRDM_1SDH_20150227T154518_20150227T154618_004811_005FB1_5EAF.SAFE',
 'S1A_EW_GRDM_1SDH_20150228T063755_20150228T063855_004820_005FEF_19C7.SAFE',
 'S1A_EW_GRDM_1SDH_20150228T063855_20150228T063955_004820_005FEF_24CC.SAFE',
 'S1A_EW_GRDM_1SDH_20150228T130958_20150228T131058_004824_005FFF_307B.SAFE']

idir = '/Data/sat/downloads/sentinel1/2015/'
odir = '/Data/sat/downloads/sentinel1/2015/processed_CLAHE_update/'

# Ensure the output directory exists
try:
    os.makedirs(odir, exist_ok=True)
    print(f"Output directory '{odir}' ensured.")
except OSError as e:
    print(f"ERROR: Could not create output directory '{odir}': {e}")
    # Decide whether to exit or continue
    exit(1) # Exit if the directory can't be created

# --- TEMPORARY: Process only specific files for this run ---
# # Find all .SAFE files in the directory
# safe_files = glob.glob(os.path.join(idir, '*.SAFE'))
#
# # Filter out any potential non-SAFE directories if needed
# safe_files = [f for f in safe_files if os.path.isdir(f) and f.endswith('.SAFE')]

# Use the predefined list of files to process
# Construct full paths for the specified files from files_to_process
#print(f"INFO: Processing specific files defined in 'files_to_process': {files_to_process}")
safe_files = [os.path.join(idir, fname) for fname in files_to_process]

# Filter to ensure these specified files exist and are valid .SAFE directories
original_count = len(safe_files)
safe_files = [f for f in safe_files if os.path.exists(f) and os.path.isdir(f) and f.endswith('.SAFE')]
if len(safe_files) < original_count:
    print(f"WARNING: Some files specified in 'files_to_process' were not found or are not valid .SAFE directories. Found {len(safe_files)} valid files.")
# --- END TEMPORARY ---

# Sort the file paths by date
# Add error handling in sorting in case extract_date fails for some files
try:
    sorted_safe_files = sorted(safe_files, key=extract_date)
except Exception as e:
    print(f"ERROR during sorting SAFE files: {e}")
    print("Proceeding with unsorted file list.")
    # Decide how to proceed: maybe process unsorted, or exit
    sorted_safe_files = safe_files # Process unsorted as a fallback

print(f"{len(sorted_safe_files)} SAFE files found")

# Use ProcessPoolExecutor to parallelize the processing with a progress bar
# Adjust max_workers based on your system's CPU cores and memory
num_workers = min(16, os.cpu_count() or 1) # Safer default
print(f"Starting processing with {num_workers} workers.")

with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = []
    # Use tqdm context manager for better cleanup
    with tqdm(total=len(sorted_safe_files), desc="Processing SAFE files") as pbar:
        for ifile in sorted_safe_files:
            future = executor.submit(process_file, ifile, odir)
            # Update progress bar when a future completes
            future.add_done_callback(lambda p: pbar.update())
            futures.append(future)

    # Wait for all futures to complete and retrieve results/exceptions
    print("Waiting for all tasks to complete...")
    completed_count = 0
    error_count = 0
    for future in futures:
        try:
            future.result()  # Raises exceptions from the worker process if any occurred
            completed_count += 1
        except Exception as e:
            # The error should have already been printed within process_file
            # but we can log a summary here if needed.
            error_count += 1
            # Optionally print a summary error message from the main process
            # print(f"ERROR: A worker process failed (see details above). Exception: {e}")

print(f"Processing complete. {completed_count} tasks finished successfully, {error_count} tasks failed.")
