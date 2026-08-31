[![Tests](https://github.com/nansencenter/arktalas_ice_drift_experiments/actions/workflows/test.yaml/badge.svg?branch=limosat)](https://github.com/nansencenter/arktalas_ice_drift_experiments/actions/workflows/test.yaml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15111936.svg)](https://doi.org/10.5281/zenodo.15111936)

# limosat - Lagrangian Ice Motion from Satellites

limosat is an open source, highly configurable algorithm for calculating sea ice drift from remote sensing imagery. It processes satellite images like so:

1.	Keypoint Detection: Identify points representing ice features in a semi-constrained grid.
2.	Interpolation: Interpolate positions for unmatched keypoints using homography.
3.	Pattern Matching: Refine keypoint positions using template matching and discard points with low correlation.
4.	Persistent Storage: Optionally save drift data for later analysis.

## Learned drift workflow

The experimental ALIKED workflow has a separate, compact facade because its
detect-on-each-image and local displacement-field architecture does not fit the
ORB trajectory classes cleanly:

```python
from limosat.learned_drift import ALIKEDDrift

tracker = ALIKEDDrift(device="cuda", cache_dir=feature_cache)
result = tracker.track_images(source_image, target_image, elapsed_hours=21.4)
```

`ALIKEDDrift.track_sequence` extracts each unique image once and can derive a
matching prior only from the immediately preceding accepted field. The
selected command-line pair entry point is
`experiments/run_learned_drift_pair.py`. PyTorch and Kornia are optional
learned-workflow dependencies and are not required by the existing ORB path.

## Citation

Chua, S. M. T., & Korosov, A. (2025). limosat - Lagrangian Ice Motion from Satellites (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.15111936

## Repository Structure

```
.
├── LICENSE              # MIT License information
├── README.md            # Project documentation
├── environment.yaml     # Conda environment specification
├── examples/            # Usage examples
├── limosat/             # LiMOSAT library source code
└── tests/               # Tests
```
## Setup Environment

```bash
conda env create -f environment.yaml && conda activate limosat
```

## Run limosat

1.	Prepare Your Data:
Organize your satellite imagery into a folder and run `preprocessing.py`.
   If you use the gridded descriptor cache, note that changing stride or descriptor
   parameters (ORB settings, border size, octave) requires regenerating the cache.
2. Build catalog:
Use create_image_gdf to build a catalog of imagery metadata.
3.	Set-up database(optional):
Enable persistence by providing a SQL engine and Zarr storage path to store both the drift keypoints and pattern matching templates. limosat can also be run without persistence.
4.	Run `examples/limosat_drift.ipynb`
5. Visualise results

## License

This project is licensed under the MIT License. See the LICENSE file for details.
