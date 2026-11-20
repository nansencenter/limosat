# Published trajectory dataset

`limosat.publish` converts the internal database outputs of a release into the public LiMOSAT trajectory dataset
(schema 2): CF-1.8 trajectory files in NetCDF4, one per processing period, fit for public use and leaving out everything an
ordinary data user does not need (internal ids and row numbers, quality-control bookkeeping, removed links, raw points, local
paths). It reads each period's internal kept-link Parquet (`<period>/<period>_qc.parquet`) and its manifest plus the period's
image catalogue, and never scores, filters or alters a link. Keep it independent of `limosat.qc`: the public format is a
stable contract, while the internal files may change. The QC SQLite files are not used.

## Build and validate

Use the LiMOSAT Conda environment from `environment.yaml` (it includes PyArrow, netCDF4 and pyproj):

```bash
python -m limosat.publish build \
  --source-root /path/to/internal/release \        # <period>/<period>_qc.parquet + manifest
  --catalog-dir /path/to/image_catalogs \
  --output-dir /path/to/staging/limosat_v1 \
  --release-id limosat_v1 \
  --metadata netcdf_metadata.json                  # title, summary, creators (ORCID), licence, acknowledgement; optional
```

`build` writes the files, `manifest.json`, and then validates them (`--no-validate` skips that). `--period YYYYMM_YYYYMM`
(repeatable) builds selected periods; `--overwrite` replaces existing files; `--skip-source-checksum` skips hashing the
sources against their manifests. The source is read month by month and checks that the observation count equals
displacements plus trajectories and that the number of primary displacements equals the number of source links starting in
the window. A 55-million-observation period needs about 8 GB of memory and a few minutes.

Internal provenance (source paths, checksums, protocol ids) is written next to, not inside, the output folder as
`<output-dir>.provenance.json`. Do not publish it. The files carry no local paths. The README of a release is written by hand.

```bash
python -m limosat.publish validate --source-root ... --catalog-dir ... --output-dir ... --sample-modulus 50
```

`validate` re-checks checksums, global attributes, the layout (`rowsize` total, unique ids, `point_index`, start-time order,
`time_start`/`time_end`, flag masks) and the manifest counts for every file, and compares every Nth trajectory (default 50, 1
checks all) with the source links: positions, displacement values, quality flags, scenes, chain continuity and `is_primary`.

## File layout

One file per processing period, `limosat_<period>.nc`, flat in the output folder, and `manifest.json` (schema version,
release id, column definitions, flag bits, the primary window of every period, and per file the checksum, size, observation,
trajectory and displacement counts, time and spatial bounds, compression and layout).

A file is a CF *trajectory* featureType in the contiguous ragged array representation (CF 9.3.3):

- dimensions `trajectory`, `obs` and `scene`; `rowsize(trajectory)` (`sample_dimension = "obs"`) is the number of observations of
  each trajectory, so trajectory *i* is `obs[offset[i] : offset[i] + rowsize[i]]` of every `obs` variable;
- trajectories are ordered by start time (`time_start`, ascending) and the observations of a trajectory by `point_index`
  (0 at the first observation), so a date window reads as a few contiguous runs and one trajectory is a single slice;
- `trajectory_id` (unique within a period, `cf_role`), `time_start`, `time_end` on the trajectory dimension;
- on the observation dimension: `time` (float64 seconds since 1970-01-01), `lat`/`lon` (WGS 84, computed from the positions),
  `x_m`/`y_m` (EPSG:3413 metres, grid mapping `crs`), `point_index`, `dx_m`, `dy_m`, `speed_m_per_day`, `interpolated`,
  `correlation`, `direct_match_share`, `quality_flags` (`flag_masks`/`flag_meanings`), `is_primary`, `scene_index`;
- `scene_name(scene)`: the Sentinel-1 product name of every scene, looked up with `scene_index`;
- the displacement variables describe the step from the previous observation and are missing (`_FillValue`: NaN, -1 for
  `interpolated`, 255 for `quality_flags`) on a trajectory's first observation.

Nothing is rounded or packed: values are the internal float64/float32 numbers, and compression is lossless (zlib level 4 with
shuffle, 65,536-observation chunks). The global attributes follow CF and ACDD (title, summary, creators with ORCID, licence,
time coverage, bounds, release id, schema version, primary window and rule, counts).

`is_primary` marks the observations whose arriving displacement starts in the period's own half-open window (a period owns start
times from 1 August of its first year to 1 August of the next; the first period also owns its first three months). Selecting
`is_primary` observations from all periods gives every date once; all observations of the three-month overlaps are published.

## Scene names

The scene name comes from the period's image catalogue (`<period>_image_catalog_*.json`, written by `limosat.catalog`, which
numbers images by acquisition time then file name). `publish` checks that every image id exists in the catalogue and that its
time equals the observation time, so a wrong catalogue fails the build.

## Reader examples

```python
import numpy as np, xarray as xr
ds = xr.open_dataset("limosat_202305_202407.nc")               # lazy; dims trajectory, obs, scene
january = (ds.time >= np.datetime64("2024-01-01")) & (ds.time < np.datetime64("2024-02-01")) & (ds.is_primary == 1)
x = ds["x_m"].isel(obs=january.values)
```

For date and space-time windows, whole trajectories, displacements with both ends, flag filters and column selection use
the `limosat-analysis` package (`open_limosat("limosat_<period>.nc")`).
