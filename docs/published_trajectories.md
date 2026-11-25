# Published trajectory dataset (EfficientLoFTR)

`limosat.publish` converts a primary composition (`limosat compose-parquet`, see `composition.md`) into the public
LiMOSAT-EfficientLoFTR trajectory dataset (schema 1): one CF-1.8 trajectory file in NetCDF4 for the composed period (for example
January to April 2020), with a `manifest.json`. The format deliberately follows the public release of the ORB pipeline
(`limosat.publish` on the `olivia` branch, schema 2): the same dimensions, ordering, naming, compression, global attributes, manifest
and build/validate commands, so that users and readers handle both alike. The observation variables differ because the pipelines
record different things. This branch does not import the ORB code; the two publishers are independent.

It reads only `trajectory-extensions-qc-v1.parquet` and `composition-manifest-v1.json` of the composition, never scores, filters or
alters an extension, and leaves out what an ordinary data user does not need: internal trajectory and pair ids, quality-control
bookkeeping, rejected extensions, created-only and dormant points and local paths.

## Build and validate

Use an environment with PyArrow, netCDF4 and pyproj (`pip install "limosat[publish]"`, or the conda environment of
`environment.yaml`):

```bash
python -m limosat.publish build \
  --source /path/to/primary-composition \         # trajectory-extensions-qc-v1.parquet + composition-manifest-v1.json
  --output-dir /path/to/staging/limosat_eloftr_v1 \
  --release-id limosat_eloftr_v1 \
  --metadata netcdf_metadata.json \               # title, summary, creators (ORCID), licence, acknowledgement; optional
  --tmp-dir /path/with/room/for/scratch           # about 80 bytes per observation, removed afterwards
```

`build` writes `limosat_eloftr_<period>.nc` (`--period` names the period; the default is the first and last start month) and
`manifest.json`, then validates the file (`--no-validate` skips that). `--overwrite` replaces an existing file;
`--skip-source-checksum` skips hashing the extensions table against the composition manifest. Internal provenance (source path,
checksums, run id, configuration hash, QC protocol) is written next to, not inside, the output folder as
`<output-dir>.provenance.json`; do not publish it. The README of a release is written by hand.

```bash
python -m limosat.publish validate --source ... --output-dir ... --sample-modulus 50
```

`validate` re-checks the checksum, global attributes, the layout (`rowsize` total, unique ids, `point_index`, start-time order, time
order within a trajectory, `time_start`/`time_end`, flag masks) and the manifest counts, and compares every Nth trajectory (default 50,
1 checks all) with the source extensions: it chooses the chain of each sampled trajectory again from all its links, then compares
positions, displacement values, match statistics, flags, scenes and the observation each displacement starts at.

### Memory and disk

The extensions table is far larger than memory (about 175 million rows for January to April 2020), so `build` works in three streaming
passes. Pass 1 reads the trajectory id, start and end time and `accepted` flag, chooses the chain of every trajectory (`select_chain`)
and builds the trajectory index (6 GB peak for the 175 million links of January to April 2020, about 10 minutes). Pass 2 reads every column and spills
the observations to bucket files, each holding whole trajectories (77 bytes per observation, so about 14 GB for 182 million
observations). Pass 3 sorts one bucket at a time in memory (`--bucket-observations`, default 20 million, about 3 GB) and writes it
into the file. The output is about 50 bytes per observation (9 GB for January to April 2020). Plan for the scratch space on a
fast disk.

## File layout

A file is a CF *trajectory* featureType in the contiguous ragged array representation (CF 9.3.3):

- dimensions `trajectory`, `obs` and `scene`; `rowsize(trajectory)` (`sample_dimension = "obs"`) is the number of observations of
  each trajectory, so trajectory *i* is `obs[offset[i] : offset[i] + rowsize[i]]` of every `obs` variable;
- trajectories are ordered by start time (`time_start`, ascending, ties by an internal id) and the observations of a trajectory in time
  order (`point_index`, 0 at the first observation), so a date window reads as a few contiguous runs and one trajectory is a slice;
- `trajectory_id` (the running number in the file, `cf_role`), `time_start`, `time_end` on the trajectory dimension;
- on the observation dimension: `time` (float64 seconds since 1970-01-01), `lat`/`lon` (WGS 84, computed from the positions),
  `x_m`/`y_m` (EPSG:3413 metres, grid mapping `crs`), `point_index`, `dx_m`, `dy_m`, `speed_m_per_day`,
  `selected_matches`, `support_radius_m`, `maximum_residual_m`, `quality_flags` (`flag_masks`/`flag_meanings`), `scene_index`;
- `scene_name(scene)`: the Sentinel-1 product name of every scene that holds an observation, looked up with `scene_index`;
- the displacement variables describe the step from the previous observation and are missing (`_FillValue`: NaN, 255 for
  `quality_flags`) on a trajectory's first observation.

Nothing is rounded or packed: times, positions and displacements are the internal float64 numbers, the three match statistics a
float32 of the internal float64, and compression is lossless (zlib level 4 with shuffle, 65,536-observation chunks). The global
attributes follow CF and ACDD (title, summary, creators with ORCID, licence, time coverage, bounds, release id, schema version, QC
policy, counts).

### Observations, gaps and chains

An observation is one feature position on one scene. A trajectory is created at a seed point and continues through the scene pairs
that support it; it has no observation on a scene that no pair from its position reaches with support, and it may continue later
through another pair. So the time step between consecutive observations varies, from the 98 minutes of two neighbouring passes to
days.

**Chains.** The composition stores a trajectory as a tree. A new scene is reached through the best supported pair from an observation
of the trajectory, preferring the most recent source image; when no pair leads from the latest observation to the new scene (a pair
was not planned), the new observation continues from an older one, and the observation that was skipped keeps no continuation (a side
observation). In January to April 2020, 14.7% of the links end in such a side observation, and in 2.8% of the observations two
continuations both go on, so the tree can fork. The release does not publish the tree. It publishes one **chain** per trajectory:
the path from its first observation to its **latest** observation (`select_chain`, rule `latest`). Every displacement then starts at
the previous row, `point_index` is the position along the chain, and a velocity is `dx_m` divided by the time difference of two
consecutive rows.

Links off the chain are not published: 19.6% of the 174.5 million links of January to April 2020 (14.7% side observations and 4.9%
that belong to branches abandoned at a fork, with their descendants); 44% of the trajectories of January lose at least one branch.
They are counted in the manifest (`links_off_chain_not_published`). The alternative rule `deepest` (`--chain-rule deepest`, the path
with the most links) keeps 0.5 percentage points more links, but it can end a trajectory before its latest observation; in January it
chose a different chain for 3% of the trajectories.

Dropping them has little effect on a regional deformation field (tested on 3-day windows in four months and six regions, 18 to 30 hour
links): median 98.5% of the cells with deformation survive, the cell values correlate with r = 0.97 with those of the full tree
(0.965 when the same share of links is removed at random), and 95% of the high-deformation cells (above 0.05 per day) stay high.

### Quality flags

`quality_flags` is a bit mask on each displacement: bit 0 `EXCEEDS_CONFIGURED_SPEED` (the speed is above the speed the matcher is
configured for, 43,200 m/day in production) and bit 1 `LOCAL_NEIGHBOUR_REVIEW` (the displacement differs strongly from the other
displacements between the same two scenes within 20 km). Both are review flags: the displacement is published. Displacements above the
hard speed limit (60,000 m/day) are rejected by the composition and are not published (none in the January to April 2020 run).
Flags never split trajectories.

## Reader example

```python
import numpy as np, xarray as xr
ds = xr.open_dataset("limosat_eloftr_202001_202004.nc")        # lazy; dims trajectory, obs, scene
february = (ds.time >= np.datetime64("2020-02-01")) & (ds.time < np.datetime64("2020-03-01"))
x = ds["x_m"].isel(obs=february.values)
```
