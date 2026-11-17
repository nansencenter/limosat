# Published trajectory dataset

`limosat.publish` converts the internal database outputs of a release into the public LiMOSAT trajectory
dataset (schema 2): a format fit for public use that leaves out everything an ordinary data user does not
need (internal ids and row numbers, quality-control bookkeeping, removed links, raw points, local paths).
It reads each period's internal kept-link Parquet and its manifest plus the period's image catalogue, and
never scores, filters or alters a link. Keep it independent of `limosat.qc`: the public format is a stable
contract, while the internal files may change.

The result is one Parquet point table per processing period (one row per observation, sorted by time, with
the Sentinel-1 scene name and an `is_primary` flag for the period's own date window), plus a manifest, a
generated data dictionary and checksums.

## Format

One Parquet file per processing period, `limosat_<period>.parquet`, flat in the output folder, and:

- `manifest.json`: schema version, release id, column definitions, flag bits, the primary window of every
  period, and per file the checksum, size, row, trajectory and displacement counts, time and spatial bounds.
- `DATA_DICTIONARY.md`: generated from the schema constants in `limosat/publish.py`. Do not edit by hand.
- `checksums.sha256`.

A row is one **observation** of a feature in one Sentinel-1 scene, carrying the displacement from the previous observation.
A trajectory is the rows with one `trajectory_id` (unique within a period) ordered by `point_index`; the first
row has `point_index` 0 and null displacement columns. Rows are sorted by `time`, with time-ordered row groups so
date queries read few row groups.

| Column | Meaning |
| --- | --- |
| `period`, `trajectory_id`, `point_index` | identity; use (`period`, `trajectory_id`) across periods |
| `time`, `scene` | UTC acquisition time and Sentinel-1 product name of the scene |
| `x_m`, `y_m` | EPSG:3413 position in metres |
| `dx_m`, `dy_m`, `speed_m_per_day` | displacement from the previous row of the trajectory |
| `interpolated`, `correlation`, `direct_match_share`, `flags` | properties of the arriving displacement |
| `is_primary` | the row lies in the period's own date window |

The full definitions, flag bits and the explanation of `is_primary` are in the generated data dictionary.

### Overlapping periods and `is_primary`

Periods are 15 months long and overlap their neighbours by three months. All rows are published. Rule
`limosat_release_august_annual_v1` gives each displacement to one period by its start time: a period owns start times
from 1 August of its first year up to 1 August of the next, the first period also owning its first three
months. `is_primary` is true on rows in that window (for a trajectory's first row, its own time; otherwise the
start time of the arriving displacement). Keep `is_primary` rows of all periods to use each date once.

## Scene names

The scene name comes from the period's image catalogue (`<period>_image_catalog_*.json`, written by
`limosat.catalog`, which numbers images by acquisition time then file name). `publish` checks that every image
id exists in the catalogue and that its time equals the observation time, so a wrong catalogue fails the build.

## Build and validate

Install PyArrow in the active environment (`pip install pyarrow`, or conda-forge), then:

```bash
python -m limosat.publish build \
  --source-root /path/to/internal/release \        # <period>/<period>_qc.parquet + manifest
  --catalog-dir /path/to/image_catalogs \
  --output-dir /path/to/staging/public_v2 \
  --release-id <public release id>
```

`--period YYYYMM_YYYYMM` (repeatable) builds selected periods; `--overwrite` replaces existing files;
`--skip-source-checksum` skips hashing the sources against their manifests. The build streams month by month
(about 3 GB of memory for a 52-million-link period) and checks that the row count equals displacements plus
trajectories and that the number of primary displacements equals the number of source links starting in the window.

Internal provenance (source paths, checksums, protocol ids) is written next to, not inside, the output folder
as `<output-dir>.provenance.json`. Do not publish it. Public outputs contain no local paths.

```bash
python -m limosat.publish validate --source-root ... --catalog-dir ... --output-dir ... --sample-modulus 50
```

`validate` re-checks checksums, schema, time order, `point_index`, and the manifest counts for every file,
and compares every Nth trajectory (default 50, 1 checks all) with the source links: positions, displacement values,
flags, scenes, chain continuity and `is_primary`.

## Reader examples

```python
import pandas as pd
frame = pd.read_parquet("limosat_202305_202407.parquet")
once = frame[frame.is_primary]                       # each date once across periods
trajectory = frame[(frame.trajectory_id == 401)].sort_values("point_index")
```

```sql
-- DuckDB
SELECT date_trunc('month', time) AS month, avg(speed_m_per_day)
FROM read_parquet('limosat_*.parquet') WHERE is_primary AND point_index > 0 GROUP BY 1;
```
