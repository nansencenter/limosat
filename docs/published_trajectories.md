# Published trajectory dataset

`limosat.publish` turns a frozen internal release into the public LiMOSAT trajectory dataset (schema 2).
It only projects existing results: it reads each period's internal kept-link Parquet and its manifest plus
the period's image catalogue, and never scores, filters or alters a link. Keep it independent of
`limosat.qc`: the public format is a stable contract, while the internal files may change.

## Format

One Parquet file per processing period, `limosat_<period>.parquet`, flat in the output folder, and:

- `manifest.json`: schema version, release id, column definitions, flag bits, the `primary` window of every
  period, and per file the checksum, size, row, trajectory and step counts, time and spatial bounds.
- `DATA_DICTIONARY.md`: generated from the schema constants in `limosat/publish.py`. Do not edit by hand.
- `checksums.sha256`.

A row is one **observation** of an ice feature in one Sentinel-1 scene, carrying the step that arrived at it.
A trajectory is the rows with one `trajectory_id` (unique within a period) ordered by `point_index`; the first
row has `point_index` 0 and null step columns. Rows are sorted by `time`, with time-ordered row groups so
date queries read few row groups.

| Column | Meaning |
| --- | --- |
| `period`, `trajectory_id`, `point_index` | identity; use (`period`, `trajectory_id`) across periods |
| `time`, `scene` | UTC acquisition time and Sentinel-1 product name of the scene |
| `x_m`, `y_m` | EPSG:3413 position in metres |
| `dx_m`, `dy_m`, `speed_m_per_day` | step from the previous row of the trajectory |
| `interpolated`, `corr`, `direct_match_share`, `flags` | properties of the arriving step |
| `primary` | the row lies in the period's own date window |

The full definitions, flag bits and the explanation of `primary` are in the generated data dictionary.

### Overlapping periods and `primary`

Periods are 15 months long and overlap their neighbours by three months. All rows are published. Rule
`limosat_release_august_annual_v1` gives each step to one period by its start time: a period owns start times
from 1 August of its first year up to 1 August of the next, the first period also owning its first three
months. `primary` is true on rows in that window (for a trajectory's first row, its own time; otherwise the
start time of the arriving step). Keep `primary` rows of all periods to use each date once.

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
(about 3 GB of memory for a 52-million-link period) and checks that the row count equals steps plus
trajectories and that the number of `primary` steps equals the number of source links starting in the window.

Internal provenance (source paths, checksums, protocol ids) is written next to, not inside, the output folder
as `<output-dir>.provenance.json`. Do not publish it. Public outputs contain no local paths.

```bash
python -m limosat.publish validate --source-root ... --catalog-dir ... --output-dir ... --sample-modulus 50
```

`validate` re-checks checksums, schema, time order, `point_index`, and the manifest counts for every file,
and compares every Nth trajectory (default 50, 1 checks all) with the source links: positions, step values,
flags, scenes, chain continuity and `primary`.

## Reader examples

```python
import pandas as pd
frame = pd.read_parquet("limosat_202305_202407.parquet")
once = frame[frame.primary]                       # each date once across periods
trajectory = frame[(frame.trajectory_id == 401)].sort_values("point_index")
```

```sql
-- DuckDB. PRIMARY is a reserved SQL word, so the column must be quoted.
SELECT date_trunc('month', time) AS month, avg(speed_m_per_day)
FROM read_parquet('limosat_*.parquet') WHERE "primary" AND point_index > 0 GROUP BY 1;
```
