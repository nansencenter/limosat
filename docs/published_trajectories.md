# Published trajectory extensions

`limosat.publish` creates the sole public scientific dataset for a frozen LiMOSAT
QC SQLite release: one Parquet file of accepted, consecutive trajectory
extensions. It uses the post-QC trajectory identifiers and segment boundaries
already materialized by `limosat.qc`; it does not apply or expose QC decisions.

The coordinate reference system is EPSG:3413 and all coordinate and displacement
values are metres. Times are UTC Parquet timestamps. The Parquet schema metadata
records the release ID, source window, CRS, units, schema version, and SHA-256 of
the frozen QC SQLite input. The adjacent `*.manifest.json` records input/output
checksums, counts, bounds, and validation results.

## Data dictionary (schema v1.0.0)

| Field | Type | Meaning |
| --- | --- | --- |
| `trajectory_id` | int64 | Stable post-QC trajectory segment identifier. |
| `source_image_id`, `target_image_id` | int64 | Consecutive source and target image identifiers. |
| `start_time_utc`, `end_time_utc` | timestamp[us, UTC] | Endpoint acquisition times. |
| `elapsed_seconds` | float64 | `end_time_utc - start_time_utc`, in seconds. |
| `x0_m`, `y0_m`, `x1_m`, `y1_m` | float64 | Source and target EPSG:3413 coordinates, in metres. |
| `dx_m`, `dy_m` | float64 | Target minus source displacement, in metres. |
| `distance_m` | float64 | Euclidean displacement magnitude, in metres. |
| `speed_m_s` | float64 | `distance_m / elapsed_seconds`. |
| `speed_m_per_day` | float64 | Speed scaled to metres per day. |
| `source_position_type`, `target_position_type` | string | `observed` or `interpolated` endpoint status. |

## Publishing and validation

Install the optional writer dependency once in the active environment:

```bash
conda install -c conda-forge pyarrow
```

Publish one selected frozen release (the Parquet and manifest are written beside
the SQLite input):

```bash
python -m limosat.publish one \
  --input /Volumes/KINGSTON/arktalas/qc/releases/20260908_v1/201905_202007/201905_202007_qc.sqlite
```

Discover and publish every eligible frozen QC SQLite release under the root:

```bash
python -m limosat.publish all \
  --release-root /Volumes/KINGSTON/arktalas/qc/releases/20260908_v1
```

Validate an existing output row-for-row against its QC SQLite input:

```bash
python -m limosat.publish validate-all \
  --release-root /Volumes/KINGSTON/arktalas/qc/releases/20260908_v1
```

The validation checks the QC release gate, expected accepted-edge count,
trajectory continuity, and that no output edge matches a recorded QC break.

## Reader examples

```python
# PyArrow
import pyarrow.parquet as pq
table = pq.read_table("201905_202007_qc_publish.parquet")
metadata = table.schema.metadata
```

```sql
-- DuckDB
SELECT AVG(speed_m_per_day)
FROM read_parquet('201905_202007_qc_publish.parquet');
```

```python
# Pandas
import pandas as pd
frame = pd.read_parquet("201905_202007_qc_publish.parquet")
```

```r
# R (arrow)
library(arrow)
trajectory_extensions <- read_parquet("201905_202007_qc_publish.parquet")
```
