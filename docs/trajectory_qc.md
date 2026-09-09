# Trajectory quality control

QC checks drift vectors after tracking finishes and writes a separate SQLite
database. Rejected vectors split trajectories; the original database is unchanged.

## Run QC

Close the tracking database before running QC. Input coordinates must use
EPSG:3413 in metres, with consistently formatted UTC acquisition times.

```bash
python -m limosat.qc \
  --input run.sqlite \
  --table run_table \
  --input-sha256 "$RAW_SHA256" \
  --configured-speed-m-per-day 35000 \
  --output-dir qc_work \
  --cleaned-output run_qc.sqlite
```

- Set `RAW_SHA256` to the closed input file's lowercase SHA-256 checksum.
- Replace `35000` with the effective tracking speed limit, in metres per day.
- Use a new work directory and output database path.
- Omit `--table` if the input contains only one trajectory table.

QC refuses an uncheckpointed SQLite write-ahead log (WAL). Close or checkpoint
the source database before retrying; do not delete its WAL file manually.

## Checks and limitations

QC checks actual-gap speed and displacement consistency with nearby vectors
from the same source and target images. Large local discrepancies can be
rejected using additional speed or geometric checks. Review flags retain the
vector for inspection. Exact rules and thresholds are defined in the
[QC v1 protocol](../limosat/qc/protocols/trajectory_link_qc_v1.json).

Both directly matched and interpolated vectors are checked. Where neighbour
support is insufficient, only the absolute speed limit of 60,000 m/day applies.
QC does not identify every incorrect match or distinguish ice from open water;
retained vectors are not guaranteed to be valid.

## Outputs and resuming

Use the `<source_table>__qc` table in the output database for analysis. Trajectory
IDs, end markers and convergence references reflect the resulting segments.
Finalization discards any leftover one-keypoint segments and audits their removal;
accepted drift vectors are unaffected. Row accounting is
`source_rows = cleaned_rows + removed_singleton_rows`.

The work directory contains QC decisions in `qc_analysis.sqlite` and an output
validation summary in `materialization_manifest.json`. Use the output only after
the command succeeds and the manifest reports `status: complete`.

To resume interrupted QC, repeat the command with `--resume`, using the same
input, configuration, code and work directory. Keep the complete work directory;
checkpoints cannot be reused after changing the code or protocol.

To extend tracking, use the original database and template Zarr store, then run
QC again. The QC database is an analysis output and cannot resume tracking.
