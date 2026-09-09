# Trajectory quality control

QC checks drift vectors after tracking finishes and writes a separate SQLite
database. Rejected vectors split trajectories. Segments with only one keypoint
are removed from the cleaned trajectory table; the original database is unchanged.

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
[QC protocol](../limosat/qc/protocols/trajectory_link_qc_v3.json).

Both directly matched and interpolated vectors are checked. Where neighbour
support is insufficient, only the absolute speed limit of 60,000 m/day applies.
QC does not identify every incorrect match or distinguish ice from open water;
retained vectors are not guaranteed to be valid.

## Outputs and resuming

Use the `<source_table>__qc` table in the output database for analysis. Trajectory
IDs, end markers and convergence references reflect the resulting segments.
The work directory contains QC decisions in `qc_analysis.sqlite` and an output
validation summary in `materialization_manifest.json`. Use the output only after
the command succeeds and the manifest reports `status: complete`.

To resume interrupted QC, repeat the command with `--resume`, using the same
input, configuration, code and work directory. Keep the complete work directory;
checkpoints cannot be reused after changing the code or protocol.

To extend tracking, use the original database and template Zarr store, then run
QC again. The QC database is an analysis output and cannot resume tracking.

## Singleton removal (protocol v3)

After splitting rejected vectors and assigning duplicate observations, the
finalizer removes every one-keypoint segment, including isolated input seeds,
duplicate singletons, and singletons created by QC breaks. Segments with at
least two keypoints remain, preserving their rowids and accepted drift vectors.
An output with no retained trajectories is valid and explicitly counted.

The output table `qc_removed_singletons` records removed source rowids and
post-split trajectory IDs. Original observations remain in the raw database.
Convergence references to removed singleton segments are set to NULL; their
previous targets are recorded in `qc_cleared_convergence`. Existing break and
duplicate assignment tables remain provenance records, including assignments
to segments that have subsequently been removed.

Publication checks require `source_rows = cleaned_rows + removed_singleton_rows`,
zero `remaining_singleton_trajectories`, and `retained_vectors` equal to scanned
vectors minus rejected vectors. The manifest also records
`cleared_convergence_references`. Consumers that require source and cleaned row
counts to be equal must adopt this accounting before using v3 products.

V3 changes finalization only: vector scoring thresholds and rejection decisions
are unchanged from v2. The v1/v2 protocol files and previously generated products
remain unchanged. Use new work/output paths and rerun with the v3 protocol;
materialization refuses scans produced under an older protocol. Existing QC
databases are not migrated in place.
