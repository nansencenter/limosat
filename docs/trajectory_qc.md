# Trajectory quality control

LiMOSAT trajectory quality control (QC) checks drift vectors after tracking has
finished and writes a separate QC database. It applies to both directly matched
and interpolated vectors.

A drift vector is the displacement between consecutive keypoints in a
trajectory. When QC rejects a vector, it splits the trajectory at the target
keypoint and keeps both parts. No keypoints are deleted, and the original
database is not modified.

## Run QC

Finish tracking and close the database before running QC. Keep the original
SQLite database and template Zarr store.

```bash
python -m limosat.qc \
  --input /path/to/run.sqlite \
  --table run_table \
  --input-sha256 "$RAW_SHA256" \
  --configured-speed-m-per-day 35000 \
  --output-dir /path/to/qc_work \
  --cleaned-output /path/to/run_qc.sqlite
```

Before running the command:

- Set `RAW_SHA256` to the original database's verified lowercase SHA-256 checksum.
- Replace `35000` with the effective `max_speed_m_per_day` used during tracking,
  in metres per day. Do not infer this value from the drift vectors.
- Use a new work directory and a new output database path.
- Ensure coordinates are in EPSG:3413, in metres, and acquisition times use a
  consistent UTC format that sorts chronologically, as written by LiMOSAT.

The table name is optional if the input contains exactly one LiMOSAT trajectory
table. If `--cleaned-output` is omitted, the output is named
`<input_name>_qc.sqlite` inside the work directory.

Do not run QC while LiMOSAT is writing to the input database. QC verifies the
checksum before and after reading the source and refuses a nonempty SQLite
write-ahead log (WAL). Use SQLite's checkpoint or backup facilities to obtain a
closed, consistent database; do not delete a WAL file manually.

## What QC checks

QC compares each drift vector with neighbouring vectors from the same source
and target images. A robust local fit estimates the displacement and its
variability without using the vector being checked.

The version 2 rules reject vectors that exceed the absolute speed limit of
60,000 m/day, or have a sufficiently large, locally supported displacement
error. Additional checks use the tracking speed limit and reversals in the
orientation of neighbouring keypoints. Vectors marked for review are recorded
but remain connected.

Correlation, sea-ice concentration and external drift or buoy data are not used
to reject vectors. Low sea-ice concentration alone therefore does not cause
rejection. Where there are too few neighbours, only the absolute speed limit
is applied. QC does not guarantee that all incorrect matches are removed;
false-rejection rates have not been quantified across all seasons and sea-ice
concentrations.

The versioned rules and thresholds are supplied in
[the QC protocol](../limosat/qc/protocols/trajectory_link_qc_v2.json) and are
checked by the command before processing.

Version 2 preserves the speed and residual thresholds. It limits the absolute
sum of affine prediction weights to 2; larger weights or a rank-deficient fit
use a robust median translation instead. This prevents narrow or one-sided
neighbour geometry from amplifying small displacement differences. The archive
prescreen now skips a fit only when a conservative residual bound rules out
every local reject and review decision. These changes require new QC runs;
the original version 1 protocol remains available for historical provenance.
Independent stratified visual validation is still required before operational
promotion; passing regression tests does not establish a false-rejection rate.

## Outputs

Use the `<source_table>__qc` table in the output database for trajectory analysis.

- Rejected vectors split trajectories. New trajectory IDs are assigned above
  the maximum original ID.
- Each resulting trajectory has one `is_last` keypoint.
- `converged_to` references are updated to the applicable trajectory segment
  at the observation time and image.
- Duplicate observations with the same trajectory, image and time are excluded
  from scoring and preserved as separate single-keypoint trajectories.

The work directory contains processing checkpoints, `qc_analysis.sqlite` with
QC decisions and summaries, and `materialization_manifest.json` with the final
validation result and output checksum. Detailed diagnostics cover all rejected
and review vectors, plus a deterministic 1-in-1,000 sample of other vectors;
they are not stored for every accepted vector. Allow disk space for the work
databases and QC output in addition to the original database.

Treat the output as ready only when the command exits successfully and
`materialization_manifest.json` reports `status: complete`. The command checks
keypoint counts, trajectory end-marker counts and positions, duplicate images
within trajectories, and SQLite integrity before completing the output.

Retain the original database, QC database and QC records together. Automated
processing should invoke this command after tracking and require successful
validation before publishing the QC output.

## Resume or extend processing

To resume interrupted QC, repeat the same command with `--resume`. Processing
continues from the last complete image checkpoint. The input, code, protocol,
configuration and work database must be unchanged. Resume requires the existing
audit database to match the run identity and checkpoint counts. Rebuild work
directories created with earlier code; checkpoints cannot cross protocol versions.
Source end markers preceding later observations are refused during preparation.
Keep the work directory private and resume only from trusted checkpoints.

For large runs, the same command also supports separate `--stage prepare`,
`--stage scan` and `--stage materialize` steps. The default, `--stage all`,
performs all three. An incomplete scan cannot produce a completed QC output.

To extend tracking, use the original database and template Zarr store, not the
QC database. Run QC again after tracking finishes. QC changes trajectory IDs
and end markers but does not update templates; IDs may also change when an
extended run is processed again.

For near-real-time processing, use closed periods or consistent database
snapshots. Include earlier keypoints needed to evaluate vectors crossing the
start of the period; isolated daily slices omit these vectors.
