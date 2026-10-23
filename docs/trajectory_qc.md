# Trajectory quality control

QC checks drift vectors after tracking finishes and writes a separate SQLite
database. Rejected vectors split trajectories; all keypoints are preserved and
the original database is unchanged.

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
[QC protocol](../limosat/qc/protocols/trajectory_link_qc_v2.json).

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

## QC v3 on a v2 release

QC v3 keeps every v2 decision and is applied to the v2 scientific edge Parquet of one period. It
excludes isolated direct-match outliers (`R1`) and links with an endpoint at least 2 km inside land
(`LAND`), splitting trajectories at every excluded link, and adds quality flags that never split
trajectories. Rules and thresholds are in the
[v3 protocol](../limosat/qc/protocols/trajectory_link_qc_v3.json).

```bash
python -m limosat.qc.v3 \
  --input 202405_202507.parquet \
  --period 202405_202507 \
  --land-raster land_3413_1km.npz \
  --sic-root noaa_nsidc_cdr_v6 \
  --output-dir qc3_202405_202507
```

- `--land-raster` is an `.npz` with `inland_km` (distance inside land, 0 at sea) on a regular
  EPSG:3413 grid, plus `x0` (west edge), `y1` (north edge) and `res` in metres.
- `--sic-root` is optional and enables the open-water flag from NOAA/NSIDC CDR v6 daily files.
- Interrupted runs resume from the per-month files in `<output-dir>/work`.

Outputs are `<period>_qc3.parquet` (kept links with `qc3_trajectory_id`, `qc3_segment`, `qc3_flags`
and `qc3_flag_reasons`), `<period>_qc3_exclusions.parquet` (excluded links with reason codes, reason
text and diagnostics) and `<period>_qc3.manifest.json`. Use the output only when the manifest reports
`status: complete`: kept plus excluded links equal the input, `edge_uid` is unique, and v3
trajectories have no continuity gaps. v2 columns such as `is_last` and `converged_to` are carried
unchanged and describe the v2 trajectory.

| `qc3_flags` bit | Name | Meaning |
| --- | --- | --- |
| 0 | `ISOLATED_INTERP` | R1 conditions met, but an endpoint is interpolated |
| 1 | `INTERP_RESIDUAL` | An endpoint is interpolated and the link is > 2 km from its local prediction |
| 2 | `PAIR_INCOHERENT` | Image pair with ≥ 20 links, at least half > 2 km from their neighbour median |
| 3 | `OPEN_WATER` | Sea-ice concentration < 15% at both endpoints |
| 4 | `COASTAL` | An endpoint less than 2 km inside land |
