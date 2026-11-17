# Trajectory quality control

QC checks drift vectors after tracking finishes and writes a separate SQLite
database. Rejected vectors split trajectories; all keypoints are preserved and
the original database is unchanged.

There are two QC tools:

- **LiMOSAT QC v1** (`python -m limosat.qc`): the current one-pass QC used for the
  archive release `limosat_qc_v1_20261029`. See [LiMOSAT QC v1](#limosat-qc-v1).
- **The streaming finalizer** (`python -m limosat.qc.archive`): the earlier QC that
  finalizes a tracking run. It is documented from [Run the streaming finalizer](#run-the-streaming-finalizer)
  on. Before this change it was started with `python -m limosat.qc`.

## LiMOSAT QC v1

LiMOSAT QC v1 runs **once** over the regular matching output of one processing period: every link
between consecutive keypoints of a trajectory. It does not build on an earlier QC. The steps run in
order, and each only sees the links that survived the earlier ones, so the rules we are most confident
in act first. A removed link gets the code of the step that removed it and splits its trajectory
(its target keypoint starts a new segment). Kept positions are never altered.

| Step | Code | A link is removed when |
| --- | --- | --- |
| 0 | `SPEED` | its speed over the actual image gap is above 60 km/day |
| 1 | `OPEN_WATER` | sea-ice concentration is below 15% at either endpoint; links with no value are kept |
| 1 | `LAND` | its deeper endpoint is at least 1 km inland on the land raster |
| 2 | `ISOLATED_UNVERIFIED` | no other surviving link of its image pair lies within 30 km, and the gap is over 54 h or the speed over 25 km/day |
| 3 | `ROGUE_VECTOR` | it is at least 5 times the neighbours' spread from the leave-one-out local-affine prediction of its image pair, and either more than 2 km off with no agreeing neighbour, or more than 5 km off with at most 2 |
| 4 | `INTERP_UNSUPPORTED` | it is interpolated and not supported by at least 3 of its 16 nearest surviving links within 50 km (within 1 km of their median, within 2 km with target correlation of at least 0.40, or within 3 km with at least 0.50) |

Thresholds are in the [protocol](../limosat/qc/protocols/limosat_qc.json), which carries a SHA-256 that
every output manifest records. Do not edit it: change the protocol only by adding a new file and id.

Kept links carry `pair_direct_share` (the share of the image pair's input links that are direct
matches) and the flags below. Flags never split trajectories.

| `qc_flags` bit | Name | Meaning |
| --- | --- | --- |
| 0 | `FEW_NEIGHBOURS` | fewer than 2 other kept links of the pair within 30 km |
| 1 | `LOW_PAIR_DIRECT_SHARE` | `pair_direct_share` below 0.40, where kept interpolated links are more often wrong |
| 2 | `MASK_FALSE_LAND` | added afterwards by a release amendment, **not set by `limosat.qc`**: an end of the link lies in false land of its own scene (see below) |

`MASK_FALSE_LAND` was added to release `limosat_qc_v1_20261029` on 2 November 2026 after the run. The
preprocessing land mask (band 2) of Sentinel-1 scenes spanning 180 degrees had false land strips over
sea ice because the Nansat `watermask()` cropped MOD44W to the wrong longitudes. The amendment flags links
with an end in such false land more than 5 km from land in the QC land raster. It is informational:
nothing was removed, the rules and protocol checksum are unchanged, and only `qc_flags` and
`qc_flag_reasons` changed. The amendment is recorded in the Parquet metadata (`limosat.qc.amendments`),
so readers must take the flags a file declares from `limosat.qc.flag_bits` plus its amendments. A QC run
of this code does not produce bit 2; a rerun of the archive needs the flag added again, or the
preprocessing fixed.

### Run LiMOSAT QC v1

```bash
python -m limosat.qc \
  --compact-database 202305_202407/work/compact_points.sqlite \
  --raw-database raw/202305_202407.sqlite \
  --period 202305_202407 \
  --output-dir qc/periods/202305_202407 \
  --land-raster land_agree_3413_1km.npz \
  --sic-root uni_bremen_amsr2_asi_n3125
```

- `--compact-database` is the `compact_points` store written by the earlier streaming QC's `prepare`
  stage. `--raw-database` supplies image orbit numbers and the tracker fields `stopped` and `converged_to`.
- `--sic-root` is the University of Bremen AMSR2 ASI 3.125 km daily GeoTIFF tree. `--sic-fallback-asi6250`
  and `--sic-fallback-nsidc` fill days it lacks, in that order; the manifest lists the product used per day.
- Build the land raster with `python -m limosat.qc.land` (`build` one raster per land-polygon source, then
  `combine`; QC v1 uses a cell as land only where Natural Earth 10 m and OpenStreetMap agree).
- Interrupted runs resume from `--work-dir` (default `<output-dir>/work`).

Outputs, per period: `<period>_qc.parquet` (kept links), `<period>_qc_removed.parquet` (removed links with
the removing step and its diagnostics), `<period>_qc.manifest.json` (protocol id and checksum, counts by
step and flag, checks, output checksums) and, when `--raw-database` is given, `<period>_qc.sqlite` (every
raw point with trajectory ids rewritten to the QC trajectories). Use an output only when the manifest
reports `status: complete`: kept plus removed links equal the input, link ids are unique and the
trajectories have no continuity gaps (`continuity_violations` is 0).

## Run the streaming finalizer

Close the tracking database before running QC. Input coordinates must use
EPSG:3413 in metres, with consistently formatted UTC acquisition times.

```bash
python -m limosat.qc.archive \
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

## Checks and limitations (streaming finalizer)

QC checks actual-gap speed and displacement consistency with nearby vectors
from the same source and target images. Large local discrepancies can be
rejected using additional speed or geometric checks. Review flags retain the
vector for inspection. Exact rules and thresholds are defined in the
[QC protocol](../limosat/qc/protocols/trajectory_link_qc_v2.json).

Both directly matched and interpolated vectors are checked. Where neighbour
support is insufficient, only the absolute speed limit of 60,000 m/day applies.
QC does not identify every incorrect match or distinguish ice from open water;
retained vectors are not guaranteed to be valid.

## Outputs and resuming (streaming finalizer)

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

## QC v3.1 on a v2 release

> **Superseded draft.** LiMOSAT QC v1 above replaces QC v3.1. This section and `limosat.qc.v3` are kept
> to reproduce earlier releases; the `qc3_flags` bits below (`PAIR_INCOHERENT`, `FEW_NEIGHBOURS`) are not
> the QC v1 flags.

QC v3.1 keeps every v2 decision and is applied to the v2 scientific edge Parquet of one period. It
excludes links, splitting trajectories at every excluded link, for these reasons:

| Reason | Condition |
| --- | --- |
| `R1` | Isolated outlier: > 2 km from the local prediction, ≥ 5× the neighbours' own spread, no corroborating neighbour (direct or interpolated) |
| `LAND` | An endpoint inside land (any land cell of the 1 km raster) |
| `OPEN_WATER` | Sea-ice concentration below 15% at both endpoints; links without a value are kept |
| `INTERP_UNSUPPORTED` | An endpoint interpolated, and fewer than 3 of the 16 nearest other links within 50 km, or more than 1 km from their median displacement |

Rules and thresholds are in the [v3.1 protocol](../limosat/qc/protocols/trajectory_link_qc_v3_1.json).

```bash
python -m limosat.qc.v3 \
  --input 202405_202507.parquet \
  --period 202405_202507 \
  --land-raster land_3413_1km.npz \
  --sic-root uni_bremen_amsr2_asi_n3125 \
  --output-dir qc3_202405_202507
```

- `--land-raster` is an `.npz` with `inland_km` (distance inside land, 0 at sea) on a regular
  EPSG:3413 grid, plus `x0` (west edge), `y1` (north edge) and `res` in metres.
- `--sic-root` points to University of Bremen AMSR2 ASI 3.125 km daily GeoTIFFs
  (`<YYYY>/<mon>/asi-AMSR2-n3125-<YYYYMMDD>-v5.4.tif`); `--sic-product nsidc` uses NOAA/NSIDC CDR v6
  25 km files instead. Without `--sic-root` no link is excluded as open water.
- `--sic-fallback-asi6250` and `--sic-fallback-nsidc` fill days missing from the main product, in that
  order (AMSR2 ASI 6.25 km, then NOAA/NSIDC CDR v6). The manifest lists which product served each day.
- Interrupted runs resume from the per-month files in `<output-dir>/work`.

Outputs are `<period>_qc3.parquet` (kept links with `qc3_trajectory_id`, `qc3_segment`, `qc3_flags`,
`qc3_flag_reasons`, `qc3_neighbours_30km`, `qc3_sic_start` and `qc3_sic_end`),
`<period>_qc3_exclusions.parquet` (excluded links with reason codes, reason text and diagnostics) and
`<period>_qc3.manifest.json`. Use the output only when the manifest reports `status: complete`: kept
plus excluded links equal the input, `edge_uid` is unique, and v3 trajectories have no continuity
gaps. v2 columns such as `is_last` and `converged_to` are carried unchanged and describe the v2
trajectory.

| `qc3_flags` bit | Name | Meaning |
| --- | --- | --- |
| 0 | `PAIR_INCOHERENT` | Image pair with ≥ 20 links, at least half > 2 km from their neighbour median |
| 1 | `FEW_NEIGHBOURS` | Fewer than 2 other links of the pair within 30 km |
