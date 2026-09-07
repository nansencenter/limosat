# LiMOSAT trajectory QC finalization

## Status and ownership

Trajectory-vector QC v1 is a frozen, non-destructive post-run finalization stage.
The scientific implementation, protocol, tests, and SQLite output contract live
in this repository under `limosat/qc/`. A deployment repository should only
invoke the CLI, supply run-specific values and paths, retain the raw product,
and gate publication on successful validation. It must not duplicate the
threshold logic.

The implementation preserves the tested v1 decisions. Independent visual
labelling in low-concentration and seasonal strata remains outstanding; the
historical experiment reports document that evidence boundary. Packaging alone
does not establish the false-rejection rate in those strata.

The frozen protocol is
`limosat/qc/protocols/trajectory_link_qc_v1.json`. The implementation checks its
fixed parameters against that file before processing. The source run's effective
`max_speed_m_per_day` remains a required run-specific input; it must not be
guessed from the data or replaced by a deployment default. Deployment must
resolve the effective value from the run configuration and its LiMOSAT version,
then record that value in its finalization manifest.

## Terminology

QC follows the production terminology in `Keypoints` and `ImageProcessor`:

- A **keypoint** is a stored observation in a trajectory.
- A **match** is a correspondence obtained by matching images or templates.
- A **drift vector** is the displacement between consecutive keypoints. QC
  evaluates both directly matched and interpolated vectors.
- A **trajectory** is the time-ordered sequence of keypoints. Rejecting a vector
  splits this sequence without deleting either endpoint.

The scoring entry point is `score_vectors`; `score_edges` remains a compatibility
wrapper for existing callers. Existing SQLite names (`qc_flagged_edges`,
`qc_link_sample`, and `links` counts) and the protocol ID/path are unchanged to
preserve report and provenance compatibility. They refer to drift vectors.
The geometric term **edge** remains appropriate for a Delaunay triangle side;
`topology_max_edge_m` is a source-triangle size limit, not a drift-length limit.

## Required lifecycle

For every finite production run:

1. Finish tracking and close the SQLite writer.
2. Preserve the raw SQLite and template Zarr store unchanged.
3. Record or verify the raw SQLite SHA-256.
4. Run trajectory QC into a new work directory and a new output SQLite.
5. Require the materialization checks to pass before publishing downstream.
6. Publish the raw database, QC database, audit sidecar, manifests, and protocol
   ID together. Downstream trajectory analysis uses the `<source_table>__qc`
   table.

The QC database is an analysis product and is never used to resume tracking:
trajectory IDs and `is_last` markers change, but template Zarr state is not
rewritten. Resume or extend the raw run, then finalize the extended result
again.

For continuous NRT processing, apply the same command only to closed periods
or immutable database snapshots. Run it again over the complete finite product
when that production period closes. Do not scan a database while LiMOSAT is
writing it.

## Official command

The streaming finalizer is the production entry point for all database sizes:

```bash
python -m limosat.qc \
  --input /path/to/raw_run.sqlite \
  --table raw_run_table \
  --input-sha256 "$RAW_SHA256" \
  --configured-speed-m-per-day 35000 \
  --output-dir /path/to/qc_work \
  --cleaned-output /path/to/raw_run_qc.sqlite
```

The default stage is `all`: it prepares a descriptor-free compact database,
scans vectors in complete image-time groups with resumable checkpoints, and
materializes the cleaned product. Large deployments may call `--stage prepare`,
`--stage scan`, and `--stage materialize` separately. A partial scan cannot be
materialized.

`RAW_SHA256` must contain the source file's verified lowercase SHA-256. The CLI
checks the file before and after preparation/materialization and rejects a
nonempty SQLite WAL. Checkpoint or snapshot SQLite using its backup facilities
before finalization. Work directories and checkpoint pickle files must be
trusted and private to the run.

To resume an interrupted run, repeat the command with `--resume`; preparation
is skipped and the scan resumes at its last complete image checkpoint. The
source, protocol, code, configuration, and compact manifest must match.
Completed stage-only commands are not publication success: deployment must
require the final `materialization_manifest.json` with `status: complete`.

V1 expects EPSG:3413 coordinates in metres and uniform UTC acquisition-time
strings that sort chronologically, as written by LiMOSAT. Mixed timestamp
formats or other projections must be normalized in a separate input copy.
Closed NRT snapshots need previous observations for vectors crossing the snapshot
boundary; isolated daily slices omit those vectors. IDs are deterministic within
one immutable input, and may change when an extended run is finalized again.

The audit records every rejected/review vector, duplicate, pair/image summary,
and a deterministic 1/1000 background sample. Detailed local diagnostics for
every accepted vector are not stored. The archive work database and publication
database are separate; allow room for both plus the preserved raw input.

The production package has one entry point: `python -m limosat.qc`.
`core.py` contains the scoring rules, `archive.py` handles streaming and
trajectory finalization, and `audit.py` stores diagnostics. Experimental SIC,
buoy, plotting, and research-report workflows are not production dependencies.
The former experimental QC launchers have been removed; historical reports
remain as evidence, not executable deployment instructions.

## Refactor verification (2026-09-07)

The lean implementation was checked against all six saved `edge_audit.csv`
files under `results/limosat_qc_20260903/`: 253,024 vectors had identical reject,
review, and decision-reason values. These checks used the 1,000 m prescreen,
50,000 m/day for the three archive windows, 35,000 m/day for Radarsat-2 2024,
and 30,000 m/day for the 2020 and NRT samples, matching the historical audits.
This is regression evidence, not new independent scientific validation or a
rerun of the full Kingston archive.

The focused tests cover scoring, sparse support, motion boundaries, duplicate
points, repeated trajectory breaks, convergence remapping, point preservation,
resume equivalence, source integrity, and publication gates. Run them with:

```bash
python -m pytest -q tests/unit/test_qc_limosat_trajectories.py \
  tests/unit/test_qc_limosat_archive_streaming.py
```

## Publication gate

A deployment job succeeds only when the command exits successfully and all of
the following are true:

- scan metadata status is `complete`;
- source and cleaned row counts agree;
- every output trajectory has exactly one `is_last` row;
- no output trajectory repeats an image;
- SQLite `quick_check` returns `ok`;
- the output SHA-256 is recorded in `materialization_manifest.json`.

The raw database must remain available even after the QC product is published.
A rejected vector removes no point: its target begins a deterministic new segment
above the maximum original trajectory ID. Review vectors remain connected and are
recorded in the audit.

The archive finalizer maps `converged_to` to the latest applicable split segment
at the convergence row's time/image. Repeated trajectory/image/time rows are
excluded from scoring and preserved as singleton trajectories. The older
in-memory analysis helper has a different nearest-time convergence fallback;
deployment should consistently use the archive CLI above.

## Protocol changes

Do not alter v1 thresholds in place. Any scientific rule or threshold change
requires a new protocol ID and protocol JSON, regression tests, a comparison
against v1, and a new deployment selection. SIC, correlation, buoy data, and
OSI SAF drift remain diagnostics rather than v1 rejection gates.
