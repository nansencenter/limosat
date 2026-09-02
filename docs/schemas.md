# Product schemas

## SQLite schema version 3

`runs` stores the resolved configuration JSON and SHA256, a checksum of all
LiMOSAT Python source, the checkpoint SHA256, status, UTC runtime boundaries,
total runtime, and manifest identity. Resume rejects changed code, model, or
configuration under an existing `run_id`. Earlier databases are rejected;
start a global run with a new database path and run ID. There is no migration
layer.

`images` stores stable image/component identity, platform, absolute orbit,
absolute path, UTC acquisition time, byte size, and SHA256. Resume rejects an
image whose identity, acquisition metadata, time, path, or content hash changed.

`candidate_pairs` stores the immutable deterministic plan: ordinal,
candidate/primary selection, source and target compute labels, UTC times,
elapsed seconds, footprint-overlap fraction, direct overlap area, and the
actual global chronological images between source and target. That last count
is provenance, not a recovery gate. `planning_counts` records deterministic
same-pass, candidate-time, recovery-elapsed-time, overlap, and
diagnostic-allowlist decisions.

`pairs` stores primary or recovery image-pair identity, source/target image and UTC
times, elapsed seconds, targeted flag, state, field checksum, product counts,
stage runtimes, matcher calls, tile/phase diagnostics, ancillary SIC checksums,
and error text. Normal execution never changes a
row whose status is `complete`.

`field_nodes` stores one EPSG:3413 source location per grid node:

| Column | Type/unit |
| --- | --- |
| `x_m`, `y_m` | float64 semantics, metres |
| `available` | boolean integer |
| `dx_m`, `dy_m` | metres; SQL `NULL` when unavailable |
| `selected_matches`, `candidate_matches` | counts |
| `support_radius_m`, `maximum_residual_m` | metres or `NULL` |

`trajectories` owns IDs that are global within a run and are derived from seed
image and millimetre-rounded seed coordinate. Neither global trajectory table
contains `component_id`. `trajectory_points` uses states `created`,
`observed`, `dormant`, and `reappeared`. Dormant coordinates are SQL
`NULL`. `position_basis` distinguishes `seed_grid`,
`primary_pair_field`, `recovery_pair_field`, and `missing`.

`trajectory_convergence_events` is a non-destructive diagnostic. It stores the
measured candidate and deterministically preferred trajectory, separation and
audit radius in metres, and both observation counts. It never changes global
trajectory identity or coordinates.

`deformation_cells` stores primary-pair Delaunay triangle centroid and area,
plus divergence, shear, total deformation, and vorticity in inverse seconds.

## Manifest schema version 3

`run-manifest-v3.json` records:

- resolved config and config SHA256;
- EfficientLoFTR checkpoint SHA256;
- LiMOSAT source checksum, code revision and dirty state;
- EfficientLoFTR repository revision/dirty state and checkpoint SHA256;
- exact command and UTC runtime boundaries;
- EPSG, dtype, distance/time/rate units;
- global chronology and component compute-planning labels;
- image size and content checksums;
- every pair state, field checksum, counts, matcher calls, and stage runtimes;
- candidate planning exclusions, orbit metadata, open-water ancillary
  checksums, selected phase/same-centre routing hypothesis, actual matcher
  calls, and tile-gate counts;
- trajectory, trajectory-point, and deformation-cell counts; and
- the explicit policy that sparse recovery fields are not deformation products.

Product schema versions are listed independently so additive trajectory or
deformation revisions do not silently change the pair-field contract.

## Field-replay provenance version 1

`field-replay-provenance-v1.json` is an analysis provenance record, not the
native run manifest. It identifies the immutable production state/plan and
ordered completed-field set, reports whether each field checksum was verified,
records SQLite schema 3 and trajectory product schema 3, and compares the new
global catalogue with the prior component-sharded summary. A separate
`render-report-v1.json` records deterministic trajectory selection, frame
timing, source checksums, and checksums for figures, MP4, and GIF outputs.
