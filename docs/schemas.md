# Product schemas

## SQLite schema version 1

`runs` stores the resolved configuration JSON and SHA256, a checksum of all
LiMOSAT Python source, the checkpoint SHA256, status, UTC runtime boundaries,
total runtime, and manifest identity. Resume rejects changed code, model, or
configuration under an existing `run_id`.

`images` stores stable image/component identity, absolute path, UTC acquisition
time, byte size, and SHA256. Resume rejects an image whose identity, time, path,
or content hash changed.

`pairs` stores adjacent or recovery edge identity, source/target image and UTC
times, elapsed seconds, targeted flag, state, field checksum, product counts,
stage runtimes, matcher calls, and error text. Normal execution never changes a
row whose status is `complete`.

`field_nodes` stores one EPSG:3413 source location per grid node:

| Column | Type/unit |
| --- | --- |
| `x_m`, `y_m` | float64 semantics, metres |
| `available` | boolean integer |
| `dx_m`, `dy_m` | metres; SQL `NULL` when unavailable |
| `selected_matches`, `candidate_matches` | counts |
| `support_radius_m`, `maximum_residual_m` | metres or `NULL` |

`trajectories` owns stable IDs derived from seed image and millimetre-rounded
seed coordinate. `trajectory_points` uses states `created`, `observed`,
`dormant`, and `reappeared`. Dormant coordinates are SQL `NULL`.
`position_basis` distinguishes `seed_grid`, `field_advected_adjacent`,
`field_advected_skip`, and `missing`.

`deformation_cells` stores adjacent-pair Delaunay triangle centroid and area,
plus divergence, shear, total deformation, and vorticity in inverse seconds.

## Manifest schema version 1

`run-manifest-v1.json` records:

- resolved config and config SHA256;
- EfficientLoFTR checkpoint SHA256;
- LiMOSAT source checksum, code revision and dirty state;
- EfficientLoFTR repository revision/dirty state and checkpoint SHA256;
- exact command and UTC runtime boundaries;
- EPSG, dtype, distance/time/rate units;
- component chronology;
- image size and content checksums;
- every pair state, field checksum, counts, matcher calls, and stage runtimes;
- trajectory, trajectory-point, and deformation-cell counts; and
- the explicit policy that sparse recovery fields are not deformation products.

Product schema versions are listed independently so additive trajectory or
deformation revisions do not silently change the pair-field contract.
