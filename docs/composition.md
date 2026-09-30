# Parquet-first primary composition

`limosat compose-parquet` reads the immutable pair-worker NPZ and JSON products
and writes trajectory points and directly measured trajectory extensions without
importing either product into SQLite. The command does not modify the native run
database or the pair-product directory.

The local field defaults are 4 km grid spacing and a 6.4 km maximum triangle
edge; pair planning remains on a fixed 4 km grid. Recomposition of an existing
archive must use that archive's frozen configuration, not current defaults.

```bash
limosat compose-parquet CONFIG OUTPUT_DIRECTORY
```

The input pair-product count must exactly match the deterministic primary plan.
Every marker, NPZ file, array-content checksum, field checksum, configuration,
model, and pair identity is verified. The producer implementation hash is
retained as provenance but does not have to equal the downstream consumer code
hash. This allows composition and publication code to evolve without
invalidating completed GPU measurements.

The command writes these files atomically:

- `trajectory-points-v1.parquet`, including created, observed and dormant rows;
- `trajectory-extensions-qc-v1.parquet`, containing only directly measured
  source-to-target extensions; and
- `composition-manifest-v1.json`, written last as the completion marker.

## Initial QC policy

This branch does not import the ORB production QC protocol wholesale. Its first
ELoFTR-specific policy is intentionally small:

- reject extensions above 60 km/day;
- mark extensions above the configured matcher speed for review;
- calculate leave-one-out diagnostics using only extensions from the exact same
  source/target image pair; and
- mark strong local anomalies for review without rejecting them.

The neighbour defaults are a 20 km search radius, 8 minimum and 24 maximum
neighbours. The ELoFTR fields are already built from locally consistent matches
on a 4 km grid, so trajectory neighbours are correlated evidence. Automatic
local rejection requires a separate validation experiment. Sparse locations
with fewer than eight neighbours receive no local decision and remain subject
only to the speed gate.

`trajectory-extensions-qc-v1.parquet` records accepted, review and rejected rows.
The `accepted` column excludes only hard-speed failures. PySIDA and public
exporters can select `accepted`, while validation can examine the review rows.

## Targeted recovery

Recovery uses the completed primary composition as an immutable reference. It
does not rewrite primary trajectory identities or fields.

Default target preparation selects two kinds of measured loss for each
eligible recovery pair. A dormant row is an identity that a primary pair from
its source image tried and failed to continue. An unscheduled loss is an
identity last measured on the pair's source image that has no row on the target
image because no primary pair from its last image reached that target. Each
unscheduled loss is nominated once, for the earliest eligible pair whose target
footprint contains its frozen source position. Identities already targeted as
dormant are not nominated again. The SQLite and Parquet recovery paths apply
the same rule; nothing is predicted, and the recovery field must still measure
the motion. In the March 2020 capped-plan archive this class was 26% of
uncensored endings, and 78% of a held-out sample gained a QC-accepted link.
For adaptive rounds, pass a frozen JSON request plan with
`pair_request_schema_version: 1`, `run_id`,
`config_sha256`, `primary_composition_manifest_sha256`, and a `pairs` list of
`{"pair_id": ..., "trajectory_ids": [...]}`. The requested source coordinates
are read from the immutable primary Parquet, not trusted from the plan. An
explicit request may target either a dormant row or no row; it cannot replace
an existing measurement.

```bash
limosat prepare-recovery-parquet \
  CONFIG PRIMARY_COMPOSITION_DIRECTORY RECOVERY_TARGET_DIRECTORY \
  --requests REQUEST_PLAN.json
```

Each nonempty pair target is stored atomically and bound to the run
configuration, image-pair identity, primary-composition checksum, requested
trajectory identities and exact coordinate-array checksum. Composition applies
each recovery field only to the identities bound to that target set; older
target markers without identities must be regenerated. Recovery workers consume
those targets without
reading or writing SQLite:

```bash
limosat pairs CONFIG --kind recovery \
  --recovery-targets RECOVERY_TARGET_DIRECTORY \
  --batch-index 0 --batch-count 4
```

After every targeted recovery pair product is complete, compose delta products:

```bash
limosat compose-recovery-parquet \
  CONFIG PRIMARY_COMPOSITION_DIRECTORY RECOVERY_TARGET_DIRECTORY OUTPUT_DIRECTORY
```

The recovery composer writes `trajectory-augmentations-v1.parquet`,
`trajectory-augmentation-extensions-qc-v1.parquet`, and
`recovery-composition-manifest-v1.json`. Consumers overlay augmentation points
on primary points by `(trajectory_id, image_id)` and append augmentation
extensions to primary extensions. Recovery fields sample only frozen primary
coordinates, inserting a row if the frozen primary has none. Later primary
fields may continue a recovered coordinate through dormant or absent rows,
but a recovery field never consumes an earlier augmentation. Recovery products
remain trajectory-only and are not deformation inputs.

`limosat.pair_queue.choose_pair_batch` provides the provisional adaptive policy:
each unresolved trajectory nominates its shortest unprocessed pair, pairs are
ranked by how many trajectories nominated them, and every unresolved position
eligible for a selected pair is submitted for field reuse. Reassess field
support after each bounded round before selecting the next, then compose the
saved products to measure actual trajectory-duration gains. This is not yet a
validated production policy.
The existing March 2020 frozen primary archive and newly targeted experiment
products have different run configurations, so the branch recovery CLI cannot
compose them together directly. The March experiment uses a separate bounded,
resumable driver and a retrospective comparison; the same-configuration CLI
above is for runs planned together from the outset.
