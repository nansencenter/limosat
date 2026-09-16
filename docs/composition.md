# Parquet-first primary composition

`limosat compose-parquet` reads the immutable pair-worker NPZ and JSON products
and writes trajectory points and directly measured trajectory extensions without
importing either product into SQLite. The command does not modify the native run
database or the pair-product directory.

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

This initial command composes primary trajectories. Recovery augmentation remains
on the native composition path until the frozen-primary augmentation semantics
are added to the Parquet implementation.
