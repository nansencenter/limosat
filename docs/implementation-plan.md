# Production hardening and experiment plan

This list records changes and decision gates needed before the next native
pan-Arctic run. It incorporates applicable safeguards from production LiMOSAT
without importing ORB tracking, interpolation, buoy truth, or implementation
code from that repository.

## Scientific status of field-consensus tolerances

The current 1,000 m displacement-agreement distance and the experimental
750/1,000/1,250 m adaptive distances are **not calibrated sea-ice-physics
thresholds**. They are engineering hypotheses constrained by existing
EfficientLoFTR fields and topology checks.

The diagnostic time-scaled formula also uses a provisional strain coefficient
of `0.7e-6 s^-1`; neither that coefficient nor its 500--1,500 m clipping bounds
have been calibrated for EfficientLoFTR. The name describes the dimensional
scaling being tested, not an evidence-based acceptance threshold.

Published evidence establishes the following constraints, but does not supply
one universal agreement distance:

- Sea-ice deformation is spatially localized and depends on sampling scale
  ([Stern and Lindsay, 2009](https://doi.org/10.1029/2009JC005380)).
- Small-scale deformation depends strongly on both spatial separation and time
  interval ([Oikkonen et al., 2017](https://doi.org/10.1002/2016JC012387)).
- Recent observations caution against extrapolating one scale law through the
  kilometre-to-floe transition
  ([Hutchings et al., 2024](https://doi.org/10.1029/2024GL108582)).
- A published Sentinel-1 feature-tracking validation reported 563 m RMSE for a
  different matcher and experiment. This is uncertainty context, not a
  calibration of EfficientLoFTR vector agreement
  ([Muckenhuber et al., 2016](https://doi.org/10.5194/tc-10-913-2016)).
- Position uncertainty propagates into deformation differently with cell size
  and elapsed time
  ([Bouchat and Tremblay, 2020](https://doi.org/10.1029/2019JC015944)).

Therefore a production tolerance must be selected from reproducible
truth-free tests at the actual 4 km output scale and catalogue time intervals.
The tests must report coverage, fold rejection, displacement roughness,
deformation distributions on identical measured cells, and consistency across
overlapping completed image pairs. Missing measurements remain missing in all
variants; no field value is smoothed or interpolated.

The initially proposed adaptive policy for the completed-field screen was:

1. Local tier: nearest 8 matches, require 6 agreeing within 3 km, using 750 m
   displacement agreement.
2. Standard tier: nearest 12 matches, require 8 agreeing within 6 km, using
   1,000 m displacement agreement.
3. Sparse tier: nearest 16 matches, require 10 agreeing within 8 km, using
   1,250 m displacement agreement.

The agreeing fraction decreases with search area, from 75% to 67% to 62.5%,
while the absolute evidence requirement increases. The first supported tier is
used, so well-supported field grid points do not inherit the larger search
area.

The all-604-primary-pair CPU screen recovered 65,565 net field grid points
(+1.781 percentage points of coverage), but increased the pooled common-cell
99th-percentile total deformation by 4.75%. Its 99th-percentile adjacent-vector
gradient increased by 11.2%, and fold rejection increased from 0.237% to
0.329%. Coverage was higher on 575 image pairs and unchanged on 29, but the
deformation change is large enough that the adaptive policy must not become a
production default yet. The next comparison should preserve every baseline
estimate and invoke a smaller or wider fallback only where the baseline is
missing. This separates coverage extension from changing already supported
motion.

The screen used the immutable completed fields as the baseline. A batched
recalculation retained for diagnostics disagreed on availability at 1,041 of
3,681,369 grid points (0.028%, 63 net points). This numerical difference is
another reason to validate any selected policy through the canonical field
builder before a production rebuild.

## Implementation list

### P0 — required before the next native run

- [x] **Acquisition-pass exclusion.** Extend catalogue image identity with
  platform and absolute orbit. Reject a candidate pair only when both values
  identify the same acquisition pass. Allow a later repeat on the same
  relative orbit. Require this metadata in a production catalogue and record
  exclusions in the manifest.
- [x] **Use an elapsed-time-only recovery bound.** Remove the global skipped
  image limit. `maximum_recovery_elapsed_hours` bounds a non-consecutive
  recovery pair; unrelated pan-Arctic images never affect eligibility. Keep
  the intervening-image count only as provenance. Schedule every candidate
  within the horizon that targets genuine measured loss; do not impose an
  image-count or per-target pair limit.
- [x] **Motion-reachable pair domain.** Compare direct footprint intersection
  with `source footprint ∩ buffer(target footprint, physical displacement
  limit)`. Pair fields remain measured source-to-target products. Record direct
  overlap fraction, reachable area, and estimated 4 km field-grid count.
- [x] **Physics-reachable tile validity.** Before EfficientLoFTR inference,
  skip a tile only when valid source and target support cannot be connected
  within the elapsed-time speed limit. Record each reason. This is independent
  of the matcher and should reduce empty GPU calls.
- [x] **Candidate-pair eligibility.** Require 5% overlap relative to the
  smaller footprint and at least 1,024 km2 direct overlap. Report both values;
  a percentage alone can admit an unusable sliver. Exclude same-pass
  acquisitions before ranking primary pairs.
- [x] **Deterministic spatial-coverage planner.** For every fixed 4 km planning
  cell, select the most recent eligible source and retain the union of image
  pairs selected by those cells. Keep equal-time alternatives so completed
  field quality can break ties. Do not add speculative redundant pairs or a
  per-target cap; report the resulting distribution before GPU inference.
- [x] **Robust compulsory phase estimate.** Estimate phase correlation for
  every independent production pair. If response is below 0.05, evaluate both
  phase-shifted and same-centre hypotheses and select after the unchanged field
  and fold gates. Record actual matcher work and both outcomes.
- [x] **Full consensus screen.** Run the baseline and adaptive policies on all
  604 immutable April-week primary-pair match files without EfficientLoFTR
  inference. The complete screen took 630.4 seconds on CPU and retained source
  checksums for every raw match file and completed field.
- [ ] **Field-consensus decision.** Do not promote the screened adaptive policy
  because its pooled common-cell deformation tail increased 4.75%. First test a
  baseline-preserving fallback: keep every baseline estimate unchanged, then
  evaluate local and sparse tiers only for missing field grid points. Report
  deformation on both identical cells and all newly created cells before
  changing `FieldConfig` defaults.

### P1 — required before full global trajectory acceptance

- [x] **Lean assessment products.** Optionally retain selected post-gate,
  pre-consensus matches as one compressed, checksummed SQLite record per
  completed image pair. Finalize a complete run into one compact global
  trajectory Parquet catalogue plus a checksummed statistics/provenance
  summary. Keep SQLite as the single authoritative resume/audit product and
  avoid per-pair output directories.
- [ ] **Measured-loss recovery audit.** Compose primary fields first, identify
  genuine dormant parcels, compute only bounded non-consecutive recovery
  pairs, and deterministically recompose. Recovery fields never generate
  deformation cells.
- [ ] **Seed-occupancy experiment.** Compare current target-time measured
  occupancy with bounded exclusion around a dormant parcel's last measured
  position. Do not predict an unmeasured target coordinate. Report duplicate
  parcel creation and later measured convergence before choosing a policy.
- [x] **Converging-identity audit.** Production LiMOSAT deterministically kept
  the longest, highest-quality trajectory when tracked positions converged.
  For the global catalogue, first report persistent measured convergence; do
  not merge or stop scientific parcel identities automatically without a
  documented rule.
- [x] **Deterministic recovery ordering.** Run every elapsed-time-eligible
  candidate that targets genuine measured loss. Order work by target time,
  most recent measured source, overlap, and stable image-pair identity. Never
  use an unrecorded random subset.
- [ ] **Global acceptance report.** Compare trajectory count, observation
  count, lifetime, dormant rows, reappearances, duplicate-seed diagnostics,
  field support, and primary-pair deformation with the baseline replay.

### P2 — optional efficiency work

- [x] **Conservative open-water gate.** Evaluate OSI SAF concentration only as
  a compute gate: skip only when complete source and target evidence is below
  the selected threshold; missing or stale data keeps the tile. Do not make it
  a trajectory observation or add a mandatory heavyweight dependency.
- [x] **Resource and failure audit.** Preserve deterministic pair claiming,
  immutable completion, checksums, and retryable failures while adding counts
  for validity, motion, orbit, open-water, and insufficient-support rejection.

## Experiment sequence

1. Completed: all-primary-pair CPU field-consensus screen from immutable raw
   matches; no EfficientLoFTR inference.
2. Run the baseline-preserving fallback screen, then review coverage and
   deformation evidence and select one policy.
3. Implement and test acquisition-pass exclusion, elapsed recovery bounds,
   motion-reachable domains, and tile validity.
4. Inspect the deterministic spatial-coverage planner's April-week dry-run
   counts without loading the matcher.
5. On the GPU host, run the new 5%/1,024 km2 April-week plan. Do not rerun
   overlap strata unless field diagnostics reveal a new failure mode.
6. Rebuild primary fields only if matcher- or domain-level settings changed;
   otherwise reuse immutable fields and recompose the global catalogue.
7. Run elapsed-time-bounded recovery, recompose, compare manifests, and render final
   pan-Arctic outputs.

The candidate and recovery horizons are currently both 96 hours. This remains
an explicit production choice, not an established physical constant.
