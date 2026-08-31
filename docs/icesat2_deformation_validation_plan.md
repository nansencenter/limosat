# ICESat-2 validation of LiMOSAT deformation

Status: alignment audit and frozen March expansion complete; ATL10 lead arm still insufficient, 20 August 2026

## Purpose and claims

ICESat-2 is not a direct two-dimensional drift-vector reference. Buoys and
manual vectors remain the endpoint-displacement reference. ICESat-2 supplies an
independent one-dimensional observation of leads and sea-ice topography. It can
therefore test whether a SAR deformation field places opening, convergence,
and shear where independently observed ice structure makes those processes
plausible.

The primary comparison is production ORB versus the selected ALIKED field on
identical laser segments. Method-specific support is reported separately as a
coverage result.

## Literature constraints carried into the design

1. Ricker et al. (2025) advect altimetry parcels to a common target time using
   the exact fractional time offset, retaining trajectories and uncertainty.
   Here the SAR displacement field is inverted at each laser segment time,
   rather than overlaying the original laser coordinates on either SAR image.
2. Ricker et al. (2023) show that sea-ice drift materially affects
   co-registration even over tens of minutes. They estimate residual motion by
   pattern correlation and remove long-wave topography with a 5 km window.
   This pilot uses a 5 km local topographic baseline and includes rigid
   along-track shift controls; it does not optimize a shift against the desired
   deformation association.
3. Ricker et al. (2023) and Duncan and Farrell (2022) use a 0.6 m sail-height
   threshold relative to surrounding level ice for ridge events. That value is
   frozen before examining the SAR associations. Standard ATL07 will smooth or
   miss some narrow ridges, so the binary ridge count is accompanied by a
   continuous roughness measure.
4. Kortum et al. (2025) hold out the near-coincident laser overpass when
   constructing a SAR-to-altimetry mapping. No ICESat-2 track used here may be
   used to tune the ORB or ALIKED field, deformation thresholds, or spatial
   smoothing.

References:

- Ricker et al. (2025), *Drift-aware sea ice thickness maps from satellite
  remote sensing*: https://tc.copernicus.org/articles/19/3785/2025/
- Ricker et al. (2023), *Linking scales of sea ice surface topography*:
  https://tc.copernicus.org/articles/17/1411/2023/
- Duncan and Farrell (2022), *Determining variability in Arctic sea ice
  pressure ridge topography with ICESat-2*:
  https://drum.lib.umd.edu/items/c6db563e-d137-45b2-aa81-9bd5faa0074a
- Kortum et al. (2025), *Sea ice freeboard extrapolation from ICESat-2 to
  Sentinel-1*: https://tc.copernicus.org/articles/19/4701/2025/
- Duncan and Farrell (2026), 10 km monthly Arctic UMD-RDA sea-ice topography
  product: https://doi.pangaea.de/10.1594/PANGAEA.990265

## Frozen pilot data

SAR pair:

- catalog image 10245: 2020-03-28 12:13:29 UTC
- catalog image 10352: 2020-03-29 11:16:05 UTC
- elapsed time: 23.043 hours
- CRS: EPSG:3413; all distances are metres
- ORB: current production Q24/I48, kilometre-scaled 15 km MAGSAC run; database
  image IDs 50 and 53; 1,244 paired material trajectories
- ALIKED: nearest-12 coherent 4 km field with fixed-point fold rejection; 8,520
  available vectors; no routine pattern matching

ICESat-2 Version 7 crossings inside the SAR interval:

| RGT fragment | UTC interval | Approximate role |
|---|---|---|
| 0030 | 2020-03-28 12:08-12:22 | near-start/pre-deformation control |
| 0039 | 2020-03-29 02:20-02:28 | additional crossing |
| 0040 | 2020-03-29 03:52-04:05 | replication/development crossing |
| 0041 | 2020-03-29 05:29-05:35 | additional crossing |
| 0044 | 2020-03-29 10:11-10:22 | near-end primary crossing |

Raw ATL07/ATL10 granules and CMR manifests are stored under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/icesat2_validation`.

## Colocation

For pair start time `t0`, end time `t1`, laser time `tL`, source location `x0`,
and pair displacement `u(x0)`, define

`alpha = (tL - t0) / (t1 - t0)` and `xL = x0 + alpha * u(x0)`.

The implementation solves this mapping for `x0` with fixed-point iteration for
each method independently. It accepts a solution only inside a valid,
non-folded source triangle and with a final forward residual no greater than
1 m. ORB interpolation uses triangles no longer than 20 km and minimum triangle
quality 0.05. ALIKED uses triangles no longer than 6.4 km. This is the local
counterpart of Ricker's drift-aware parcel registration; no external coarse
drift product replaces the measured LiMOSAT displacement.

Displacement uncertainty should later be propagated as approximately
`alpha * sigma_u`, with an additional residual-registration sensitivity from
rigid along-track and cross-track shifts. A best shift must not be selected by
maximizing the expected validation association.

### Drift-aware v2 audit

The material-coordinate inversion is now explicit and auditable in every point
output. For each tracker it retains the pair-start material coordinate, the
full pair displacement and pair-end coordinate, the displacement already
accumulated at the individual laser timestamp, the remaining displacement to
the pair end, and the numerical inversion residual. It also retains a
deliberately incorrect no-advection control that samples the source-time field
at the laser coordinate. This control measures whether drift correction is
material; it must not be used to tune the tracker or ICESat-2 thresholds.

Uncertainty is reported as `alpha` times the frozen 130-estimate buoy endpoint
error P90: 1,310 m for ORB and 1,301 m for ALIKED. This is an approximate
position-error scale, not a complete confidence interval. In particular, it
does not represent unresolved acceleration within a SAR pair, spatially
correlated displacement errors, or ICESat-2/SAR geolocation error.

The physical spatial-null and bootstrap distance is now specified in metres
and converted to bins. The earlier implementation silently encoded five bins
as 20 km, which was correct only for 4 km reporting bins. The 1 km sensitivity
now retains the intended 20 km shift and block length.

## ICESat-2 observables

### ATL07 ridge and roughness arm

- use strong beams only;
- require `height_segment_quality == 1`, fit quality 1 or 2,
  `height_segment_ssh_flag == 0`, and finite surface height;
- subtract a centred robust 5 km local baseline;
- identify a ridge peak at relative height at least 0.6 m, separated from
  another counted peak by at least 20 m;
- aggregate ridge density and the 90th-minus-10th percentile relative-height
  roughness to 4 km along-track bins.

The continuous roughness statistic is primary for standard ATL07 because the
product can smooth narrow sails. A later ATL03/UMD-RDA arm is required before
making a claim about individual narrow-ridge detection.

The v2 pilot additionally reports two literature-aligned structural measures:

- relative-height standard deviation within the reporting support, while the
  interpretive laser-height plots remain unsmoothed;
- ridging intensity as ridge frequency per kilometre multiplied by mean
  detected sail height.

Maximum compressive principal strain is accumulated from the SAR pair start to
the laser time and compared with ridging intensity. This is a more direct
local compression measure than negative divergence alone. It is retained
alongside, not substituted for, the frozen shear/roughness comparison.

### ATL10 lead arm

- use quality 1-2 freeboard segments and lead lengths from strong beams;
- aggregate lead length to 4 km bins;
- compare lead fraction with cumulative positive divergence (opening).

A crossing with fewer than 20 lead-containing bins is reported as insufficient
and is not interpreted through an enrichment ratio.

## Metrics and controls

Primary metrics on common ORB/ALIKED support:

- Spearman cumulative convergence versus ridge density;
- Spearman shear versus relative topographic roughness;
- ridge density in the highest convergence quintile divided by all other bins;
- Spearman opening versus lead fraction;
- paired ALIKED-minus-ORB association, with 20 km moving-block bootstrap
  intervals.

Spatial significance uses 999 within-beam circular shifts of at least 20 km.
This preserves much of the along-track autocorrelation. A nominal result must
replicate on another crossing or pair; one track is not a final accuracy claim.

Sensitivity arms, run only after the frozen primary result, are 8 and 12 km
reporting bins, fit quality 1 only, and rigid track offsets. Threshold selection
must not be based on which arm favours either tracker.

## First two crossing results: ATL07 0044 and 0040

ATL10 is a negative feasibility result for this crossing. Only 18 lead records
fall inside either method's support, and none lie on strong-beam common support.
No lead/deformation inference is made.

ATL07 0044 provides 87 common 4 km bins, 480 km of accepted strong-beam
topography, and 231 fixed-threshold ridge peaks. ATL07 0040 independently
provides 75 common bins, 297 km, and 123 ridge peaks:

| Common-support metric | 0044 ORB | 0044 ALIKED | 0040 ORB | 0040 ALIKED |
|---|---:|---:|---:|---:|
| Convergence vs ridge-density Spearman | -0.064 | 0.089 | -0.236 | -0.087 |
| Shear vs roughness Spearman | -0.014 | 0.217 | 0.189 | 0.296 |
| Top-convergence ridge enrichment | 0.82x | 1.47x | 0.63x | 1.21x |
| Circular-shift p, shear vs roughness | 0.627 | 0.023 | 0.018 | 0.001 |

For 0044, ALIKED covers 89.9% of laser observations in the union of method
support, versus 73.9% for ORB. For 0040 the difference is 98.5% versus 25.3%.
The positive ALIKED shear-versus-roughness result repeats on both tracks. ORB is
positive only on 0040. On both tracks the ALIKED-minus-ORB association
differences are positive, but their 20 km block-bootstrap intervals include
zero. This is evidence that the ALIKED deformation field preserves plausible
local shear structure and much more along-track support; it is not yet a
statistically resolved paired accuracy advantage.

The convergence-versus-ridge relation does not repeat as a positive result.
This is physically plausible because most observed ridges record deformation
before the 23 h SAR interval, and standard ATL07 smooths narrow sails. Current
convergence enrichment must not be used as a tracker selection metric.

## Next gates

1. Add a crossing/pair with sufficient leads for the ATL10 opening test.
2. Run 8 and 12 km support sensitivity and rigid-offset controls.
3. Evaluate UMD-RDA/ATL03 ridge retrieval and the
   independent monthly 10 km UMD-RDA deformation/topography product.
4. Extend to more SAR pairs and bootstrap by granule and pair, not by laser
   segment.

## Multi-image extension

Four additional Version 7 ATL07 granules were downloaded to Kingston and
opened as complete six-beam ATL07 products. The CMR search and pair-specific
manifests are retained under `icesat2_validation/manifests`. Applying the frozen
method to the March sequence produced the following support audit. Spearman
values below are descriptive on exact ORB/ALIKED common bins; a row marked
insufficient is not an accuracy result.

| SAR images | ATL07 | Common bins / ridges | ORB shear vs roughness | ALIKED shear vs roughness | Result |
|---|---:|---:|---:|---:|---|
| 10107 -> 10217 | 0024 | 9 / 4 | -0.321 | 0.310 | Insufficient spatial length and ridge events |
| 10107 -> 10217 | 0025 | 0 / 0 | n/a | n/a | No quality-controlled topography on common support |
| 10229 -> 10245 | 0029 | 30 / 25 | -0.099 | 0.156 | Descriptive positive ALIKED difference; spatial-null shear test not significant |
| 10245 -> 10341 | 0039 | 96 / 297 | -0.020 | 0.008 | No shear association for either method |
| 10245 -> 10352 | 0039 | 102 / 304 | -0.044 | 0.136 | Positive ALIKED difference, but spatial-null `p=0.143` |
| 10245 -> 10352 | 0040 | 75 / 123 | 0.189 | 0.296 | Positive ALIKED association, `p=0.001` |
| 10245 -> 10352 | 0044 | 87 / 231 | -0.014 | 0.217 | Positive ALIKED association, `p=0.023` |

The positive ALIKED shear relationship is therefore strongest and repeatable
for the 23 h 10245-to-10352 deformation image, but is not universal across
component intervals. This is consistent with ATL07 roughness recording
deformation accumulated before as well as during a SAR interval. ICESat-2 is
useful for detecting structural regressions, but the current association must
not be treated as direct drift truth or as the sole tracker-selection metric.

Two temporally coincident products, ATL07 0040 on 10245-to-10341 and ATL07 0044
on 10341-to-10352, contained millions of in-interval laser observations but no
observations inside either valid deformation footprint. They are retained as
`insufficient_method_support` results rather than silently dropped.

## ALIKED speed-path validation

The same frozen ATL07 tracks were used to check whether faster ALIKED matching
preserves deformation, always on the exact same laser observations for every
field:

- matching-only feature thinning is rejected for deformation. A 640-feature
  spatially balanced cap reduced matching by 23-28% with little buoy or coverage
  loss, but weakened the two positive full-pair shear associations. A
  768-feature cap saved only 12-15% and also weakened them;
- prior-guided target windowing retains all 1,024 detected features. The prior
  is the preceding ALIKED pair's median velocity scaled to the new time gap,
  with a fixed 15 km uncertainty. It reduced matching from 101.05 to 50.85 s
  on 10245-to-10352 and from 41.47 to 27.40 s on 10245-to-10341. Coverage and
  buoy availability were unchanged;
- 99.27% and 99.45% of common 4 km vectors in those two tests changed by no
  more than 100 m. Exact-common-bin deformation agreement with full matching
  was 0.81-0.83 for shear on ATL07 0040/0044 and 0.96 on ATL07 0039;
- the 15 km window preserved positive full-pair shear associations on both
  tracks (`0.228`, `p=0.033`; `0.154`, `p=0.058`), although both were weaker
  than full matching. This is a promising speed candidate, not yet a production
  default.

A 20 km uncertainty arm saved only 25% on the long pair and produced
non-monotonic ATL07 changes. The window width must be fixed from prior-error
coverage and independent buoy/manual controls, not tuned to maximize ICESat-2
association.

## Drift-aware v2 result

The frozen 10245-to-10352 pair was rerun for ATL07 0040 and 0044 at 1 and 4 km
reporting supports. Outputs are under
`icesat2_validation/results/drift_aware_v2/pair_10245_10352` on Kingston.

Material-point correction is not a small adjustment. Median source-reference
shifts are 11.64 km (ORB) and 13.56 km (ALIKED) for 0040, and 13.33 km and
13.41 km for 0044. The corresponding buoy-scaled position-error P90s are about
0.90 km for 0040 and 1.24-1.25 km for 0044. At 4 km, drift-aware and static
ALIKED shear fields on identical bins have Spearman correlation only 0.104 for
both tracks; ORB correlations are negative. A static geographic overlay is
therefore not an acceptable approximation for these events.

The original ALIKED shear/roughness association survives the corrected,
scale-aware analysis:

| ATL07 track | Support | ORB shear/roughness | ALIKED shear/roughness |
|---|---:|---:|---:|
| 0040 | 1 km | 0.098 (`p=0.028`) | 0.164 (`p=0.003`) |
| 0040 | 4 km | 0.189 (`p=0.018`) | 0.296 (`p=0.001`) |
| 0044 | 1 km | 0.010 (`p=0.686`) | 0.138 (`p=0.060`) |
| 0044 | 4 km | -0.014 (`p=0.627`) | 0.217 (`p=0.023`) |

The new maximum-compression/ridging-intensity result is promising only for the
near-end 0044 crossing: ALIKED is 0.101 (`p=0.051`) at 1 km and 0.196
(`p=0.052`) at 4 km, whereas 0040 does not replicate it. This timing pattern is
physically plausible but remains exploratory because existing ridges can
predate the SAR interval. It is not yet a tracker-selection criterion.

### Next drift-aware gates

1. Apply fixed, symmetric along-track and cross-track offsets derived from the
   buoy position-error scale; report a sensitivity envelope rather than
   selecting the best registration.
2. Replace the constant-within-pair trajectory assumption where overlapping
   intermediate SAR fields provide enough spatial support, and compare the
   piecewise trajectory with the direct long-pair trajectory.
3. Repeat at the SAR-pair/granule level. Bootstrap whole pairs or granules;
   laser segments and beams within one crossing are not independent samples.
4. Retain raw, unsmoothed ATL07 heights for interpretation and keep 1 km
   morphology statistics separate from the 4 km tracker-support comparison.

## Alignment audit and frozen March expansion

The shared machine-readable event ledger is implemented in
`experiments/multisensor_event_ledger.py` and is written by both ICESat-2 and
CryoSat-2 validators. It records EPSG:3413 coordinates in metres, pair-total
displacements in metres, per-observation UTC and interval fraction, material
and advected coordinates, field/interpolation identities, boundary outcomes,
and the complete selection flow. Tests cover zero and constant motion,
reverse-time sign, CRS round trips, exact common support, unique bin assignment,
and boundary/missing support.

The frozen existing-event audit is under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/multisensor_alignment_audit_v1_20260819`
(event-manifest SHA-256
`76705f957d75d653f8b6244fe52ab5a2a333d65741c07812322e294c970528d3`).
It retains 13 named variants, including four insufficient-support controls,
with selection tables, exact-common bins, numerical checkpoints, and compact
track/field figures. Five variants used the 8,523-node pre-final field rather
than the selected 8,520-node fixed-point fold-rejected field. Corrected ATL07
outputs are under `icesat2_validation/results/selected_fold_rejected_v3`.
Their common-support comparisons, spatial nulls, and paired association
statistics are identical to the earlier outputs; the correction only removes
the three rejected nodes from method-specific support.

The symmetric sensitivity output is under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/multisensor_alignment_sensitivity_v1_20260819`
(manifest SHA-256
`e9f15da4cdfb59c21817d82123bdd6bdfbac0402c6d529cec00343edef3651f3`).
At 4 km, 9 of 12 event-method envelopes change sign or span at least 0.20 in
Spearman rho. ATL07 0040 remains positive under all tested registrations, but
its magnitude is sensitive; 0044 changes sign. The previously weak 0024,
0029, and 0039 results are also generally alignment-sensitive. These envelopes
are diagnostics, not registration choices.

Six missing Version 7 products were selected from time and CMR geometry before
association calculation: ATL07 0030 and 0041 and ATL10 0030, 0039, 0040, and
0041. The frozen selection manifest is
`experiments/configs/multisensor_march_expansion_20260819.json`; all files and
outputs are on Kingston. At exact-common 4 km support for 10245-to-10352,
ATL07 0030 is positive (ORB `rho=0.182`, ALIKED `rho=0.414`) and ATL07 0041 is
negative (ORB `rho=-0.202`, ALIKED `rho=-0.258`). Neither has sufficient
component-pair support, so neither is a cross-pair replication. Every added
ATL10 application has zero to two lead-containing bins and is retained as
lead-insufficient. The per-track synthesis is under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/multisensor_expansion_summary_v1_20260820`.

The expanded evidence therefore does not establish a universal ALIKED
shear/roughness advantage. Positive long-pair tracks coexist with a negative
track, alignment sensitivity, and failed component-pair replication. The
appropriate role remains structural regression checking alongside buoy/manual
displacement accuracy, not tracker selection from altimetry alone.

The fair runtime benchmark carried the prequalified ATL07/ATL10 0039 products
onto the exact common support of its warm repetition 1 component field. ATL07
has 96 exact-common 4 km bins and is null for both methods (ORB
shear/roughness `rho=-0.002`, ALIKED `rho=0.029`; neither spatial-null
significant). ATL10 has 137 exact-common bins but only one lead-containing bin
and remains insufficient. These results are under
`fair_orb_aliked_runtime_v3_20260819/comparison/multisensor_warm_rep1` on
Kingston and confirm that the timing arm did not acquire an outcome-selected
multisensor advantage.
