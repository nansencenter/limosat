# LiMOSAT descriptor contract audit

Date: 2026-08-15

Scope: Arctic LiMOSAT with the existing standard VAE-preprocessed uint8 band.
This audit describes the current code contract before any alternative descriptor
is integrated. It is not a proposal to change the production pipeline yet.

## Executive findings

1. The configured ORB model leaves `WTA_K=2`, for which OpenCV reports
   `NORM_HAMMING` as the default norm. LiMOSAT instead defaults to
   `NORM_HAMMING2`.
2. The cross-check matcher uses the configured norm, while the Lowe-ratio branch
   is hardcoded to FLANN-LSH. Those two branches therefore do not have an
   explicit shared distance contract.
3. `descriptor_distance_max=120` is not portable. For a 32-byte ORB descriptor,
   the maximum distance is 256 under Hamming and 128 under Hamming2. The same
   threshold consequently means very different acceptance regions.
4. Grid and recomputed keypoints hardcode `size=31`, external image orientation,
   and a configured `octave`. These values can disagree with ORB `patchSize` and
   `nlevels` (for example, `patchSize=48`, `nlevels=5`, `octave=5`).
5. Persistence reloads every descriptor as `uint8`. A float descriptor would be
   silently corrupted even though the in-memory `Keypoints` table can hold it.
6. Every accepted trajectory replaces its descriptor and template with the
   newest observation. There is no anchor descriptor, descriptor bank, update
   confidence, or rollback after a bad match.

OpenCV's ORB documentation states that Hamming2 is required for `WTA_K` 3 or 4,
where two-bit bins are used; ordinary binary ORB (`WTA_K=2`) uses Hamming:

- https://docs.opencv.org/4.13.0/dc/dc3/tutorial_py_matcher.html
- https://docs.opencv.org/4.0.0-alpha/db/d95/classcv_1_1ORB.html

## Current contract by stage

| Stage | Current assumption | Consequence for alternatives |
| --- | --- | --- |
| Input | Band 1 is a uint8 VAE product; mask value 2 is excluded | Keep this fixed during descriptor attribution |
| Seeding | Detector finds one response-ranked keypoint per window | A dense learned descriptor does not reproduce ORB response or keypoint size |
| Buoy seeding | ORB is redetected near the buoy, up to 300 m from its centre | Validation must distinguish buoy position from seeded feature position |
| Grid | Keypoints are made at fixed pixel stride with `size=31` | Descriptor must support evaluation at arbitrary supplied locations or provide a different candidate generator |
| Orientation | Grid and updated points use `img.angle` | Rotation-invariant and orientation-free descriptors need their own policy |
| Scale | Grid and updated points receive configured `octave` | `octave` is an ORB pyramid field, not a generic scale interface |
| Descriptor array | Matching stacks rows into a single NumPy matrix | All active descriptors must share shape and dtype |
| Cross-check | OpenCV BFMatcher with configured norm | Binary and float descriptors need different norms and calibrated distances |
| Extra candidates | FLANN algorithm 6 (LSH) and a Lowe-like ratio | LSH is binary-specific; float descriptors require a different index or native matcher |
| Threshold | Absolute distance `< 120` | Must be replaced by backend-specific normalized or calibrated confidence |
| Model filter | Candidate matches are grouped by source image and fitted with MAGSAC homography | A candidate graph should retain rejected alternatives until multi-frame scoring |
| PM refinement | A stored image template refines the selected point | Learned candidates can still use PM, but descriptor position must be recomputed after correction |
| Update | Corrected position entirely replaces descriptor and template | One false association can contaminate all later steps |
| Storage | Descriptor is JSON text and reloaded as `uint8` | Float type, dimension, backend, normalization, and version are lost |
| Cache | Key includes image basename, stride, border, octave, and a partial model tag | It omits preprocessing identity, band, orientation policy, `WTA_K`, `fastThreshold`, dtype, and descriptor version |

## ORB configuration inconsistencies to test explicitly

The repository default configuration uses `nlevels=8`, `patchSize=31`,
`edgeThreshold=31`, and `octave=8`. The RADARSAT configuration path commonly
uses `nlevels=5`, `patchSize=48` or 64, `edgeThreshold=16`, and `octave=5`.
OpenCV accepts supplied keypoints with those octave values, but acceptance is not
evidence that the scale semantics are intended.

The controlled ORB matrix should therefore include:

- `WTA_K=2 + Hamming` versus the existing `WTA_K=2 + Hamming2`;
- zero, geographic, and detector-derived orientation;
- a valid level range `0..nlevels-1` plus the current `octave=nlevels` arm;
- keypoint size tied to `patchSize` versus the hardcoded 31 pixels;
- borders derived from the effective descriptor footprint versus the current
  independent border;
- calibrated normalized distance and margin rather than only absolute 120.

### First Arctic contract matrix

The first matrix holds the VAE images, buoy paths, candidate grid, 50 km/day
gate, and beam-anchor graph fixed. It is only ten transitions and must not be
treated as a final parameter selection.

| Contract arm | Within 2 km | Errors over 50 km | Long-path final error |
| --- | ---: | ---: | ---: |
| Current scale/orientation + Hamming | 7/10 | 0/10 | 19.4 km |
| Current scale/orientation + Hamming2 | 6/10 | 0/10 | 19.4 km |
| Change octave 5 to 4 | 5/10 | 2/10 | 91.5 km |
| Default nlevels 8, patch 31, octave 8 | 5/10 | 3/10 | 133.3 km |
| Default nlevels 8, patch 31, octave 7 | 4/10 | 2/10 | 122.3 km |
| Octave 0 | 2/10 | 2/10 | 120.7 km |
| Zero orientation | 1/10 | 7/10 | 147.9 km |

Changing the supplied keypoint size from 31 to 64 at octave 5 produced identical
results in this experiment, while doing so at octave 0 did not improve the poor
octave-0 result. This suggests OpenCV's supplied-keypoint ORB scale behavior
needs a direct descriptor-footprint test rather than inference from field names.

The immediate conclusion is narrow: switch Hamming/Hamming2 into the held-out
matrix, preserve geographic orientation, and do not “repair” octave values
before broader evidence exists.

## Descriptor adapter requirements

No backend should be passed directly into `KeypointDetector` or `Matcher` unless
it declares the following metadata and operations:

```text
DescriptorSpec
  backend_name and version
  candidate_generation: supplied_grid | detected_sparse | semi_dense
  dtype and dimension
  metric: hamming | hamming2 | cosine | l2 | learned_pair_score
  normalized_distance_range or score calibration
  input_channels, input_range, and resize policy
  rotation and scale policy
  supports_descriptor_at_location
  supports_descriptor_update
  cache_fingerprint
```

```text
CandidateObservation
  trajectory_id
  image_id and time
  pixel and projected coordinates
  appearance_cost and raw backend score
  source state id
  uncertainty or score margin
  backend metadata
```

This separates candidate generation from state selection. ORB can generate a
dense supplied grid; XFeat can generate sparse or semi-dense candidates; a
pairwise learned matcher can emit candidate edges without pretending that its
score is an ORB descriptor distance.

## Initial challenger compatibility

| Backend | Native output | Required LiMOSAT adapter | Update candidates |
| --- | --- | --- | --- |
| ORB | 32-byte `uint8`, binary, supplied or detected keypoints | Correct norm; explicit size/orientation/scale; normalized distance | rolling, anchor, bit majority, descriptor bank |
| XFeat sparse | learned keypoints, float descriptors and reliability | float/cosine matcher; resize-coordinate mapping; sparse coverage handling; float-safe cache/storage | rolling, anchor, normalized mean, descriptor bank |
| XFeat + LightGlue | pairwise match indices and learned confidence | candidate-edge backend; do not route through BF/FLANN or distance 120 | retain per-image XFeat descriptors; LightGlue score is edge evidence |
| DISK/ALIKED/SuperPoint + LightGlue | detector-specific float descriptors and pairwise confidence | backend-specific pretrained weights, image normalization, license, dimension, and LightGlue feature configuration | same as XFeat/LightGlue after separate transfer testing |
| KeyNet/AffNet/HardNet | scale/orientation-aware LAFs and float HardNet descriptors | preserve LAF geometry; L2/cosine matching; do not force ORB angle/octave fields | normalized mean or bank, with LAF state retained |

XFeat supports sparse and semi-dense modes; LightGlue consumes sparse feature
sets and has weights tied to specific feature families. These are separate
backends rather than interchangeable descriptor arrays:

- https://github.com/verlab/accelerated_features
- https://github.com/cvg/LightGlue

## Multi-frame state policies to benchmark first

All policies start from the same candidates and physics gate.

1. `rolling`: replace appearance state after every selected observation. This
   reproduces the current failure mode most closely.
2. `anchor`: always compare to the initial accepted descriptor. It avoids drift
   but cannot adapt to genuine appearance change.
3. `majority_bit`: for ORB, maintain a bitwise majority prototype over accepted
   observations. Byte-wise averaging is invalid for a binary descriptor.
4. `bank_min`: keep a bounded bank and score a candidate by its closest stored
   descriptor. This supports multiple appearances but can preserve a bad member.
5. `anchor_rolling`: combine anchor and latest-descriptor costs, requiring both
   or penalizing disagreement.
6. `confidence_gated`: update only when multi-frame and local-coherence evidence
   is strong; otherwise retain the previous state.

The first graph experiment will use a layered directed acyclic graph over image
time. Nodes are candidate observations, edges are allowed motions, and path cost
contains appearance evidence plus optional velocity, cycle, and neighbourhood
terms. Beam search is required once the descriptor state becomes path-dependent.

## Integration rule

Production code should not be generalized until a backend and update policy
improve held-out buoy paths and full deformation metrics. The first graph and
adapter work remains under `experiments/` and consumes the standard VAE images.
