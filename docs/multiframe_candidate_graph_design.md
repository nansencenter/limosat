# Multi-frame candidate graph design for LiMOSAT

Date: 2026-08-15

Status: experimental Arctic design. Preprocessing remains the standard VAE
product. SIFT and optical flow are outside the current scope.

Naming in this document: **first view reference** is the descriptor from the
first SAR observation (legacy internal name `anchor`); **latest confirmed
reference** is the newest descriptor accepted into persistent memory; and
**previous selected reference** is the immediately previous match even if it
was not confirmed (legacy `provisional`).

Later empirical results across March 2020 development, February 2020
validation, and N-ICE2015 holdout are consolidated in
`docs/arctic_descriptor_graph_empirical_report.md`. They show that
confidence-gated updates repair the March long path but are not universally
better than rolling updates on the 2015 holdout. The graph must therefore keep
candidate selection and permanent state updates separate until PM, cycle, and
neighbour evidence are available.

A later frozen replay identified a separate implementation assumption: the
production-safe 128-pixel candidate-grid border removes 38 buoy neighbourhoods
that are inside the raster but may not safely contain later PM/template windows.
A descriptor-only 32-pixel sweep improves the all-transition <=2 km fraction
from 65.4% to 79.0% on February and from 56.9% to 63.8% on N-ICE. This is
evidence for adaptive edge handling, not for lowering the production border.

## Empirical motivation

The first exact-time buoy graph uses 13 coincidences, three buoys, nine SAR
images, and ten transitions. One buoy supplies an eight-transition path; the
other two supply one transition each.

With the current ORB scale/orientation settings and corrected Hamming distance:

| State selection | Within 2 km | Errors over 50 km | Eight-step final error |
| --- | ---: | ---: | ---: |
| Greedy, replace descriptor every frame | 1/10 | 7/10 | 84.0 km |
| Greedy, fixed initial descriptor | 1/10 | 7/10 | 180.6 km |
| Beam 4, fixed initial descriptor | 7/10 | 1/10 | 128.4 km |
| Beam 8, fixed initial descriptor | 7/10 | 0/10 | 19.4 km |
| Beam 32, fixed initial descriptor | 7/10 | 0/10 | 19.4 km |

For the long path, beam 8 plus the initial descriptor remains between 0.36 and
0.83 km for seven consecutive observations, then misses the final two-day
transition by 19.4 km. This small fixture is evidence for a hypothesis, not a
selection result: LiMOSAT currently commits too early and destroys useful
appearance history after every accepted match.

The ORB contract matrix also shows that this result depends on existing
keypoint semantics. Changing the current `octave=5` to 4, removing geographic
orientation, or switching wholesale to default ORB parameters degrades the
path. These settings must be attributed, not normalized by assumption.

## Graph definition

For acquisition time `t`, let candidate node `i` contain:

```text
z[t,i]
  pixel_xy: float64[2] pixels, (column, row)
  map_xy: float64[2] metres, EPSG:3413
  descriptor: uint8[32] for current ORB
  response or reliability: optional float
  mask and border status
  image_id and UTC acquisition time
```

A directed edge from `z[t-1,j]` to `z[t,i]` exists only when both observations
are valid and:

```text
norm(map_xy[t,i] - map_xy[t-1,j]) <= vmax * delta_time_days
```

The current first-pass edge cost is normalized descriptor distance. Later terms
are added one at a time:

```text
C(edge) =
    wa * appearance_cost
  + wv * velocity_prediction_residual
  + wc * forward_backward_or_skip_cycle_residual
  + wn * local_neighbour_deformation_residual
  + wm * missing_observation_penalty
```

The aim is not to force globally rigid ice. The neighbour term should permit
real divergence and shear while penalizing isolated, topologically implausible
vectors and unsupported triangle flips.

## Why beam search

For a fixed anchor descriptor and no path-dependent terms, the problem is a
layered shortest path. Descriptor replacement, descriptor banks, velocity
state, missing observations, and template history make two paths arriving at
the same image position different states. Beam search is therefore the simplest
bounded, auditable first implementation.

At each image:

1. Predict a physics-bounded region for every retained state.
2. Retrieve the best `K` candidate observations per state.
3. Refine candidates with pattern matching only if needed.
4. Calculate appearance and optional motion/consistency evidence.
5. Retain the best `B` full states, not only the best current position.
6. Delay permanent commitment for a short lag.

The current fixture indicates `B=8` is sufficient while `B=4` is not. This
value must be retested across held-out sequences.

## Two-tier state rather than unconditional replacement

Each trajectory should distinguish confirmed history from provisional graph
state:

```text
TrajectoryState
  immutable_seed_descriptor
  confirmed_descriptor_bank
  provisional_paths[B]
  confirmed_template_bank
  last_confirmed_position/time
  velocity and uncertainty
  consecutive_missing_count
```

An observation can update the confirmed bank only after independent evidence:

- strong descriptor score and margin;
- agreement with the immutable anchor or at least two confirmed appearances;
- acceptable PM correlation;
- forward/backward or skip-frame cycle support;
- local-neighbour support;
- no impossible speed or mask/border condition.

If evidence is insufficient, the node may remain in a provisional path without
changing the descriptor or template. A missing node carries predicted state and
a penalty but does not masquerade as an observation.

The exact-time buoy archive now gives an empirical update signal. ORB distance
from the immutable buoy descriptor discriminates errors over 2 km with AUC 0.839
on February validation and 0.903 on N-ICE2015, stronger than previous-frame ORB
distance. This supports retaining the seed/confirmed bank as an explicit guard.
The values are diagnostics with buoy-cluster uncertainty, not production
thresholds; PM, cycle, and neighbour evidence still have to make the update
decision without buoy truth.

Candidate-level forensics adds a narrower update result. In six failed
transitions, an accurate but uncommitted previous observation ranked the next
truth-near candidate 1st-3rd, while confirmed memory ranked it 11th-265th. The
next experiment should therefore keep one-frame provisional appearance as an
additional proposal source. It must not replace confirmed memory; PM and
forward/backward or skip-frame support decide whether it is later promoted.

Buoy-supervised threshold training retains the existing conservative policy.
Across 36 direct graph replays, February selects a 0.032 best-match lead and
0.40 maximum descriptor difference under a 95% safe-update requirement. It is
behaviorally identical to the existing 0.032/0.35 rule on N-ICE: 23 safe memory
updates, no false updates, and 33 of 58 transitions within 2 km. The threshold
varies across grouped February folds, so this is not evidence for changing the
default. ORB also separates same-buoy paths from physically plausible buoy
distractors much better than the current sparse XFeat setup.

## Update policies

### First-view reference (legacy: immutable anchor)

Strength: prevents model drift and currently performs best.

Failure mode: genuine appearance change eventually makes the seed obsolete.

### Replace memory after every match (legacy: rolling descriptor)

Strength: adapts immediately.

Failure mode: one incorrect association changes the model used for every later
match. The current graph demonstrates this failure strongly.

### Bitwise majority

For ORB, aggregation must operate per bit. Averaging bytes is not a valid binary
prototype. Majority is still vulnerable when false observations outnumber early
correct observations.

### Descriptor bank

Keep several confirmed appearances and score a candidate using minimum or robust
aggregate distance. The first unrestricted bank failed because it admitted
unconfirmed matches. Bank membership, not merely bank scoring, is the important
decision.

### First-view-guarded update (legacy: anchor-guarded update)

Require a rolling candidate to retain support from the immutable anchor before
updating. A hard immutable anchor may be too restrictive, so the later version
should compare against a small set of independently confirmed anchors.

## Cycle and path consistency experiments

For images `A`, `B`, and `C`:

- compare `A -> B -> C` with direct `A -> C` retrieval;
- compare forward `A -> B` with reverse `B -> A`;
- compare paths produced with and without the intermediate image;
- measure closure in metres and descriptor space;
- retain a candidate if multiple paths support it, rather than requiring a
  single global transform.

This follows the broader idea of path consistency: associations should remain
compatible when intermediate observations are skipped. Recent tracking work
explicitly studies this formulation:

- Lu et al., *Self-Supervised Multi-Object Tracking with Path Consistency*,
  CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Self-Supervised_Multi-Object_Tracking_with_Path_Consistency_CVPR_2024_paper.html

ReTracker is particularly relevant conceptually. It reports that sequential
pairwise matching accumulates drift, while matching only the first and current
frames fails under large temporal appearance change; it uses intermediate
history to bridge those regimes. LiMOSAT can test the same principle with an
explicit, interpretable graph before considering another learned tracker:

- Tan et al., *ReTracker: Exploring Image Matching for Robust Online Any Point
  Tracking*, ICCV 2025:
  https://openaccess.thecvf.com/content/ICCV2025/html/Tan_ReTracker_Exploring_Image_Matching_for_Robust_Online_Any_Point_Tracking_ICCV_2025_paper.html

COLMAP provides a useful correspondence-graph analogy: it supports sequential,
spatial, and transitive pair selection and builds tracks from pairwise feature
relations. LiMOSAT differs because the surface deforms, times are irregular, and
state carries physics rather than camera geometry:

- https://github.com/colmap/colmap
- https://github.com/colmap/colmap/blob/main/doc/tutorial.rst

## Local coupling between trajectories

Single-trajectory beams do not prevent two trajectories from selecting the same
feature. Once individual path candidates are available, the coordinator should:

1. build a sparse cost matrix for nearby active trajectories and candidate
   observations;
2. solve one-to-one assignment within local connected components;
3. add neighbour compatibility from previous mesh adjacency;
4. retain alternate assignments when costs are close;
5. commit only after later frames or PM/cycle evidence resolve ambiguity.

`scipy.optimize.linear_sum_assignment` is already available for dense local
components; `scipy.sparse.csgraph.min_weight_full_bipartite_matching` is the
larger sparse alternative. A new graph dependency is not required initially.

## Missing observations and re-identification

An active trajectory should be allowed to be unobserved for a bounded number of
acquisitions:

- propagate position and uncertainty without writing an observed point;
- expand the next search region according to elapsed time and uncertainty;
- query immutable and confirmed-bank descriptors for re-identification;
- require stronger cycle/neighbour evidence after a gap;
- keep interpolated/predicted state distinct from measured state.

This removes the current pressure to repair every missing match through
interpolation and prevents predicted points from contaminating descriptor
updates.

## Implementation milestones

1. Completed: exact-time, single-trajectory ORB beam graph with explicit update
   policies and timing.
2. Completed: confidence-gated descriptor updates with an immutable anchor.
3. Completed: one-frame missing-observation nodes with elapsed-time physics;
   missing state never updates the descriptor.
4. Completed: candidate-level failure replay and frozen grid-border sweep;
   border 32 is descriptor-only evidence for adaptive edge handling, not a
   production setting.
5. Next: test a one-frame provisional descriptor proposal source against the
   confirmed-only graph, preserving both path histories.
6. Add direct skip-frame and forward/backward cycle evidence without PM.
7. Couple nearby trajectory beams with one-to-one assignment and neighbour
   deformation evidence.
8. Feed selected candidates through existing PM as refinement, retaining the
   pre-PM candidate and all rejection reasons.
9. Run complete trajectories and deformation metrics on sequence-held-out
   Arctic fixtures before modifying production state or persistence.
