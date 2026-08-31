# Arctic descriptor and multi-frame graph experiments

Date: 2026-08-17

Status: experimental. No production LiMOSAT code, database schema, or public API
has been changed. All tests use the existing `balanced_q2q98_clahe25` VAE band;
there is no additional preprocessing arm.

## Questions tested

1. Which hidden ORB contracts affect buoy tracking?
2. Does retaining candidate paths reduce the damage from early commitment?
3. When should an accepted observation update appearance memory?
4. Can a recent learned feature, XFeat, replace the ORB descriptor/grid contract?
5. Do graph improvements survive sequence-held-out Arctic data?
6. How does appearance at the exact buoy location evolve, and which changes
   precede tracking failure?

## Plain-language memory names

- **First view reference**: descriptor from the first SAR observation of the
  tracked ice. Older experiment code and files call this the `anchor`.
- **Latest confirmed reference**: newest descriptor accepted into persistent
  memory. Older code calls this the confidence-gated `rolling` descriptor.
- **Previous selected reference**: descriptor at the immediately previous
  selected location even when it was not accepted into persistent memory. Older
  reports call this `provisional` appearance.
- **Best-match lead**: normalized descriptor-cost gap between the best and
  second-best candidates. Older code calls this the update margin.
- **Safe to remember**: evaluation label for a selected point within 2 km of the
  exact-time buoy. Buoy truth supplies this training label but is not available
  to the matcher.

The plain-language terms are used in new reports. Legacy configuration names are
retained only where changing them would break reproducibility.

## Frozen fixtures

Every buoy position is linearly interpolated in EPSG:3413 metres to the exact
UTC SAR acquisition time. Temporal extrapolation is forbidden. Splits are by
whole image sequence and buoy path.

| Sequence | Role | Exact-time coincidences | Buoys | Images | Paths >=3 | Transitions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| March 2020 | development | 13 | 3 | 9 | 1 | 10 |
| February 2020 | validation | 292 | 86 | 12 | 55 | 206 |
| N-ICE2015 | holdout | 65 | 7 | 15 | 7 | 58 |

Three February records are excluded because their SAR time is outside the
recorded buoy interval. One additional exact-time point is outside its SAR
raster. The N-ICE GeoJSON declares EPSG:4326 even though its point coordinates
are already EPSG:3413 metres; the fixture builder detects and records that CRS
override instead of reprojecting the false metadata.

## ORB contract result

The active ORB model uses `WTA_K=2`, so ordinary Hamming is the correct native
distance. LiMOSAT defaults to Hamming2. This is not a label-only difference: an
absolute threshold of 120 spans almost the entire Hamming2 range but less than
half of the Hamming range.

The matrix was extended from the small March development set to February and
N-ICE2015 with every transition retained in the denominator. Current Hamming is
the best confidence-update arm on both: 65.4% within 2 km in February and 56.9%
in N-ICE2015. Current Hamming2 falls to 63.9% and 31.0%. The full LiMOSAT-default
combination (`nlevels=8`, `patchSize=31`, out-of-range `octave=8`, Hamming2)
reaches 56.6% and 48.3%; 32.8% of all N-ICE transitions are catastrophic.

Geographic orientation remains essential: zero angle gives only 28.8% within
2 km in February and 19.0% in N-ICE2015. Changing the accepted-but-out-of-range
current `octave=5` to 0 gives 27.3% and 6.9%. Changing keypoint size from 31 to
64 while retaining octave 5 produces bit-identical paths, and size 31 versus 64
at a fixed detected position changes zero descriptors. Under this OpenCV
contract the overwritten octave, not the size field, controls the effective
descriptor footprint. The out-of-range value should therefore not be silently
“fixed”; it should eventually be replaced by an explicit, valid scale contract
that is proven equivalent on held-out sequences. No tested replacement yet
matches current Hamming on both 2020 and 2015.

Other production blockers for learned descriptors are structural:

- persistence reloads every descriptor as `uint8`;
- the ratio branch is hardcoded to binary FLANN-LSH;
- the threshold 120 is backend-specific;
- grid/update keypoints hardcode size, angle, and octave;
- the cache fingerprint omits several descriptor and preprocessing fields;
- every accepted observation replaces the descriptor and template immediately.

The full line-by-line contract is in `docs/orb_descriptor_contract_audit.md`.

On the full-70 fixture, the fair exact-point comparison remains mixed. Switching
the graph from the current research contract (`nlevels=5`, `patchSize=64`,
`octave=5`, Hamming) to LiMOSAT's descriptor defaults (`nlevels=8`,
`patchSize=31`, overwritten `octave=8`, Hamming2) lowers the March fixed-first
beam from 92.7% to 87.6% within 2 km and February from 84.4% to 81.3%, while
raising January from 79.4% to 86.9% and April from 89.7% to 93.1%. The default
contract also has more catastrophic March/February beam paths.

The contracts are complementary at transition level: for the March fixed-first
beam, 27 transitions succeed only with the current contract and three only with
the default; in January, one succeeds only with current and 17 only with default.
This motivates retaining multiple scale/distance hypotheses in the candidate
graph, not choosing a descriptor contract from one month. An oracle union is
only an upper bound; PM/cycle/neighbour evidence must select between hypotheses.
This comparison supplies keypoints exactly at buoy/grid locations. It does not
repair the separate LiMOSAT local-detection-window failure described below and
is not a complete production LiMOSAT comparison.

## Exact-time buoy appearance archive

The 370 frozen fixture records are now represented explicitly: 369 have a valid
SAR sample and one February point is recorded as `outside_scene_footprint`.
Nothing is silently removed. For each extractable observation the archive stores:

- north-up EPSG:3413 patches covering 2.5, 5, and 10 km at 129 by 129 pixels;
- native-orientation patches of 31, 65, and 129 pixels;
- validity masks, raw uint8 standard-VAE pixels, and per-patch texture statistics;
- the exact current ORB uint8 descriptor and nearest sparse XFeat float32
  descriptor, with availability and distance-to-feature flags;
- previous-frame and immutable-anchor NCC, gradient NCC, SSIM, normalized mutual
  information, histogram Jensen-Shannon distance, RMSE, phase correlation, ORB
  Hamming distance, and XFeat cosine distance.

Observation IDs are identical and ordered consistently across CSV, patch, and
descriptor archives. The centre of every map-aligned patch equals the centre of
its corresponding native patch. All tracking results are joined after appearance
extraction so the buoy truth cannot affect candidate selection.

For the confidence-update ORB graph, immutable-anchor Hamming distance is the
strongest consistent diagnostic of error over 2 km:

| Split | Transitions with descriptor | Buoys | Failure AUC | Buoy-cluster 95% interval |
| --- | ---: | ---: | ---: | ---: |
| February validation | 173 | 57 | 0.839 | 0.743-0.910 |
| N-ICE2015 holdout | 58 | 7 | 0.903 | 0.647-0.969 |

The corresponding exact-location 5 km anchor NCC is also informative, but less
stable: validation AUC 0.691 (0.577-0.802) and holdout AUC 0.856
(0.483-0.947). Previous-frame ORB distance is weaker than anchor distance. This
supports keeping immutable appearance memory while allowing guarded updates; it
does not justify a fixed threshold yet. Intervals use 1,000 deterministic
resamples of whole buoy paths rather than treating transitions as independent.
These exact-buoy descriptors are evaluation diagnostics, not matcher inputs. An
operational gate must use evidence available at each candidate location and be
retested on sequence-held-out data.

The paired graph comparison keeps untrackable transitions in the denominator.
Confidence-gated updates make 26 N-ICE transitions trackable that the fixed
anchor graph cannot reach, improve the <=2 km fraction from 0.379 to 0.569 on
that holdout, and rescue five paired errors without harming a paired <=2 km
result. The February change is small (0.649 to 0.654), which reinforces that
update policy should remain conditional rather than unconditional.

## Buoy-supervised descriptor and memory training

Exact-time buoy paths now provide two explicit training signals without exposing
target truth to the matcher:

1. descriptor retrieval: rank the same buoy against other buoy locations in the
   next image that satisfy the 50 km/day motion bound and are more than 2 km from
   the positive;
2. safe-to-remember label: a selected graph candidate is safe to add to
   persistent memory when it lies within 2 km of the buoy.

ORB is currently the supported descriptor. Using the previous true view as the
reference, it ranks the same buoy first in 92.3% of eligible February cases and
100.0% of eligible N-ICE cases. Exact-location BRISK reaches 83.5% and 95.7%
with fewer eligible February cases; it remains a useful binary control. Sparse
XFeat's nearest feature within 5 km reaches 45.9% and 58.3%. The XFeat
comparison has a different localization contract, but it is sufficiently far
behind that it should not replace ORB in the next memory experiment.

Thirty-six ORB memory rules were replayed with the production-safe 128-pixel
raster border. February selection used whole-buoy grouped folds and required at
least 95% safe updates on each training partition. The full February data select
a 0.032 best-match lead and 0.40 maximum descriptor difference. On N-ICE this
produces exactly the same paths, 23 updates, and errors as the existing
0.032/0.35 hand-set rule: 33 of 58 transitions are within 2 km and all 23 memory
updates are safe. Replacing memory after every match gives only 6 of 58 within
2 km and makes 52 false updates.

The selected thresholds vary across February folds, and a richer learned
classifier did not preserve its false-update rate on N-ICE. The evidence
supports the existing conservative rule and use of previous-view ORB as
proposal evidence; it does not support changing a production threshold yet.

### Full-70 one-step descriptor retrieval

The official Level-1 fixture also isolates the descriptor from multi-frame
error propagation. At each transition, the source descriptor is extracted at
the known buoy point and ranked against the next image's fixed 16-pixel
candidate grid. Target truth is used only for rank and endpoint scoring. Every
method has 743/743 descriptor-available cases after VAE-mask filtering.

| Descriptor contract | <=2 km, 50 km/day gate | <=2 km, scene-wide | Median gated error |
| --- | ---: | ---: | ---: |
| ORB, geographic angle, Hamming | 94.3% | 89.0% | 0.55 km |
| ORB, geographic angle, Hamming2 | 93.4% | 86.3% | 0.56 km |
| BRISK, geographic angle, Hamming | 43.1% | 11.7% | 3.00 km |
| ORB, zero angle, Hamming | 29.6% | 8.3% | 3.39 km |

Geographic ORB/Hamming remains strong in every temporal split: 95.7% within
2 km in March, 87.5% in February, 92.1% in January, and 96.6% in April. Ordinary
Hamming is equal to or better than Hamming2 in every full split. The 50 km/day
gate improves ORB/Hamming by 5.4 percentage points relative to scene-wide
retrieval without changing descriptor ranking inside the retained set.

The one-step result is materially better than complete multi-frame paths. This
localizes much of the remaining loss to state propagation, candidate commitment,
and memory use rather than inability of the supplied-point ORB descriptor to
recognize the next buoy patch. BRISK is faster in this implementation (155 s
versus 224 s for each ORB arm over 743 pairs), but the accuracy loss is too large
to treat it as a speed replacement. The median 16-pixel grid quantization floor
is about 0.50 km and must remain separate from descriptor error.

## Exact buoy point versus detected-near-buoy keypoint

The buoy appearance archive and graph seed use an OpenCV keypoint supplied
exactly at the interpolated buoy pixel. They call `compute`, not `detect`. This
was compared directly with LiMOSAT's `KeypointDetector.keypoint_from_point`,
which detects ORB features in a local window and selects the feature nearest the
rounded window centre.

A direct call to the production method on one real observation from each
sequence selected exactly the same current-ORB pixel as the experiment helper.
For the LiMOSAT config-default ORB it returned no keypoint. This is explained by
a hardcoded contract: `patchSize=31` produces a 47-pixel detection window while
`edgeThreshold=31` excludes a 31-pixel band on both sides. There is no valid ORB
detection interior. The result was zero local candidates in all 369 extractable
buoy observations, not a single-scene anomaly.

Using the research ORB contract (`patchSize=64`, `edgeThreshold=16`) made local
detection possible, but did not improve tracking:

| Sequence | First-seed method | Memory method | Seed unavailable paths | Tracked, all transitions | <=2 km, all transitions | >50 km, all transitions |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| March 2020 | exact buoy pixel | confidence-gated update | 0 | 100.0% | 80.0% | 0.0% |
| March 2020 | nearest detected feature | confidence-gated update | 2 | 80.0% | 50.0% | 10.0% |
| February 2020 | exact buoy pixel | confidence-gated update | 3 | 86.8% | 65.4% | 5.4% |
| February 2020 | nearest detected feature | confidence-gated update | 12 | 75.1% | 57.6% | 5.4% |
| N-ICE2015 | exact buoy pixel | confidence-gated update | 0 | 100.0% | 56.9% | 3.4% |
| N-ICE2015 | nearest detected feature | confidence-gated update | 0 | 100.0% | 44.8% | 5.2% |

Conditional descriptor retrieval changes little: with the previous true view,
February top-1 rank changes from 92.3% at the exact point to 91.5% for a detected
source against exact-point candidates. The main cost is coverage and the change
of physical centre. Median nearest-feature offset is 126 m in February and 98 m
in N-ICE2015. The detected-size field itself is not responsible: because
LiMOSAT overwrites the octave, replacing the reported detector size with the
fixed size changed zero of 324 current-ORB nearest descriptors.

The 300 m production gate also measures distance from a rounded, inverse-mapped
window centre rather than from the buoy geometry. Twenty-one February centres
are more than 300 m from the exact buoy after the map-pixel-map round trip; six
accepted nearest features consequently lie more than 300 m from the buoy, with
a maximum of 659 m. These rows must be treated as geolocation-contract failures,
not evidence that a farther visual feature is attached to the buoy's ice patch.

The evidence supports exact supplied-point extraction when a buoy is the seed.
Local detection can remain an additional hypothesis only if missing detections
fall back to the exact seed and the actual buoy-to-feature offset is enforced.
It should not replace the exact seed.

## Additional Arctic sequence availability

The connected January-April 2020 IABP/Sentinel-1 catalogue contains much more
potential exact-time validation data than the frozen pixel fixtures:

| Month | Exact-time coincidences | Buoys | Paths >=3 | Potential transitions | Median gap | Local standard-VAE images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| January 2020 | 7,342 | 208 | 206 | 7,134 | 21.40 h | 0 |
| February 2020 | 4,013 | 194 | 161 | 3,819 | 19.77 h | 12 |
| March 2020 | 8,588 | 226 | 217 | 8,362 | 3.27 h | 12 |
| April 2020 | 12,521 | 220 | 216 | 12,301 | 1.64 h | 0 |

The drive contains catalogues, buoy tracks, trajectory outputs, and
sea-ice-concentration products, but not the recorded
`/Data/sat/downloads/.../processed_VAE_2_16_ELU_64` January or April rasters.
January is the best cadence-matched extension to February; April is the best
high-frequency within-2020 stress test. They should be restored or regenerated
with the same standard VAE before descriptor selection is revisited. N-ICE2015
remains a separate physical-regime holdout and should not be pooled into 2020
threshold training.

### Stratified coverage and IABP on-ice QC

Raw coincidence rows are not an effective sample size because the source file
contains repeated buoy observations for the same SAR scene. After exact-time
interpolation and deduplication by buoy/scene, January-April contains 32,464
positions from 277 buoys, 5,543 scenes, 2,027 acquisition passes, and 196
200 km EPSG:3413 blocks. The existing standard-VAE fixtures contain only 305
positions, 89 buoys, 21 scenes, and seven blocks. They are adequate for the
current failure analysis, but not for choosing a general descriptor/update rule.

Daily NOAA/NSIDC CDR v6 SIC sampled at the exact SAR position/date places
27,851 observations in the primary >=80% pack-ice pool. Another 228 are in
15-80% SIC and require platform-level evidence. Complete official IABP Level-1
files were downloaded for the eight selected MIZ buoy sequences. All eight
official tracks agree with the local exact positions and pass continuity QC,
but only two have adequate on-ice evidence: USIABP/AARI SVP-B platforms with
all four surface temperatures below -1.8 C. Four AOML BD2GHI ocean-drifter
sequences are rejected and two platforms with missing type and no surface
temperature remain on hold. This reduces the acquisition longlist from 94 to
70 Sentinel-1 scenes while leaving the 32-scene January/April pack-ice tier
unchanged.

The 128-pixel whole-image safety border was converted to metres empirically.
Across all 21 local standard-VAE rasters, the median per-raster GCP scale is
80.50 m/pixel and the observed maximum is 81.60 m/pixel. The catalogue audit
therefore uses a conservative 82 m/pixel, or 10.496 km for 128 pixels. Exact
interpolation moves 38 nominal matches just outside their scene footprint and
3,693 positions fail the full edge-safety buffer. These positions must not be
counted as descriptor/update training examples unless raster-specific window
geometry proves every required patch fits.

The 50 km/day candidate gate is nearly, but not completely, truth-preserving:
15 of 22,790 eligible <=72 h acquisition-pass transitions exceed it. The
expanded experiment should keep 50 km/day as the primary arm and report
40/50/75/100 km/day sensitivity. This separates descriptor ranking failures
from a small real tail excluded by the physics setting.

The readable selected sequence names, Level-1 evidence, exact coverage strata,
and QC-filtered acquisition queue are under
`results/iabp_s1_stratified_coverage/`. Repeated buoy IDs and acquisition passes
must remain in a single split. April remains the high-cadence temporal stress
set; N-ICE2015 remains the external physical-regime holdout.

### Full-70 official Level-1 fixture

The targeted extension is now a pixel-ready experiment rather than an
availability estimate. All 70 Sentinel-1 archives and all 70 frozen standard-VAE
rasters are present on KINGSTON. Official IABP Level-1 files are available for
all 154 linked buoys. Across 1,103 buoy/image links, final Level-1 QC retains
1,000 observations and assigns explicit reasons to every other row: 52 fail the
128-pixel whole-image border, 23 are not bracketed by Level-1 track samples, 22
belong to AOML ocean drifters, five exceed the six-hour track-gap threshold, and
one lacks adequate on-ice platform evidence. No link fails the 100 km/day track
speed or 500 m catalogue-position agreement thresholds.

The 1,000 accepted observations form 228 split-safe paths and 748 transitions at
most 72 hours apart. Five February points fall on invalid VAE mask support, so
the descriptor graph evaluates 743 transitions. March is development, February
is validation, January is temporal evaluation, and April is the season-edge/
high-cadence evaluation. A second `month_exclusive_buoy` view removes buoy-ID
overlap between months; it is reported beside the full temporal sample rather
than substituted silently for it.

The fixture uses official Level-1 `x/y` as truth and keeps the prior catalogue
coordinates for audit. Only the first truth position seeds a path. Future buoy
positions cannot enter candidate creation, graph costs, descriptor updates, or
path selection. Eighteen transitions skip one selected but unusable image; this
is recorded per transition rather than reconnecting rows invisibly.

One VAE mask contains values `0, 1, 253`. Frozen preprocessing defines usable
support as values below 2, but LiMOSAT and several experiments previously
excluded only value 2. All affected paths now use `mask < 2`/`mask >= 2`; value
253 is excluded and retained as a noncanonical-mask warning. This is a concrete
hidden-assumption failure, not a change to preprocessing.

## Multi-frame ORB result

The graph uses a 16-pixel candidate grid, Hamming distance, an explicit
50 km/day speed-scaled hard gate, and beam width 32. Buoy positions are visible
only to the initial seed and evaluation.

| Sequence | Policy | Completed paths | Median error | <=2 km | >50 km | Long-path final median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| March development | rolling greedy | 3/3 | 78.29 km | 10.0% | 70.0% | 84.02 km |
| March development | beam + anchor | 3/3 | 0.78 km | 70.0% | 0.0% | 19.43 km |
| March development | beam + confidence update | 3/3 | 0.81 km | 80.0% | 0.0% | 1.16 km |
| February validation | rolling greedy | 59/66 | 0.97 km | 73.5% | 4.1% | 1.29 km |
| February validation | beam + anchor | 60/66 | 0.65 km | 74.7% | 7.3% | 0.87 km |
| February validation | beam + confidence update | 60/66 | 0.70 km | 75.3% | 6.2% | 0.83 km |
| February validation | confidence + one null node | 63/66 | 0.72 km | 75.3% | 5.8% | 0.83 km |
| N-ICE2015 holdout | rolling greedy | 7/7 | 1.35 km | 60.3% | 0.0% | 18.76 km |
| N-ICE2015 holdout | beam + anchor | 5/7 | 0.91 km | 68.8% | 9.4% | 40.21 km |
| N-ICE2015 holdout | beam + confidence update | 7/7 | 1.34 km | 56.9% | 3.4% | 32.61 km |

Three of 66 February paths have no valid ORB seed descriptor and remain explicit
failures. The one-null-node graph rescues all three graph-search failures, but
uses six skipped observations and has 96.9% observation coverage. On N-ICE2015,
null nodes rescue two anchor failures but produce much worse errors. Missing
nodes therefore improve continuity; they are not evidence of correct tracking.

The important result is not a universal winning policy. Confidence-gated updates
repair the March long path, but rolling is safer on the 2015 holdout. Candidate
history, descriptor update, and path selection must remain separate until PM,
cycle, and neighbourhood evidence can confirm an observation.

### Full-70 graph result

The same descriptor-only graph was rerun on the Level-1 fixture with the frozen
128-pixel border. Fractions below retain untracked transitions in the denominator.
The readable memory name is shown; legacy configuration names remain in CSVs for
reproducibility.

| Split | Transitions | Best memory/search arm | Median error | <=2 km | >50 km |
| --- | ---: | --- | ---: | ---: | ---: |
| March development | 468 | beam, fixed first descriptor | 0.56 km | 92.7% | 0.6% |
| February validation | 32 | fixed first / beam / confidence tie | 0.57-0.58 km | 84.4% | 0.0% |
| January evaluation | 214 | greedy previous-selected descriptor | 0.66 km | 86.0% | 0.0% |
| April season edge | 29 | all four arms tie at 2 km | 0.51-0.62 km | 89.7% | 0.0% |

March alone suggests retaining a beam of fixed-first candidates: it rescues 26
baseline failures and regresses two, for a net gain of 24 transitions within
2 km. January reverses that conclusion. The same beam rescues one but regresses
15, while confidence-gated updating rescues one and regresses ten. On the strict
month-exclusive-buoy subset, the previous-selected baseline reaches 74.1% in
January versus 66.7% for all three alternatives. The strict February and April
subsets contain only nine transitions each and are reported as sensitivity
checks, not stable estimates.

The result rules out a universal “always use the updated descriptor” decision.
It supports candidate memory with explicit fixed-first and previous-selected
sources, leaving commitment to later PM, cycle, and neighbourhood evidence. It
also confirms that development-only beam gains can overfit a month even without
learning descriptor weights.

A 40/50/75/100 km/day sensitivity sweep reused identical candidate descriptors,
so only the physics gate changed. Forty km/day equals 50 on the March fixed-first
beam (92.7%) and improves the previous-selected baseline from 81.3% to 87.5% in
February and from 86.0% to 88.8% in January. The strict unseen-buoy January
subset improves from 74.1% to 85.2%. In contrast, 75-100 km/day reduces March
performance and creates 9-10% catastrophic paths for the January beam arms.

A hard 40 km/day production limit would still be wrong: five of 748 Level-1
truth transitions exceed 40 km/day and one exceeds 50. The evidence instead
supports a 50 km/day hard safety ceiling with a soft speed prior or cost above
40 km/day. This preserves the observed fast tail while suppressing the large
ambiguous candidate set that harms the wider hard gates.

Transition-level attribution explains the gap from the 94.3% one-step result.
Across 743 transitions, exact one-step ORB succeeds while the complete graph
fails on 56 previous-selected greedy cases, 48 fixed-first beam cases, and 46
confidence-update cases. The graph rescues only one to four one-step failures.
These are state-propagation failures: the descriptor still recognizes the ice
when started from the true previous position, but the graph no longer searches
from that state or selects its path.

The 38-41 cases where both one-step retrieval and the graph fail have much
larger exact-buoy descriptor change (median normalized Hamming 0.45-0.46 versus
0.14 for joint successes). In contrast, graph-only failures do not have a
larger median local-mean change than joint successes, and their local-standard-
deviation change is only modestly larger. A simple “the image became bright or
dark/noisy” rule therefore does not explain the propagation failures.

There is a real but narrower update-poisoning signature. In the March
confidence-update path, 18 transitions follow a descriptor update made more
than 2 km from the buoy, and all 18 subsequent transitions fail. However, the
fixed-first beam also fails 16 of those 18 cases. Wrong updates mostly mark and
continue an already-wrong state; only two cases are clean evidence that avoiding
the update preserves a <=2 km result. This supports keeping unconfirmed
appearance provisional, but makes restoring/reconnecting the correct spatial
state the higher-priority intervention.

## Candidate failure forensics and border ablation

The frozen confidence-update graph was replayed exactly while recording every
pre-pruning candidate and retained beam state. Buoy truth was not used during
matching. It was joined afterward to identify the nearest grid node, whether it
was physically reachable, its descriptor ranks, and the stage at which it was
lost. Replayed node indices match the frozen paths and the maximum map-position
difference is `5.9e-11` m.

The requested candidate grid excluded a 128-pixel band around every raster:
38 truth-near neighbourhoods were covered by the SAR image but removed before
ORB extraction. Every one was less than 128 pixels from the raster boundary;
their exact-buoy patches had ordinary valid fractions and texture statistics.
This is a deliberate production safety border for later template extraction,
interpolation, and pattern matching, not an arbitrary descriptor failure.

A frozen sweep changed only this requested border. February selects 32 pixels;
N-ICE2015 remains untouched as holdout:

| Sequence | Border | Candidate within 2 km | Tracked | <=2 km, all eligible | >50 km, all eligible |
| --- | ---: | ---: | ---: | ---: | ---: |
| February validation | 128 px | 83.4% | 86.8% | 65.4% | 5.4% |
| February validation | 32 px | 99.5% | 95.6% | 79.0% | 2.9% |
| N-ICE2015 holdout | 128 px | 86.2% | 100.0% | 56.9% | 3.4% |
| N-ICE2015 holdout | 32 px | 100.0% | 100.0% | 63.8% | 1.7% |

The descriptor-only improvement transfers to the holdout and costs about 4-6%
additional precomputation plus tracking time. It does not establish that the
full template and PM windows fit safely near the edge. The result motivates
candidate-specific edge handling: descriptors may propose a point nearer the
edge, but PM and template updates run only when their complete windows fit.

With the 32-pixel border, 207 of 273 eligible transitions are within 2 km and
66 fail or are untracked. Their primary mechanisms are:

| Mechanism | Transitions |
| --- | ---: |
| appearance ranks the truth-near candidate outside the local branch | 32 |
| state or selected-path physics gate excludes it downstream | 17 |
| initial descriptor unavailable | 9 |
| truth-near state survives the beam but loses final path selection | 6 |
| residual border or candidate ranking failure | 2 |

The temporal replay does not support descriptor poisoning as the dominant
explanation. Eight selected observations over 2 km updated memory, but none
met the strict next-frame poisoning signature. There was one case matching the
“poor image followed by a clearer image” recovery opportunity and it recovered.
Gate exclusions mostly begin after an earlier appearance or selection error.
Across 29 contiguous failure episodes, the initiating transition is appearance
ranking in 20, final-path selection in four, missing seed in three, and a
candidate-border or candidate-ranking issue in one each; no episode begins as a
gate exclusion.

There is, however, direct evidence for a short provisional appearance bridge.
For six failed transitions (four February, two N-ICE), the previous selected
position was only 14-728 m from the buoy and had not been committed. Confirmed
memory ranked the next truth-near candidate 11th-265th, while the actual
descriptor from that uncommitted observation ranked it 1st-3rd. Two successful
transitions show the same signature. This is a truth-labelled diagnostic, not
a deployable update rule, but it motivates a controlled one-frame provisional
descriptor bank: use provisional appearance to propose candidates while the
immutable/confirmed bank, cycle evidence, and PM decide whether to commit it.

## XFeat result and requirements

XFeat was tested as its native sparse detector plus 64-dimensional float32
descriptors and cosine distance. It was not forced through ORB keypoint, dtype,
Hamming, threshold, FLANN, or persistence assumptions.

At max side 1536 and top-k 16,000:

| Sequence | Best XFeat policy | Completed paths | Median feature floor | Median error | <=2 km | >50 km |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| March development | beam anchor | 1/3 | 0.96 km | 2.47 km | 37.5% | 0.0% |
| February validation | beam anchor+rolling | 60/66 | 1.71 km | 3.00 km | 32.4% | 1.6% |

Two March low-texture seeds have no detected feature within 5 km. Increasing
resolution reduces the feature-coverage floor but does not close the error gap
to ORB. XFeat extraction is fast in this fixture (4.6-5.0 seconds for 9-12
scenes), so the blocker is accuracy and sparse coverage, not runtime.

This does not reject XFeat + LightGlue. It rejects raw sparse XFeat/cosine as a
drop-in replacement for the supplied ORB grid. LightGlue should be tested as a
pairwise candidate-edge producer or hard-pair fallback, because its learned
confidence is not a persistent descriptor distance.

## Candidate graph state and updates

Each layer contains candidate position, time, mask state, descriptor, backend
score, and source state. Edges exist only inside the elapsed-time physics gate.
Beam states carry immutable and provisional appearance memory, position,
velocity, missing count, cumulative cost, and a complete rejection trace.

The implemented confidence rule updates appearance only when the selected
candidate is the locally best appearance match, its normalized margin exceeds
the configured threshold, and its cost remains below a ceiling. Otherwise the
candidate may survive provisionally without modifying memory. In the March long
path, five of eight observations update memory; this changes final error from
19.43 km for the immutable anchor to 1.16 km.

A null node writes no observed position or descriptor. Its next physics gate
uses elapsed time since the last measured point, and it cannot update appearance.

## Next experimental milestones

1. Keep 128 pixels as the frozen production-safe control. Test adaptive edge
   handling that separately records descriptor availability, PM-window safety,
   and template-update safety; do not lower the global production border.
2. Test a one-frame provisional descriptor as an additional proposal source;
   never let it overwrite confirmed memory without later evidence.
3. Add direct `A -> C` and reverse `B -> A` edges. Score closure of `A -> B -> C`
   against the skipped path in metres and descriptor space.
4. Feed each retained candidate through the existing PM refinement while keeping
   pre-PM position, PM correlation, descriptor margin, and rejection reason.
5. Build sparse one-to-one assignment within local trajectory components so two
   tracks cannot claim the same candidate. Retain close alternative assignments.
6. Add neighbour evidence from the previous Delaunay/mesh topology: relative
   displacement residual, edge-length change, triangle orientation, and robust
   support for real divergence/shear rather than a rigid global transform.
7. Run the full LiMOSAT deformation products for direct-only and all-linked
   trajectories. Compare buoy error, survival, cycle closure, divergence, shear,
   total deformation, spatial coverage, outlier topology, and runtime.
8. Only after the SAR tracker is stable, add independent sensor validation.

## Multisensor validation, not displacement substitution

- ICESat-2 ATL07 provides along-track sea-ice surface height and type, including
  fixed 10 m segments for strong beams in Version 7. Use colocated height peaks
  and high-relief sections as ridge evidence after advecting the SAR feature or
  deformation field to the altimeter time. ATL10 freeboard and lead information
  can provide additional surface-state context. These tracks validate structural
  consistency; they are not point drift truth.
- CryoSat-2 elevation/freeboard/thickness/roughness products are suitable for
  broader roughness and thickness regime checks. Their footprint and sampling do
  not support treating individual ridges as coincident SAR control points.
- AMSR2 Level-2/Level-3 sea-ice concentration is a coarse regime, ice-edge, and
  compactness covariate. Use it to stratify failures and reject open-water
  interpretations, not to validate kilometre-scale deformation vectors.

Official product references:

- https://nsidc.org/data/icesat-2/products
- https://nsidc.org/data/atl07/versions/7
- https://nsidc.org/data/icesat-2/related-data
- https://gportal.jaxa.jp/gpr/information/product

## Targeted data acquisition status

The complete 70-product Sentinel-1 EW GRDM collection is stored directly on
KINGSTON at
`/Volumes/KINGSTON/arktalas/experiments/limosat_descriptor_update_2020/sentinel1/raw/`.
It occupies 18,021,755,594 archive bytes. A full member-by-member CRC pass found
all 70 expected archives with no missing, duplicate, unexpected, partial, or
corrupt archive. All 70 two-band uint8 standard-VAE rasters pass the frozen
intensity, mask, and geolocation contract under
`sentinel1/standard_vae/2020/MM/`.

The four apparent repeat-publication controls in the older local catalogue are
not downloadable repeats. Every candidate URL returns 404 and official ASF
wildcard search returns only the already-selected primary GRD product. They are
recorded as stale catalogue duplicates and excluded from pixel-consistency
claims. The two real same-pass controls are adjacent 60-64 second slices with no
pixel overlap. Across 16, 32, and 64-pixel inward offsets, cross-seam mean
absolute intensity change is 0.92-1.05 times equal-distance within-scene change.
This finds no gross seam discontinuity, but cannot measure repeat-pixel descriptor
stability.

## Reproducible outputs

- Fixture ledger: `results/arctic_fixture_ledger/q2q98_clahe25/`
- Stratified exact-time coverage, empirical raster scale, Level-1 on-ice QC,
  readable sequence manifests, QC-filtered Sentinel-1 acquisition queue, and
  the completed download/CRC verification ledgers:
  `results/iabp_s1_stratified_coverage/`
- ORB matrices: `results/orb_multiframe_graph/final_arctic_matrix/`
- XFeat matrices: `results/xfeat_buoy_graph/`
- Exact-time buoy patches, descriptors, tracking joins, clustered uncertainty,
  contact sheets, and paired update effects:
  `results/buoy_patch_evolution/q2q98_clahe25/`
- Candidate-border sweep with all eligible transitions retained in denominators:
  `results/orb_border_sweep/q2q98_clahe25/`
- Truth-joined candidate, beam, patch, and descriptor failure archive:
  `results/orb_candidate_forensics/q2q98_clahe25_border32/`
- Buoy-grouped descriptor separability and memory-rule training:
  `results/buoy_supervised_update_training/q2q98_clahe25/`
- Consolidated machine-readable comparison:
  `results/arctic_descriptor_graph_summary/comparison.csv`
- Candidate/update design: `docs/multiframe_candidate_graph_design.md`

The next production-facing decision requires the PM-refined, locally coupled
graph to improve the held-out deformation metrics. These descriptor-only graph
results are not sufficient to change LiMOSAT defaults.
