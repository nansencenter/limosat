# Arctic tracking next-experiment plan

Status: approved; development candidate frozen, confirmation passed, final unopened  
Created: 2026-08-17  
Scope: evidence-bounded improvements to LiMOSAT tracking plus small learned-matcher pilots

## 1. Decision this run must support

Determine which of the following produces the best operational sea-ice displacement
and deformation measurements per unit computation:

1. the current LiMOSAT ORB, MAGSAC, and pattern-matching architecture;
2. targeted changes to candidate retention, trajectory state, and pattern-matching
   commitment;
3. lightweight learned pairwise candidate sources: DeDoDe v2-B as the
   accuracy-first arm and ALIKED plus LightGlue as the speed-first arm;
4. an accuracy-neutral but materially faster form of one of the above.

This run is not a broad matcher competition. Each experiment must address a
measured loss in the preceding stage. RoMa v2 remains a deferred dense-warp pilot
for a CUDA/HPC environment and will not block the local work.

## 2. Established evidence and resulting priorities

- On the full-70 buoy fixture, supplied-point ORB plus the physics gate recognizes
  the next buoy neighbourhood on 94.3% of isolated transitions at the current
  2 km diagnostic threshold.
- Complete paths are worse and the preferred memory policy changes between
  temporal subsets. Candidate commitment, spatial-state propagation, appearance
  updates, and recovery are therefore leading current hypotheses.
- The existing full-70 graph is descriptor-only. It has not yet included the
  operational pattern-matching refinement, MAGSAC, local trajectory coupling, or
  reverse/direct cycle evidence.
- Existing results do not establish whether global MAGSAC preferentially removes
  valid high-shear or divergent motion.
- XFeat sparse detection and cosine matching have already underperformed ORB.
  They will not be repeated. SIFT and optical flow remain out of scope.
- The standard VAE/DN-CLAHE image preparation is frozen. Preprocessing is not an
  experimental factor in this run.
- The reproduced 70-image B0 has 94,890 persisted rows and 37,450 trajectories.
  Its corrected local-Zarr run has final-state SHA256
  `efb3030ec8afe987f9d34d4d1244694c055f1f808690a349eda65650e6e118bc`.
- Full instrumentation preserved that hash exactly. Across 104,166 eligible
  source-trajectory/image opportunities, the current matcher proposed 94,458
  candidates. Of those, 40,866 (43.3%) were discarded by the motion gate and
  1,428 were MAGSAC outliers. Cross-check supplied 94,426 proposals; the current
  `knn_k=8` Lowe path added only 32. This makes descriptor-first commitment before
  spatial physics the first targeted candidate-generation hypothesis.
- Warm-cache B0 took 489 seconds locally. Pattern matching consumed 251 seconds
  (51% of wall time), versus 33 seconds for all ORB grid matching, making pattern
  matching the first accuracy-neutral speed target.

Priority order:

1. reproduce and instrument the operational baseline without changing behaviour;
2. localize losses by processing stage;
3. test only the architecture changes implicated by those losses;
4. screen DeDoDe v2-B and ALIKED plus LightGlue as pairwise proposal sources
   with different declared roles;
5. consider RoMa v2 on HPC after the local evidence is reviewed.

## 3. Frozen operational baseline

The following is the authoritative operational configuration supplied on
2026-08-17. File paths will be adapted to local/KINGSTON storage, but algorithmic
values must remain unchanged in the baseline arm.

```yaml
run_settings:
  run_name: limosat_dense_30000
  clear_existing_data: true
  max_cpu_cores: null

image_processor_params:
  persist_updates: true
  persist_interval: 50
  pruning_interval: 25
  temporal_window: 4
  convergence_radius_pixels: 7
  max_speed_m_per_day: 30000
  window_size: 128
  window_border: 6
  border_size: 128
  border_matched: 48
  border_interpolated: 48
  stride: 15
  octave: 5
  min_correlation: 0.30
  response_threshold: 0.001
  use_interpolation: true
  max_interpolation_time_gap_hours: 96

keypoint_detector:
  type: ORB
  orb_params:
    nfeatures: 200
    scaleFactor: 1.20
    nlevels: 5
    edgeThreshold: 24
    firstLevel: 0
    patchSize: 88
    scoreType: HARRIS_SCORE

matcher_params:
  norm_type: NORM_HAMMING2
  descriptor_distance_max: 120
  model_threshold: 15000
  lowe_ratio: 0.80
  knn_k: 8
  plot_matches: false
  geometric_model:
    type: AffineTransform
    use_model_estimation: true
    estimation_method: USAC_MAGSAC
    min_homography_inliers: 3
```

Inherited code defaults must be recorded in the run manifest. In particular,
`template_size` is not specified above and currently inherits the code default
of 16 pixels (a 33 by 33 template). `WTA_K` is not specified and inherits the
OpenCV ORB default of 2. The operational Hamming2 choice is retained in the
baseline even though ordinary Hamming is native for `WTA_K=2`; changing it would
confound baseline reproduction with an algorithmic change.

The 30 km/day limit is frozen. This run will record truth transitions outside
the gate but will not perform another day-gate sweep.

B0 instrumentation must also record the effective units of spatial thresholds.
The current convergence and new-seed proximity code queries EPSG:3413 geometry
with `convergence_radius_pixels=7`; its effective radius is therefore about
7 metres rather than 7 image pixels. This is frozen in B0 and treated as a
separate density/collision hypothesis only after baseline reproduction.

## 4. Data and validation design

### 4.1 Available evidence

The primary 2020 dataset contains:

- 70 Sentinel-1 products with frozen standard-VAE/DN-CLAHE rasters;
- 1,000 QC-accepted official Level-1 buoy/image observations;
- 748 consecutive transitions in 228 split-safe paths;
- 143 linked buoys after final tracking-fixture construction;
- January-April temporal, cadence, appearance, and spatial variation.

The existing four-sequence local benchmark was used to tune the operational
configuration. It is baseline-reproduction and regression data, not a clean
final holdout.

N-ICE2015 is materially different in region, acquisition availability, cadence,
and sea-ice regime. It must not serve as the transfer holdout for the 2020 set or
be pooled into 2020 parameter selection. It may be reported as a separate
applicability dataset. Any holdout claim for N-ICE2015 must be constructed within
N-ICE2015 itself.

### 4.2 Within-dataset split rule

Before inspecting new method scores, write a frozen split manifest for each
dataset independently.

For the 2020 full-70 set:

- use three within-dataset folds: development, confirmation, and final holdout;
- assign whole buoy IDs, including all their trajectory segments, to one fold;
- stratify folds as closely as possible by month, 200 km spatial block,
  acquisition gap, path length, and ice-concentration regime;
- store the assignment, objective, diagnostics, and random seed in a
  machine-readable manifest;
- cluster uncertainty by whole buoy trajectories and acquisition pass/image.

Images and acquisition passes may occur in more than one primary fold because
many buoys share the same SAR scene. A diagnostic graph of buoy-image-pass
linkage found only eight connected components; the largest contains 836 of the
1,000 observations. Requiring both buoy and image exclusivity would therefore
make a representative three-way split impossible. The primary holdout tests
generalization to different buoy paths under the same within-2020 image
distribution, not transfer to unseen imagery.

Create a secondary image/pass-disjoint sensitivity subset where sufficient
within-fold transitions survive. It is reported beside the primary split and is
not used to choose thresholds. The final buoy holdout remains unopened until all
method choices and thresholds are frozen.

N-ICE2015 results are reported separately and descriptively until an adequate
within-N-ICE split is demonstrated.

The 2020 split was frozen on 2026-08-17 with seed `20260817` before evaluating
any new method. It contains 333/334/333 observations, 48/48/47 whole buoys,
249/250/249 transitions, and 84 trajectory segments in each fold. All folds
contain all four months. Cadence proportions differ by at most 1.8 percentage
points and the largest spatial-block deviation from the full dataset is 2.0
percentage points. The manifest and joined tables are in
`results/arctic_tracking_next_experiment/splits/full70_2020/`.

Protocol deviation recorded 2026-08-17: an exploratory console group-by that
joined the legacy one-step retrieval table to the new fold labels accidentally
printed aggregate confirmation and final-holdout rows. No method or threshold
choice was made from those values and they are excluded from all development
analysis, but the current final fold must not be described as analyst-unseen.
Before a formal final claim, use an independently executed sealed evaluation or
freeze a replacement holdout without consulting these aggregates.

### 4.3 Evidence tiers

No single validation source is sufficient. Use three complementary tiers:

1. **Buoy truth:** point displacement, path survival, recovery, and update
   contamination in EPSG:3413 metres.
2. **Controlled real-SAR warps:** known dense translation plus physically
   plausible localized shear and divergence applied to real SAR imagery. This
   tests mechanism and MAGSAC bias with dense truth; it is not treated as
   real-world performance proof.
3. **Real sequence consistency:** forward/reverse cycle closure, local neighbour
   compatibility, deformation noise/topology, valid spatial coverage, and
   trajectory persistence.

ICESat-2, CryoSat-2, and AMSR2 remain later independent interpretation and
stratification sources. They are not needed to choose the first tracking change.

## 5. Stage-attribution baseline

The first executable change is logging and saved intermediates only. It must not
change selected matches or final outputs.

For every candidate/trajectory, record:

1. initial trajectory state and source image;
2. all descriptor candidates inside the operational search support;
3. descriptor distance, ratio-test result, response, spatial displacement, and
   source descriptor/template identity;
4. candidates entering MAGSAC;
5. MAGSAC inlier/outlier status, fitted model, residual, and rejection reason;
6. pre-pattern-matching position;
7. pattern-matching correction, correlation, rotation offset, availability, and
   hard-threshold result;
8. topology/interpolation checks and rejection reason;
9. descriptor and template written after acceptance;
10. final trajectory decision and runtime for every stage.

Every stage should preserve stable candidate IDs so the fate of a truth-near
candidate can be traced without re-matching tables heuristically.

Primary attribution questions:

- Was a truth-near candidate generated?
- Was it ranked below a distractor?
- Did MAGSAC reject it while retaining a less accurate candidate?
- Did pattern matching refine or damage it?
- Was a good candidate discarded only during path commitment?
- Did an accepted observation contaminate the next descriptor/template state?
- Did interpolation create continuity without an accurate observation?

### 5.1 Controlled deformation check

Use development images only. Apply a small preregistered family of smooth warps
to real SAR pixels and masks:

- uniform translation control;
- translation plus localized shear;
- translation plus localized divergence/convergence;
- one combined field with magnitudes bounded by observed Arctic deformation
  regimes.

Do not tune warp magnitudes to make a method fail. Store the exact float64
displacement field in pixels and metres, raster transform, masks, interpolation
method, and random seed. Compare spatial rejection probability before and after
MAGSAC against known local deformation magnitude and gradient.

## 6. Targeted LiMOSAT architecture sequence

Changes are sequential ablations, not a factorial sweep. A later arm is opened
only if the preceding attribution shows that it targets a material loss.

### B0: operational baseline

Exact supplied configuration, existing hard pattern-matching threshold, current
MAGSAC and trajectory-update behaviour.

Run two explicitly labelled views of B0. `dense_operational` uses normal
window-based spatial seeding and no buoy injection; it is authoritative for
runtime, coverage, trajectory persistence, and deformation output.
`dense_plus_buoy_probes` retains the same normal seeding but also injects
buoy-linked keypoints through the existing `keypoint_from_point` path. It is
authoritative for tracing known ice parcels through the operational stages, but
not for production runtime or candidate-population counts because the probes
slightly alter the candidate set. Exact-coordinate supplied-point retrieval
remains a separate diagnostic and is never presented as operational behaviour.

### B1: instrumented operational baseline

Logging/intermediate-output change only. B1 final trajectories must be identical
to B0. Any difference is a regression and blocks later experiments.

### A1: retained-candidate pattern matching

- retain multiple descriptor candidates until pattern matching;
- refine the same fixed number of candidates for all methods;
- preserve pre-refinement coordinates;
- initially retain the operational hard correlation threshold of 0.30;
- do not update descriptor/template memory for rejected or missing candidates.

Purpose: measure whether pattern matching can select/refine a truth-near
alternative that current early commitment loses.

### A2: soft pattern-matching evidence

Use the same A1 candidates. Treat correlation as a calibrated score rather than
an immediate rejection. Keep three states distinct:

- pattern matching succeeded with measured correlation;
- pattern matching was geometrically unavailable;
- pattern matching produced an invalid/degenerate result.

Thresholds or score transforms are selected only on development data and frozen
before confirmation.

### A3: fixed-lag reconnection and cycle evidence

- retain spatially diverse hypotheses for three acquisitions initially;
- preserve the last confirmed position and appearance memory;
- allow direct skip-frame edges from the last confirmed observation;
- add reverse and direct-cycle residuals in metres;
- allow future evidence to reinstate an alternative inside the fixed lag;
- mark intervening unobserved states as missing/provisional, never as measured;
- commit descriptor/template updates only after confirmation.

The fixed-lag length is not swept initially. Three acquisitions is the declared
pilot value; it changes only if failure attribution demonstrates a specific
longer ambiguity.

### A4: local trajectory coupling

Open only if A3 leaves collision or spatial-coherence failures.

- solve one-to-one candidate assignment in local connected components;
- compare relative displacement and edge-length changes with previous
  neighbourhood topology;
- use robust local evidence that permits real shear and divergence;
- do not impose a single rigid local transformation everywhere.

### A5: appearance statistics, conditional

Open only if remaining failures are attributable to appearance memory.

Compare a small preregistered set:

- immutable first descriptor;
- previous confirmed descriptor;
- one-frame provisional descriptor used only to propose candidates;
- Hamming medoid of confirmed observations;
- bit-stability-weighted expected Hamming cost.

Buoy truth may build teacher-forced diagnostic banks, but operational banks may
contain only information available at inference time. No selected but
unconfirmed descriptor may overwrite confirmed memory.

## 7. Learned sparse pairwise pilots

### 7.1 Rationale and declared roles

DeDoDe v2-B is the accuracy-first arm. Its v2 detector specifically addresses
clustered detections, while its detector and descriptor are decoupled. Use the
upright v2 detector, the smaller B descriptor, and the native DualSoftMax matcher.
Do not combine DeDoDe with LightGlue unless a compatible feature-specific
LightGlue model is demonstrated.

ALIKED plus LightGlue is the speed-first arm. ALIKED is a lightweight learned
detector/descriptor with sub-pixel keypoints, and LightGlue provides official
ALIKED-specific weights and adaptive sparse matching.

Both target pairwise proposal quality without requiring a dense CUDA warp.
Neither solves multi-frame state management, guarantees a detection at a buoy
or grid coordinate, or may bypass trajectory evaluation.

Primary references:

- ALIKED: https://arxiv.org/abs/2304.03608
- LightGlue: https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html
- Official inference implementation: https://github.com/cvg/LightGlue
- DeDoDe v2: https://arxiv.org/abs/2404.08928
- Official DeDoDe implementation: https://github.com/Parskatt/DeDoDe

The official LightGlue implementation includes ALIKED extraction and
ALIKED-specific matcher weights. DeDoDe is also integrated in Kornia. These
implementations do not promise Apple MPS support for every operation. Therefore
local execution begins with separate device compatibility smoke tests and must
report the actual device and any CPU fallback. CUDA benchmarks from the
papers/repositories are not used as estimates of M3 runtime.

### 7.2 Method-neutral sequence interface

Do not force learned methods through ORB's detector-window-to-fixed-grid
contract. Every candidate source instead exports a common pairwise edge table:

- method and model revision;
- source and target image IDs/times;
- source and target pixel coordinates as float64 `(column, row)`;
- source and target EPSG:3413 coordinates in metres;
- confidence/matchability and native descriptor distance where defined;
- source/target detection scores and descriptor IDs where defined;
- mask, edge, tile, and pattern-matching availability;
- forward/reverse match identity and cycle residual;
- extraction and matching runtime.

For native learned detection, persistent trajectories are built from matched
feature endpoints, not from a requirement that an arbitrary grid coordinate be
detected. At an intermediate image, the target endpoint from `A -> B` is joined
to a source endpoint from `B -> C` using pixel/map proximity, compatible feature
identity or descriptor evidence, and cycle/neighbour evidence. The join may
retain alternatives. Detector repeatability and sequence survival are measured
explicitly rather than assumed.

For existing LiMOSAT trajectories, pairwise learned matches may propose local
displacements from nearby source features. This association records the offset
between the trajectory location and learned source feature and must be evaluated
against buoy truth. A learned feature is not silently relabelled as the exact
buoy/grid point.

Spatial evenness is restored after candidate generation using deterministic
cell/window selection or Poisson-disc thinning of valid matched features. Keep
the best supported candidates per spatial cell while retaining alternates for
the graph. Report occupied-cell fraction, largest coverage hole, nearest-neighbour
distance distribution, Delaunay edge-length distribution, and persistent-track
coverage. This replaces ORB-specific window detection without requiring a
perfectly uniform grid.

DeDoDe's decoupled descriptor also permits a later supplied-point diagnostic if
native detection coverage is the measured bottleneck. It is not the primary
DeDoDe arm and is opened only after the native pipeline is evaluated.

### 7.3 Isolation and dependency rule

- Do not add PyTorch or learned-matcher packages to the production LiMOSAT
  environment initially.
- Create a separate reproducible environment after plan approval.
- Record Python, PyTorch, model revision/checksum, device, dtype, resize policy,
  and all fallback operations.
- Cache per-image learned features so sequence runtime does not repeatedly
  charge extraction for the same image.

### 7.4 Pilot data

Select cases from the 2020 development fold only:

- two operational successes as controls;
- two cases where a truth-near ORB candidate existed but the complete path lost
  it;
- two difficult appearance/noisy-image transitions;
- two high local-deformation or MAGSAC-suspect transitions identified by B1;
- at least one short multi-image sequence containing recovery after a poor
  observation.

The selection rule and case IDs are frozen before learned results are viewed.
Confirmation and final holdout cases remain unopened.

### 7.5 Frozen first learned configurations

#### DeDoDe v2-B accuracy arm

- official upright v2 detector weights;
- official descriptor-B weights;
- native DualSoftMax matcher;
- one primary feature budget selected from the device/memory smoke test before
  accuracy is viewed;
- known relative orientation applied before extraction;
- descriptor-G/DINOv2 remains out of the local pilot.

#### ALIKED plus LightGlue speed arm

- use the official ALIKED plus LightGlue weights;
- use the standard non-rotation-specific ALIKED model because relative image
  orientation is known and images will first be placed in the same geographic
  orientation;
- use one primary feature budget chosen from memory feasibility before accuracy
  is viewed;
- use standard-VAE uint8 input converted deterministically to the model's stated
  normalized tensor format;
- do not introduce RGB pseudo-colour channels; replicate the frozen SAR band only
  if the implementation requires three channels;
- retain invalid-mask provenance and exclude invalid support consistently.

One capacity reduction may be evaluated only for speed after the primary run.
There is no broad threshold or feature-count sweep.

### 7.6 Pipelines

#### D0: DeDoDe v2-B plus DualSoftMax direct

Known orientation alignment, native detection/description/matching, and direct
sub-pixel displacement. This is the learned sparse accuracy probe.

#### D1: DeDoDe v2-B plus operational pattern matching

Use D0 matches as candidate proposals, then apply the same pattern-matching
refinement and availability rules used by A1.

#### L0: ALIKED plus LightGlue direct

Known orientation alignment, ALIKED extraction, LightGlue matching, direct
sub-pixel displacement. This determines native point accuracy and the maximum
possible speed benefit from omitting pattern matching.

#### L1: ALIKED plus LightGlue plus operational pattern matching

Use learned matches as candidate proposals, then apply the same pattern-matching
refinement and availability rules used by A1. This isolates proposal quality.

#### L2: learned proposals with non-global consistency

Open only if B1 demonstrates MAGSAC deformation bias and L0/L1 supply valid
matches in the affected areas. Replace the single global rejection decision with
forward/reverse matching, local spatial consistency, and the A3 trajectory
evidence. Do not implement L2 merely because L1 contains more matches.

LightGlue is a pairwise edge producer. Its descriptors or confidence values will
not be treated as persistent ORB-compatible trajectory descriptors without a
separate demonstrated update rule.

### 7.7 Learned sparse stopping rules

Stop an arm after the development pilot if any of the following holds:

- the detector lacks adequate feature support around the buoy/reference paths;
- candidate recall is materially worse than ORB at matched spatial support;
- post-pattern-matching accuracy does not recover;
- pairwise gains disappear in a short sequence;
- local runtime or memory is clearly incompatible with its declared accuracy or
  speed role.

Advance an arm to the confirmation fold only if it does at least one of the
following:

- recovers truth-near candidates absent from operational ORB;
- improves valid coverage in B1-identified difficult/high-deformation areas
  without increasing false paths;
- preserves baseline accuracy and deformation quality while indicating a
  plausible end-to-end speedup of at least 2x;
- materially improves trajectory persistence after the same downstream checks.

## 8. Deferred RoMa v2 pilot

RoMa v2 remains the preferred qualitatively different dense-warp question, but
its representative runtime/memory assessment belongs on the available CUDA/HPC
environment. No local implementation is planned in this run.

When resumed, use exactly the B1 development cases and compare:

- direct dense warp;
- dense warp plus operational pattern matching;
- B0 operational baseline.

Query the dense warp at existing trajectory positions and retain confidence,
forward/reverse closure, runtime, memory, and native displacement. Do not reduce
RoMa v2 to another sparse detector before its dense-warp value is measured.

## 9. Metrics and units

### 9.1 Point and trajectory metrics

- projected error in EPSG:3413 metres;
- empirical CDF and fractions within 0.5, 1, 2, and 5 km;
- median, p90, p95, and maximum error;
- explicit untracked/missing fraction retained in denominators;
- catastrophic-error fraction above 50 km;
- survival through 2, 3, and 5 or more images;
- final-path error, gap count, recovery count, and time to recovery;
- forward/reverse and direct/skip cycle residuals in metres.

The 2 km measure remains a candidate-survival diagnostic, not the final
deformation-accuracy criterion.

### 9.2 Candidate-stage metrics

- truth-near candidate recall before each processing stage;
- candidate rank and margin;
- learned-detector nearest-feature coverage floor, reported separately for each
  method;
- MAGSAC acceptance as a function of known/estimated local deformation;
- pattern-matching availability, correction magnitude, correlation, and paired
  change in error;
- false-candidate and duplicate-trajectory assignment rates.

### 9.3 Deformation metrics

- divergence, shear, and total deformation at the operational output scales;
- error against controlled dense-warp truth;
- spatial noise and outlier topology in real sequences;
- valid spatial coverage overall and in high-deformation strata;
- triangle flips, neighbour-order violations, and nonphysical discontinuities;
- sensitivity to missing trajectories without silently interpolating truth.

### 9.4 Runtime and resource metrics

Measure cold and warm runs separately:

- preprocessing/input conversion;
- feature extraction per image;
- pairwise matching;
- MAGSAC/local consistency;
- pattern matching;
- graph/trajectory coordination;
- persistence and total end-to-end sequence runtime;
- peak resident memory and cached-output size;
- device, threads, dtype, image dimensions, feature count, and pair count.

A speed claim must use end-to-end sequence runtime and include feature extraction,
not matcher-only throughput.

## 10. Statistical comparison and decision rules

- Use paired comparisons on identical eligible transitions and paths.
- Resample whole buoy trajectories or grouped sequence units, never individual
  transitions as independent samples.
- Report confidence intervals and the complete denominator.
- Development selects methods/thresholds; confirmation checks the selected
  method once; final holdout is opened once after the plan is frozen.
- Do not choose a method from one month and describe another month as transfer.
- Preserve failure categories and inspect regressions as well as net gains.

An accuracy-improvement arm advances only if gains survive confirmation without
material regression in deformation quality, catastrophic paths, or spatial
coverage.

An accuracy-neutral speed arm advances if held-out point, trajectory, and
deformation metrics remain statistically and practically equivalent while
end-to-end runtime improves by at least 2x. The practical equivalence bounds must
be frozen from the operational deformation requirement before final holdout.

## 11. Outputs and storage

Bulk images, model caches, candidate tables, and learned feature tensors go on
KINGSTON. Only compact manifests, summaries, plots, and code belong in the
repository.

Proposed drive root after mount:

```text
/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/
  inputs/
  split_manifests/
  operational_baseline/
  stage_attribution/
  controlled_warps/
  graph_ablations/
  aliked_lightglue/
  caches/
```

Proposed repository summary root:

```text
results/arctic_tracking_next_experiment/
```

Every run receives a manifest containing code revision, dirty-worktree status,
config, input checksums/IDs, split ID, environment, device, random seed, timing,
and output paths. No existing result is overwritten.

## 12. Execution milestones

### M0: storage and baseline recovery

- mount KINGSTON;
- verify the full-70 archives/rasters and checksums against existing ledgers;
- locate the four-sequence operational outputs and any exact run-specific config;
- adapt paths only and reproduce a small baseline fixture.

### M1: frozen split and B0/B1 equivalence

- build and review within-dataset split manifests;
- select development pilot cases without viewing new method scores;
- implement the stage ledger;
- prove B1 outputs are identical to B0.

### M2: causal stage attribution

- run B1 on development cases;
- run controlled real-SAR deformation checks;
- quantify descriptor, MAGSAC, pattern-matching, topology, update, and path losses;
- decide which A-arm is justified.

### M3: targeted architecture ablations

- run only justified A1-A5 arms sequentially;
- select one architecture on development;
- run it once on confirmation.

### M4: learned sparse screen

- create isolated environment and device smoke tests;
- run D0/D1 and L0/L1 on the frozen development pilot;
- stop or promote one arm using the declared rules;
- do not touch final holdout.

### M5: final comparison

- freeze method/config and practical equivalence bounds;
- run selected operational, architecture, and learned arms on final within-2020
  holdout;
- generate full trajectory/deformation products only for surviving methods;
- report accuracy, coverage, failure topology, runtime, and storage together.

## 13. Confirmation checklist

Execution begins only after confirming:

- [ ] this planning document is accepted or amended;
- [ ] KINGSTON is mounted and visible;
- [ ] the supplied `limosat_dense_30000` configuration is the baseline;
- [ ] within-dataset grouped holdouts replace N-ICE-as-transfer evaluation;
- [ ] 30 km/day is frozen and no gate sweep is required;
- [ ] DeDoDe v2-B and ALIKED plus LightGlue may be installed in an isolated
      local environment;
- [ ] RoMa v2 is deferred to a later HPC session;
- [ ] the 2x threshold is acceptable for an accuracy-neutral speed result;
- [ ] practical deformation-equivalence bounds are identified before final
      holdout.

## 14. Change log

- 2026-08-17: initial draft. Incorporated the operational
  `limosat_dense_30000` config, within-dataset holdout requirement, frozen
  30 km/day gate, local ALIKED plus LightGlue pilot, and deferred RoMa v2 HPC
  work.
- 2026-08-17: made whole-buoy within-2020 splits primary after the mounted-data
  linkage audit showed an 836-observation buoy-image-pass component; added an
  image-disjoint sensitivity subset. Added a method-neutral pairwise/trajectory
  interface, explicit spatial-evenness metrics, DeDoDe v2-B as the accuracy arm,
  and ALIKED plus LightGlue as the speed arm.
- 2026-08-17: completed development-stage attribution and the first learned
  native-crop screen. Added the execution evidence and resulting decisions in
  Section 15.
- 2026-08-17: isolated numerical instability in metre-coordinate homography
  estimation, completed three full-70 geometric pilots, and froze the
  kilometre-coordinate homography as the only confirmation candidate.
- 2026-08-17: invalidated the first ALIKED screen after finding randomly
  initialized extractor weights and insufficient target-crop support; completed
  corrected 2,048- and 1,024-feature pretrained pilots and advanced only the
  2,048-feature method as a selective local-proposal fallback.
- 2026-08-17: added guarded sub-pixel correlation refinement and fractional
  state/template handling as a bounded implementation task; made the selected
  kilometre-conditioned homography the base for subsequent candidate tests; and
  added direct ALIKED without pattern matching as an explicit accuracy/runtime
  arm.
- 2026-08-18: completed complete-footprint ALIKED fields, exact anchored feature
  caching, the physics-subset speed arm, and controlled rigid/affine/lead warps.
  Selected nearest-12 consensus with flipped-node rejection as the provisional
  learned raw-measurement policy; retained adaptive-eight as its control.
- 2026-08-18: completed a frozen 26-path, five-transition March sequence,
  repeat/jump buoy-QC sensitivity, and ORB comparison. Stopped graph
  reconnection and guarded local-affine smoothing for lack of causal evidence.
  Recorded and fail-closed an accidental mixed-split aggregate evaluation; the
  original final holdout must be replaced for independent testing.

## 15. Development execution record and decisions

### 15.1 Operational and supplied-point evidence

- Dense B0 and instrumented dense B1 produce the identical final SHA256
  `efb3030ec8afe987f9d34d4d1244694c055f1f808690a349eda65650e6e118bc`.
- The corrected 70-image exact-buoy development diagnostic linked all 333
  observations and all 249 expected one-step transitions. It generated a
  descriptor candidate on 222/249 transitions; 199/249 had a candidate within
  2 km. Interpolation produced a position for 240/249.
- Only 135/249 probe trajectory identities reached final acceptance, but this
  initially overstated failure. Seventy-five were deliberately pruned by
  convergence; the retained replacement trajectory reached the target image in
  68 cases and was within 2 km in 66. Following replacements gives 203/249
  represented measurements and 190/249 within 2 km.
- Exact probe injection is not an operational score. It creates nearby
  trajectories in the dense field: 29/164 pattern-accepted probes were then
  rejected by the topology stage, although 28 of those rejected pattern
  positions were within 2 km of truth. This is evidence that the diagnostic
  perturbs density/topology, not evidence to disable topology operationally.

### 15.2 Unchanged dense field at buoys

The unchanged dense B0 field was evaluated without inserting buoy points.
Displacements were transferred from trajectories present at both endpoints.

- Nearest surviving displacement within 10 km: 222/249 available, 207/249
  within 2 km, 93.2% within 2 km among available estimates, and 107 m median
  error.
- Inverse-distance averaging of up to four surviving vectors within the same
  10 km radius preserves 222/249 availability and raises the within-2-km count
  to 210/249 (94.6% among available), while lowering median error to 73 m.
- Of the 27 transitions without a surviving vector inside 10 km, 13 had no
  source point seeded in that radius and 14 had source points but none survived
  to the target. These are separate spatial-sampling and tracking-survival
  problems.
- The 721-to-731 pair dominates survival loss: 16/18 transitions had a source
  point within 10 km, but only 3/18 had a surviving local vector. Exact supplied
  ORB represented 12/18 on the same pair, so this pair is the next targeted
  candidate/trajectory test rather than a general descriptor sweep.

The local average is a development-selected output estimator, not yet a held-out
gain. Its 10 km radius and four-neighbour cap are frozen before confirmation.

### 15.3 Candidate and speed pilots

- Adding one physics-local ORB fallback candidate to the two-image smoke raised
  pre-model candidates from 1,248 to 1,779 but reduced final accepted points
  from 817 to 815. The single-fallback arm is stopped; future candidate work
  must retain alternatives through commitment rather than select one more local
  descriptor minimum early.
- A two-image profile attributes 3.99 seconds to 3,844 OpenCV template
  correlation calls and 1.19 seconds to 2,375 ORB window detections. OpenCV
  thread counts 1, 2, 4, and 8 produced identical hashes; the best timing gain
  over the current eight threads was only 1.2%, so thread tuning is stopped.

### 15.4 Corrected learned sparse pilot

The original three-case learned screen is invalid. The ALIKED pilot constructed
`ALIKED(...)`, which creates a randomly initialized network in the installed
Kornia version, instead of calling `ALIKED.from_pretrained(...)`. An
identical-image test consequently produced zero standard-threshold matches.
Furthermore, the true target patch was 13 pixels outside one 512-pixel crop and
only 1.9 pixels inside another. These crop errors also invalidate the two
high-drift DeDoDe results. DeDoDe remains deferred rather than rejected by that
screen.

The corrected ALIKED development pilot used official pretrained ALIKED-N16 and
ALIKED-specific LightGlue weights, frozen VAE input, 768-pixel crops shifted
inside image bounds using only the source position, and 2,048 features. Eight
cases were frozen before viewing the corrected results: two ORB controls, the
known pattern-correlation and topology losses, one no-descriptor case, one
convergence replacement, and two consecutive short-sequence steps.

- The integration invariant produced 2,044 same-index standard-threshold
  matches from 2,048 keypoints when an image was matched to itself.
- Every case contained at least one physics-valid ALIKED vector originating
  within 10 km of the buoy and producing no more than 2 km endpoint error. The
  closest local vector was correct in 8/8 cases, with 230 m median and 1.58 km
  maximum endpoint error. The highest-confidence local vector was also correct
  in 8/8, with 153 m median and 1.45 km maximum error.
- Across the eight cases, 555/556 physics-valid matches originating within
  10 km produced endpoints within 2 km of buoy truth. The median matched-cell
  occupancy was 100% on a 4-by-4 grid and the minimum was 62.5%.
- Only 6/8 cases had a matched feature originating within 2 km of the buoy.
  ALIKED is therefore not yet a direct substitute for forced ORB extraction at
  each deformation-grid point. It is promising as a local displacement proposal
  that can be associated with a nearby grid point and refined there by the
  operational pattern matcher.
- The eight selected exact supplied-point ORB probes generated a truth-near
  descriptor candidate in 5/8 cases and produced a represented correct final
  measurement in 5/8. This targeted comparison is evidence of complementary
  failure recovery, not an aggregate claim that ALIKED is more accurate than
  ORB.

The 2,048-feature CPU path took 1.59 seconds for two-crop extraction and 0.51
seconds for LightGlue, or 2.10 seconds per pair. The existing full-70
supplied-point ORB retrieval benchmark took about 0.30 seconds per buoy/image
pair while searching a substantially larger target grid, so corrected ALIKED is
approximately seven times slower on the present local CPU workload. A single
allowed 1,024-feature reduction lowered ALIKED to 1.42 seconds per pair (32%
faster than 2,048), but lost the difficult topology case and reduced tight
source support from 6/8 to 5/8. The 2,048-feature configuration remains the
accuracy pilot; neither configuration is an operational CPU speed replacement.

ALIKED advances only to a selective fallback pilot: retain standard LightGlue
thresholding, use local confidence and spatial consistency to propose one or
more displacement hypotheses where ORB has no reliable direct candidate, and
apply the proposed displacement to the exact grid point before the existing
pattern-matching refinement. Do not tile every full scene on CPU or replace the
ORB grid until this fallback is shown to improve a larger stratified development
sample and short-sequence survival.

### 15.5 Geometric-filter attribution

The 721-to-731 trace found that descriptor recognition was not the dominant
failure. All 27 rejected buoy-near candidate instances were within 2 km of
truth, but the hardcoded projective homography assigned them residuals of
15.2-21.6 km and rejected them. The supplied config names `AffineTransform`,
while the matcher always called `findHomography` and then wrapped the resulting
projective matrix as an affine transform.

Three opt-in alternatives were tested without changing the legacy default:

- the configured global affine estimator recovered the target pair, but its
  full-70 run lost four previously correct 10 km development cases and reduced
  longer trajectory survival;
- the union of homography and affine inliers retained more legacy candidates,
  but still lost three previously correct 10 km cases;
- fitting the same homography after expressing EPSG:3413 coordinates and the
  15 km residual threshold in kilometres recovered all 43 model outliers on the
  motivating match group. Across the fixed B1 candidate audit it increased
  inliers in 39/115 estimable image groups and decreased only four groups, by
  one inlier each. This establishes numerical scale, rather than a necessary
  change of geometric model, as the narrower cause.

### 15.6 Frozen development candidate

The full-70 kilometre-coordinate homography run is the only candidate advanced
to confirmation. ORB, frozen VAE preprocessing, descriptor matching, motion
limit, pattern matching, interpolation, topology checks, and output estimator
remain unchanged.

- At the frozen 10 km local-average evaluation, availability rose from 222/249
  to 242/249 and correct estimates rose from 210/249 to 229/249. The paired
  result contains 19 gains and zero losses.
- At 50 km, availability rose from 248/249 to 249/249 while all 234 correct
  baseline cases were retained.
- The correct gains are concentrated in the January 721-to-731-to-740 short
  sequence. Other development strata were unchanged in correct count, although
  one additional March estimate became available. Confirmation must therefore
  be interpreted as a check on new buoys within this 2020 dataset, not as proof
  of transfer to another acquisition regime.
- End-to-end runtime was 482.9 seconds versus 489.3 seconds for B0. Persisted
  rows rose from 94,890 to 96,539; trajectories of length at least two, three,
  and four rose by 608, 537, and 127 respectively. Length-at-least-five fell by
  four (726 to 722), so no universal long-path improvement is claimed.
- The unchanged legacy path still reproduces the expected two-image SHA256
  `29890d5435c62512587330fe50d9e7f8f00dbf78219c3cc6b234b67070778f4b`.

The confirmation decision is now frozen: compare B0 and the
kilometre-coordinate homography once on the confirmation buoys with the same
10 km/up-to-four-vector estimator. Do not inspect the final split.

### 15.7 Confirmation result and next gate

The frozen candidate passed the one allowed confirmation evaluation on 250
transitions from different buoys in the same 2020 dataset.

- Ten-kilometre local-average availability increased from 225/250 to 244/250;
  correct estimates increased from 218/250 to 236/250. The paired comparison
  has 18 gains and zero losses.
- The 50 km correct count remained 240/250. Candidate availability was 249/250
  versus 250/250 because one incorrect baseline estimate became unavailable.
- As in development, all correct-count gains occur in the January 0-3 hour
  cadence stratum. This confirms the result across held-out buoys, but not
  across an independent image sequence or year.

The next gate is output-quality validation, not another descriptor or model
sweep: expose the coordinate scale explicitly in a candidate LiMOSAT config,
compare deformation/topology distributions on the baseline and selected dense
fields, and prepare an independent final evaluator. The final split remains
unopened here because legacy final aggregates were accidentally exposed earlier
in the session; an independent evaluation is preferable to another nominally
blind score in this process.

### 15.8 Selected forward path: geometric scale and pattern-matching cost

The kilometre-coordinate homography is retained as the implementation path to
move forward with. The coordinate scale must be exposed explicitly in the
LiMOSAT config, and the 15 km MAGSAC residual threshold must not be treated as
revalidated merely because its physical value was preserved. It was tuned when
the estimator received metre-scale coordinates, which this experiment showed
to be numerically unstable. Before operational adoption, inspect the residual
distribution and compare a narrow, preregistered threshold bracket on the
development data only (initially 10, 15, and 20 km). Freeze that decision before
an independent sequence/year evaluation. Do not choose the threshold to reduce
runtime: it is a global-model residual tolerance, not the 30 km/day physical
motion limit, and making it too tight can suppress real local deformation.

The full-70 stage audit also localizes the pattern-matching workload. With the
selected kilometre-coordinate homography, 61,608 proposals reached pattern
matching. The 0.30 correlation threshold rejected 1,844 (2.99%) and unavailable
windows rejected 142 (0.23%). The result is strongly conditional on proposal
type:

- direct descriptor matches: 563/52,670 (1.07%) failed the correlation
  threshold and 26 (0.05%) were unavailable;
- interpolated proposals: 1,281/8,938 (14.33%) failed the correlation threshold
  and 116 (1.30%) were unavailable.

For comparison, the instrumented baseline had 538/52,150 (1.03%) direct and
937/7,526 (12.45%) interpolated correlation-threshold failures. The selected
geometry therefore added 1,932 pattern-matching evaluations but also added
1,461 correlation-accepted positions and 1,266 final accepted positions. The
extra work is productive overall; most of its failures come from the additional
interpolated proposals rather than weaker direct ORB matches.

This evidence does not support lowering the 0.30 correlation threshold or
reducing the number of direct candidates solely to avoid pattern matching.
Direct proposals already pass at about 98.9%, so indiscriminate pruning would
mostly remove usable measurements. Pattern matching is not only a sub-pixel
refiner for interpolated points; its 14.3% rejection rate is providing material
validation there.

The targeted speed pilot is therefore adaptive search extent while retaining
the same correlation check:

1. Measure the pattern-matching correction-vector distribution separately for
   direct and interpolated proposals, including buoy-error dependence.
2. Give high-confidence direct descriptor matches a smaller matched search
   border, while retaining the current 48-pixel border for interpolated and
   low-confidence proposals.
3. Compare runtime, buoy paired gains/losses, final accepted positions,
   trajectory survival, topology rejection, and deformation distributions.
4. Accept the smaller border only if buoy accuracy and practical deformation
   outputs remain equivalent; otherwise retain the current pattern matcher.

A future descriptor should be credited with a pattern-matching speed gain only
if it either replaces interpolated proposals with correct direct matches or
supports a demonstrably smaller refinement search area. Producing fewer direct
matches is not itself a useful speed result.

### 15.9 North-up ALIKED versus supplied-point ORB

The corrected eight-case ALIKED screen was expanded without opening confirmation
or final data. The primary development panel contains one deterministic buoy
transition from each of 39 eligible acquisition-pair clusters. A separate
spatial sensitivity panel contains one transition per acquisition-pair by
200 km source block: 57 spatial units, still clustered within the same 39 image
pairs. Seven rare ORB failures are reported as a challenge panel and are not
allowed to change the representative aggregate.

Both methods use the standard VAE image. ALIKED receives one channel in
`[0, 1]`; Kornia broadcasts this internally. On the first frozen case, explicit
one-channel and manually repeated three-channel inputs produced exactly equal
keypoints, descriptors, and scores. Repeating the VAE channel is therefore
unnecessary and adds no information.

The learned-feature comparison also corrects four confounders in the earlier
screen: both images are resampled north-up in EPSG:3413, the target extent grows
with the 30 km/day physical limit and elapsed time, target features are extracted
in overlapping 512-pixel tiles at constant density, and detections whose 16-pixel
support intersects invalid data are removed. The accelerated geolocation grid
was checked against 49 exact GDAL transforms in each image; maximum discrepancy
was 0.00053 native pixels. All 39 primary and all 57 spatial-sensitivity truths
fell inside their physics-sized target tiles.

Primary results, with correctness defined before the run as endpoint error no
greater than 2 km:

| Proposal | available | correct / 39 | median error when available | p90 error |
|---|---:|---:|---:|---:|
| ALIKED, inverse-distance average of four closest local vectors | 36 | 34 (87.2%) | 46 m | 282 m |
| supplied-point ORB proposal | 30 | 30 (76.9%) | 473 m | 757 m |
| ALIKED after current pattern matching | 35 | 33 (84.6%) | 64 m | 283 m |
| ORB after current pattern matching | 29 | 29 (74.4%) | 58 m | 116 m |

Against the ORB proposal, the primary ALIKED local average has seven paired
gains and three losses, a 10.3 percentage-point difference. The image-pair
bootstrap interval is -5.1 to 25.6 percentage points and the exact paired
two-sided p-value is 0.344. This is promising evidence, not a conclusive
aggregate win. The spatial sensitivity result is consistent: ALIKED is correct
on 48/57 units versus 43/57 for ORB. Equal-weighting the 39 image-pair clusters
gives a 12.8-point difference with a -0.9 to 26.9 point cluster-bootstrap
interval.

The challenge panel is useful attribution but not prevalence evidence. ALIKED
produced a correct raw proposal in all seven rare ORB failure cases. Across the
primary panel, the gain is strongest at relative native rotations above 20
degrees (12/12 correct ALIKED versus 9/12 ORB), but north-up orientation was not
varied independently of crop coverage, tiling, and invalid-mask handling, so
this does not isolate rotation as the sole cause.

The local-average policy is spatially defensible: its four selected source
features have a median maximum distance of 1.12 km and a maximum of 1.45 km.
Selecting the highest LightGlue-confidence vector is correct in 36/39 primary
cases, but its source feature is a median 8.44 km from the buoy. That result is
an upper bound on translation recognition, not an acceptable deformation-grid
policy: borrowing a remote coherent vector could erase real shear or ridge
signals. The next policy test must constrain confidence selection to tight
source support and retain vector-coherence diagnostics.

Current wide-border pattern matching is not automatically beneficial after a
precise ALIKED proposal. It changes 34 correct raw ALIKED cases to 33 correct
accepted cases: two correct proposals are lost or made incorrect and one
incorrect proposal is recovered. One 54 m proposal was moved to a false 4.52 km
peak at correlation 0.382. The current implementation locates an integer
correlation maximum; it does not perform sub-pixel peak interpolation. ALIKED
should therefore be tested with no pattern step or a tightly bounded fractional
refinement rather than inheriting the 48-pixel ORB/interpolation search blindly.

The correction distribution supports an adaptive rather than universal border.
Of 35 correlation-accepted ALIKED refinements, 32 moved no more than two native
pixels; the remaining three moved 53-58 pixels and contained one useful recovery
and two incorrect results. Across the full selected ORB geometry run, accepted
direct proposals have 480 m median, 860 m p95, and 1.91 km p99 pattern
corrections; 97.5% move no more than 1 km. Interpolated proposals are materially
different, with 1.48 km median and 4.41 km p95 corrections. The next pattern
experiment therefore keeps 48 pixels for interpolated proposals, tests 12, 16,
and 24 pixels for direct ORB proposals, and tests 2, 4, and 8 pixels for
high-confidence/coherent ALIKED proposals. Each arm adds a fractional
correlation-peak fit and reports peak sharpness and second-peak ambiguity. The
production default remains unchanged until this paired test is complete.

CPU feasibility remains the principal disadvantage. Physics-complete tiled
ALIKED takes a median 7.99 seconds per local pair (3.59 seconds extraction,
2.53 seconds LightGlue, and 1.55 seconds north-up resampling), versus about 0.30
seconds for the earlier supplied-point ORB retrieval benchmark. These workloads
are not identical enough for an operational speed ratio, but ALIKED is not a
local-CPU speed replacement. Any dense pilot must cache per-image/tile features
and use the available GPU/HPC path.

The next bounded development experiment is candidate-policy replay, not another
descriptor sweep: persist the tight ALIKED match vectors once, compare the
four-nearest local average with (a) highest confidence restricted to 2 km and
(b) a preregistered vector-consensus rule, then carry only the best spatially
local rule into short sequential tracking. Confirmation and final data remain
unopened for this learned-feature work.

### 15.10 Sub-pixel pattern matching and fractional state consistency

This is a bounded foundational task to complete before interpreting the next
pattern-matching border or ORB-versus-ALIKED sequence comparison. At the native
approximately 80 m pixel spacing, an integer correlation peak can contribute up
to about half a pixel of localization error per axis even when the correct peak
has been selected. This principally limits displacement and deformation
precision. It may also affect sequence survival when the estimated position is
used to extract the next appearance state.

The current implementation has a second issue that must be isolated in the same
test. Pattern matching truncates the proposed floating-point pixel coordinate to
an integer search centre, but adds the selected integer correction back to the
original floating-point proposal. Template extraction separately casts its
centre to an integer. A fractional ALIKED proposal can therefore retain a
fraction that was not estimated from the correlation surface, while the next
template does not preserve the same fractional centre. ORB grid proposals are
more nearly integer-valued, so this coordinate convention may also bias a direct
ORB-versus-ALIKED comparison.

Implement and validate the following without changing the legacy default:

1. Define one explicit pixel-coordinate convention for the response-map origin,
   integer peak, fractional peak offset, corrected map coordinate, and projected
   EPSG:3413 coordinate. Add analytic tests that would fail under truncation,
   rounding, or row/column sign mistakes.
2. Add a configured peak-refinement method with `none` as the exact legacy arm,
   a full two-dimensional quadratic fit over the winning 3-by-3 correlation
   neighbourhood as the fast candidate, and bounded continuous masked NCC as a
   validation reference rather than an assumed production default.
3. Require a finite interior neighbourhood, a negative-definite fitted Hessian,
   a fitted maximum within one pixel of the integer peak, and an explicit
   integer fallback. Record the fractional correction, fit residual, curvature,
   search-boundary distance, chosen rotation, and peak-to-sidelobe ambiguity.
4. Add a separate configured template-sampling arm: exact legacy integer slicing
   versus interpolation at the corrected fractional centre. Do not claim a
   sequence-stability benefit unless the fractional position is carried through
   template/state extraction rather than only written to the output geometry.
5. Validate on known analytic correlation surfaces and controlled fractional
   translations of real VAE ice patches. Include masks, rotations, intensity
   change, noisy/low-texture patches, and an interpolation method different from
   the reference estimator used to generate the shift.
6. On the frozen development panels, hold descriptor proposals, geometry model,
   search border, correlation threshold, and candidate policy fixed. Compare
   integer, quadratic, and continuous-reference endpoint error, fallback rate,
   catastrophic changes, runtime, forward/reverse closure, and deformation
   noise. Then test integer versus fractional template sampling on short
   sequences.

The first bounded replay is complete on the 36 ALIKED-available representative
development cases. Correcting the response-map coordinate convention and using
a bilinearly centred source template retains all 34 correct accepted cases.
The guarded 2-D quadratic fit gives 18.4 m median and 65.7 m p90 endpoint error,
versus 42.1 m and 93.6 m for the aligned integer peak and 63.6 m and 282.8 m
for the legacy 48-pixel matcher. A separately optimized continuous NCC
reference accepts the same 34 correct cases but is slightly worse overall
(19.9 m median and 83.7 m p90). The quadratic fit is therefore retained as the
production candidate; continuous NCC remains a validation reference. The
4-pixel and 8-pixel quadratic arms are numerically equivalent here, so 4 pixels
is sufficient for the tightly localized ALIKED proposal arm.

This replay also isolates fractional template sampling from fractional peak
fitting. With the same aligned integer peak, bilinear source templates improve
the median from 74.2 m to 42.1 m. Adding the quadratic fit then improves it to
18.4 m. Both coordinate/state consistency and peak fitting matter; reporting
only the latter would misattribute most of the gain.

This task does not address a correctly scored correlation peak in the wrong
spatial basin. The 4-5 km ALIKED pattern-matching jumps remain candidate/search
commitment failures and are handled by the adaptive-border and no-pattern arms.
The isolated quadratic arithmetic is negligible relative to template
correlation, but integrated runtime must still be recorded.

### 15.11 Geometry base for all subsequent tests

The kilometre-coordinate change is selected by development and confirmation
evidence but is not yet a complete operational implementation. Complete it
before running the next candidate-policy, pattern-matching, or sequence arms:

- expose the homography input scale in the LiMOSAT config and run manifest;
- retain EPSG:3413 metres for stored positions, errors, physical limits, and
  outputs--kilometres are only the numerically conditioned estimator space;
- preserve an explicit legacy metre-coordinate mode for exact reproduction;
- convert the configured physical residual threshold into estimator units in
  one tested location rather than through scattered implicit conversions;
- record fitted-model residuals in both estimator units and metres;
- run the preregistered 10, 15, and 20 km residual-threshold diagnostic on
  development only, freeze the result, and do not reopen confirmation to tune it;
- use the selected kilometre-conditioned configuration as the common geometry
  base for all subsequent ORB and ALIKED paired tests.

This implementation is complete in commit `147fced`. The physical threshold
diagnostic is also complete on the same 249 development transitions. For the
primary local-average estimate within 10 km, 15 km retains 242 estimates and
229 correct estimates; 10 km retains 240 and 227, while 20 km retains 240 and
225. The 15 km threshold is frozen for subsequent development tests. This does
not promote it to a universally optimal physical tolerance; it is the selected
value for the current Arctic operational configuration and must be checked on
an independent sequence after the remaining choices are frozen.

### 15.12 Full-70 pattern-matching window selection

The guarded quadratic peak fit and bilinearly centred template extraction are
implemented in commit `c055ceb`; their explicit configuration is documented in
commit `2e02f04`. A full-70 development sweep then held the 1 km homography
coordinate scale, 15 km physical MAGSAC threshold, correlation threshold, ORB
configuration, images, and buoy transitions fixed.

For direct descriptor matches, 12, 16, 24, and 48 pixel search borders all gave
approximately 50.1 m median local-average buoy error within 10 km. The 24-pixel
arm was the best accuracy/coverage compromise: 241/249 transitions available,
230/249 correct within 2 km, 50.1 m median error, 1.28 km p90 error, and 58,331
accepted links. It gained one correct transition over both the 48-pixel arm and
the legacy baseline while retaining all but 24 of the 48-pixel arm's links. In
a clean back-to-back run it took 405 seconds, versus 395 seconds for 16 pixels.
The 10-second saving at 16 pixels cost 228 links, so 24 pixels is frozen for the
direct arm.

With the direct border fixed at 24 pixels, interpolated borders of 32, 48, and
64 pixels were compared. The 32-pixel arm retained the same 230 correct buoy
transitions and ran in 385 seconds, but lost 429 accepted links and 175 paths
with at least two observations relative to 48 pixels. The 64-pixel arm gained
177 links but lost two correct buoy transitions and worsened p90 error to
1.46 km, with no material runtime benefit. The selected development
configuration is therefore:

```yaml
image_processor_params:
  border_matched: 24
  border_interpolated: 48
  pattern_matching_subpixel_method: quadratic
  template_sampling: bilinear
matcher_params:
  model_threshold: 15000
  geometric_model:
    coordinate_scale_m: 1000.0
```

Relative to the legacy full-70 baseline, the selected arm keeps 230 versus 229
primary correct transitions, reduces median error from 71.3 m to 50.1 m
(29.7%), and has 58,331 versus 58,706 links (-0.64%). It also has 9,681 versus
9,673 paths with at least four observations. The loss is concentrated in
shorter path retention rather than the longest observed paths. These are
development results; the configuration remains subject to independent
sequence evaluation and external deformation validation.

### 15.13 ALIKED local consensus and propagated paths

The corrected ALIKED run now persists all 54,168 deduplicated match vectors,
including projected source and target positions, displacement, LightGlue
score, source distance, speed, and the preregistered physics gate. This allowed
the candidate rule to be tested without recomputing features or changing the
frozen 39-case representative panel.

The 1 km displacement-consensus cluster among matches originating within 2 km
of the current point is materially better than the original four-nearest
inverse-distance average. It is available in 36/39 representative cases and
all 36 are correct within 2 km, with 22.6 m median and 161 m p90 error. The
four-nearest rule has the same availability but only 34 correct cases, 46.0 m
median, and 282 m p90 error. Highest-confidence selection also has 36 correct
cases but is less precise than consensus. The consensus rule uses the current
source state, LightGlue scores, and agreement among displacement vectors; it
does not use the target buoy position.

Tight 4-pixel quadratic/bilinear refinement preserves all 36 correct consensus
proposals and improves the one-step distribution to 17.9 m median and 63.5 m
p90. Direct consensus therefore remains a valid no-pattern-matching speed arm,
while q4 is the precision arm.

The sequential check was then made progressively less truth-dependent. A
cached-vector replay used each estimated endpoint as the next source state. A
separate full-crop run resampled every following source and target SAR patch
around that propagated estimate and reran ALIKED plus LightGlue. Across 27
transitions forming 11 consecutive buoy paths, direct consensus completed 10
paths; all 24 available steps remained within 2 km. Median final error across
complete paths was 32.8 m, median path maximum error was 36.4 m, and the largest
carried source-state error on a complete path was 228 m.

The one failed three-step path has no physics-valid ALIKED match at any step,
including under truth-reinitialized extraction. Its first step has 19 matches
originating within 10 km, but every one exceeds the 30 km/day speed gate; its
second step has only 109 valid source features and no local match; and its third
step's nearest matched source feature is 11.8 km away. This is a feature/match
coverage failure rather than accumulated state error or a consensus-policy
failure. It motivates spatially balanced feature allocation or a bounded
fallback, not wider pattern matching.

On the fully propagated paths, q4 alone accepts 23/24 direct proposals. A
learned-arm-specific fallback--use q4 when correlation accepts it, otherwise
retain the direct consensus endpoint--keeps all 24 correct steps, gives 21.6 m
median and 78.0 m p90 error, and completes the same 10/11 paths with 28.3 m
median final error. It adds 2.13 seconds per evaluated transition on local CPU.
Direct consensus is therefore the speed candidate and q4-with-direct-fallback
the precision candidate. This fallback is not applied to ORB, where a failed
pattern check can still be important evidence against the descriptor proposal.

The propagated local experiment is promising but not a dense deformation
test. Median local CPU costs were 1.60 seconds for north-up resampling, 3.66
seconds for ALIKED extraction, and 2.59 seconds for LightGlue matching per case,
before optional q4 refinement. The next learned-feature test must measure
spatial coverage and evenness across image-pair spatial blocks, then determine
whether feature caching, GPU execution, or selective fallback makes the method
operationally feasible.

That spatial-block check is now complete on 57 deterministic representative
cases. The frozen consensus policy is available for 52 and correct for 50,
versus 48 correct for the original four-nearest policy. Consensus reduces the
median from 54.4 m to 25.9 m and p90 from 526 m to 251 m. Tight q4 preserves
the 50 correct cases and improves the median to 19.0 m, although the two
remaining discrepancies stay in coherent wrong basins. Thus the perfect 36/36
one-per-image-pair result was optimistic, but the policy improvement survives
spatial balancing.

Both remaining discrepancies lack any correct ALIKED vector within the 2 km
source neighbourhood. ORB and high-correlation pattern matching converge to
nearly the same wrong displacement, so these are not failures q4 can repair.
Buoy QC explains part of this. One SIMB3 transition already has
`track_qc_pass: false` because its source and target context gaps are about
20 hours. The other transition passes the old frame-local QC but its raw IABP
record repeats the exact same latitude/longitude for 20 hours, followed by a
421 km/day one-hour jump. The SAR estimates follow the spatially coherent ice
motion rather than the frozen telemetry plateau.

A reproducible repeat-position diagnostic now records exact-fix run length and
the speeds immediately before and after each run. Using the project's existing
6-hour maximum gap and 100 km/day maximum track-speed values as a sensitivity
diagnostic flags only that stale-jump transition; it does not flag a correct
case whose position repeats for 9 hours with only 0.1 km/day adjacent motion.
Across the 57 spatial cases, the existing source-and-target track QC plus this
stale-jump diagnostic retains 52 labels. This combined-label-QC stratum has 47
available consensus estimates and all 47 are correct, with 24.3 m median and
158 m p90 direct error. Q4 gives 18.6 m median and 62.3 m p90. On the same 52
labels, ORB plus pattern matching has 39 available/correct estimates, 64.7 m
median, and 116 m p90. The paired comparison is eight ALIKED-only correct,
zero ORB-only correct, and 39 both correct.

The combined-label-QC result is explicitly a diagnostic sensitivity analysis,
not a post-hoc replacement primary endpoint. It shows that buoy telemetry
staleness is a real source of apparent matching error and must become a frozen,
method-independent data-quality rule before independent evaluation. It also
changes the interpretation of ALIKED: with spatial consensus and tight PM it
is currently more accurate and more available than supplied-point ORB around
the validated buoy locations, but remains substantially slower on local CPU
and has not yet demonstrated dense, spatially even deformation coverage.

The feature-budget comparison used the identical 63-case spatial panel. Reducing
ALIKED from 2,048 to 1,024 features lowers summed local case time from 379 to
233 seconds (-38.7%) and median case time from 7.09 to 3.80 seconds (-46.4%).
On the combined-label-QC subset, 1,024-feature q4-with-direct-fallback retains
all 47 correct available cases and the same 18.6 m median as 2,048 features,
but its p90 increases from 62 to 109 m. Reducing again to 512 features saves
only another 8% of summed time, loses two centre estimates and two correct
cases, and is rejected as dominated. The retained learned arms are therefore
2,048 features for accuracy and 1,024 features plus q4 fallback for compute.

A cached-vector spatial-evenness diagnostic then queried a 5 by 5 grid at 4 km
spacing around every one of the 57 representative buoy/image-pair blocks. It
does not use buoy truth away from the centre: it measures proposal availability
and local vector roughness only. The 2,048-feature arm covers 1,296/1,425
queries (90.9%), with 51/57 blocks fully covered and 47 m median displacement
difference between adjacent 4 km queries. The 1,024-feature arm covers
1,288/1,425 (90.4%), with 45 blocks fully covered and 65 m median adjacent
difference. All eight queries lost by 1,024 features are covered by the
2,048-feature arm. The 512-feature arm falls to 1,209/1,425 (84.8%), only
10 fully covered blocks, and 90 m median adjacent difference. Four blocks have
no coverage at any budget, which feature-count tuning cannot repair.

This is evidence that 1,024 features is a credible compute arm, but not yet a
dense deformation result: the 16 km local grids overlap the buoy-centred crops,
and adjacent-vector differences combine real deformation with matching noise.
The next pilot must allocate features over a complete image-pair footprint,
build a spatially regular displacement field, and evaluate coverage, triangle
topology/closure, and deformation distributions before independent adoption.

### 15.14 Complete-footprint ALIKED fields

A complete-footprint CPU pilot now allocates ALIKED features evenly in
non-overlapping 35.8 km cores extracted from overlapping 512-pixel north-up
tiles. Target tiles are matched only when reachable under 30 km/day, so the
search grows with acquisition interval. Matches are cached once and replayed
onto a regular 4 km field without using buoy locations. The field is compared
with the selected dense ORB field on identical nodes; buoys are used only as
labelled point checks.

Two development pairs were run. Images 721 to 731 span 1.63 hours and about
120,751 square kilometres, using 112 tiles per image. The 1,024-feature run
took 172 seconds on local CPU: 69 seconds for extraction and 90 seconds for
matching. Images 740 to 849 span 21.41 hours and about 136,792 square
kilometres, using 133 source and 134 target tiles. It took 279 seconds: 81
seconds extraction and 183 seconds matching. The increase is caused by the
physics search expanding from about 2 km to 27 km. This confirms that target
search/matching, not feature extraction, is the main long-gap compute target.

With the original fixed 2 km support, the short pair covers 6,995/7,550 grid
nodes (92.6%), has no flipped triangles, and is correct for all seven available
buoy cases with 37 m median and 175 m maximum error. The long pair covers
7,239/8,548 nodes (84.7%) and is correct for all three buoy cases with 102 m
median and 126 m maximum error, but contains 34 flipped triangles. ALIKED is
also rougher than ORB: adjacent 4 km vectors differ by 85 m median and 270 m
p90 on the long pair, versus 39 m and 117 m for ORB.

The failure is support-limited rather than a general descriptor failure. At
2 km, 1,303 short-pair nodes use only one or two agreeing vectors. Increasing a
single fixed radius improves mean agreement and coverage, but on the long pair
causes neighbouring nodes to switch between coherent hypotheses: a fixed 6 km
radius increases flipped triangles from 34 to 145 and produces a 1.61 p99
triangle-area ratio. A strict adaptive rule performs better. It tests 2, 3, 4,
then 6 km and publishes the first hypothesis supported by at least eight
agreeing vectors; if eight are never found, the node remains unavailable.

That accuracy-first rule gives 96.7% short-pair coverage, no flips, and 8/8
available buoy cases correct with 17 m median and 45 m p90 error. On the long
pair it gives 86.4% coverage, improves buoy median/p90 error to 48/84 m, reduces
median/p90 adjacent-vector difference to 55/160 m, and reduces 34 flips to 4.
Its triangle-area p01/p99 is 0.942/1.075, close to the ORB field's
0.959/1.052. Tight q4 pattern refinement accepts five of the six nodes involved
in the remaining flips but removes none of the flips, confirming that
correlation often shares the same wrong basin. Marking those six nodes
unavailable removes every flip and changes coverage only from 86.38% to
86.31%; interpolation is not required in the raw measurement field.

The provisional learned dense policy is therefore: 1,024 features per tile,
direct LightGlue vectors, adaptive 2/3/4/6 km support requiring eight coherent
vectors, and explicit rejection of nodes in flipped triangles. Pattern matching
is not run routinely and is not used as a topology repair. This remains a
development candidate, not an operational selection: it needs at least one
additional high-deformation pair, sequence propagation, and external
deformation evidence. A candidate graph is now narrowly motivated only for
recovering rejected/unsupported spatial hypotheses; it is not needed to replace
the accurate direct field everywhere.

Before opening a third dense learned result, pair 10245 to 10352 is frozen as
the next stress case. Among 18--30 hour development pairs in a different
temporal block, it has two representative buoy labels, complete median source
and target validity, 1,244 paired ORB trajectories, and a 606 m p90 difference
between neighbouring ORB vectors. This is materially more dynamic than pair
740 to 849 (240 m p90) without selecting one of the apparently catastrophic
multi-kilometre ORB fields. Selection uses baseline metadata only; ALIKED dense
performance for this pair had not been opened.

The frozen third pair is now complete. It covers about 155,180 square
kilometres with 150 source and 148 target tiles and took 310 seconds on local
CPU (89 seconds extraction, 204 seconds matching). The fixed 2 km field covers
87.5% of 9,701 nodes, has 79 flipped triangles, and gives 324 m median and
616 m maximum error across its two available buoy labels. ORB has 73 flips on
the same dynamic pair, so widespread deformation/topology complexity is not
specific to the learned method.

Without retuning, adaptive-eight raises ALIKED coverage to 89.2%, reduces
flips from 79 to 3, lowers adjacent-vector p90 from 349 m to 222 m, and gives
295/506 m median/p90 buoy error. ORB's adjacent p90 is 414 m and its triangle
area p01/p99 is 0.808/1.517; adaptive ALIKED gives 0.809/1.306. Rejecting the
six nodes involved in the three residual flips removes all flips and retains
89.16% coverage. This preselected result supports the adaptive-eight plus
flip-node rejection policy across a short consistency control, a moderate
21-hour pair, and a more dynamic 23-hour pair. The next evidence target is no
longer another local radius sweep; it is sequence reuse/caching, GPU runtime,
and external validation of deformation magnitude.

### 15.15 Feature caching and physics-conditioned matching speed

A globally anchored EPSG:3413 tile grid now makes per-image ALIKED features
reusable across image pairs. The cache identity includes the image path, tile
centre, pixel size, tile/core geometry, feature budget, support radius, model,
and detection threshold. Replaying pair 731 to 740 from cache reproduces the
match CSV and 4 km field SHA256 hashes exactly. Extraction time falls from
84.7 to 3.0 seconds and total time from 239.5 to 130.2 seconds (-45.6%) with no
algorithmic change. The current cache contains 709 real tile files for five
images and occupies 710 MiB, about 142 MiB per image in this sample. Float32
descriptors are retained to preserve exactness.

A separate speed arm restricts each LightGlue target-tile call to source
features that can physically reach that target core under 30 km/day. This
removes comparisons that would necessarily fail the later speed gate, but it
does alter LightGlue context and is therefore evaluated as an algorithmic arm,
not assumed equivalent. On the fully cached 1.63-hour pair it reduces matching
from 112.2 to 31.3 seconds and total time from 130.2 to 49.0 seconds. Relative
to the cold full-context run this is a 4.9-fold speedup. After adaptive-eight,
coverage is 97.81% versus 97.85%, median ORB disagreement is 24.17 versus
24.30 m, all three buoy errors are identical (14.7 m median), and both fields
have zero flips.

Cross-pair reuse was then tested rather than inferred. For pair 740 to 849,
126/134 source tiles were cache hits from the preceding 731 to 740 run while
the newly observed target was extracted once. Total time was 220.6 seconds,
versus 278.7 seconds for the earlier non-cached/full-context implementation.
After the target entered the cache, an identical repeated-window run took
131.8 seconds; its match and field hashes equal the first cached run. The
adaptive-eight speed field has 86.75% coverage, zero flips, and 43 m median
buoy error, slightly improving the earlier full-context/non-anchored result.

The preselected dynamic March pair gives the important caveat. Physics-subset
matching reduces cold matching from 204 to 151 seconds, but cache writes and a
slightly larger anchored layout limit total cold speedup to 310 versus 283
seconds. After adaptive-eight, full-context versus the speed arm gives 89.22%
versus 88.75% coverage, 295 versus 317 m median buoy error, and 3 versus 1
flipped triangles. Both buoy cases remain within 2 km, and the topology tail is
essentially unchanged, but this is not exact accuracy neutrality.

Decision: globally anchored feature caching is retained as a safe operational
optimization. Physics-subset matching remains an explicit speed candidate; it
must stay paired with full-context matching through independent evaluation.
Long-gap LightGlue calls remain the dominant local-CPU cost, so CUDA/HPC is
still the appropriate next runtime test.

### 15.16 Controlled deformation and local-policy selection

A controlled-warp fixture now applies known deformations to a real 1,024 by
1,024 standard-VAE SAR patch and estimates a 4 km field without using the
truth during matching. The three preregistered cases are a fractional rigid
translation, smooth affine divergence/shear, and a 600 m vertical lead opening.
This isolates estimator behaviour from buoy interpolation error and permits
direct checks of displacement, deformation gradient, topology, and lead
opening.

The adaptive-eight policy is accurate for the rigid and affine cases (6.2 and
7.7 m median error) and recovers the lead opening as 602 m. It nevertheless
assigns two nodes 0.9 km left of the lead to the right-hand motion mode, giving
a 569 m maximum error. This is a boundary-selection failure: a wider spatial
support contains more votes from the wrong side even though each motion mode
is internally coherent.

A bounded nearest-candidate rule resolves this failure. At each output node it
uses at most the 12 nearest matched source features within 6 km, requires at
least eight vectors in the selected 1 km displacement cluster, and otherwise
publishes no value. On the same controlled cases it gives:

- rigid translation: 100% coverage, 6.1 m median and 11.9 m p90 error;
- affine deformation: 100% coverage, 7.8 m median and 14.3 m p90 error, with
  deformation-gradient Frobenius error of 4.9e-5;
- lead opening: 100% coverage, 2.7 m median, 9.9 m p90, 24.5 m maximum error,
  and 602 m recovered opening.

The policy was then replayed without tuning on the four available dense field
pairs. Compared with adaptive-eight, nearest-12 slightly lowers coverage by
0.00, 0.01, 0.42, and 0.90 percentage points, respectively. It improves field
smoothness and ORB agreement on every pair and retains all 16 available buoy
estimates within 2 km. Across those paired buoy cases, median error changes
from 21.3 to 19.5 m and p90 from 86.0 to 72.8 m; nearest-12 wins 10 cases and
adaptive-eight wins six. Mean error is essentially unchanged (63.7 versus
64.2 m) because the two dynamic March buoy cases worsen modestly.

Nearest-12 therefore supersedes adaptive-eight as the provisional
accuracy-first local estimator; adaptive-eight remains the frozen comparison
policy. The choice prioritizes lower displacement/deformation tails and correct
motion-boundary assignment over retaining the last fraction of spatial
coverage. Tight q4 correlation still does not repair the one residual flipped
triangle on the dynamic pair. Rejecting its three participating nodes removes
the fold, changes coverage only from 87.857% to 87.826%, and slightly improves
neighbour consistency. The learned raw-measurement policy is now nearest-12
plus explicit flipped-node rejection, with no routine pattern matching.

### 15.17 First complete-footprint sequence propagation

The first dense sequence reuses the cached pair matches for images 721, 731,
740, and 849. Two development buoys are linked through all three transitions;
the final transition spans 21.4 hours. Each pair is estimated either from the
true buoy position (the one-step lower bound) or from the preceding estimated
endpoint (operational propagation). No feature crops are recentered on truth.

Both nearest-12 paths remain available and within 2 km for all three steps.
The final errors are 57 and 77 m, compared with 9 and 49 m when the final pair
is reinitialized at truth. This difference is ordinary accumulated vector
error rather than a candidate-basin failure: propagated source offsets grow to
31 and 83 m, while the final displacement proposals change by only 10 and 0 m.
The smooth control therefore gives no evidence that a reconnection graph would
improve these paths. Such a graph remains conditional on observing unsupported
states or a discrete hypothesis switch in a more dynamic sequence.

Before opening further dense learned results, the March development chain
10107-10217-10229-10245-10341-10352 is frozen as the next sequence fixture.
Baseline metadata gives 26 complete five-step buoy paths over 47.7 hours, with
29-30 labels on each early pair and 28 on each late pair. Gaps alternate between
21.4, 1.64, 1.64, 19.8, and 3.27 hours, and the chain spans two to three spatial
blocks per pair. Five early labels exceed the frozen 30 km/day motion limit and
will be reported separately rather than used to retune the gate. Selection used
only split, cadence, buoy, speed, and spatial metadata; dense ALIKED results for
the five constituent pairs had not been opened.

The frozen chain is now complete. Nearest-12 provides 26,766/32,494 valid 4 km
nodes (82.4%) versus 24,028/32,494 for the selected ORB field (73.9%). This is
an 8.43 percentage-point or 11.4% relative coverage increase. ALIKED has no
flipped triangles across the five pair fields; ORB has 15, all on the second
long gap. ALIKED is noisier at the centre of the neighbour-difference
distribution (37.7 versus 17.5 m median), but its deformation tails are lower:
p90 is 169 versus 189 m and p99 is 527 versus 793 m. External deformation
validation is therefore still required; the coverage/topology improvement does
not by itself prove that every additional small-scale gradient is physical.

At the 130 one-step buoy measurements belonging to the 26 complete paths,
nearest-12 and ORB are both available everywhere and fail on exactly the same
five labels. ALIKED has 125/130 within 2 km, 47.8 m median error, and 1,301 m
p90; ORB has 125/130, 52.0 m, and 1,310 m. ALIKED wins 77 paired errors and ORB
wins 53. The learned method's demonstrated advantage is therefore coverage and
topology, with only a small point-accuracy improvement rather than recovery of
different buoy cases.

All 26 ALIKED paths remain available through all five propagated transitions.
The final errors have 71 m median and 647 m p90, and 24/26 paths are within
2 km. One added propagation failure is caused by accumulated vector error, not
a discrete candidate switch: even with source-state errors as large as 9.9 km,
the propagated and truth-reinitialized displacement proposals differ by at
most 153 m. A candidate-reconnection graph is therefore stopped for now; no
recoverable alternate motion basin has been observed in either dense sequence.

The repeat/jump Level-1 diagnostic explains the largest tail. Buoy
`300234010307830` contains a 20-hour exact-position run followed by a
421 km/day jump, and later a 14-hour exact-position run followed by a
200 km/day jump, while the one-hour bracket test marks the individual fixes as
valid. Removing no primary labels but reporting the existing combined-QC
sensitivity leaves 23 complete five-step paths: 22/23 finish within 2 km, with
64 m median and 321 m p90 final error. The remaining valid DTOP path finishes
at 2.66 km; ALIKED and ORB make nearly identical errors on its two long gaps.
This supports carrying exact-repeat/jump context into future buoy validation
rather than attributing every large residual to image matching.

A local affine displacement fit was tested as a targeted response to ALIKED's
higher median neighbour difference. It improved the controlled smooth-affine
warp, but an unconstrained fit smeared the 600 m lead. Requiring interpolation
geometry, less than half a pixel (40 m) weighted residual, and less than 4%
local displacement gradient protected the lead, but admitted only 166/26,766
real nodes (0.62%) and changed no buoy or aggregate p90/p99 metric. This arm is
stopped and its implementation was removed; nearest-12 remains the minimal
selected estimator.

The five full-context CPU pair runs took 770 seconds in total despite 803/1,045
tile extractions (76.8%) being cache hits. This is not an operational speed
win over ORB. It reinforces the existing decision that exact caching is safe,
but full-context learned matching needs CUDA/HPC or the separately evaluated
physics-subset speed arm before deployment is computationally credible.

One split-handling incident is recorded explicitly. The first attempt at pair
10107-10217 used the shared transition table without filtering
`within_dataset_split`; its aggregate summary therefore included 29 labels
from each of development, confirmation, and final holdout. That run was
invalidated before individual non-development rows were inspected, and the
dense field itself is label-free. The runner now fails closed when a requested
pair spans multiple splits and requires an explicit split. Nevertheless, the
aggregate final-holdout result was exposed, so the original final holdout is no
longer described as pristine; a new independent sequence/split is required for
the final evaluation.

### 15.18 Frozen N-ICE2015 buoy-array deformation validation

N-ICE2015 is the next external deformation check, kept separate from the 2020
accuracy estimates because its region, ice regime, cadence, and data
availability are materially different. The split below was frozen before any
SAR deformation result from these pairs was opened:

| Role | fixture images | operational catalog IDs | common buoys | gap |
|---|---|---:|---:|---:|
| implementation diagnostic only | 2 to 4 | 6603 to 6689 | 7 | 39.56 h |
| excluded temporal buffer | 4 to 7 | 6689 to 6775 | 7 | 32.84 h |
| independent evaluation | 7 to 8 | 6775 to 6801 | 6 | 14.89 h |
| independent evaluation | 8 to 11 | 6801 to 6901 | 6 | 32.83 h |
| independent evaluation | 11 to 15 | 6901 to 6998 | 5 | 47.73 h |

The buffer transition is not scored. It prevents the implementation diagnostic
and independent evaluation from sharing an image. No parameter may be changed
from the image-2-to-4 result; if that diagnostic exposes an implementation
error, the error is fixed and the diagnostic rerun, but images 7, 8, 11, and 15
remain unopened until the implementation is frozen again.

The primary comparison is the selected ORB configuration against direct
ALIKED/LightGlue with nearest-12 and flipped-node rejection. Both use the same
standard `VAE_2_16_ELU_64` rasters, 30 km/day physics gate, projected
EPSG:3413 coordinates, and 4 km reporting grid. The selected ORB arm carries
24-pixel direct and 48-pixel interpolated pattern-matching windows, quadratic
fractional refinement, bilinear template sampling, and kilometre-conditioned
15 km MAGSAC geometry. The learned arm does not use pattern matching.

The buoy array is used for deformation rather than only endpoint accuracy.
For each evaluation pair, report:

- displacement error at every buoy, with median, p90, maximum, and vector bias;
- error in every pairwise relative buoy displacement, which removes common
  translation and directly tests deformation;
- affine displacement-gradient, divergence, and shear-rate error from identical
  fits to the observed and estimated buoy vectors;
- triangle orientation/area-ratio errors and spatial coverage inside the buoy
  array convex hull;
- leave-one-buoy-out and buoy-bootstrap sensitivity, because only five to six
  simultaneous buoys are available on each evaluation pair.

The six required standard-VAE rasters are now reconstructed from the exact raw
Sentinel-1 products and stored on Kingston. The historical preprocessing code
is fixed at arktalas_vae commit `321e9d6`, with weight SHA256
`28ab116a7bbbc6613f05e16595e6bdcd1d7c26a702cb62e5f200a7177d8e18b2`
and normalization SHA256
`016381dbaa9e2279ae1bc6f653572fc24ebc9d8cc9f2cd898ff0372dfd722336`.
It uses DN plus CLAHE, the 2-layer/16-hidden-unit ELU VAE, and q0.2-q99 output
scaling. A reconstructed reference scene has the identical mask, 4,483 of
25.95 million fused pixels differing by one DN, MAE 0.000173 DN, and
correlation 0.99999997 against the surviving historical output. All six new
rasters have two uint8 bands and 441 GCPs. Their paths and image identities are
recorded in `results/nice_deformation_validation/standard_vae_stage_manifest.csv`.

The fixture builder also exposed and fixed a staging error: the source buoy
table's obsolete `image_filepath` could survive alongside the renamed Kingston
path, causing downstream code to choose the wrong preprocessing. The builder
now drops the source path before installing the verified mapped path.

### 15.19 Descriptor-independent buoy-array metrics and March control

A common buoy-array evaluator now accepts long-form estimated vectors from any
method. It retains unavailable estimates in the expected denominators and
reports endpoint error and bias, every pairwise relative-displacement error,
longitudinal/transverse strain-rate error, affine displacement-gradient and
divergence/shear/vorticity error, triangle orientation and area-ratio error,
leave-one-buoy-out sensitivity, and deterministic buoy bootstrap intervals.
All coordinates are required to be EPSG:3413 metres and elapsed time is hours.
Truth and estimate fits always use the identical available buoy subset. Input
and upstream source files are SHA256-recorded.

The validator was exercised on the March development chain before N-ICE data
staging. The comparison uses ALIKED nearest-12 truth-reinitialized vectors and
the selected ORB field sampled by the same inverse-distance average of up to
four trajectories within 10 km used in the dense-pair comparison. Across all
130 paired estimates, ALIKED has 47.8 m median and 1,301 m p90 endpoint error
versus 52.0 and 1,310 m for ORB, winning 77 cases to 53. Across all 1,625 buoy
pairs, ALIKED has the lower relative-displacement error 910 times versus 715;
its median paired improvement is 6.1 m. It has the lower affine-gradient error
on three of five image pairs, but the median advantage is only 2.3e-5 in
dimensionless gradient magnitude. Neither method produces an incorrect buoy
triangle orientation.

The most important result is reference quality. The primary buoy truth has
median affine residuals of 138 to 755 m across the five pairs. Applying the
independently defined combined label QC retains 119/130 estimates and lowers
that range to 76 to 509 m. It reduces ALIKED gradient error from 0.043 to 0.012
on the first long gap, 0.053 to 0.016 on the second long gap, and 0.011 to 0.004
on the final short gap; ORB changes similarly. In the QC sensitivity, ALIKED
still wins 71/119 endpoint cases and 750/1,357 buoy-pair errors, with median
advantages of 4.7 m and 6.3 m, respectively.

Decision: the higher ALIKED median neighbour difference does not translate
into worse 80 to 95 km buoy-array gradients in this development control. ORB
and ALIKED are much closer to one another than either is to noisy short-cadence
buoy deformation truth. This is evidence to keep ALIKED as a frozen candidate,
but not to add Torch/Kornia to core LiMOSAT yet: its local CPU path remains too
slow, Apple MPS is unavailable on this machine, and its accuracy advantage is
small. The shared deformation validator is retained as infrastructure. The
next discriminating evidence is the frozen N-ICE array on standard VAE data,
followed by the same learned arm on CUDA/HPC.

### 15.20 Independent N-ICE2015 result

The image-2-to-4 diagnostic was opened first. ORB was available at all seven
buoys within 10 km, with 492 m nearest-vector and 431 m up-to-four-vector
median error. ALIKED nearest-12 was available at all seven with 120 m median
and 169 m p90 error, 63.6% grid coverage versus ORB's 61.5%, and zero flipped
triangles. No ALIKED parameter was changed before the three independent pairs
were opened.

The isolated four-image ORB operational run and three pairwise ALIKED runs use
images 6775, 6801, 6901, and 6998 only. The buffer and diagnostic images are
not in the operational run. Across the 17 independent buoy transitions:

| metric | operational ORB, local average within 10 km | ALIKED nearest-12 |
|---|---:|---:|
| available / expected | 15 / 17 | 17 / 17 |
| correct within 2 km | 11 / 17 | 17 / 17 |
| median endpoint error | 620 m | 113 m |
| p90 endpoint error | 6,278 m | 252 m |
| maximum endpoint error | 9,047 m | 304 m |
| available buoy pairs | 31 / 40 | 40 / 40 |
| median relative-displacement error | 1,152 m | 93 m |
| p90 relative-displacement error | 6,943 m | 287 m |
| median affine-gradient error | 0.03096 | 0.00218 |
| maximum affine-gradient error | 0.05536 | 0.00301 |

ALIKED wins 29 of the 31 jointly available buoy-pair deformation errors; the
median paired improvement is 1,076 m. Its affine-gradient error is lower on
all three pairs by 93-98%. The buoy arrays are about 191-194 km across, and the
truth affine residual medians are 115, 181, and 148 m, so the ALIKED endpoint
and deformation residuals are commensurate with rather than far below the
reference's internal non-affine variability. Bootstrap upper tails are unstable
when resampling only five or six buoys into nearly degenerate geometries;
leave-one-buoy-out results are retained as the more interpretable sensitivity.

The post-hoc state-reset control separates descriptor failure from propagation.
Fresh two-image ORB improves the final local-average median from 5.53 to 2.47
km but remains much worse than ALIKED's 185 m. On the middle pair, fresh ORB's
nearest vector gives 260 m median while averaging up to four vectors gives
1.22 km. State propagation therefore contributes to the operational loss, but
the long-gap failure remains after reset and local averaging can itself mix
incompatible vectors. This supports a fresh-detection learned backend and a
bounded coherent local estimator rather than inserting ALIKED descriptors into
ORB's existing trajectory/update architecture.

Nearest-12 initially left 2, 6, and 7 flipped grid triangles. The original
one-pass vertex rejection was not guaranteed to finish: on the final pair,
retriangulation created two new folds. Rejection now iterates to a fixed point.
It removes 4, 14, and 23 nodes, retains 7,944/26,079 grid nodes (30.46%), and
leaves zero folds. ORB has 10,305/26,079 nodes within 10 km (39.51%) but 49
flipped triangles. ALIKED's accuracy/topology gain therefore comes with a real
9.05 percentage-point spatial-coverage loss on these scenes.

Decision: ALIKED is promoted from an exploratory descriptor to the leading
accuracy candidate for N-ICE-like deformation. It is not yet added as a core
LiMOSAT dependency. The operational form is fixed-grid cached ALIKED features,
physics-conditioned tile matching, nearest-12 coherent estimation, and
fixed-point fold rejection, with no routine pattern matching.

### 15.21 N-ICE CPU speed arm

The previously defined physics-reachable LightGlue subset was evaluated after
the full-context result. It changes only one of 17 buoy vectors, by 1.16 m;
endpoint, pairwise-deformation, and affine metrics are otherwise unchanged.
After fixed-point rejection it retains 7,937 grid nodes versus 7,944 for full
context and leaves zero folds in all three fields.

Descriptor comparisons fall from 3.05 million to 1.78 million (-41.6%), but
measured matching time falls only from 432 to 350 seconds (-19.0%). Total wall
times are not compared because the full-context runs populated the cache while
the speed runs were fully cache-warm. The limited CPU conversion of comparison
savings indicates that model-call and data-movement overhead now dominate.

Decision: retain physics-conditioned matching as the local CPU candidate, but
do not present it as an operational speed solution. The next implementation
step is a single selected-policy sequence runner and a CUDA/HPC timing run;
further local threshold sweeps are lower value.

### 15.22 Selected-policy sequence runtime and feature budget

A single-process selected-policy runner now loads ALIKED and LightGlue once,
loads every unique image feature set once, applies physics-conditioned matching,
and writes nearest-12 fields with fixed-point fold rejection. On the three
independent N-ICE pairs, grid availability is identical to the earlier
pairwise runs and the maximum displacement-vector difference is
`3.0e-11 m`. This validates orchestration without changing the selected
algorithm.

The measured four-image CPU run took 564.5 seconds, of which 511.4 seconds
(90.6%) was pair matching. This is slower than the earlier 436.7-second
aggregate because the unchanged LightGlue calls took 511 versus 350 seconds in
the new session. It is not evidence that sequence reuse slows the algorithm.
Non-matching work fell from 86.6 to 53.1 seconds; normalizing to the earlier
matching time gives about 403 seconds, a 7.7% end-to-end saving. Preparing all
descriptors and local affine frames once per pair changed a back-to-back CPU
match from 80.82 to 80.05 seconds, which is too small to distinguish from
runtime variation. It remains enabled by default only for CUDA/MPS, where it
also avoids repeated host-to-device transfers.

Feature-count reduction was tested only on the frozen implementation-diagnostic
pair 6603 to 6689, not selected using the independent N-ICE result:

| features/tile | matching time | selected pair time | fold-free coverage | buoy available/correct | median / p90 buoy error |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 126.1 s | 133.9 s | 63.02% | 7/7 | 120 / 169 m |
| 768 | 99.7 s (-20.9%) | 107.4 s (-19.8%) | 60.46% | 7/7 | 144 / 222 m |
| 512 | 65.4 s (-48.2%) | 72.9 s (-45.5%) | 52.90% | 6/7 | 135 / 232 m among available cases |

The 512 arm is rejected because its speed gain removes one labelled result and
10.12 percentage points of field coverage. The 768 arm is a valid explicit
speed/coverage trade, but its 20% matching reduction costs 2.56 percentage
points of coverage and raises median/p90 buoy error by 20%/32%; it is not
promoted over the 1,024-feature accuracy arm. The dominant remaining target is
therefore LightGlue execution on CUDA and reduction or batching of model calls,
not further local feature-count tuning. Exact pair-match caching should be used
when assembling multiple overlapping deformation windows so matching is paid
once per image pair.

## 16. Current prioritized work list

| Priority | Work item | Current state | Next evidence or decision |
|---:|---|---|---|
| 1 | Productionize kilometre-conditioned homography | Complete; commit `147fced`, 15 km frozen from the 10/15/20 km development diagnostic | Carry the explicit scale and 15 km threshold into every subsequent arm and independent evaluation |
| 2 | Fix and validate PM fractional coordinates | Complete; guarded quadratic implementation in `c055ceb`, full-70 control complete | Carry the opt-in method into independent sequence and deformation evaluation |
| 3 | Test fractional template/state propagation | Complete on the full-70 development sequence; bilinear sampling retained in the selected PM arm | Audit update contamination on independent paths rather than reopen the selected development arm |
| 4 | Replay the local ALIKED candidate policy | Complete; controlled warps and nine dense pairs select nearest-12 within 6 km with at least eight coherent vectors | Freeze nearest-12; retain adaptive-eight only as the comparison policy |
| 5 | Compare ALIKED direct versus tightly refined | Complete; direct nearest consensus is retained and selective q4 does not repair flipped topology | Omit routine PM from the dense learned arm; reject the small residual flipped-node set |
| 6 | Reduce direct-ORB PM cost safely | Complete; direct 24 pixels and interpolated 48 pixels selected on development | Freeze these windows for subsequent tests and independent evaluation |
| 7 | Carry the best local proposal policy into sequences | Complete on two development chains; all 26 March paths survive five steps and no recoverable hypothesis switch is observed | Stop reconnection work; revisit only after an unsupported or switched state is demonstrated |
| 8 | Validate deformation/output quality | Complete on the frozen N-ICE diagnostic and three independent pairs; ALIKED is 17/17 within 2 km and reduces median pairwise-deformation error from 1,152 to 93 m, but has 9.05 percentage points less grid coverage | Retain the independent result; next add altimetry/lead/ridge validation rather than tuning on these pairs |
| 9 | Measure operational speed paths | Physics-conditioned matching preserves the N-ICE result but reduces local CPU matching time only 19%; exact caching remains safe and Apple MPS is unavailable | Build one selected-policy sequence entry point and run it on CUDA/HPC |
| 10 | Independent final evaluation | Original final holdout is compromised by one accidental aggregate mixed-split evaluation | Define a new independent sequence/split before any final claim |

The immediate execution order is now consolidating the selected ALIKED policy
into a reproducible sequence entry point, running the learned arm on CUDA/HPC,
and defining a new independent 2020 sequence/split. Altimetry/ridge validation
is the next external deformation target; the N-ICE pairs must not be reused for
further parameter selection.
Priorities 6-9 use the selected geometry and sub-pixel conventions so their
results do not need to be repeated.
ALIKED without pattern matching is a first-class arm: its sub-pixel direct
locations may be accurate enough that omitting pattern matching improves both
accuracy and runtime. The potential operational saving is material because PM
used 251 seconds, or 51% of the measured full-70 ORB baseline, but there is not
yet a comparable dense ALIKED runtime from which to claim a net speedup. This is
tested rather than assumed; the comparison must include cached feature cost,
sequence persistence, and deformation quality, not only one-step buoy endpoint
error.

## 17. ICESat-2 structural deformation validation

The frozen method, selected March 2020 pair, literature basis, exact-time
advection, quality criteria, spatial nulls, and first ATL07/ATL10 results are in
`docs/icesat2_deformation_validation_plan.md`.

The 0044 and frozen 0040 ATL07 crossings both support a positive association
between ALIKED shear and ICESat-2 roughness: Spearman 0.217 (`p=0.023`) and
0.296 (`p=0.001`) under within-beam circular shifts. ORB gives -0.014
(`p=0.627`) and 0.189 (`p=0.018`). ALIKED-minus-ORB 20 km block-bootstrap
intervals still include zero on both tracks, so this is repeatable structural
evidence rather than a resolved paired accuracy advantage. ALIKED covers 89.9%
and 98.5% of method-union laser observations versus 73.9% and 25.3% for ORB.
Convergence-versus-ridge association does not repeat and is not used for model
selection. The corresponding ATL10 0044 crossing contains no strong-beam lead
events on common support and is retained as an insufficient-data result.

## 18. Multi-image ICESat-2 and ALIKED prior-window result

The frozen ATL07 comparison now covers three usable deformation images and five
usable image/track combinations, plus two explicit no-spatial-support controls
and two insufficient short-track controls. The positive ALIKED
shear-versus-roughness relationship repeats across three tracks for the 23 h
10245-to-10352 field, but not for the 19.8 h component field. This prevents the
ICESat statistic from being overinterpreted as direct deformation truth. Full
details and the support ledger are in
`docs/icesat2_deformation_validation_plan.md`.

The first spatial-thinning speed design is rejected: it preserved buoy error
and gross coverage but altered local gradients enough to weaken both frozen
ATL07 associations. The replacement is prior-guided target windowing:

1. retain all 1,024 ALIKED detections per tile;
2. scale the preceding accepted pair's median velocity to the new image gap;
3. shift the source tile footprint by that displacement and expand it by a
   fixed 15 km uncertainty;
4. run LightGlue only against target tiles and source features inside that
   reachable window; and
5. retain the original 30 km/day absolute motion gate and fixed-point fold
   rejection.

On two long, dynamic pairs this reduced matching by 49.7% and 34.0%, retained
or slightly increased 4 km coverage, preserved every buoy endpoint available
to the full matcher, and kept 99.3-99.5% of common vectors within 100 m. The
next gate is a within-sequence held-out test in which the prior is generated
only from the preceding accepted ALIKED field. Required failure handling is a
fallback to the full physics window when the prior is absent, stale, or its
residual audit exceeds the fixed uncertainty. Do not tune the uncertainty on
the ATL07 association.

## 19. Alignment-first multisensor and runtime work block

The first CryoSat-2 pilot adds useful but different evidence from ICESat-2. For
the frozen 10245-to-10352 pair, eight RDWES1B granules supplied 41,152
quality-controlled footprints. Exact ORB/ALIKED common support contained 1,534
footprints, four tracks, and 208 four-kilometre bins. Drift-aware shear versus
lead fraction was positive for both ORB (`rho=0.361`, spatial-null `p=0.001`)
and ALIKED (`rho=0.328`, `p=0.001`). ALIKED had greater method-specific support
(2,481 footprints on five tracks versus 1,609 on four for ORB), but did not beat
ORB on the common-support association. Floe-only roughness had no useful shear
association for either method. Median footprint advection was about 3 km and
the P95 was 15-17 km; removing advection reduced the shear/lead association to
0.157 for ORB and 0.149 for ALIKED. The CryoSat result therefore validates that
the drift-aware fields contain lead-related spatial structure, not that ALIKED
is uniformly more accurate.

CryoSat-2 and ICESat-2 have complementary roles. CryoSat-2 provides many
footprints and repeated tracks suitable for lead-fraction and opening/shear
tests. ICESat-2 provides finer along-track morphology suitable for ridge,
roughness, and localized linear-kinematic-feature interpretation. Buoys remain
the displacement reference. No altimetry association is a direct displacement
truth or a tracker-selection metric by itself.

This multisensor framework is method-independent and remains useful regardless
of whether ALIKED becomes operational. The existing production ORB setup has
already generated a ten-year deformation archive; that archive is a primary
scientific product to validate and interpret, not a temporary baseline that is
discarded if a learned method improves future processing. ALIKED is evaluated
as a prospective higher-accuracy or complementary product. Exact common support
is mandatory when comparing ORB with ALIKED, but it must not restrict validation
of the much larger ORB-only archive. The multisensor pipeline must therefore
support both single-product validation and paired method comparison as separate
declared analysis modes.

### 19.1 Mandatory alignment and data-selection ledger

Small registration and inclusion choices can materially change a multisensor
result. Every comparison must therefore write a machine-readable event ledger
before its outcome metric is inspected. For each SAR pair and altimeter
granule/beam/track, record:

1. product identifiers, exact UTC acquisition times, SAR interval, and the
   reason the event entered the candidate set;
2. source CRS, target CRS, axis order, coordinate units, vector units, and the
   exact forward/inverse transformations used;
3. observed altimeter coordinate, material-reference coordinate, advection
   vector, advection fraction, and whether direct or piecewise SAR motion was
   used;
4. counts after temporal selection, product QC, surface classification,
   spatial support, common-method support, binning, and the final metric;
5. deformation-field identity and hash, interpolation method, boundary rule,
   reporting resolution, minimum-observation rule, and all missing-data reasons;
6. whether an event is development, confirmation, independent evaluation, or
   an explicit insufficient-support control.

Candidate events must be selected using time/geometry/support rules that do not
use the observed correlation. All eligible events, including null and
insufficient-support cases, remain in the ledger. ORB/ALIKED comparisons use
the identical altimeter observations and bins; method-specific coverage is
reported separately. Results are shown per track and per SAR pair before any
pooled statistic. Uncertainty and resampling operate on whole tracks, granules,
or SAR pairs rather than treating laser or radar footprints as independent.
Single-product analyses of the ten-year ORB archive use the same alignment,
selection, null, and support rules, but do not require ALIKED coverage.

Before expanding the dataset, add small invariance tests for zero motion,
constant translation, reverse-time sign, CRS round trips, exact common-support
identity, unique bin assignment, and boundary exclusion. Save a visual audit
for every retained event showing observed and advected tracks over the
deformation field, with a small number of labelled checkpoints that can be
verified numerically.

### 19.2 Predeclared alignment sensitivities

The primary method remains exact-time drift-aware registration to the SAR pair
start. The following are symmetric diagnostics, never alternatives selected by
which gives the strongest association:

- static/no-advection control;
- direct long-pair motion versus piecewise motion from intermediate accepted
  SAR fields where coverage exists;
- along-track and cross-track offsets of zero and fixed positive/negative
  distances derived from the independent buoy registration-error scale;
- 1 km morphology support and 4 km tracker support, with 8 and 12 km reporting
  sensitivities where sample counts permit;
- strict common-method support, alongside method-specific coverage;
- leave-one-track-out and leave-one-pair-out summaries.

The output is a sensitivity envelope and support ledger. Do not choose an
offset, bin size, product threshold, SAR pair, or granule because it improves a
correlation. If a result changes sign under plausible alignment uncertainty,
report it as alignment-sensitive.

### 19.3 Fair ORB-versus-ALIKED runtime gate

There is still no valid end-to-end ORB/ALIKED speed percentage. The warm-cache
production ORB B0 processed 70 images in 489 seconds. The selected four-image,
three-pair ALIKED CPU sequence took 564.5 seconds, while prior-guided matching
reduced two individual ALIKED pair totals from 133.0 to 83.9 seconds and from
55.7 to 36.9 seconds. These workloads cannot be divided into a defensible
cross-method ratio.

Run both methods on the same frozen image sequence, evaluated pair set, spatial
footprint, 4 km output support, and persistence requirements. The ALIKED arm is
1,024 features per tile, cached per-image features, physics-conditioned
prior-guided matching with a fixed 15 km uncertainty and audited fallback,
nearest-12 estimation, fixed-point fold rejection, and no pattern matching.
The ORB arm is the current production config with selected direct/interpolated
pattern windows. Measure cold cache and warm cache separately, use at least
three repetitions after one untimed setup run, and report:

- model/setup, image preparation, detection/description, matching, pattern
  matching, field estimation, topology/QC, persistence, and total wall time;
- time per unique image, evaluated pair, square kilometre of common valid
  support, and 1,000 accepted 4 km vectors;
- buoy endpoint and pairwise-deformation errors, fold-free coverage, cycle
  closure, and multisensor statistics on exactly common support.

CPU is the reproducible local baseline. A later CUDA/HPC run uses the same
manifest and outputs; it must not silently change feature counts, precision,
tile support, or LightGlue settings. The held-out prior-window sequence and
fallback audit must pass before calling the faster ALIKED arm operational.

### 19.4 Immediate execution order

1. Implement and test the event-ledger/alignment invariants without changing
   scientific outputs.
2. Rebuild the current ICESat-2 and CryoSat-2 pilot ledgers and verify every
   retained and rejected event, coordinate transformation, timestamp, and
   common-support count.
3. Run the predeclared alignment sensitivities and produce per-track/per-pair
   plots plus a concise sensitivity table; do not tune from the result.
4. Search the existing March data first for additional prequalified CryoSat-2
   and ICESat-2 crossings. Download only products identified by a frozen
   geometry/time manifest, and store them on Kingston.
5. Run the fair local CPU timing gate on a bounded common sequence. If it is
   clean and reproducible, prepare the exact CUDA/HPC command and manifest.
6. Update this plan and the sensor-specific validation notes with evidence,
   limitations, failed cases, output paths, and the next single decision.

### 19.5 Alignment implementation and audit record

The shared ledger and invariance tests are complete. The implementation is
`experiments/multisensor_event_ledger.py`; both sensor validators now emit the
same schema without changing their outcome calculations. The ledger makes UTC,
EPSG:3413 axis order and metre units, pair-total displacement, temporal
reference, observed/material/advected coordinates, interpolation and boundary
rules, field hashes, selection counts, and missing-support reasons explicit.
Focused tests cover zero/constant motion, reverse-time sign, CRS round trips,
exact common-support identity, unique along-track bins, and boundary handling.

The frozen reconstruction of 13 existing variants is under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/multisensor_alignment_audit_v1_20260819`;
its event-manifest SHA-256 is
`76705f957d75d653f8b6244fe52ab5a2a333d65741c07812322e294c970528d3`.
Four insufficient-support cases remain named controls. Every event has a
selection flow, exact-common bins, numerical checkpoints, and a compact visual
audit. Timestamp, advection direction, field reconstruction, and bin counts
passed the audit.

Five variants had used the 8,523-node pre-final ALIKED field rather than the
selected 8,520-node fold-rejected field. Corrected outputs are under
`icesat2_validation/results/selected_fold_rejected_v3` and
`cryosat2_validation/results/selected_fold_rejected_v2` on Kingston. ICESat-2
common-support statistics are identical after correction. CryoSat-2 loses one
common footprint but retains exactly the same primary shear/lead correlations
and spatial-null results. This is a provenance correction, not a scientific
result change.

### 19.6 Alignment sensitivity and March expansion result

The predeclared symmetric sensitivity is complete under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/multisensor_alignment_sensitivity_v1_20260819`;
its manifest SHA-256 is
`e9f15da4cdfb59c21817d82123bdd6bdfbac0402c6d529cec00343edef3651f3`.
At 4 km, 9 of 12 interpretable event-method envelopes meet the sign-change or
0.20-rho-span sensitivity rule. CryoSat-2 shear/lead is stable and positive for
both methods. ICESat-2 0040 stays positive but changes materially in magnitude,
whereas 0044 changes sign; most weaker ICESat-2 events are also sensitive.
Piecewise motion supplies 146 unique exact-common event bins and zero-support
cases remain explicit. No sensitivity arm was promoted as a preferred
registration.

The frozen March selection added ATL07 0030/0041 and ATL10
0030/0039/0040/0041 only after the time/geometry manifest was written. The
selection manifest SHA-256 is
`ae0f64641df7a44c983d970c06a1ca378e4f7aebf77fef7bc5c3c954e033cabe`;
the component-application manifest SHA-256 is
`d2b62bf1b5c5a25412b660d7fd306bb6ef80433a72708266c77ca66d715449c4`.
All six products passed HDF5 verification and remain on Kingston.

The per-track synthesis is under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/multisensor_expansion_summary_v1_20260820`.
On 10245-to-10352, new ATL07 0030 is positive (ORB 0.182, ALIKED 0.414)
and new 0041 is negative (ORB -0.202, ALIKED -0.258). Their component-pair
applications are insufficient, so neither replicates across pairs. All eight
new ATL10 event/pair applications have zero to two exact-common lead bins and
are insufficient. No second CryoSat pair was added. The existing CryoSat
shear/lead relationship remains the most alignment-stable structural result,
but it is still a single-pair result. The ICESat evidence does not establish a
general ALIKED advantage.

### 19.7 Fair CPU runtime and exact-common accuracy gate

The frozen three-image/two-adjacent-pair benchmark is complete under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/fair_orb_aliked_runtime_v3_20260819`.
The benchmark-manifest SHA-256 is
`50614fb7c7cb0a6de08ba4b7367be95c47cb2db1bd966448e40f71557cd31aa4`.
It uses the production ORB configuration and selected ALIKED policy on images
10245, 10341, and 10352, with one excluded setup followed by three valid cold
and three valid warm measurements. Warm repetition 2 is retained but excluded
because a separately frozen validation job overlapped it; uncontaminated warm
repetition 4 restores the predeclared count, and `protocol_deviation.json`
records the deviation.

| Cache | ORB median total | ALIKED median total | ALIKED / ORB |
|---|---:|---:|---:|
| Cold | 27.63 s | 189.91 s | 6.87x |
| Warm | 22.47 s | 106.47 s | 4.74x |

These totals include the persisted trajectory/field products and the common
4 km post-processing needed for the declared outputs. ALIKED feature caching
removes detection/description from warm runs, but pair matching still takes a
median of roughly 85-91 seconds for the two pairs and dominates its runtime.
The selected ALIKED CPU arm is therefore not operationally faster than current
ORB on this bounded common workload.

Accuracy and topology do pass the bounded gate. On exact-common buoy support,
10245-to-10341 endpoint medians are 86.4 m for ORB and 77.3 m for ALIKED;
pairwise-relative medians are 632.2 and 612.5 m. On 10341-to-10352 the endpoint
medians are 44.6 and 44.8 m and relative medians are 61.5 and 67.6 m. Median
ORB/ALIKED vector differences on the exact-common grids are 37.1 and 24.8 m,
and both products have zero eligible folded triangles. ALIKED has 4,209 versus
3,330 available nodes on the first pair and 4,267 versus 3,289 on the second,
about 26-30% more support, without losing these accuracy invariants.

The sequential-prior audit passes all six reported repetitions: the first pair
uses the full-physics fallback because no preceding field exists, and the
second uses only the immediately preceding accepted fold-free field. Its
matched-residual P90 is 1,134 m, within the fixed 15 km uncertainty. The exact
common ATL07 0039 check remains null (ORB -0.002, ALIKED 0.029 on 96 bins), and
ATL10 has only one lead-containing common bin. Cycle closure is unavailable
because the frozen bounded sequence has no independently evaluated closing
edge; this is an explicit missing metric rather than a zero result.

### 19.8 Current decision

ALIKED is not established as generally more accurate: it provides materially
more fold-free spatial support and comparable buoy accuracy on these pairs,
but its small accuracy advantages are mixed by pair and its ICESat-2 structural
advantage does not replicate robustly. It is also 4.74x slower than ORB in the
valid warm CPU comparison. Production ORB therefore remains the operational
method and the ten-year archive remains fully in scope for single-product
validation. ALIKED remains a promising complementary high-coverage product.

This was the decision before the matcher-call audit in section 19.10, which now
places a within-2020 CPU confirmation ahead of CUDA execution. Local CUDA and
MPS are unavailable.
The exact, hash-checked handoff is
`experiments/configs/fair_aliked_cuda_handoff_20260820.json`, executed by
`experiments/run_fair_aliked_cuda_hpc_20260820.sh`. No GPU performance is
estimated here. If the CUDA output fails vector/support/buoy/topology parity or
remains operationally unattractive, keep ALIKED off the production path and
use it only as a complementary field in independently selected validations.

### 19.9 Faster ALIKED matcher pilot

The matcher was isolated while retaining the frozen 1,024-feature extraction,
physics search, nearest-12/require-8 field estimator, topology rejection, and
buoy evaluation. Exact cosine mutual-nearest-neighbour (MNN) and symmetric
ratio-filtered MNN use the unit-normalized cached ALIKED descriptors directly;
three-layer LightGlue uses the same released ALIKED-LightGlue weights as the
nine-layer reference but caps contextual refinement at three transformer
layers. The implementation exposes the matcher and LightGlue layer count in
the run manifest; the nine-layer default is unchanged.

MNN is rejected as the dense operational matcher. It reduces the three-pair
N-ICE matching time from 511.42 to 3.63 seconds (140.8x), but fold-free support
falls from 7,937 to 5,919 nodes (-25.4%) and buoy availability from 17/17 to
11/17. On the March sequence it is accurate on the short prior-guided pair but
loses 443 nodes, or 9.24 coverage percentage points, on the longer prior-free
pair. Symmetric ratio filtering does not improve the controlled-warp result.
This shows that descriptor distance alone is fast enough, but does not replace
LightGlue's contextual disambiguation on ambiguous or long-gap SAR pairs.

Three-layer LightGlue passes the controlled rigid, affine shear/divergence,
and 600 m lead-opening cases without a material change from nine layers. On
the frozen three-pair N-ICE sequence it gives:

| Metric | 9-layer LightGlue | 3-layer LightGlue | Change |
|---|---:|---:|---:|
| Matching time | 511.42 s | 348.69 s | -31.8% |
| End-to-end time | 564.52 s | 378.34 s | -33.0% |
| Fold-free nodes | 7,937 | 7,817 | -1.5% |
| Buoys available / within 2 km | 17 / 17 | 17 / 17 | unchanged |
| Pooled buoy median / p90 | 112.8 / 251.8 m | 112.8 / 228.9 m | median unchanged; p90 lower |

The three March warm repetitions are 85.32, 99.11, and 114.28 seconds, for a
99.11-second median. This is 6.9% faster than the valid nine-layer warm median
of 106.47 seconds, retains 8,447/8,476 fold-free nodes (99.66%), and retains
the same 52/56 buoy results within 2 km. It is still 4.41x the production ORB
warm median of 22.47 seconds. The smaller March gain is consistent with
adaptive early stopping already limiting work on easier matcher calls; the
harder N-ICE calls benefit more from the explicit three-layer cap.

This was an intermediate decision. Section 19.10 supersedes it: three-layer
LightGlue did not retain enough field support on the longer 2015 development
pair, whereas five-layer LightGlue with MNN target-tile ranking did.
MNN remains rejected as the final feature matcher. Primary outputs from this
stage are:

- `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/learned_feature_pilots/selected_sequence_6775_6998_aliked1024_lightglue3_cpu_v1_20260820`;
- `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/learned_feature_pilots/selected_sequence_6775_6998_aliked1024_mnn_cpu_v1_20260820`;
- `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/dense_sequence_10245_10341_10352_lightglue3_seqprior_v1_20260820` and its two warm repetitions;
- `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/dense_sequence_10245_10341_10352_mnn1024_seqprior_v1_20260820`.

### 19.10 LightGlue call reduction and five-layer result

The expensive unit in the tiled matcher is one LightGlue call for every
source-tile/physically-reachable-target-tile edge, not one call per image pair.
The two March pairs contain 1,122 such calls; the three N-ICE pairs contain
4,845. The N-ICE source tiles have medians of 7, 13, and 15 reachable target
tiles per pair, with maxima of 9, 21, and 25. This explains why changing only
LightGlue's transformer depth produced an inconsistent speed gain: the call
graph itself remained large.

The diagnostic implementation now exposes the following LightGlue config in
the pair and sequence manifests while retaining the former nine-layer Kornia
wrapper as the default:

- transformer layers, depth confidence, width confidence, and match-filter
  threshold;
- the former Kornia LAF adapter or a direct adapter that supplies only ALIKED
  keypoints, descriptors, and image sizes;
- optional compile mode;
- per-call timings, feature counts, raw and physics-valid support, adaptive
  stopping layer, pruning counts, and exact-MNN comparison;
- an optional MNN-ranked limit on the number of target tiles sent to
  LightGlue.

The direct adapter was numerically identical to the wrapper on all controlled
warp fields and was 14.2% faster in that bounded run. The more aggressive
`depth_confidence=0.90`, `width_confidence=0.95` speed config also passed the
controlled warps, but failed the real 6603-to-6689 development diagnostic: it
lost 5.9% of fold-free nodes and increased the seven-buoy median endpoint error
from 120.1 to 159.0 m. Controlled warps alone were therefore insufficient for
matcher selection. Three layers with default confidences also lost 1.8% of
field nodes on this pair. Five layers with default confidences restored the
nine-layer result (4,292 versus 4,291 nodes and the same seven-buoy median) and
became the fixed base config:

```yaml
matcher: lightglue
lightglue_layers: 5
lightglue_depth_confidence: 0.95
lightglue_width_confidence: 0.99
lightglue_filter_threshold: 0.10
lightglue_adapter: direct
lightglue_compile: false
mnn_candidate_limit: 8
```

For each source tile, exact cosine MNN cheaply scores every target tile already
allowed by the physical search. Target tiles are ranked first by their count of
physics-valid MNN matches, then by median descriptor similarity and stable tile
ID. LightGlue is run only on the best eight target tiles. The limit is on
target-tile hypotheses, not on ALIKED features or final matches. LightGlue
still performs the final contextual matching; exact MNN alone remains rejected.

The limit was selected only on the 6603-to-6689 development pair. Limits of
four, six, eight, and ten retained 83.8%, 98.5%, 100.1%, and 100.0% of the
five-layer/all-candidate field nodes respectively. Four also reduced buoy
availability to 5/7. Eight was the smallest setting that retained all seven
buoys and the complete field. The actual eight-tile run cut LightGlue calls
from 1,585 to 784 (-50.5%), matching time from 126.30 to 78.89 seconds (-37.5%),
and pair total from 134.68 to 87.37 seconds (-35.1%). It retained 4,297 versus
4,292 fold-free nodes, identical buoy median/p90 errors, and a 5.5 m P90 vector
difference on common field nodes.

The fixed config then passed the separate three-pair N-ICE evaluation without
further tuning:

| Metric | 9-layer, all candidates | 5-layer, MNN top 8 | Change |
|---|---:|---:|---:|
| LightGlue calls | 4,845 | 2,388 | -50.7% |
| Matching time | 511.42 s | 224.64 s | -56.1% |
| End-to-end time | 564.52 s | 283.75 s | -49.7% |
| Fold-free nodes | 7,937 | 7,941 | +4 nodes |
| Buoys available / within 2 km | 17 / 17 | 17 / 17 | unchanged |
| Pooled endpoint median / p90 | 112.79 / 251.79 m | 112.79 / 251.79 m | unchanged |

The MNN ranking itself took 3.60 seconds across the three pairs. Common-node
vector differences from the nine-layer field had per-pair medians of 1.0-2.3 m
and P90 values of 32.2-37.4 m. The largest differences were 721-1,105 m, so the
localized tail must be mapped and attributed before promotion even though the
field-wide statistics pass.

The exact buoy-array deformation audit is also close and mixed rather than
uniformly favourable. Both configs provide all 17 endpoints and all 40 buoy
pairs. The top-eight config changes pooled relative-displacement median error
from 92.56 to 95.45 m, improves its P90 from 287.24 to 280.86 m, changes median
gradient Frobenius error from 0.002176 to 0.002024, and changes median triangle
area-ratio error from 0.004043 to 0.004749. Each has one incorrectly oriented
buoy triangle. These small mixed changes do not indicate a deformation-accuracy
loss, but the sample is too small to claim improvement.

Primary output and the source-hashed deformation comparison are under:

- `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/learned_feature_pilots/selected_sequence_6775_6998_aliked1024_lightglue5_direct_mnntop8_cpu_v1_20260821`;
- its `buoy_array_deformation_pairwise_comparison_metrics/` subdirectory.

Decision: the five-layer/direct/MNN-top-eight config supersedes three-layer
LightGlue as the leading CPU ALIKED candidate, but production remains ORB and
the frozen CUDA handoff remains unchanged. This is a call-routing improvement,
not evidence that ALIKED is yet CPU-operational: the March fair benchmark has
not been rerun with a workload that contains enough target-tile candidates to
exercise the limit.

Section 19.11 completes this predeclared 2020 confirmation run without opening
the final buoy holdout. The task was to select a sequence containing a long-gap
pair whose median reachable-target count exceeds eight, then run these fixed
arms on identical cached features: five-layer/direct/all-candidate and
five-layer/direct/MNN-top-eight, with the existing nine-layer result as the
scientific reference. Require repeated warm timing, endpoint and buoy-array
deformation parity, fold-free/common coverage, and a map of every common-node
vector difference above 500 m. If it passes, update the CUDA handoff and test
GPU-specific mixed precision, FlashAttention, and compile settings as separate
runtime ablations with CPU vector/support parity. Keep the score-before-physics
candidate-ordering issue as a separate diagnostic; changing that order did not
improve the development field and must not be folded into this promotion test.

### 19.11 Within-2020 confirmation of MNN target-tile routing

The selection and evaluation are complete. The config was frozen before match
evaluation at
`experiments/configs/aliked_lightglue_confirmation_20260822.json`. Selection
used only confirmation-fold transition counts, elapsed time, EPSG:3413 image
geometry, and physics-reachable tile counts. It did not use descriptor matches,
buoy error, field coverage, or the final buoy holdout.

The selected January chain is 721→731→740→849. Its three adjacent pairs provide
18, 16, and 15 confirmation-buoy transitions. The last gap is 21.41 hours,
with 1,051 geometrically reachable source/target tile edges and a median of nine
target tiles per source tile. The primary split holds out whole buoy paths, not
SAR images, so this is within-2020 confirmation on different buoys rather than
generalization to unseen imagery.

The nine-layer/Kornia reference and fixed five-layer/direct config agree well:

| Pair | 9-layer nodes | 5-layer nodes | Common-vector P90 | Maximum difference |
|---|---:|---:|---:|---:|
| 721→731 | 7,312 | 7,307 | 10.5 m | 268.1 m |
| 731→740 | 7,947 | 7,945 | 11.9 m | 405.7 m |
| 740→849 | 7,379 | 7,391 | 18.4 m | 318.4 m |

Both have 47/49 confirmation buoy estimates and 345 available buoy-pair
deformation comparisons. Their pooled endpoint medians are both 33.74 m;
endpoint P90 changes from 132.35 to 136.13 m. Pairwise-relative error median
changes from 42.51 to 42.85 m and P90 remains 196.34 m. Median buoy-array
gradient Frobenius error changes from 0.001318 to 0.001176. No common field
difference exceeds 500 m. This confirmation supports five-layer/direct as an
adequate scientific replacement for nine-layer/Kornia on this bounded 2020
sequence; it is not evidence of a general accuracy improvement.

The MNN-top-eight arm is scientifically indistinguishable from
five-layer/all-candidate here. It retains exactly the same fold-free node count
on all three pairs, the same 47 buoy endpoints and 345 buoy pairs, and identical
pooled endpoint and deformation errors. The two short-pair fields are exactly
identical. On 740→849 the common-vector median and P90 are zero, P99 is 0.006 m,
and the maximum is 74.9 m. All predeclared scientific parity gates pass.

The runtime gain does not meet the predeclared 10% normalized promotion gate:

| Warm CPU metric, three repetitions | 5-layer all | 5-layer MNN top 8 | Change |
|---|---:|---:|---:|
| Matching wall median | 146.57 s | 134.55 s | -8.2% |
| Core elapsed median, excluding cache read | 181.96 s | 165.34 s | -9.1% |
| Raw end-to-end median | 215.28 s | 171.31 s | -20.4% |

The raw end-to-end change is not used because Kingston feature-cache reads
varied from 5.85 to 33.45 seconds and happened to favour two top-eight runs.
The exact call-audited 740→849 pair reduces LightGlue calls from 986 to 903
(-8.4%) and audited matching wall time from 82.99 to 77.27 seconds (-6.9%). The
larger 30.1% reduction in routing-stage target hypotheses is not a call count:
it includes candidates with fewer than four physics-valid source features that
the all-candidate arm also skips.

Decision: retain five-layer/direct as the base CPU LightGlue config. Do not
promote a fixed top-eight limit as a universal default and do not update the
CUDA handoff yet. Top eight remains useful for wide no-prior searches such as
N-ICE, where source tiles can reach 21-25 target tiles; it offers limited
leverage when geometry already caps candidates at nine.

The next logical implementation task is an exact no-op fast path: when a source
tile has no more than the configured candidate limit, skip MNN ranking entirely
and call LightGlue on the existing candidates. Then measure runtime by actual
candidate-count stratum and test whether several target-tile hypotheses can be
batched into one LightGlue invocation without altering match support. The
bounded gate remains the same confirmation sequence plus the wider N-ICE
sequence. Primary outputs are under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/confirmation_lightglue_candidate_routing_v1_20260822`;
the concise report, source hashes, deformation tables, and spatial difference
map are in its `comparison/` subdirectory.

### 19.12 Variable-length target batching is rejected on CPU

The proposed target batching was implemented as an experimental direct-
LightGlue path. Reachable target-tile hypotheses for one source tile are padded
to the largest feature count in the group and processed with explicit
self/cross-attention masks. Per-hypothesis adaptive stopping, pruning, match
filtering, target-tile identity, and physics filtering remain separate. A
small variable-length fixture gives the same match indices, scores, stopping
layers, and pruning counts as individual calls.

Real CPU timings reject the approach. Matching times below exclude feature-
cache reads, field estimation, writing, and the MNN candidate-audit time:

| Pair and routing policy | Target batch | LightGlue attempts | Matching time | Change from local batch-1 control |
|---|---:|---:|---:|---:|
| 2020 740→849, all candidates | 1 | 992 | 82.23 s | reference |
| 2020 740→849, all candidates | 2 | 554 | 110.43 s | +34.3% |
| 2020 740→849, all candidates | 4 | 337 | 127.73 s | +55.3% |
| N-ICE 6801→6901, MNN top 8 | 1 | 832 | 72.16 s | reference |
| N-ICE 6801→6901, MNN top 8 | 2 | 430 | 94.17 s | +30.5% |

Thus halving the number of Python/model invocations does not halve the
transformer work. On CPU, padding every hypothesis to the largest source and
target feature set, constructing dense attention masks, and retaining padded
tensor dimensions after per-hypothesis pruning costs more than the removed
invocation overhead. Batch size four makes this worse. This also explains why
ORB can remain faster despite more named stages: its Hamming search, local
correlations, and interpolation operate on much smaller arrays, whereas this
ALIKED design loads roughly 95,000 features per image and evaluates hundreds
of attention problems whose cost grows approximately quadratically with the
features in each tile pair.

The batched fields are close but not bitwise identical. Masked batched linear
algebra changes floating-point ordering, and near-threshold score changes can
alter the consensus input:

| Comparison | Valid matches | Fold-free nodes | Common-vector P90 / max | Availability churn | Buoy result |
|---|---:|---:|---:|---:|---|
| 2020 batch 1→2 | 45,715→45,428 | 7,391→7,391 | 7.21 / 411.90 m | 4 gained, 4 lost | 15/15 retained; error changes <0.001 m |
| N-ICE batch 1→2 | 20,678→20,541 | 2,707→2,709 | 17.85 / 1,010.46 m | 15 gained, 13 lost | 6/6 retained; median 120.44→108.70 m, mixed per buoy |

Both outputs remain fold-free after rejection and retain every buoy endpoint,
but the localized tail means batching is not an exact architectural rewrite.
Its mixed six-buoy change is not evidence of an accuracy improvement.

Decision: keep production ORB unchanged and do not promote CPU target
batching. The experimental option defaults to batch size one. GPU batching is
a separate hypothesis because higher device utilization may reverse the
throughput result; any CUDA test must repeat batch sizes one, two, and four and
require CPU parity in coverage, vector differences, buoy errors, and topology.
The next local task returns to the smaller exact optimization from Section
19.11: bypass MNN ranking when a source tile already has no more candidates
than the configured limit, then report time by candidate-count stratum.

Raw outputs are under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/lightglue_target_batching_v1_20260823`.
The evaluated source SHA-256 values are `2339e5d8...ed8` for
`aliked_matchers.py`, `6d3de2b6...64d6` for `run_aliked_dense_pair.py`, and
`2be4bbde...794f` for `run_aliked_selected_sequence.py`.
The uncorrected `2020_pair_740_849_batch2` directory is superseded by
`2020_pair_740_849_batch2_fixed` and is safe to remove after review.

### 19.13 Selected ALIKED workflow refactor

The selected five-layer/direct workflow is now a standalone
`limosat.learned_drift` package rather than an experiment-script call graph.
Its explicit stages are north-up tile sampling, validity-aware ALIKED
extraction/cache, physics-routed LightGlue, speed filtering, nearest-12 grid
consensus, and fixed-point fold rejection. The optional top-eight MNN policy
has an exact no-op path when physical routing already produces eight or fewer
target tiles. Rejected CPU target batching is not in the core.

Two frozen real-pair gates pass. Pair 740→849 retains 45,715 matches and all
7,391 fold-free nodes with maximum grid-vector difference `1.4e-11` m. N-ICE
pair 6801→6901 with top-eight routing retains 20,678 matches, all 2,707
fold-free nodes, the same seven fold rejections, and maximum grid-vector
difference `2.9e-11` m. The refactor therefore changes architecture, not the
measured deformation product. Full details and the cleanup inventory are in
`docs/aliked_refactor.md`.

The adjacent-sequence gate is also complete using the frozen three-layer March
run as an orchestration control. Both pairs retain their exact match counts,
availability, and support. The first pair falls back to the full physics
window; the second receives the exact preceding-field prior
`(2521.500310992539, -996.010649034501)` m. The remaining cleanup is to retain
old scripts only where they still provide experimental audits or buoy
evaluation not yet backed by the package. This is separate from CUDA
performance work and does not affect use of the ten-year ORB archive for
multisensor validation.

### 19.14 Learned pair persistence prototype

The refactored workflow now has resumable pair persistence without forcing
redetected ALIKED features into the ORB template/trajectory schema. SQLite
stores run/config identity, exact image-pair inputs, processing state, timings,
counts, and array location. One zipped Zarr archive per pair stores every raw
match, regular-grid drift value, support diagnostic, and fold-rejected index.
Feature-cache tensors remain reproducible intermediates and are not duplicated.

The pair key includes stable source/target image IDs, elapsed time, preceding-
field prior, and prior uncertainty. A config SHA-256 locks each run. Writes use
`writing → complete` state only after the temporary Zarr archive has been
closed and atomically renamed; explicit failures remain non-loadable and pass
the retry test. This makes an interrupted pair safe to recompute and prevents a
changed sequential prior from silently reusing an incompatible field.

The real 2020 pair 740→849 round-trips exactly: 45,715 matches, 8,548 grid
nodes, 7,391 available nodes, and all stored numeric arrays are unchanged. The
final Kingston layout saves in 0.342 s, loads in 0.037 s, and occupies 1.315 MB
versus 7.155 MB for the two CSV products. A conventional directory Zarr took
10.7 s for the same pair and is rejected because its small-file overhead is
poor on Kingston and likely poor on a parallel filesystem. The validated store
is under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/aliked_refactor_persistence_v2_20260823`.

This is pair/field persistence, not a claim that ALIKED detections retain
feature identity across images. Virtual material-point trajectories remain a
deterministic derived product of successive stored fields. The next concrete
task is a small manifest-driven sequence runner that loads completed pairs,
computes only missing pairs, propagates the immediately preceding accepted
field prior, and verifies that a stop/restart reproduces the uninterrupted
three-image sequence exactly. Multi-writer SQLite claiming is deferred until
that single-writer restart gate passes.

### 19.15 Optimized EfficientLoFTR sequence gate

The official optimized EfficientLoFTR checkpoint was run on the frozen
721→731→740→849 confirmation chain. Each 35.84 km source core receives one
512-pixel north-up standard-VAE source/target match at 80 m per pixel. The
downstream 30 km/day gate, nearest-12/eight-vector consensus, 4 km grid, and
fixed-point fold rejection are identical to the selected learned-drift field
stages. No confidence threshold, pattern matching, MAGSAC, or buoy-derived
motion prior is used.

Two target-window controls were completed before interpreting buoy results:
same-centre tiles and a global target shift derived only from the immediately
preceding accepted field. The shift changes long-pair common vectors by 24 m
median and 65 m p90 but does not improve coverage (87.38% shifted versus
87.42% same-centre). This chain's true displacement remains within the 2.56 km
tile margin, so it does not test recovery from a genuinely displaced search
window. Same-centre is retained as the simpler result for this gate; a coarse
router is still required before testing larger drift.

| Method | Buoys available / expected | Within 2 km | Median endpoint | P90 endpoint |
|---|---:|---:|---:|---:|
| Production ORB, up to four within 10 km | 45 / 49 | 45 / 49 | 20.0 m | 333.1 m |
| ALIKED nearest-12 | 47 / 49 | 47 / 49 | 33.7 m | 136.1 m |
| Optimized EfficientLoFTR nearest-12 | 47 / 49 | 47 / 49 | 34.4 m | 62.6 m |

Truth-free propagation from each preceding predicted buoy position retains
14/15 complete ORB paths and 15/15 for both learned methods. Final-position
median/P90 errors are 219/797 m for ORB, 57/187 m for ALIKED, and 40/92 m for
EfficientLoFTR. Across 345 available learned-method buoy pairs, relative-
displacement median/P90 errors are 42.9/196.3 m for ALIKED and 45.5/134.5 m for
EfficientLoFTR. EfficientLoFTR has the lower buoy-array affine-gradient error
on all three pairs; the median is 0.000662 versus 0.001176 for ALIKED and
0.001191 for ORB. This is encouraging confirmation evidence, not an independent
image or year result.

EfficientLoFTR displacement coverage is 97.10%, 98.60%, and 87.42%, exceeding
ALIKED by 0.32, 0.79, and 0.96 percentage points. Every accepted displacement
field is fold-free. ALIKED-versus-EfficientLoFTR common-vector medians are
33.0, 33.4, and 38.1 m. On the 21.41 h pair their total-deformation log
correlation is 0.542 on 7,236 common cells; both reproduce similar linework,
but this is method agreement rather than dense deformation truth. A tile-seam
audit finds only 1.9–6.7% higher median adjacent-vector change across learned
tile boundaries, with no corresponding P90 inflation, so the mapped linework
is not explained by tile boundaries.

The complete optimized EfficientLoFTR MPS run took 163.1 seconds versus 215.3
seconds for the warm five-layer/direct ALIKED CPU run: 24.2% less wall time or
1.32× throughput. Device parity is not available. Production ORB remains much
faster: its four relevant CPU image updates took 22.3 seconds inside the
70-image run, although that timing includes preceding operational state rather
than an isolated four-image reset.

Decision: EfficientLoFTR is now a credible learned dense-matching candidate,
not merely a smoke test. Its one-call-per-source-tile architecture is simpler
than ALIKED target-candidate routing, gives comparable or better confirmation
accuracy tails, slightly higher coverage, and useful MPS speed. Do not promote
it to production from this reused chain. The next discriminating test is an
independent within-2020 chain containing displacement beyond one tile margin,
with a non-buoy coarse router, followed by a CUDA comparison on the frozen
N-ICE sequence and altimetry/deformation validation.

### 19.16 EfficientLoFTR long-sequence trajectories and routing

The next gate is complete locally on Apple MPS. A matcher-neutral trajectory
layer now advects fixed material-point IDs through successive fold-free 4 km
fields. It samples each field at the point's current predicted EPSG:3413
position with local piecewise-affine interpolation. Triangles wider than 1.6
grid spacings, unsupported triangles, and orientation reversals are rejected;
no buoy truth enters propagation. The same field and trajectory code accepts
ALIKED, EfficientLoFTR, or future dense-match inputs.

The EfficientLoFTR sequence runner now makes one model call per source tile,
uses local velocity from only the preceding accepted field to centre the next
target tile, falls back explicitly to the preceding global velocity outside
local support, and persists matches, fields, tile-level routing, buoys, strict
trajectories, and support diagnostics. On N-ICE this is 400 model calls across
three pairs versus 2,388 LightGlue calls for the selected ALIKED run. Pair
outputs have config/checkpoint/prior identities and canonical field hashes.
A two-pair stop/restart gate resumed both real pairs with zero model setup in
5.6 seconds after an 80.8-second first computation. An earlier byte hash was
correctly rejected because CSV round trips changed values by up to 2e-12 m;
the final hash quantizes at one micrometre, far below scientific precision,
and the fresh v3 restart passes.

The independent N-ICE 6775→6801→6901→6998 chain includes 14.9, 32.8, and
47.7-hour gaps. The selected same-centre-start/local-routing result is:

| Method | Available / expected | Within 2 km | Endpoint median / P90 | Fold-free field coverage by pair |
|---|---:|---:|---:|---:|
| Production ORB, local average within 10 km | 15 / 17 | 11 / 17 | 620 / 6,278 m | 39.51% pooled, with 49 folds before learned-field QC |
| ALIKED five-layer/direct MNN-top-eight | 17 / 17 | 17 / 17 | 113 / 252 m | 34.20%, 29.24%, 28.92% |
| EfficientLoFTR, local routing | 17 / 17 | 17 / 17 | 83 / 228 m | 33.50%, 29.73%, 30.20% |

EfficientLoFTR retains 8,062/26,079 fold-free nodes (30.91%), 121 more than
ALIKED and 0.46 percentage points more coverage. Its N-ICE MPS run takes
142.7 seconds versus 283.8 seconds for the selected ALIKED CPU run: 49.7% less
wall time or 1.99x throughput, but this is not device parity. A global-only
routing control is slightly worse than local routing: pooled endpoint median
and P90 are 103/235 m, and it retains 104 fewer field nodes.

The unchanged frozen buoy-array deformation validator also favours
EfficientLoFTR slightly on relative motion, while the affine result remains
mixed:

| Method | Buoy pairs | Relative-error median / P90 | Affine-gradient median / maximum |
|---|---:|---:|---:|
| ORB | 31 / 40 | 1,152 / 6,943 m | 0.03095 / 0.05536 |
| ALIKED selected | 40 / 40 | 95 / 281 m | 0.00202 / 0.00296 |
| EfficientLoFTR | 40 / 40 | 83 / 256 m | 0.00226 / 0.00268 |

Thus EfficientLoFTR improves the relative-displacement median/P90 and worst
gradient error, while ALIKED retains the slightly lower median gradient error.
All methods have one incorrect estimated triangle orientation on the middle
pair. These are only three five-to-six-buoy arrays, so the result supports
continued development rather than a claim of general superiority.

Strict N-ICE trajectories retain 1,455/2,342 initial grid points (62.1%)
through all four images. Two of six buoy paths remain field-supported end to
end; their final errors are 33 and 39 m. The low count is an honest support
limitation, not a matching failure at the individual pair level: all 17
pairwise buoy estimates are present and correct. It shows why a long-duration
system needs alternative observation edges rather than requiring every
adjacent field to be present.

A deliberately labelled 96-hour constant-velocity gap arm was tested and is
rejected as a scientific trajectory product. It raises retained N-ICE buoy
paths from 2/6 to 5/6 but degrades final median error from 33 m to 4.42 km.
Field resupport after a blind extrapolation does not restore material identity.
Strict trajectories remain primary; the gap files are diagnostic only. The
next trajectory task is therefore a time-directed image graph with observed
skip edges (for example A→C alongside A→B and B→C), uncertainty-aware path
selection, and explicit reconnection only when a later EfficientLoFTR field
supports the dormant spatial hypothesis.

That observed-edge graph now has an initial N-ICE pilot. Adding independently
matched 6775→6901 (47.7 h) and 6801→6998 (80.6 h) fields to the three adjacent
edges raises complete grid paths from 1,455 to 1,581, an 8.7% relative gain or
5.38 percentage points. Exactly 190 graph rows use observed skip edges; none
uses temporal extrapolation. The graph prefers an available adjacent edge and
uses a skip edge only when that adjacent field is unsupported. It does not yet
recover additional buoy paths because the skip fields have no strict local
triangle at the exact failed buoy hypotheses. This is still useful evidence:
an observed graph recovers real spatial support without the 4-9 km errors of
the velocity-bridging arm, but candidate-field support and multiple hypotheses
must be improved before claiming buoy-path reconnection.

A direct four-day 6775→6998 edge also exposes the limit of the current coarse
router. Its phase response is only 0.007 and the inferred 52.4 km shift is
wrong; the resulting field has 0.23% coverage and no buoy estimates. A
same-centre control retains 6.30% coverage and one correct buoy, so it is better
but still sparse. The 80.6-hour 6801→6998 phase response is also low (0.030),
although its near-zero shift effectively behaves like the same-centre case and
retains 3/5 buoy estimates. Phase response must therefore be stored and audited.
Do not tune a hard response threshold on these five examples. The next robust
start design should retain both same-centre and coarse-shift hypotheses for
uncertain starts, then let coherent field support resolve them; the extra first-
pair cost is amortized over a long sequence.

The six-image March development chain tests a different failure: its first
21.4-hour median buoy drift is 23.6 km, beyond the useful same-centre target
window. A truth-free 1 km projected phase-correlation prior estimates
`(23.87, -12.10) km`, 3.48 km from the median buoy drift, in 0.69 seconds. It
raises first-pair field coverage from 42.3% to 59.5%, buoy availability from
11/29 to 28/29, and correct estimates from 8/29 to 25/29. Across all six
images, strict grid trajectories improve from 1,370/3,277 (41.8%) to
2,456/4,606 (53.3%); 28/29 buoy paths survive, with final median/P90 errors of
108/1,404 m. The 13.7 km maximum remains a serious spatial tail.

The same coarse start is not universally beneficial. On N-ICE, where the
initial median drift is only 7.9 km and same-centre matching already succeeds,
it changes 17/17 available endpoints to 16/17 and pooled median error from 83
to 94 m. Keep phase correlation as an audited start hypothesis, not an
unconditional default. More start pairs must define a geometry- and
response-based selection rule without using buoy outcomes.

Primary outputs:

- March phase/local sequence:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/efficientloftr_sequence_10107_10217_10229_10245_10341_10352_phase_local_mps_v1_20260823`;
- independent N-ICE selected sequence and deformation comparison:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/learned_feature_pilots/efficientloftr_sequence_6775_6801_6901_6998_localprior_mps_v1_20260823`;
- global and coarse-start N-ICE controls are sibling directories ending in
  `globalprior_mps_v1_20260823` and `phase_local_mps_v1_20260823`;
- exact restart gate:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/learned_feature_pilots/efficientloftr_resume_gate_6775_6801_6901_mps_v3_20260823`.
- observed-edge graph pilot:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/learned_feature_pilots/efficientloftr_trajectory_graph_6775_6998_v1_20260823`.

Decision: EfficientLoFTR is the learned method to develop. Keep ALIKED frozen
as the principal learned control and production ORB unchanged for the ten-year
archive. Before CUDA, the highest-value local task is the method-neutral
skip-edge trajectory graph and recovery audit. CUDA remains a performance and
parity gate, not a prerequisite for continuing the scientific architecture.

### 19.17 Full-70 EfficientLoFTR component run

The full 2020 set has now been processed with the optimized EfficientLoFTR
matcher, sequential-local routing, phase-correlation initialization, the
30 km/day physics gate, and the unchanged 4 km nearest-12/eight-agreeing field.
The 70 products are not one spatially connected time series. Their buoy-linked
image graph contains 16 independent overlap components: one eight-image
branched component, one six-image component, and fourteen four-image
components. The branched January 25–29 component was represented by two valid
six-image swath paths. The resulting experiment therefore contains 70 unique
images, 17 valid paths, 57 pair runs, and 55 unique pair edges. No zero-overlap
chronological edge was forced into a sequence.

Across the 693 unique edge/buoy cases, 672 (97.0%) have a learned field estimate
and 652 (94.1% of all expected; 97.0% of available) are within 2 km. Available
endpoint errors have a 40.3 m median, 389 m P90, 1.32 km P95, and 13.64 km
maximum. These conditional error quantiles must be read alongside availability.
The pair-median fold-free field coverage is 84.5%, node-weighted coverage is
72.7%, and 47/57 pair runs retain at least 50% coverage. Four fields fall below
10%, all below 1%. Fold rejection removes 493 nodes and every final field is
orientation-preserving.

| Cadence | Available / expected | Within 2 km / expected | Median / P90 available |
|---|---:|---:|---:|
| 0–3 h | 300 / 306 | 300 / 306 | 31 / 166 m |
| 3–12 h | 86 / 86 | 85 / 86 | 58 / 364 m |
| 12–30 h | 286 / 299 | 267 / 299 | 45 / 825 m |
| 30–72 h | 0 / 2 | 0 / 2 | unavailable |

The only three source-marginal-ice cases and all four target-marginal-ice cases
are unavailable; the remaining 690/689 cases are pack ice. This experiment is
therefore strong evidence for pack ice, not evidence that the current
eight-neighbour field works in lower-concentration ice. The validation month
also has lower availability (26/33) than the other month-defined splits. These
strata are small, but they prevent the pooled 97% availability from being
misread as spatially general performance.

The exact-common comparison with the current production ORB local average
within 10 km contains 230 development-split buoy transitions on these path
edges. Missing predictions remain in the denominator:

| Method | Available / expected | Within 2 km / expected | Median / P90 available |
|---|---:|---:|---:|
| Production ORB q24/i48 quadratic-bilinear | 222 / 230 | 213 / 230 | 48 / 1,249 m |
| EfficientLoFTR phase/local | 220 / 230 | 213 / 230 | 40 / 1,016 m |

Thus the learned run does not improve the 2 km success count and loses two
available cases, but reduces conditional median error by 16.7% and P90 by
18.7%. On the 216 cases available to both methods it has lower error on only
50.5%; the gain is a distribution/tail improvement, not uniform domination.
This is the first larger comparison that supports the qualitative impression
of cleaner learned deformation while retaining the conclusion that production
promotion is premature.

ALIKED was not run over all 55 unique pair edges, so it cannot be inserted into
the 230-case table with the same denominator. The largest valid exact-common
three-method subset is the frozen March 27--29 central chain: 130 development
transitions from 26 complete buoy paths over five adjacent pair fields. An
exact join on buoy ID, source image ID, and target image ID contains all 130
cases for every method:

| Method | Available / expected | Within 2 km / expected | Median / P90 available |
|---|---:|---:|---:|
| Production ORB q24/i48 quadratic-bilinear | 130 / 130 | 125 / 130 | 52.0 / 1,310 m |
| ALIKED nearest-12 | 130 / 130 | 125 / 130 | 47.8 / 1,301 m |
| EfficientLoFTR phase/local | 130 / 130 | 125 / 130 | 45.3 / 1,293 m |

All three methods fail the same total number of cases, and their P90 values
differ by only 17 m. On this subset the main learned-method difference is
therefore spatial support and trajectory retention, not buoy success count.
The following counts refer to published displacement-field nodes, not the much
larger and method-dependent raw descriptor/match populations:

| Method | Native/output sampling | Available output nodes over five pairs | Pooled field coverage | First-image trajectory cohort | Measured runtime for these images |
|---|---|---:|---:|---:|---:|
| Production ORB | Irregular persisted points (about 5 km median nearest-neighbour in the larger archive); queried onto the common 4 km grid | 24,028 / 32,494 | 73.9% | 1,421 | 42.43 s CPU for the six image updates inside the full-70 run |
| ALIKED nearest-12 | Virtual material points on a regular 4 km grid | 26,766 / 32,494 | 82.4% | 4,603 | 769.60 s CPU for the five historical full-context pair runs |
| EfficientLoFTR phase/local | Virtual material points on a regular 4 km grid | 27,285 / 32,494 | 84.0% | 4,606 | 269.71 s Apple MPS sequence total; 262.62 s pair compute |

For the same six images, trajectory length below means the number of images in
which a first-image material-point ID is observed. The percentages form the
full distribution within each first-image cohort:

| Method | Mean / median observations | 1 image | 2 images | 3 images | 4 images | 5 images | All 6 images |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production ORB | 3.96 / 4 | 7.7% | 16.2% | 12.5% | 22.5% | 17.7% | 23.4% (333) |
| ALIKED nearest-12 | 4.70 / 6 | 2.8% | 9.6% | 12.1% | 17.9% | 5.3% | 52.2% (2,402) |
| EfficientLoFTR phase/local | 4.76 / 6 | 2.4% | 7.9% | 13.1% | 18.0% | 5.3% | 53.3% (2,456) |

The trajectory definitions are not identical. The learned rows use strict
adjacent-field trajectories that end permanently when local field support is
lost. The ALIKED row was derived by passing its five stored nearest-12 fields
through the same matcher-neutral strict advection layer used for
EfficientLoFTR. ORB may skip an image and reconnect the same stored trajectory
ID later;
although only 333 first-frame ORB IDs occur in all six images, 1,128/1,421
(79.4%) are linked again in the final image. The distribution therefore
measures continuous observation more strictly than ORB's operational
persistence capability. Runtime is also descriptive rather than device-fair:
ORB and ALIKED used CPU, EfficientLoFTR used Apple MPS, and the ORB timings are
incremental updates within a longer operational graph. The table should not be
used to infer a hardware-normalized speed ratio.

Strict material-point survival has a 47.7% median across the 17 paths and a
45.0% point-weighted complete fraction; the range is 0–93.4%. Four paths end
with zero original points. Two are genuine routing collapses: the 47.7 h
1793→2039 buoy moves at 35.7 km/day, outside the frozen 30 km/day gate, and the
71.6 h 9535→9872 inherited-prior field has zero coverage. The other two are the
January branched-swath paths: their one-step fields remain mostly accurate, but
no point seeded in the first footprint stays inside every later swath. Zero
full-path survival is therefore not equivalent to zero useful deformation.
Operational learned trajectories need per-image reseeding, observed skip edges,
dormant hypotheses, and reconnection; strict first-image survival remains an
audit rather than the sole product definition.

The run also identifies two concrete router defects. A failed preceding field
currently falls back to a same-centre target; after the 71.6 h collapse this
recovers only 0.77% coverage on the next 24 h pair. It should instead start a
fresh multi-hypothesis coarse search. Conversely, unconditional phase routing
can bias short first pairs while still producing near-complete coverage. On
6440→6450 it applies a 1.65 km shift over 1.63 h and produces 0.80 km median
buoy error; on 13520→13527 a 0.59 km shift produces 0.52 km median error. Field
coverage alone is therefore insufficient to select phase versus same-centre.
The next gate should run both first-pair hypotheses and select using truth-free
match/field consistency, while keeping the buoy result sealed for evaluation.

The 17 completed sequence manifests sum to 3,258 s (54.3 min) on Apple MPS;
matching accounts for 2,168 s and image sampling for 682 s. The production ORB
full-70 run took 404.8 s, so this learned experiment is 8.05× longer end to end.
This is not device parity and the workloads differ (57 learned pair fields
versus the operational ORB temporal graph), but EfficientLoFTR remains far from
the production speed target.

The long restart exposed a persistence boundary case: an in-memory CSV field
changed by only 7.3e-12 m, but its one-micrometre-rounded hash landed on the
opposite rounding boundary. The writer now reloads the persisted CSV, verifies
that displacement changed by no more than one micrometre, hashes that exact
representation, and propagates it to the next pair. The affected four-image
path was regenerated; all current pair outputs pass their stored content hash.

Primary output:
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/efficientloftr_full70_components_phase_local_mps_v1_20260824`.
The batch manifest contains pooled metrics and the exact-common ORB comparison;
pair, sequence, buoy, cadence, split, and SIC summary CSVs are stored alongside
it.

Decision: retain these full-70 fields as the main local EfficientLoFTR evidence
set. The next implementation task is a minimal truth-free router reset and
same-centre/phase hypothesis competition, followed by a rerun only of the
identified start/collapse cases. The next trajectory task is method-neutral
reseeding and graph reconnection. Do not tune either change on the final buoy
errors, and do not yet expand to lower-SIC claims from these data.

### 19.18 Dedicated EfficientLoFTR branch and assumption gate

Development now continues on `codex/efficientloftr`. The working tree was
preserved and no commit was made. EfficientLoFTR has its own configuration
rather than instantiating `ALIKEDConfig`. It retains only parameters it uses:
projected sampling, tile geometry, endpoint validity erosion, the radial speed
gate, field consensus, score weighting, output-grid spacing, and an explicit
maximum triangle edge. ALIKED detector and LightGlue settings are absent.

The selected parity configuration remains 80 m pixels, 512-pixel tiles,
32-pixel tile margins, 16-pixel endpoint-validity erosion, 30 km/day,
nearest 12 within 6 km, at least eight vectors agreeing within 1 km, raw matcher
score weighting, a 4 km output grid, and a 6.4 km maximum triangle edge. The
sequence runner records all of these values in its identity and manifest. A
new `--maximum-speed-km-per-day` option permits controlled 30/40/50 km/day
runs without changing the default.

Seven local synthetic assumption tests pass. They establish unit-correct
30/40/50 km/day and elapsed-time behaviour; show that eight coherent 20 km/day
vectors win over four internally coherent 46.6 km/day false vectors even when
the 50 km/day gate admits both; expose that raw score weighting can select a
smaller, high-score false cluster whereas uniform weighting selects the larger
coherent cluster; verify endpoint erosion, rounding, and its independence from
the non-overlapping source-tile core margin; and show that a 5 km
grid has 64% as many nodes as a 4 km grid over a fixed rectangular domain.
These are invariants and failure demonstrations, not evidence that 50 km/day,
uniform scores, or 5 km spacing improve real SAR results.

Grid spacing no longer silently defines topology or trajectory interpolation.
The former default behaviour, 1.6 times grid spacing, remains unchanged for
existing callers. EfficientLoFTR passes an explicit 6.4 km limit. A synthetic
5 km-grid test shows why this separation matters: its approximately 7.1 km
cell diagonals cannot be interpolated under a fixed 6.4 km limit, but are
supported under an explicit 8 km limit. Any 5 km experiment must therefore
declare the physical gap-bridging change rather than receive it implicitly.

The 50 km/day question is not only a threshold sweep. Raising the gate expands
the physics-reachable pair domain and may increase tile count. It cannot recover
motion outside the routed target patch: a 512 by 512 tile at 80 m is 40.96 km
wide, so same-centre matching cannot observe a 50 km translation. Sequential
local or coarse phase routing must first remove most of that translation. A
coherent SAR ambiguity can also pass both the speed and eight-vector consensus
tests. The real-image gate must therefore report new high-speed coverage and
new false matches together.

When Kingston is available, run only the affected development cases first:

1. freeze the current 30 km/day outputs and pair identities;
2. rerun the known 30--50 km/day buoy transitions at 30, 40, and 50 km/day with
   identical routing, then repeat only routing-failed cases with the frozen
   same-centre/phase hypothesis competition;
3. use exact-common and missing-in-denominator buoy counts, median/P90/maximum
   error, within-2-km count, field coverage, tile count/runtime, support count,
   residual, fold rejection, and newly admitted-vector speed distributions;
4. inspect every case available only above 30 km/day, including whether its
   local vector cluster is consistent with surrounding deformation and the
   reverse or skip-edge observation when available;
5. retain 30 km/day unless 40 or 50 recovers accurate transitions without a
   material catastrophic-error or topology increase. Prefer 40 over 50 if it
   captures the useful cases with fewer ambiguous candidates;
6. only after selecting the speed rule, test endpoint erosion (8/16 pixels),
   consensus support (6/8/10 of 12), raw versus uniform score weighting, and
   4 km versus explicitly configured 5 km output on the frozen development
   subset. Change one assumption family at a time.

This milestone is locally verified only with synthetic data and focused unit
tests because Kingston is unavailable. The next concrete action remains the
small real-image 30/40/50 km/day recovery audit; full-70 processing is not
justified until that audit separates recovered fast ice from false matches.

### 19.19 New trajectories and observed reconnection

The first production-LiMOSAT concept transferred to the EfficientLoFTR branch
is adding new points in every image without ORB descriptor or template memory. The strict
first-image trajectory product remains unchanged. A second product now uses a
time-directed graph of observed pair fields and starts new trajectory IDs at
valid outgoing-field source nodes in later images when no current or dormant
point occupies the neighbourhood.

The graph has explicit, interpretable states:

- `seed`: part of the initial trajectory set;
- `new_trajectory`: first measured point in a trajectory added to cover a later
  image footprint;
- `observed_adjacent`: propagated through a directly measured adjacent edge;
- `observed_skip_edge`: propagated through an independently matched longer
  edge;
- `dormant`: the point has no observation in this image and receives no
  interpolated or velocity-predicted coordinate.

Every row also records its seed image and whether an observed edge reconnected
the point after at least one missing image. Earlier observed positions remain
available to later skip edges. Reseeding never changes an existing ID. To
reduce obvious duplicate seeds, current observed positions and dormant points'
last observed positions exclude new seeds within an explicit 2 km radius. This
half-grid value is provisional and must be audited at 1, 2, and 3 km on real
fields. It does not claim to solve material identity under large unobserved
motion; possible duplicates after long gaps should be reported rather than
silently merged.

The regular EfficientLoFTR sequence runner now writes both
`trajectories_4km.csv` and
`trajectories_with_new_points_adjacent_graph.csv`. The latter initially contains only
adjacent observed edges and therefore improves spatial cohort coverage but
cannot reconnect a gap. The separate observed-graph evaluator accepts
independently matched skip-pair runs and now writes a graph with new points alongside
the strict graph. No MAGSAC, pattern matching, or temporal extrapolation is
introduced.

Three synthetic graph mechanisms pass:

1. a later field expanding from three to five grid columns adds exactly ten new
   trajectories and carries all 25 through the next image;
2. full repeated coverage adds no redundant seeds when propagated positions
   remain within 2 km of the next grid;
3. a point unsupported on A→B remains dormant, is not replaced by a B seed at
   its last observed location, and is reconnected by an independently measured
   A→C edge with its original ID.

The focused learned-trajectory suite has 24 passing tests. Real coverage gain,
nearest-neighbour distribution, duplicate rate, deformation topology, buoy
trajectory retention, and runtime remain unmeasured until Kingston is
available. The next real-data action is to replay the existing N-ICE adjacent
plus skip-edge pilot through this graph without rerunning the matcher, then
apply the same replay to the March six-image chain. Only after that audit should
the 2 km new-point exclusion radius be changed or the graph be incorporated
into a full-70 rerun.

### 19.20 Leave-one-image-out temporal-edge audit

A controlled omission experiment now tests whether an independently observed
skip edge is consistent with the two adjacent fields it replaces, and whether
changing the incoming routing field alters later results. All coordinates and
displacements are EPSG:3413 metres. Direct and composed vectors are compared
on exact-common supported nodes using the fixed 6.4 km interpolation limit;
missing support remains explicit. The unchanged prefix reruns are exact to
floating-point persistence precision, ruling out matcher nondeterminism as the
source of the observed differences.

Two N-ICE omissions and two independent March omissions were run:

| Chain and omitted image | Direct skip | Direct vs composed vector median / P90 | Exact-common end-point median / P90 | Full vs omission end support |
|---|---:|---:|---:|---:|
| N-ICE, 6801 | 6775→6901, 47.7 h | 61 / 298 m | 82 / 349 m | 1,452 vs 1,730 (+19.1%) |
| N-ICE, 6901 | 6801→6998, 80.6 h | 121 / 600 m | 97 / 477 m | 1,452 vs 1,046 (-28.0%) |
| March, 10217 | 10107→10229, 23.0 h | 43 / 98 m | 54 / 109 m | 2,475 vs 2,480 (+0.2%) |
| March, 10245 | 10229→10341, 21.4 h | 43 / 97 m | 41 / 94 m | 2,480 vs 2,977 (+20.0%) |

The March skip fields are strong: 10107→10229 retains 62.7% field coverage
and 10229→10341 retains 92.3%. Their limited exact skip-edge buoy fixtures are
all correct within 2 km (2/2 and 1/1; median errors 389 m and 53 m), but those
counts are too small for policy tuning. N-ICE remains the harder cadence test:
the 47.7-hour edge improves end support, while the 80.6-hour edge loses it.

Later matching is stable in position but not identical. After the first March
omission, common downstream displacement differences have medians of 25, 30,
and 3 m over the next three edges. After the interior omission, the following
3.27-hour edge differs by 28 m median and 50 m P90. N-ICE's following edge
differs by 43 m median and 194 m P90. These are small absolute offsets, but a
spatial derivative over a 4 km grid and a short elapsed time amplifies them:
direct-versus-composed total-deformation Spearman correlations are 0.43--0.64
for March and 0.51--0.59 for N-ICE, and the short downstream March edge is only
0.25 after the interior omission. Therefore displacement agreement does not
by itself establish deformation equivalence, and direct/composed fields should
not be averaged blindly.

Decision: observed skip edges are worth carrying into the trajectory graph,
but not as an unconditional replacement for adjacent edges. The evidence
supports a sparse temporal observation graph in which adjacent and skip edges
remain separate measurements. A truth-free local closure residual compares a
direct A→C vector with A→B→C composition, alongside support, consensus
residual, routing confidence, and topology. A skip edge may reconnect an
unsupported trajectory; where both paths exist and disagree, retain the
alternative hypothesis or flag the deformation rather than silently commit.
Do not use constant-velocity interpolation as the substitute.

The next concrete experiment is to replay both chains with adjacent and skip
edges simultaneously, rank or fuse paths using only these truth-free closure
diagnostics, and then unseal buoy errors and deformation differences. If that
passes, generate skip edges only around support collapses within the existing
temporal window rather than matching every image combination.

Primary outputs:

- N-ICE inference and analysis:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/learned_feature_pilots/efficientloftr_leave_one_out_nice_6775_6998_v1_20260827`;
- March inference and analysis:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/efficientloftr_leave_one_out_march_v1_20260827`;
- frozen March experiment plan:
  `experiments/configs/efficientloftr_march_leave_one_out_20260827.json`.

### 19.21 Truth-free closure-fused temporal graph

The adjacent and skip fields have now been replayed simultaneously. This is a
field replay only; it does not rerun EfficientLoFTR. The policy was frozen on
N-ICE before reading buoy outcomes and then applied unchanged to March.

This gate follows one fixed first-image grid cohort so that recovery and
accuracy are interpretable; per-image addition of new trajectories is disabled.
The policy is:

1. the available observation skipping the fewest images is the primary path;
2. alternative endpoints within the existing 1 km field-consensus distance are
   treated as locally consistent;
3. consistent endpoints are averaged with weight
   `selected_matches / max(maximum_residual_m, 80 m)^2`;
4. a conflicting alternative remains in the candidate ledger but does not move
   the primary endpoint;
5. only fold-free fields and the existing 6.4 km interpolation limit are used;
   there is no velocity extrapolation.

The 1 km distance, 80 m floor, and 6.4 km topology limit are existing physical
scales, not buoy-tuned values. The sealed and unsealed N-ICE trajectory CSVs
are byte-identical, confirming that reading truth did not change the replay.
Synthetic tests cover consistent fusion, conflict exclusion, and observed
skip-edge reconnection.

| Dataset | Adjacent-only complete | Shortest observed graph | Closure-fused graph | Relative fused gain | Reconnected rows | Multi-path conflicts |
|---|---:|---:|---:|---:|---:|---:|
| N-ICE | 1,455/2,342 (62.13%) | 1,581 (67.51%) | 1,582 (67.55%) | +8.73% | 37 | 27/2,378 |
| March | 2,456/4,606 (53.32%) | 2,983 (64.76%) | 2,982 (64.74%) | +21.42% | 611 (608 IDs) | 7/6,739 |

The coverage gain comes from carrying observed skip edges, not from averaging:
fusion changes final support by only +1 trajectory on N-ICE and -1 on March.
It changes positions by 29 m median/138 m P90 on N-ICE and 13 m/40 m on
March. Final cumulative total-deformation correlations between shortest-path
and fused trajectories are 0.929 and 0.910, with median absolute differences
of 0.00312/day and 0.00317/day. Thus fusion is controlled but not deformation-
neutral.

Buoy truth was unsealed only after fixing the policy:

| Dataset | Available / expected | Within 2 km | Median, shortest → fused | P90, shortest → fused | Maximum, shortest → fused |
|---|---:|---:|---:|---:|---:|
| N-ICE | 10/17 | 10/17 | 70.4 → 55.9 m | 97.4 → 97.4 m | 102.4 → 102.4 m |
| March | 137/140 | 125/140 | 69.7 → 57.7 m | 1,068.6 → 1,073.0 m | 15,073.7 → 15,027.8 m |

The frozen policy reduces median buoy error by 20.6% on N-ICE and 17.3% on
March without changing availability or the within-2-km count. March final-
image median error falls from 108.4 to 70.8 m, while its P90 changes from
1,404 to 1,417 m. The catastrophic tail is therefore not solved. No available
buoy comparison falls on one of the 34 conflicting multi-path states, so buoy
truth does not yet validate the conflict action. Also, the skip fields recover
many grid trajectories but no additional buoy trajectory; the exact failed
buoy hypotheses remain outside locally supported triangles.

Decision: retain the temporal observation graph and the closure candidate
ledger as the leading trajectory architecture. Treat closure fusion as a
promising accuracy option, not yet a production default. The next operational
experiment should generate skip matches selectively around newly dormant
spatial hypotheses or major support collapses, rather than compute every image
combination. Measure the extra matcher calls and recovered accurate support on
the full-70 development paths. Separately inspect the small conflict set using
reverse edges, another independent skip path, or deformation continuity; do
not tune the conflict rule on the current buoy sample.

Primary outputs:

- frozen truth-free policy:
  `experiments/configs/efficientloftr_closure_policy_v1_20260828.json`;
- N-ICE sealed and unsealed graph audit:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/learned_feature_pilots/efficientloftr_closure_graph_nice_v1_20260828`;
- independent March graph audit:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/efficientloftr_closure_graph_march_v1_20260828`.

### 19.22 Targeted non-consecutive matching around lost trajectories

The first matcher-call reduction gate is complete. Instead of matching every
tile for a non-consecutive image pair, the runner first replays the adjacent-
only trajectory graph. It selects trajectories that are observed at the later
pair's source image but absent at its target image, buffers their last observed
positions by the existing 6.4 km fold-free interpolation distance, and runs
EfficientLoFTR only for intersecting source-tile cores. Tile identities and
routing are unchanged. The selected positions and buffer are included in the
pair identity, so a partial run cannot be mistaken for a full-source run.

The gate was designed on N-ICE, frozen, and then replicated on the March chain.
Each targeted run was compared with a full-source control using the same
routing and the same shortest-observation trajectory policy.

| Dataset | Adjacent-only complete | Full non-consecutive complete | Targeted complete | Gain recovered | Matcher calls, targeted / full | Matching time reduction | Correct buoy comparisons, targeted / full | Post-rejection folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N-ICE | 1,455 | 1,565 | 1,565 | 100.0% | 104 / 303 | 71.0% | 12 / 12 | 0 |
| March | 1,370 | 2,061 | 2,063 | 100.3% | 122 / 245 | 60.6% | 106 / 106 | 0 |

Buoy availability, within-2-km counts, median error, and P90 error are exactly
unchanged from the routing-matched full controls in both datasets. The
targeted/full trajectory positions have zero median difference at every image;
the largest P90 is 18.4 m. Cumulative trajectory-derived total-deformation
Spearman correlation is at least 0.894 on N-ICE and 0.884 on March. N-ICE pair
wall time falls by 67.1%, and March by 56.4%.

The sparse non-consecutive recovery field itself has different outer support
and triangulation from a full-source field, so it is retained as an internal
reconnection measurement rather than emitted as a standalone deformation
product. The deformation gate is applied to the resulting cumulative
trajectory graph, which is the scientific output. Direct sparse-field
comparisons remain in the reports as diagnostics.

Decision: the targeted recovery policy passes the frozen gate and should
replace full-footprint non-consecutive matching in the next EfficientLoFTR
sequence pilot. The next task is a long-sequence scheduler that requests these
matches only after a measured support collapse, persists the resulting image-
pair links, and reports total calls per recovered trajectory. It should retain
a full-source diagnostic pair at fixed intervals to detect selection bias.

Primary outputs:

- frozen gate:
  `experiments/configs/efficientloftr_targeted_recovery_gate_v1_20260828.json`;
- N-ICE gate report:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/learned_feature_pilots/efficientloftr_targeted_recovery_nice_v1_20260828/gate/report.md`;
- March gate report:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/learned_feature_pilots/efficientloftr_targeted_recovery_march_v1_20260828/gate/report.md`.

### 19.23 Scientific product definition and release readiness

The learned trajectory architecture is now classified as **SAR-derived
advected Lagrangian sea-ice trajectories**. Fixed virtual material-point IDs are
carried through independently measured pairwise displacement fields. They are
valid Lagrangian trajectories, but they are not direct records of one ALIKED or
EfficientLoFTR feature being recognized in every image. Production ORB remains
the directly matched feature-trajectory reference.

The authoritative product definition, four-level release lineage, planned
trajectory-schema additions, uncertainty requirements and public-release gates
are in section 5.4 of
`docs/method_neutral_ice_drift_benchmark_handoff.md`. The four levels are:

1. pair correspondences and matcher diagnostics;
2. pairwise displacement fields;
3. advected material-point trajectories; and
4. deformation and Lagrangian-cell products.

This distinction strengthens rather than invalidates the present experiments.
The pair field is the fundamental measurement; the trajectory is a reproducible
composition of supported fields. The targeted recovery field remains an
internal observed reconnection measurement rather than a standalone deformation
product, as already decided in section 19.22.

Hutter et al. (2018) identify two release-critical risks for this architecture:
position error accumulates during field advection, and inconsistent or
interpolated gridded fields can create artificial deformation. Current
EfficientLoFTR fields use one source-target image interval across every tile,
which avoids mixing acquisition times within a pair. It does not eliminate
interpolation across motion discontinuities, tile/routing seams, sequential
position error, or amplification of small displacement errors by spatial
differentiation.

The existing evidence already demonstrates why displacement and deformation
must be gated separately. In the leave-one-image-out experiments, downstream
displacement differences are often only tens of metres while direct/composed
total-deformation correlations can fall to 0.25--0.64. Direct and composed
fields therefore remain separate observations unless a truth-free closure rule
and independent deformation validation justify fusion.

The current method-neutral `v1` schema and the Olivia smoke run stay unchanged.
After that run is collected, a versioned additive schema experiment should add
`position_basis`, `position_uncertainty_m`, and `uncertainty_method`. Temporal
states such as observed, dormant and reconnected remain separate from whether a
position came from a direct feature match, an adjacent field, a skip field, or a
diagnostic prediction. Until uncertainty propagation is implemented and
calibrated, missing uncertainty must be reported explicitly rather than inferred
from consensus residuals alone.

The next release-readiness tasks are therefore:

1. complete the pinned Olivia smoke without changing its schema;
2. run the long-sequence targeted scheduler and retain periodic full-source
   controls for selection-bias detection;
3. calibrate per-step and cumulative uncertainty against held-out buoys and
   direct/composed closure;
4. test discontinuity and seam behaviour near leads, ridges and multiple local
   motion modes; and
5. apply the frozen N-ICE deformation, topology, coverage and restart gates
   before describing the output as a public Lagrangian archive.
