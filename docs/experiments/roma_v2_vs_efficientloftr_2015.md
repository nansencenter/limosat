# Pilot campaign: RoMa v2 versus EfficientLoFTR on N-ICE2015

Owner: `seachu`

Status: reviewed; approved for the read-only Milestone 0 audit. Each later
milestone still requires separate human approval.

Science repository: `limosat`

Intended tracked path: `docs/experiments/roma_v2_vs_efficientloftr_2015.md`

Execution profile: Research Workbench `profiles/vega.toml`

Bulk outputs: outside Git under the Vega result root

This document is the durable experiment contract. A Codex task should execute
only the named milestone requested by the user and stop at every human-review
gate. Do not infer permission for later milestones from approval of an earlier
one.

## 1. Question and allowed decision

Determine whether official RoMa v2 produces more coherent **end-to-end valid
4 km material-point trajectories** than official optimized EfficientLoFTR on
the fixed N-ICE2015 chain `6775 -> 6801 -> 6901` when imagery, projected tiles,
routing windows, physical gates, and downstream geometry are held fixed.

This is a two-pair pilot on reused evaluation scenes, not a final holdout and
not evidence of general superiority. It may support only one of these decisions:

- advance RoMa direct-grid matching to a broader independent benchmark;
- retain RoMa only as a complementary or diagnostic method; or
- reject it from further work at the tested resolution and settings.

It cannot by itself authorize a production matcher change.

## 2. Scope and protected conditions

In scope:

- a small experiment-only RoMa adapter;
- exact reproduction of the current EfficientLoFTR pair baseline on Vega;
- RoMa direct-grid as the primary new arm;
- RoMa sampled correspondences as a secondary adapter and stochasticity arm;
- shared evaluation, figures, manifests, and restart/storage checks.

Out of scope:

- MatchAnything installation or use;
- production matcher changes;
- route optimization, model fine-tuning, confidence calibration on buoy truth,
  or post-hoc selection of a winning threshold;
- pooling this result with unrelated 2020 experiments;
- Git commits unless separately requested.

Protected and immutable during this campaign:

- source scene identity and preprocessing;
- pair times and ordering;
- the current reference `tiles.csv` routes, keyed and validated by tile ID,
  row, column, and source centre;
- analysis CRS, tile geometry, validity masks, speed gate, output grid, and
  downstream evaluation code once its synthetic integrity tests pass;
- pair-specific buoy query positions and truth vectors;
- primary denominator and metric definition after Milestone 0 approval.

Any protected change creates a new campaign version and requires explicit
approval before results are compared.

## 3. Source identity

### Limosat

- Local audit source: `/Users/seachu/projects/limosat`
- Branch: `archive/arctic-tracking-research-20260831`
- Commit: `939c6d42c3f6d9d9b512c53db4a865fa263aba29`
- The local checkout has unrelated uncommitted work. Do not copy, reset, stage,
  or edit it as part of remote source preparation.
- Prepare a clean clone or source archive at the named commit on Vega.
- Campaign code edits may be prepared in a separate reviewed worktree or branch
  after Milestone 0; do not modify production matching modules.

Relevant existing modules include:

- `experiments/run_efficientloftr_sequence.py`
- `limosat/learned_drift/config.py`
- `limosat/learned_drift/efficientloftr.py`
- `limosat/learned_drift/field.py`
- `limosat/learned_drift/imagery.py`
- `limosat/learned_drift/trajectory.py`

### EfficientLoFTR

- Repository: `https://github.com/zju3dv/EfficientLoFTR.git`
- Required commit: `ffd4a4644064354468eb1f0c7a3e732233cb732f`
- Use the official optimized configuration and outdoor checkpoint.
- The official model must be reparameterized before evaluation.
- Preserve the current Limosat confidence normalization only for the baseline
  arm; do not compare it numerically with RoMa overlap as if calibrated.

### RoMa v2

- Repository: `https://github.com/Parskatt/RoMaV2.git`
- Required release tag/package version: `v2.0.1` / `2.0.1`
- Expected release asset:
  `https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt`
- During Milestone 0 resolve and record the full commit targeted by `v2.0.1`.
- During Milestone 1 download once into the Vega model cache, then record and
  verify byte size and SHA-256 before any model run.
- Confirm that the checked-out constructor and the staged weight asset agree.
  Do not silently use a different torch-hub cache entry.
- Record that RoMa code is MIT except for incorporated DINOv3 components, whose
  custom license must be accepted as suitable before download/use.

Official API assumptions that must be tested against the pinned checkout:

- `model.match(A, B)` returns dense `warp_AB`, `warp_BA`, `overlap_AB`,
  `overlap_BA`, `precision_AB`, and `precision_BA` for the precise bidirectional
  setting;
- normalized coordinates are in `[-1, 1]`;
- official interpolation uses `align_corners=False`;
- `model.sample(preds, 5000)` returns sampled matches, overlap confidence, and
  directional 2x2 precision matrices;
- `precise` means 800 x 800 low resolution, 1280 x 1280 high resolution, and
  bidirectional matching.

If the pinned source contradicts any assumption, stop and document the exact
source lines and proposed amendment before implementing around it.

## 4. Vega layout

Resolve paths through the Vega profile. For this pilot use these physical
locations unless Milestone 0 finds an existing verified equivalent:

```text
/Data/sat/seachu/projects/limosat
/Data/sat/seachu/projects/external/EfficientLoFTR-ffd4a464
/Data/sat/seachu/projects/external/RoMaV2-v2.0.1
/Data/sat/seachu/data/limosat/nice2015/roma-v2-pilot/
/Data/sat/seachu/cache/models/efficientloftr/
/Data/sat/seachu/cache/models/romav2/
/Data/sat/seachu/envs/
/Data/sat/seachu/results/limosat/roma-v2-vs-efficientloftr/
```

Do not use `/Home/seachu` for new bulk material. Do not use `/Data/limosat` for
experimental output. Do not put data, models, environments, logs, or results in
Git.

The result directory for each execution must be unique:

```text
YYYYMMDDTHHMMSSZ-roma-v2-vs-eloftr/
  record/       # manifests, exact commands, compact logs, checksums, summaries
  results/      # tables and selected figures
  resume/       # optional restartable dense prediction shards
  scratch/      # replaceable intermediate material
```

Before a substantial run require at least 100 GiB free and write a pre-run
estimated output size. Default campaign cap is 20 GiB unless the user approves
an amended estimate. Cache and environment sizes are reported separately.

## 5. Fixed inputs

The shared 2020 IABP/SIC bundle on Vega is not used.

### N-ICE2015 Sentinel-1 rasters

The raster array convention is `(bands, rows, columns)`. Band 1 is the `uint8`
VAE image. Band 2 is the `uint8` validity mask; values greater than or equal to
2 are invalid. The rasters use ground-control-point geolocation rather than a
single trustworthy affine transform. Use the existing Limosat geolocation and
north-up sampling functions. Analysis coordinates are float64 EPSG:3413 metres.

| ID | Rows x columns | Bytes | SHA-256 |
| ---: | ---: | ---: | --- |
| 6775 | 4994 x 5189 | 33,535,016 | `749006d5c2b63ba64a14cd7fefa70ef5eeffba97374d4312c8997814a9c5c312` |
| 6801 | 4994 x 5213 | 29,193,725 | `4537d99cf2f61abc653f03c9bb97e8f791fb35ffd50307e1e29b82e5ce0cde5c` |
| 6901 | 4994 x 5189 | 32,214,444 | `9d563d3dd174b6f27ab584eaa99d51a77b638cfd2e8cdc4e59626b4a74cd14bf` |

Kingston sources:

```text
/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/sentinel1/standard_vae/2015/04/S1A_EW_GRDM_1SDH_20150428T154520_20150428T154620_005686_0074B8_CD78.tiff
/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/sentinel1/standard_vae/2015/04/S1A_EW_GRDM_1SDH_20150429T063857_20150429T063957_005695_0074F1_9411.tiff
/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2015/sentinel1/standard_vae/2015/04/S1A_EW_GRDM_1SDH_20150430T152851_20150430T152951_005715_007562_59BC.tiff
```

Elapsed time is exactly 14.893611111111111 hours for `6775 -> 6801` and
32.83166666666666 hours for `6801 -> 6901`.

### EfficientLoFTR checkpoint

- Kingston source:
  `/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/model_cache/efficientloftr_official/weights/eloftr_outdoor.ckpt`
- Bytes: 192,797,577
- SHA-256:
  `0af6291141c736e75e94b7f8aae4399b77c7731b3f08758212b2cfe370188878`

### Controlled EfficientLoFTR reference

Transfer all six named files from each directory, excluding AppleDouble files:

```text
/Users/seachu/projects/limosat/results/efficientloftr_reference_pair_6775_6801_20260903
/Users/seachu/projects/limosat/results/efficientloftr_reference_pair_6801_6901_20260903
```

Expected file SHA-256 values:

| Pair | File | SHA-256 |
| --- | --- | --- |
| 6775-6801 | `buoy_results.csv` | `9e537bbb2fb478fc197669aff83715be6674e94bdab9fe50a7aed678d35d7ee5` |
| 6775-6801 | `field_4km.csv` | `619aca77c92da2ca6f7e44220b50325d4be9cde53c86334ce5fe480fa3e58952` |
| 6775-6801 | `matches.npz` | `e067a2df9aa4be105fcb293fa8fe0d84ebbc5c7927786d6cae6bd952fd3d4442` |
| 6775-6801 | `run_manifest.json` | `2ca9a6af384bcfe074986583ce17dcc156f1e346942962165e594231b0629c2d` |
| 6775-6801 | `summary.json` | `e7b753dabd9f0250498f723fc37fc5c9e0546304226109b9a878b83a032242fa` |
| 6775-6801 | `tiles.csv` | `b4c3daa54ebc2f1d61e479ed3be6ff4b119007d8cc6629c67f7e510bbcdd9b49` |
| 6801-6901 | `buoy_results.csv` | `d66d2906fd5fe18c97c1fbade155ca5777fe38f9d6966fdfe9cf6f315a5c169c` |
| 6801-6901 | `field_4km.csv` | `c8312e69e7deae00b9f3be0e12ec125da8e4f2ac65f4ffc88581375a36291723` |
| 6801-6901 | `matches.npz` | `e9ec4757e64bdc566b93d13a99726f2a2ae8105535874df82fc72ed9e7f71f25` |
| 6801-6901 | `run_manifest.json` | `c210448d9aa545d373576e81804ed2d0fa5a0bbb13855a8d3112475652a92715` |
| 6801-6901 | `summary.json` | `9e63f98d4d4d372c8fb463676f802463fe0256c9a7e78f922617d92450f545b6` |
| 6801-6901 | `tiles.csv` | `b229db334828842f65854f28a5a47f8937ba9b159dd00302b18cd54c208495e2` |

The pair-specific buoy inputs for this pilot are the immutable columns
`buoy_id`, source time/position, target time/position, and truth displacement
from these reference CSVs. Existing method prediction columns are not inputs to
RoMa and must not influence thresholds. Milestone 0 must propose a compact
truth-only fixture plus a provenance reference before staging.

### Transfer requirements

- Create a local source manifest before transfer.
- Dry-run the exact copy; exclude `._*`, `.DS_Store`, partial files, caches, and
  unrelated result trees.
- Copy into a temporary destination, verify every expected hash and file count,
  then promote atomically.
- Keep the transfer record in the campaign run `record/` directory.
- Do not remove or alter any source after transfer.

## 6. Fixed geometry and downstream configuration

Use:

- EPSG:3413, metres, `(x, y)` order;
- 80 m projected image pixels;
- 512 x 512 pixel tiles;
- 32 pixel tile margin;
- 32 pixel transform-grid spacing;
- 16 pixel endpoint support radius;
- 30,000 m/day radial maximum speed;
- 4,000 m reporting grid;
- 12 neighbours, at least 8 agreeing;
- 6,000 m maximum neighbour distance;
- 1,000 m agreement distance;
- 6,400 m maximum triangle edge;
- 2,000 m new-point exclusion radius;
- no SIC/open-water gate.

Freeze routes from the current reference `tiles.csv`. Join by tile ID, row,
column, and source centre, and fail if the keys or source geometry are not
one-to-one and identical. Do not recompute pair-2 routes from either new method.

The initial pair's expected median route is
`(1138.8927347788638, -7149.500772510293) m`. This is a check, not a substitute
for validating every route row.

## 7. Evaluation contract

### Fixed pre-matcher population

Before running either matcher, construct a versioned table of 4 km source-grid
identities using only source/target footprints, validity masks, frozen tile
cores/routes, and the speed-reachable domain. The table must include stable
`grid_id`, EPSG:3413 source coordinates, unique tile ownership, pair support
flags, and reasons for geometric exclusion.

Milestone 0 must specify how pair-2 support is evaluated for an advected point
without using a matcher's outcome to redefine the initial denominator. The
definition and table hash are frozen before opening RoMa full-pair results.

### Primary endpoint

`end_to_end_valid_fraction`:

```text
number of fixed initial grid IDs with a valid, coherent estimate for both pairs
-------------------------------------------------------------------------------
number of fixed initial grid IDs in the pre-matcher population
```

Missing estimates remain in the denominator. Report the count and fraction for
EfficientLoFTR and RoMa direct-grid, plus their absolute percentage-point and
relative differences.

The previously reported EfficientLoFTR conditional survival is a secondary
reference: 1,827 complete of 2,388 pair-1 accepted seeds, or 76.51%. Recompute
it on Vega and report any definition mismatch before comparison.

### Secondary endpoints

In priority order:

1. coverage and survival on exactly common IDs;
2. adjacent-vector jump distribution and directional disagreement;
3. topology before and after fixed-point fold rejection;
4. RoMa forward-backward cycle error in pixels and metres;
5. buoy availability, median, p90, maximum, and vector bias, with six expected
   cases per pair retained in denominators;
6. gained-only versus common-node coherence;
7. runtime, model setup, preprocessing, per-tile latency, and peak VRAM;
8. raw prediction counts.

No score scale is assumed comparable across matchers. RoMa overlap and
precision are diagnostic variables, not EfficientLoFTR confidence substitutes.

### Threshold rule

Do not select overlap, precision, cycle, neighbour, or topology thresholds by
looking at buoy error or by choosing the best complete-pair result. During
Milestone 0 either:

- propose externally defensible fixed values and their rationale for explicit
  approval; or
- freeze a small sensitivity grid and report every arm as exploratory without
  selecting a threshold as primary.

The unthresholded physical/domain-valid direct-grid result must always be
reported so the effect of quality gates is visible.

## 8. Method arms

### A. EfficientLoFTR baseline

Use official optimized EfficientLoFTR, the verified checkpoint, fp32, and the
existing Limosat adapter. Reproduce both pairs with frozen routes on CUDA.

Expected current-reference values:

| Metric | 6775 -> 6801 | 6801 -> 6901 |
| --- | ---: | ---: |
| physics-valid matches | 68,910 | 58,077 |
| grid nodes | 6,992 | 9,258 |
| available before fold removal | 2,392 | 2,772 |
| available after fold removal | 2,388 | 2,754 |
| coverage | 34.1533% | 29.7472% |
| fold removals | 4 | 18 |
| buoys available | 6/6 | 5/6 |
| buoy median error | 99.44 m | 91.62 m |
| buoy p90 error | 125.56 m | 193.56 m |
| buoy maximum error | 134.89 m | 226.17 m |

Before proceeding, the implementation must define baseline parity tolerances
for discrete counts, coordinates, and floating metrics. A count mismatch or
failure outside approved tolerance stops the campaign. Do not replace these
references with older archived sequence values.

### B. RoMa sampled correspondences: secondary

- One-tile smoke with official `fast`.
- Main arm with official `precise`; do not manually recreate it if
  `apply_setting("precise")` is present and verified.
- Duplicate the exact band-1 grayscale north-up patch into three RGB channels,
  preserving `float32` values in `[0, 1]` and the same invalid mask treatment.
- Set `torch.set_float32_matmul_precision("highest")`.
- For each tile, call `match`, then `sample(..., 5000)`.
- Convert normalized coordinates with the pinned official convention; apply
  the shared source-core, valid-support, domain, endpoint, and speed gates;
  convert to EPSG:3413; and feed accepted correspondences into the unchanged
  field estimator.
- Run fixed sampling seeds declared in the manifest. Cache one deterministic
  dense prediction per tile so seed arms do not repeat model inference.
- Report overlap-weighted and uniform-weight downstream sensitivity separately.

This arm is not the primary RoMa result because coverage depends on stochastic
sampling and the existing neighbour estimator.

### C. RoMa direct-grid: primary

For each unique fixed grid identity and its unique source tile core:

1. sample `warp_AB`, `overlap_AB`, and `precision_AB` at the exact normalized
   source location using `align_corners=False`;
2. convert the predicted destination with the pinned RoMa pixel convention,
   then to EPSG:3413 with the same tile geometry as the baseline;
3. sample `warp_BA` at the predicted destination and calculate cycle error;
4. apply predeclared domain, support, speed, and quality arms;
5. preserve the original `grid_id` into pair 2;
6. apply separately labelled spatial/topological validation.

Do not use `match_keypoints` for landmark transport. Do not pass these
one-estimate-per-grid predictions through the 12-neighbour consensus as if they
were independent raw correspondences. Do not interpolate between overlapping
tile predictions: tile ownership is unique and fixed before inference.

## 9. Milestones and stopping points

### Milestone 0 — audit and plan only

Read-only. Do not transfer data, clone repositories, download weights, create
environments, edit code, or run a GPU model.

Deliver:

- repository status and preservation plan for unrelated work;
- verification of all paths, hashes, source versions, and current reference
  metrics available locally;
- proposed fixed grid denominator and pair-2 support rule;
- proposed parity tolerances and threshold/sensitivity contract;
- environment strategy, estimated disk use, and exact Vega staging plan;
- exact files that later milestones would add or edit;
- unresolved decisions requiring approval.

Stop for review.

### Milestone 1 — staged inputs and environments

After approval, create the run record, perform exact transfers, verify hashes,
resolve source tags/commits and licenses, create the minimum compatible
environment or separated matcher environments, and run import/version/GPU
preflight only. No scientific pair inference.

Stop if a license, download, dependency conflict, hash, disk floor, or GPU check
fails. Preserve compact evidence and request the smallest needed action.

### Milestone 2 — implementation integrity and one-tile diagnostics

Add experiment-only adapter code and small tests for:

- normalized/pixel round trips including half-pixel boundaries;
- `align_corners=False` interpolation;
- displacement direction and north-up y sign;
- tile-centre offsets and unique core ownership;
- identity warp and forward-backward cycle;
- mask, endpoint, and speed rejection;
- frozen-route joins and fixed grid IDs;
- restart equivalence for one cached tile.

Then run, in order:

1. same-image identity;
2. one central tile with EfficientLoFTR;
3. the same tile with RoMa `fast`;
4. the same tile with RoMa `precise`.

Save full dense warp/overlap/precision only for the identity and representative
tile. Record runtime and peak VRAM. Stop for visual/numerical review before
full-pair runs.

### Milestone 3 — frozen baseline and pair 1

Reproduce both EfficientLoFTR pair summaries first and pass parity gates. Then
run pair 1 for both RoMa modes. Write pair outputs and validate restart. Do not
open buoy metrics until all threshold/sensitivity arms and integrity checks for
pair 1 are complete.

Stop if baseline parity fails or RoMa `precise` exceeds memory/storage limits.
`base` may be proposed as a separately labelled fallback but never substituted
silently.

### Milestone 4 — pair 2 and end-to-end trajectories

Run pair 2 using only the frozen reference routes. Propagate stable grid IDs,
calculate the primary endpoint and all secondary metrics, then open the fixed
buoy evaluation. Do not tune or rerun a preferred threshold after seeing it.

### Milestone 5 — report and retention review

Produce machine-readable summaries and matched figures. Mark authoritative
records/results, resumable caches, and scratch. Verify required checksums and
completion markers. Generate a cleanup dry run listing superseded scratch and
optional dense caches; delete nothing without explicit review.

## 10. Diagnostics and figures

Use identical spatial axes, masks, vector scales, and colour limits across
method panels. Produce:

- fixed pre-matcher population and common evaluable mask;
- per-method availability and end-to-end survival;
- common, RoMa-only, EfficientLoFTR-only, and neither-valid nodes;
- displacement vectors and adjacent-vector jump;
- RoMa overlap, precision-derived scalar summaries, and cycle error;
- topology/fold diagnostics before and after rejection;
- spatial distribution of complete trajectories;
- buoy residual vectors after the blind method outputs are frozen;
- runtime and VRAM table.

Create a fixed zoom around EPSG:3413 `(975000 m, -715000 m)` with the same
extent and scales for all methods, showing vectors, availability, neighbour
jumps, and available RoMa quality diagnostics. Report common-node and
gained-only statistics separately.

## 11. Required records

The initial manifest is written before substantial execution and the completion
marker last. Record:

- run/campaign version and status;
- exact commands, working directories, UTC times, and exit codes;
- host, A100 identity, driver, CUDA, PyTorch, Python, package inventory, and
  peak VRAM;
- every repository URL, full commit, branch/tag, dirty state, and patch/source
  hash for experiment code;
- input, reference, config, model, and fixture hashes;
- data shapes, dtypes, band/mask semantics, CRS, axis order, pixel spacing,
  pair times, and missingness denominators;
- resolved routes and fixed-grid table hash;
- per-tile timing/count summaries and compact error evidence;
- output paths, sizes, checksums, and artifact classes;
- interpretation, limitations, and next action.

Do not rely on chat history for any of these facts.

## 12. Final handback

The final report must distinguish:

- environment/setup success;
- baseline reproduction;
- implementation diagnostics;
- exploratory threshold arms;
- frozen primary and secondary results; and
- scientific interpretation.

Recommend `advance`, `complement`, `reject`, or `inconclusive`, with the
two-pair/reused-data limitation stated beside the recommendation. Report what
was retained, what is eligible for cleanup, and the next smallest independent
experiment.

## 13. Official references

- RoMa v2 repository and usage: <https://github.com/Parskatt/RoMaV2>
- RoMa v2 pinned source path to inspect:
  <https://github.com/Parskatt/RoMaV2/blob/v2.0.1/src/romav2/romav2.py>
- EfficientLoFTR repository and official inference setup:
  <https://github.com/zju3dv/EfficientLoFTR>
