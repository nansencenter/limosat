# EfficientLoFTR Vega performance audit — 2026-09-06

## Scope

This is an isolated performance investigation. No production implementation was
changed. Tests used the completed NICE-2015 Vega experiment, its 185 frozen
512x512 uint8 SAR tile pairs, the official optimized outdoor checkpoint, PyTorch
2.6.0/CUDA 12.6, and an NVIDIA A100-PCIE-40GB.

The completed run itself took 10.437 s, including 3.532 s of model setup. Its two
pair inference totals were 3.728 s for 86 tiles and 1.650 s for 99 tiles. The
first CUDA call took 2.210 s; steady per-tile medians were 16.9–17.0 ms.

## Conclusions

1. Reparameterization is already correct and beneficial.
2. The frozen rasters are already 512x512 tiled LZW GeoTIFFs. Misaligned reads
   are measurable, but they are not starving this A100 when four readers are
   used, and production currently caches whole scenes instead of reading one
   raster window per tile.
3. Full-model CUDA Graph padding works but wastes too much fine-stage work.
   Capturing only the static backbone and coarse transformer is the useful graph
   strategy.
4. Outer FP16/BF16 autocast, dual-softmax, disabling SDPA, and higher match
   thresholds all fail either throughput or output-preservation criteria.
5. The best new low-risk candidate is vectorizing the transform-grid
   interpolation used by `north_up_patch`. It is mathematically and empirically
   exact against the current implementation and reduces tested geospatial remap
   time by 2.33–3.73x.

## Model/configuration audit

| Test | Throughput/result | Output evidence | Decision |
|---|---:|---|---|
| Reparameterization | 20.85 to 18.04 ms, 1.156x | Exact coarse matches; coordinate p95 below 0.0005 px | Keep; already active |
| Optimized matcher, eager batch 1, TF32 | 65.6 tiles/s | Reference | Test baseline |
| Dual-softmax, official threshold 0.2 | 38.7 tiles/s; 164,581 vs 144,841 matches | 0/64 exact sets; median Jaccard 0.213 vs optimized mode | Reject; skip-softmax is already active |
| Disable SDPA | 53.5 tiles/s | 41/64 exact sets; minimum Jaccard 0.890 | Reject |
| FP16 autocast | 55.7 tiles/s | 22/64 exact; median Jaccard 0.997; coordinate p95 3.87 px | Reject |
| BF16 autocast | 56.3 tiles/s | 11/64 exact; median Jaccard 0.941; coordinate p95 6 px | Reject |
| Threshold 27 vs 25 | 86.5 vs 80.7 prefix-graph tiles/s | 9.5% fewer total matches; 8/64 exact sets | Do not adopt without downstream validation |
| Threshold 30 vs 25 | 91.8 vs 80.7 prefix-graph tiles/s | 29.9% fewer total matches; 6/64 exact sets | Reject as a generic optimization |

The pinned model has no adaptive, data-dependent token-pruning threshold. Its
coarse transformer always uses fixed stride-4 depthwise aggregation/max pooling.
The dynamic tensor is created later by mutual-nearest-neighbour selection with
`torch.where`; it is the number of coarse matches sent to fine refinement.

Source inspection confirms that SDPA is selected when `NO_FLASH=False`, as in
the active optimized configuration. A kernel-level profiler table was not
captured because the separate temporary profiler upload was not authorized.

## CUDA Graph and batch audit

All same-batch graph comparisons below were exact against eager execution on 64
tiles.

| Execution | Batch | TF32 matmul | Tiles/s | Peak reserved VRAM |
|---|---:|---:|---:|---:|
| Eager | 1 | on | 65.6 | 1.64 GB |
| Static-prefix graph | 1 | on | 80.7 | 1.80 GB |
| Static-prefix graph | 2 | on | 89.6 | 2.19 GB |
| Static-prefix graph | 4 | on | 94.4 | 4.10 GB |
| Static-prefix graph | 8 | on | 95.7 | 8.30 GB |
| Static-prefix graph | 16 | on | 95.5 | 16.37 GB |
| Eager, Vega backend defaults | 1 | off | 55–58 | 1.53 GB |
| Static-prefix graph | 1 | off | 63.6 | 1.63 GB |
| Static-prefix graph | 8 | off | 72.7 | 8.30 GB |

Batch 8 is the throughput knee. It is not identical to batch 1: with TF32 off,
53/64 tiles had exact coarse sets, median Jaccard was 1.0, minimum Jaccard was
0.975, and common-match fine-coordinate p95 was 2.07 px. With TF32 on, 38/64
sets were exact and minimum Jaccard was 0.891. A batch change therefore requires
an end-to-end drift-field/trajectory regression.

The completed adapter leaves matmul TF32 off. Enabling it alone increased eager
batch-1 throughput from 57.8 to 66.3 tiles/s, but only 39/64 coarse sets were
exact against strict FP32. Enabling cuDNN autotuning alone did not help (55.1
tiles/s). TF32 is an optional accuracy/performance arm, not a safe default.

Prefix capture at strict FP32 and batch 1 is exact, but its 2.35 s capture/setup
cost makes it approximately break-even for this 185-tile job. It is useful when
one resident model processes a longer stream.

### Full graph padding

At a 4096-match cap, the full graph was lossless but achieved only 70.4 tiles/s.
Median real matches were 123, p90 2,567, p95 2,987, p99 3,181, and maximum
3,328. Overall slot utilization was 19.1%, while the median tile used only 3.0%
of its slots. A cap of 2,048 reached 76.7 tiles/s but overflowed 38/185 tiles and
lost 23,045 matches. Full-model padding is rejected.

## COG, NPZ, RAM, and remap audit

All 185 frozen inference fixtures are exact 512x512 uint8 images; no loader
padding or resize artifact was found. The source rasters used in this experiment
have 512x512 blocks, LZW compression, and pixel interleaving.

### Native COG windows

| Read pattern | Workers | Amplification | Tile pairs/s |
|---|---:|---:|---:|
| Block aligned | 1 | 1x | 137.7 |
| Block aligned | 4 | 1x | 536.2 |
| Block aligned | 8 | 1x | 568.1 |
| Shifted 256 px in x/y | 1 | 4x | 66.0 |
| Shifted 256 px in x/y | 4 | 4x | 221.8 |

The asynchronous four-reader plus prefix-graph batch-8 pipeline sustained 93.1
tiles/s for both aligned and deliberately misaligned native windows; read work
was hidden behind GPU work. These are native-pixel throughput fixtures and are
not scientifically equivalent to the GCP/TPS north-up tiles.

### Exact geospatial window reads versus whole-scene caching

On the 86-tile first pair, cached full-scene load plus production remapping took
3.586 s with one worker. Block-aligned windowed remapping took 4.840 s, fetched
236.5 million spatial block pixels, and was 35% slower. It also differed from a
same-process full-scene remap in 691 of 45.1 million image pixels (0.0015%, max
4 DN), although masks were exact. Whole-scene caching is preferable for this
small scene working set.

### Vectorized transform-grid interpolation

The production transform-grid expansion performs 17 row-wise and 512
column-wise `np.interp` calls for each map. A vectorized endpoint-aligned
bilinear expression produces bit-identical float32 grids.

| Pair | Production remap, 4 workers | Vectorized remap, 4 workers | Speedup | Live parity |
|---|---:|---:|---:|---|
| 6775→6801, 86 tiles | 2.742 s | 1.177 s | 2.33x | All images/masks exact |
| 6801→6901, 99 tiles | 3.540 s | 0.948 s | 3.73x | All images/masks exact |

Including scene preload, the combined accepted-tile preparation time fell from
7.369 s to 3.218 s (2.29x). Four workers were best; eight workers regressed.

Both the production and vectorized recomputations differ slightly from the
older frozen NPZ images (less than 0.1% of pixels; p95 absolute difference 0),
while agreeing exactly with each other in the same process. This is pre-existing
cross-process GCP/TPS/remap variability, not a vectorization regression.

### NPZ cache

The 185 compressed fixtures occupy 63.7 MB and decode to 194.0 MB. Preload time
was 1.169 s with one thread, 0.568 s with two, 0.340 s with four, and 0.739 s
with eight. NPZ is a good immutable A/B-test cache when its identity includes
scene hashes and all warp/preprocessing parameters. It should not replace the
source-of-truth COG pipeline.

### Larger-run RAM cache trace

For the separately inspected 2,355-pair trace (9,420 scene accesses, 143 unique
scenes, 7.38 GB of files), the current typed LRU-4 keying produced 5,210 misses.
Canonical string keys reduced this to 2,583; LRU-32 to 1,537; LRU-64 to 499; and
full 143-scene capacity to 143. In a page-warmed 20-pair test, canonical LRU-4
was 2.72x faster than current keying and full capacity was 3.88x faster. The
estimated whole-run saving is about four minutes (roughly 1–2%), potentially
more on cold NFS. The completed three-scene Vega experiment already fits in
LRU-4.

## Ranked next actions

1. Review the now-validated vectorized `_interpolate_grid` branch. Its unit,
   full-cohort, raw-match, field, and trajectory checks are exact.
2. Localize the rare batch-4 field/trajectory outliers and repeat the final
   field and trajectory comparison across more completed image pairs before
   changing the batch-1 default.
3. For long streams, trial static-prefix capture at batch 1 for strict output
   preservation. Do not use a full padded graph.
4. For the 143-scene workflow, canonicalize cache keys and size the scene cache
   from an explicit RAM budget. Keep the current simple LRU-4 for this three-scene
   experiment.
5. Retain optimized skip-softmax, SDPA, FP32, threshold 25, reparameterization,
   and 512x512 inputs as defaults.

## Candidate disposition

The following decisions apply to the current 512-pixel SAR production plan.
Rejected candidates should not be retested unless the model, tile geometry, or
hardware changes materially.

| Candidate | Disposition | Reason |
|---|---|---|
| Vectorized transform-grid interpolation | **Proceed to review** | Exact through tiles, raw and accepted matches, 4 km fields, and final trajectories; 2.29x faster preparation. |
| Reparameterized optimized model | **Retain** | Already active; 1.156x matcher improvement with negligible coordinate change. |
| Skip-softmax coarse matching | **Retain** | Already active; dual-softmax was 41% slower and changed the match population substantially. |
| SDPA attention | **Retain** | Already active; explicit attention was slower and changed outputs. |
| FP32, coarse threshold 25 | **Retain** | FP16/BF16 and thresholds 27/30 failed output-preservation criteria. |
| Static-prefix CUDA Graph, batch 1 | **Conditional candidate** | Exact output; useful only when a resident matcher processes enough tiles to amortize capture. |
| Static-prefix CUDA Graph, batch 4 | **Expanded testing** | Best performance/VRAM compromise, but rare downstream differences remain. |
| Static-prefix CUDA Graph, batch 8 | **Discard for production** | Only 1.3% faster than batch 4, twice its VRAM, and a worse terminal p99 difference. |
| Full-model padded CUDA Graph | **Discard** | Low slot utilization; the lossless cap was slower than the prefix graph and smaller caps lost matches. |
| Outer FP16 or BF16 autocast | **Discard** | Slower than FP32 here and produced multi-pixel coordinate differences. |
| Re-enable dual-softmax | **Discard** | Large throughput and match-population regression. |
| Disable SDPA | **Discard** | Throughput regression without exact equivalence. |
| Raise coarse threshold | **Discard as a generic optimization** | Speed came from removing 9.5–29.9% of matches. |
| Tighten adaptive token pruning | **Not applicable** | The pinned model has fixed stride-4 aggregation and no configurable adaptive token-pruning stage. |
| Block-aligned native windows for scientific tiles | **Discard** | Native windows are not equivalent to GCP/TPS north-up sampling; exact windowed remapping was 35% slower. |
| NPZ as production source | **Discard** | Useful as an immutable benchmark cache, not as the source-of-truth COG pipeline. |
| Whole-scene cache | **Retain** | Faster for the measured scene working set. |
| Canonical cache keys and larger RAM budget | **Separate candidate** | Promising for the 143-scene workflow, but independent of matcher batching and requiring an explicit memory budget. |

Batch-4 acceptance requires all of the following on a larger frozen cohort:

1. Exact repeatability for identical ordering and no meaningful sensitivity to
   deterministic batch regrouping.
2. No new folded triangles and no material degradation in field availability,
   deformation tails, or adjacent-vector gradients.
3. Reported common-cell vector-error quantiles plus every outlier above 250 m,
   including support, boundary distance, and source pair.
4. No loss of previously complete trajectories and no material degradation in
   terminal-position error or truth-referenced errors.
5. At least 10% sustained GPU-throughput gain after graph setup and data
   transfer, with peak reserved memory compatible with the production worker
   count.

## Follow-up review-branch validation

The interpolation-only change is isolated on
`codex/eloftr-vectorized-remap-review`, based on Vega revision `299b4f1`.
The branch changes only `limosat/imagery.py` and its exact-equivalence unit
test. The complete local suite passed (73 tests); the affected suite also
passed in the Vega environment (9 tests).

The full 185-tile frozen cohort was regenerated in one process with the old
`np.interp` implementation and the review-branch implementation. All source
images, target images, and validity masks were byte-identical. Their common
ordered array-content SHA-256 was
`dc427c5e35f88a749336c7ac36c3df8878e95ecefa8a18ba87a94f0dcd840e28`.
Batch-1 inference and the recorded field/trajectory evaluators then produced
the following identical old-versus-vectorized checksums:

| Artifact | SHA-256 |
|---|---|
| 6775→6801 raw matches | `243b1f5935e41b098342aa0209b729a5a9955668f20d6d928437b956b2358931` |
| 6801→6901 raw matches | `261c926b7adbe246993a22ab76fce50bf91122c66a726a03a9c9f2b097625b2e` |
| 6775→6801 accepted matches | `a1057867d68b7a7d5db44525d70d935c5a1b2eb449aeda539fb3d9cdd8ab966b` |
| 6801→6901 accepted matches | `39ae92401ece0376a04b03f6594ac696e79a94948f82449b4071b35237c7ec04` |
| 6775→6801 final 4 km field | `02253a334889b7463deead39c326a9f4833acada4d292da7cee7aa7da8d9ac04` |
| 6801→6901 final 4 km field | `867ae309ef57b42cbba5d8b627ceec840d8ddc25362ad8817404931027fc05bc` |
| Final two-step trajectory | `364f6fe9d3c2a153071ba1c7c6402c4048fc3ac92b95600dc92c1011d956f03c` |

The original frozen field, match, and trajectory files also passed their
recorded SHA-256 manifests. A fresh remap differs from the older frozen NPZs in
89,472 of 96,993,280 image pixels (0.0922%) and six mask pixels, while the old
and vectorized implementations remain exact within the fresh process. This is
the previously observed GCP/TPS cross-process variability, not a change caused
by vectorization.

## Batch-4 and batch-8 downstream evaluation

The separate batch arms retained the frozen backend settings: matrix-multiply
TF32 disabled, cuDNN TF32 enabled, SDPA, skip-softmax, threshold 25, and the
reparameterized official model. Static
prefix CUDA Graph output was exactly equal to eager output at the same batch
size on all 185 tiles: 185/185 exact coarse sets, zero confidence difference,
and zero fine-coordinate difference. Therefore the eager batched fields below
also represent prefix-graph output at the same batch size.

| Prefix graph | Tiles/s | Speedup vs batch 1 | Reserved VRAM | Exact vs same-batch eager |
|---|---:|---:|---:|---:|
| Batch 1 | 63.77 | 1.000x | 1.63 GB | 185/185 |
| Batch 4 | 71.69 | 1.124x | 4.10 GB | 185/185 |
| Batch 8 | 72.59 | 1.138x | 8.30 GB | 185/185 |

Final-field differences below are relative to the immutable completed
batch-1 experiment, restricted to cells available in both outputs.

| Arm/pair | Availability symmetric difference | Vector error median / p95 / p99 / max (m) |
|---|---:|---:|
| Batch 4, 6775→6801 | 1 / 6,992 | 1.20 / 28.88 / 68.37 / 761.88 |
| Batch 4, 6801→6901 | 10 / 9,258 | 1.35 / 32.99 / 75.27 / 581.45 |
| Batch 8, 6775→6801 | 1 / 6,992 | 1.21 / 29.05 / 72.08 / 827.99 |
| Batch 8, 6801→6901 | 12 / 9,258 | 1.38 / 32.31 / 77.56 / 756.58 |

Both arms produced 1,824 complete frozen-denominator trajectories versus
1,820 for batch 1, with four availability differences and no lost completed
trajectory. Common terminal-position error relative to batch 1 was:

| Arm | Median / p95 / p99 / max (m) |
|---|---:|
| Batch 4 | 4.21 / 40.95 / 110.89 / 873.48 |
| Batch 8 | 4.05 / 41.73 / 145.03 / 878.98 |

The independent six-buoy truth set retained exactly the same availability as
batch 1: five trajectories at image 6801 and three at image 6901. Terminal
median truth errors were 127.08/51.29 m for batch 1, 126.48/51.78 m for batch
4, and 125.75/51.51 m for batch 8. The small truth sample shows no degradation,
but is not sufficient to dismiss the rare 0.6–0.9 km batch-sensitive field and
trajectory outliers.

Batch 4 is the preferred next experimental arm: it captures most of the batch
speedup, uses half the reserved VRAM of batch 8, and has the lower p99 terminal
difference from the frozen run. Neither batch size should replace batch 1 yet;
the next acceptance test should localize the outliers and repeat the comparison
across more completed image pairs.

## Expanded batch-4 acceptance test

The reusable evaluator is
`experiments/evaluate_efficientloftr_batch_acceptance.py`. It compares ordered
final fields and trajectories, reports every field outlier above 250 m,
measures availability gains/losses and adjacent-vector tails, checks
post-rejection topology, and optionally gates truth-error tables. Provisional
limits are explicit in
`experiments/configs/efficientloftr_batch_acceptance_v1_20260906.json`; they are
engineering regression limits, not calibrated geophysical uncertainties.

Vega currently has only two image pairs with retained, replayable 512-pixel
input fixtures. The larger production archives retain matches and fields but
not the prepared image tiles, so multi-pair batch inference must be performed
as part of the next production-length diagnostic rather than reconstructed
from those outputs.

Four batch-4 executions covered the natural tile order, an exact repeat, the
reverse order, and two deterministic shuffles. The exact repeat reproduced all
raw arrays and archive SHA-256 values. Regrouping preserved every per-pair raw
match count but changed some floating-point match coordinates and confidence
values. Downstream consensus strongly attenuated those changes:

| Regrouping vs natural batch 4 | Field availability changes | Field error p95 / p99 / max (m) | Trajectory error p95 / p99 / max (m) |
|---|---:|---:|---:|
| Reverse | 0 | 0.000 / 4.14 / 66.26 | 0.000 / 3.29 / 62.14 |
| Shuffle seed 11 | 0 | 0.0004 / 7.75 / 109.28 | 0.0006 / 8.57 / 99.85 |
| Shuffle seed 29 | 0 | 0.0028 / 6.92 / 133.40 | 0.364 / 5.22 / 26.80 |

The table reports the worse of the two pair-field values for each quantile.
All regroupings produced the same 1,824 complete trajectories, identical buoy
availability, and identical buoy-error tables. All four arms passed the
provisional field, trajectory, truth, roughness, and no-new-fold gates relative
to the immutable batch-1 run. Adjacent-vector p99 changed by +0.63% on pair 1
and -0.05% on pair 2.

Five common field cells differed from batch 1 by more than 250 m: three lie on
the reference availability boundary and two are 16.5 and 37.9 km inside it.
The maximum difference was 761.9 m. All had the full 12 candidate vectors; the
large changes arose where a small numerical match change altered which 8--12
vectors survived consensus. No folded triangle or truth-error regression was
introduced.

Current disposition: batch 4 passes the complete available frozen cohort and
is stable enough for a production-length diagnostic, but not yet for a default
change. The next run must preserve its input tiles or a versioned preprocessing
identity so this same gate can be applied across many image pairs.
