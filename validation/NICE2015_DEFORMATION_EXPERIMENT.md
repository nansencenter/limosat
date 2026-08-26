# N-ICE2015 trajectory-deformation experiment

Status: design draft
Created: 2026-08-10
Intended destination: LiMOSAT validation work, with the issue-ready summary below suitable for a future GitHub issue.

## Issue-ready summary

### Goal

Determine whether trajectory-aware matching and quality control in LiMOSAT produce **more accurate sea-ice deformation fields** than the current LiMOSAT configuration and a reconstructed 2017-style feature-tracking plus pattern-matching baseline, while retaining useful spatial coverage and buoy displacement accuracy.

The primary target is accuracy against deformation calculated from the N-ICE2015 nested buoy array over the exact satellite acquisition intervals. A visually smoother or cleaner deformation map is not, by itself, success.

### Why this is different from earlier work

LiMOSAT has already been tuned and evaluated against buoy positions, including N-ICE2015. That work tested whether individual satellite trajectories followed individual buoys. This experiment asks a different question: whether the **spatial gradients among neighbouring trajectories** reproduce buoy-observed divergence and shear.

LiMOSAT is also a trajectory system rather than only a two-image drift product. Persistent identities can support deformation estimates over more images, independent pair combinations, and larger regional mosaics. The experiment must separately quantify:

1. deformation accuracy on the common buoy-supported area;
2. displacement accuracy at individual buoys;
3. additional coverage and temporal continuity supplied by trajectories.

### Minimum deliverable

- A frozen, reproducible baseline on a fixed N-ICE2015 scene catalogue.
- A paper-to-code audit of Korosov and Rampal (2017) versus current LiMOSAT.
- Exact-time buoy deformation reference data with documented QC.
- Blocked development and validation partitions with leakage controls.
- Baseline, 2017-component, trajectory-QA, and deformation-aware ablation runs.
- A machine-readable comparison report with uncertainty intervals and coverage guardrails.
- One final evaluation on data not used to select parameters. Because N-ICE2015 has already informed earlier LiMOSAT tuning, a separate array or campaign is required for the strongest claim of generalisation.

## 1. Research question and claim boundary

### Primary research question

Can information available from LiMOSAT trajectories and independent image-pair combinations be used to select or repair uncertain matches so that held-out, colocated sea-ice deformation agrees better with a nested buoy array, without merely smoothing away real deformation or sacrificing too much coverage?

### Primary hypothesis

A trajectory-aware QA or rematching method will reduce held-out error in the symmetric horizontal velocity-gradient tensor relative to the frozen current LiMOSAT baseline.

### Secondary hypotheses

- Components from the 2017 FT+PM method that are absent or materially different in LiMOSAT may recover useful matching skill.
- Independent temporal connections, forward/backward agreement, and trajectory cycle closure can identify erroneous high-correlation matches that correlation alone cannot detect.
- Deformation-aware selection can improve divergence and shear while preserving individual-buoy endpoint accuracy.
- Persistent trajectories can extend coherent deformation mapping beyond any one pair footprint, provided results are compared on common support for accuracy and on full support only for coverage.

### Claims this experiment cannot establish by itself

- That N-ICE2015-tuned parameters generalise to all regions, sensors, seasons, or ice regimes.
- That the buoy array provides dense truth everywhere in a regional LiMOSAT deformation mosaic.
- That a smoother field is more correct.
- That repeated detection of a feature at the same fixed map coordinate is expected; deformation features and the ice carrying them move.

## 2. Existing work and provenance

### Earlier local LiMOSAT buoy validation

The existing work must be treated as prior exposure to N-ICE2015, not as an untouched test:

- Reference buoy tracks: `/Home/seachu/N-ICE2015buoy/N-ICE2015_buoy_tracks_reference_epsg3413.geojson`
  - 28,036 observations in the current local file.
- Existing image-to-buoy match catalogue: `/Home/seachu/N-ICE2015buoy/N-ICE2015buoy_arktalas_image_buoy_matches_filtered_epsg3413_withSIC.geojson`
  - 721 records in the current local file.
- Archived trajectory run: `persistence_parameter_tune_2015_test` under
  `/Home/seachu/arktalas-archive/20260730-precommit-cleanup/repo-ignored/`.
- Existing parameter-tuning notebooks include
  `/Home/seachu/arktalas_ice_drift_experiments/examples/scratch/limosat_parameter_tuning_validation.ipynb`.

The archived run evaluated individual trajectory-to-buoy distances and match rates. Its outputs are useful for reproducing the endpoint baseline, but they do not constitute deformation validation.

### Closest published precedents

- [Korosov and Rampal (2017)](https://www.mdpi.com/2072-4292/9/3/258) combined feature tracking and pattern matching, examined parameter sensitivity, and validated displacement against buoys. It is the methodological predecessor that should be audited against LiMOSAT.
- [Itkin et al. (2017)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016JC012403) describes the nested N-ICE2015 buoy arrays and their deformation calculation.
- [Muckenhuber and Sandven (2017)](https://tc.copernicus.org/articles/11/1835/2017/) validated Sentinel-1 FT+PM displacement against GPS buoys.
- [Itkin (2025)](https://tc.copernicus.org/articles/19/1135/2025/tc-19-1135-2025.html) compared Sentinel-1 deformation with N-ICE2015 buoy and ship-radar deformation statistics. It compared scale-dependent behaviour rather than using colocated buoy deformation as recursive matcher feedback.

## 3. Definitions

- **Pairwise baseline:** drift estimated directly for a specified image pair without using later trajectory history to judge the match.
- **Trajectory estimate:** a LiMOSAT point identity with positions at multiple acquisition times.
- **Buoy triangle:** three quality-controlled buoys present at both endpoints of a satellite interval.
- **Deformation tensor:** the symmetric part of the horizontal velocity-gradient tensor. Report divergence, the two shear components, total deformation, and principal-axis orientation. Vorticity is a useful motion-gradient diagnostic but is not part of the symmetric deformation tensor.
- **Common support:** spacetime and scale samples for which all compared methods have a valid estimate.
- **Full support:** all valid estimates from one method. Use this to report coverage, not to make an unpaired accuracy claim.
- **Iteration:** one frozen algorithm/configuration run followed by evaluation on the development partition. An iteration must not inspect the final test labels.

## 4. Experimental principles

1. **Freeze data and baseline first.** Record scene IDs, checksums, preprocessing, code commit, environment, configuration, and random seeds before tuning.
2. **Use exact satellite intervals.** Interpolate buoy positions only when observations bracket each SAR time and pass a predeclared maximum observation gap.
3. **Split by time/event, not individual buoy observations.** Nearby buoys and adjacent image pairs are strongly correlated.
4. **Purge temporal leakage.** Separate scored blocks by at least the maximum trajectory/template memory used by the run. Warm-up imagery may initialise a block but must not contribute scored observations across a partition boundary.
5. **Tune globally, never correct to a known buoy locally.** Buoy labels may select algorithm parameters on development folds. They must never directly move an individual satellite vector toward a buoy.
6. **Compare accuracy on common support.** Report any extra trajectory-derived area separately as a coverage benefit.
7. **Preserve extremes.** A method that lowers error by deleting or smoothing all high-deformation events fails the scientific objective.
8. **Change one factor at a time before combining factors.** Every final candidate needs an ablation trail.
9. **Evaluate the final test once.** Repeated inspection converts the test set into development data.

## 5. Stage 0 — feasibility, provenance, and 2017 parity audit

This stage makes no scientific comparison and changes no matching parameters.

### 5.1 Input inventory

- Build the definitive N-ICE2015 buoy inventory, including array/deployment membership, quality flags, observation cadence, gaps, and positional precision.
- Build the Sentinel-1 catalogue overlapping each usable nested-array interval.
- Identify the exact preprocessed raster for every catalogue record and verify projection, resolution, nodata definition, footprint, orbit, and acquisition time.
- Confirm whether archived local scenes reproduce the scene selection in earlier LiMOSAT validation.
- Quantify how many image intervals contain at least one geometrically acceptable buoy triangle at each spatial scale.

### 5.2 Korosov and Rampal (2017) versus LiMOSAT audit

For each item, record: paper behaviour, current LiMOSAT behaviour, equivalence status, expected effect, and whether it is independently configurable.

- feature detector and descriptor;
- initial feature-match acceptance and spatial constraint;
- global/local geometric model and outlier rejection;
- interpolation or first-guess construction;
- pattern-matching template dimensions;
- matched and interpolated search radii;
- rotation search and early exit;
- correlation threshold;
- correlation-surface peak-shape or Hessian test;
- nodata handling for templates and candidate windows;
- forward/backward consistency;
- neighbourhood/deformation consistency filters;
- template update policy;
- same-orbit and temporal-gap handling;
- displacement/speed limits;
- grid spacing and seeding policy.

Do not label an experimental arm “2017” until this audit shows what has actually been reproduced. If exact reproduction is impractical, name it “2017-component ablation” and list the implemented components.

### 5.3 Stage gate

Proceed only if there are enough valid buoy triangles and satellite intervals for blocked evaluation. The feasibility report must show sample counts by deployment, time block, triangle scale, and deformation magnitude—not just a total count.

## 6. Stage 1 — build and test the deformation reference harness

### 6.1 Buoy reference construction

For every eligible satellite interval `[t0, t1]`:

1. interpolate each buoy to exactly `t0` and `t1` using only bracketing observations;
2. reject endpoints whose bracketing gap exceeds the frozen tolerance;
3. form Delaunay triangles within each contemporaneous array;
4. reject poorly conditioned triangles, initially using the published 15-degree minimum-angle rule;
5. record side lengths, area, aspect ratio, array/deployment, and interpolation gaps;
6. calculate vertex velocity from endpoint displacement divided by exact elapsed time;
7. calculate a constant velocity gradient over each triangle and derive divergence, shear components, total deformation, vorticity, and principal-axis orientation;
8. propagate plausible buoy position uncertainty through the calculation or estimate it by perturbation.

Triangulation and QC rules must be frozen before comparing algorithm variants.

### 6.2 Colocated LiMOSAT estimate

The primary comparison should use a scale-matched local affine velocity fit centred on each buoy triangle:

- include only LiMOSAT trajectories with valid positions at both exact scene times;
- use a frozen neighbourhood definition tied to the buoy triangle scale;
- require a minimum number and spatial distribution of satellite vectors;
- estimate the local velocity gradient with the same coordinate and sign conventions as the buoy calculation;
- record support count, condition number, residual, distance to the triangle, and effective scale.

A secondary sensitivity calculation should interpolate satellite velocity to the three buoy vertices and then apply the identical triangle operator. Agreement between the two formulations will test dependence on the comparison operator.

### 6.3 Harness tests

- Analytic rigid translation must return zero deformation.
- Analytic rotation must return near-zero symmetric deformation and the expected vorticity.
- Prescribed divergence and pure shear fields must recover their known components and signs.
- Results must be invariant to coordinate translation and consistent under rotation of axes.
- Duplicate, collinear, time-reversed, and poorly conditioned cases must fail explicitly.

### 6.4 Stage gate

Proceed only after the buoy and satellite operators recover synthetic fields within numerical tolerance and a manual sample verifies timestamps, units, triangle geometry, and sign conventions.

## 7. Stage 2 — frozen baselines

Run all baselines on identical imagery and preprocessing:

| Arm | Purpose |
|---|---|
| B0: current LiMOSAT | Operational/reference configuration frozen before this experiment |
| B1: pairwise current components | Separates the benefit of trajectory persistence from the underlying matcher |
| B2: 2017-component baseline | Adds only audited 2017 components missing or materially changed in LiMOSAT |
| B3: trajectory aggregation without deformation tuning | Tests the trajectory/coverage advantage before deformation labels influence selection |

For every arm, retain raw candidates and rejection reasons where feasible. A later method cannot be diagnosed if only accepted vectors survive.

## 8. Stage 3 — controlled deformation-aware iterations

### 8.1 Evidence allowed at inference time

Candidate QA features may include:

- correlation and the full local correlation-surface shape, including peak sharpness and peak ambiguity;
- finite and valid-data fractions in the template and every candidate match window;
- descriptor distance and geometric-model residual;
- forward/backward error;
- alternative temporal-path or cycle-closure error;
- agreement across independent image-pair combinations;
- local neighbour residual after a robust motion fit;
- trajectory age, gaps, interpolation state, template age, and template update history;
- physical speed checks.

Buoy position or buoy-derived deformation is **not** an inference-time feature. It is a development label only.

### 8.2 Recommended iteration order

1. Calibrate existing QA variables against development-fold deformation error without changing matches.
2. Add the audited correlation-peak/Hessian test if it is absent.
3. Add forward/backward and independent-path consistency.
4. Add trajectory-neighbour evidence while protecting coherent discontinuities.
5. For points flagged as uncertain, compare alternative PM candidates or rematch settings rather than immediately deleting them.
6. Rebuild only the affected trajectories when technically possible, then regenerate deformation.
7. Combine individually successful components only after their ablations are understood.

### 8.3 Avoid circular smoothing

Neighbour agreement must not force the field toward uniform motion. A discontinuity may be real if it:

- appears in independent image-pair combinations;
- advects consistently with surrounding trajectories;
- has coherent spatial orientation and finite length;
- is supported by a strong, unambiguous image match on both sides;
- is consistent under forward/backward or cycle tests.

Compare such features in a Lagrangian frame by advecting their positions to a common reference time. Do not require a moving lead or shear zone to remain at a fixed map coordinate.

## 9. Data partitioning and leakage control

### N-ICE2015 role

N-ICE2015 is a development and internal-validation dataset because previous LiMOSAT design and parameter tuning have already used its buoy endpoints. New blocked splits still provide a rigorous comparison of deformation methods, but they do not erase that prior exposure.

### Internal partition

After Stage 0, define contiguous, event-aware time blocks grouped by deployment/floe. Assign complete blocks—not individual triangles or buoys—to:

- development/training folds;
- model/parameter-selection folds;
- a locked N-ICE2015 internal test block.

Use nested blocked cross-validation where sample counts permit. Purge imagery and trajectory state around boundaries by the maximum temporal dependence of the configuration. Stratify reporting by season/deployment, triangle scale, time gap, deformation magnitude, and satellite geometry.

### External confirmation

Reserve a second array or campaign that has not informed LiMOSAT development. Candidate data sources include SEDNA/APLIS-type nested arrays or a suitable MOSAiC distributed array, subject to a separate availability, licence, satellite-overlap, and geometry audit. The external experiment should be run only after the algorithm and thresholds are frozen.

## 10. Metrics

### Primary metric

For each matched buoy triangle and SAR interval, define the deformation component vector

`q = [divergence, normal_shear, simple_shear]`.

The primary loss is equal-weighted, scale-stratified robust component error:

1. calculate absolute component errors in `day^-1`;
2. standardise each component only by a robust scale estimated from the development buoy data;
3. take the median within each predeclared triangle-scale bin;
4. average equally across components and populated scale bins.

This supplies one scalar for autonomous selection without dividing by individual near-zero buoy deformation values. Always report the unstandardised component errors alongside it.

### Secondary deformation metrics

- bias, MAE, RMSE, and correlation for each tensor component;
- total-deformation magnitude error;
- principal-axis angular error, evaluated only above a frozen deformation-magnitude floor;
- event detection precision/recall for high-deformation periods using a threshold defined from development data;
- scale dependence and spatial power-law diagnostics;
- error versus time gap, triangle scale, geometry, support count, and deformation magnitude.

### Displacement guardrails

- individual-buoy endpoint MAE/RMSE and along-/cross-track bias;
- trajectory survival duration and gap distribution;
- forward/backward and cycle-closure errors.

### Coverage and preservation guardrails

- common-support sample count;
- full-support area, time coverage, and valid-vector density;
- fraction of baseline high-deformation samples retained;
- deformation quantiles and spatial intermittency;
- rejection count by reason and regime;
- computational cost and restart/resume behaviour.

Uncertainty intervals must resample independent time/event blocks, not individual correlated triangles.

## 11. Provisional success and stopping rules

Freeze numerical tolerances after Stage 1 exposes reference uncertainty and sample size, but before Stage 3 tuning.

A candidate can replace the baseline only if, on locked data:

1. the primary deformation loss improves by a predeclared practically meaningful margin and its block-bootstrap interval supports an improvement;
2. median buoy endpoint error does not worsen beyond its predeclared guardrail;
3. common-support coverage and regional full-support coverage do not fall beyond their predeclared guardrails;
4. improvement is not explained solely by removal of high-deformation cases;
5. no major deployment, scale bin, or deformation regime shows a severe unexplained regression;
6. the result is reproduced from the frozen manifest and configuration.

Stop iterative development when any one condition is met:

- two consecutive iterations fail to improve parameter-selection loss;
- the predeclared maximum number of iterations is reached;
- remaining errors are dominated by estimated reference or sampling uncertainty;
- the method violates a displacement, coverage, or extreme-preservation guardrail.

Do not use the locked external confirmation set to decide whether to continue iterating.

## 12. Autonomous staged execution contract

Each stage should be independently runnable and resumable. A future runner should require only a stage name and a frozen experiment configuration.

### Required inputs

- `experiment.yaml`: paths, code revision, environment, parameters, partitions, QC, metrics, and stage limits;
- `scene_manifest.csv`: scene IDs, timestamps, footprints, paths, hashes, preprocessing, and availability;
- `buoy_manifest.csv`: source, buoy IDs, deployment, timestamps, quality, CRS, and hashes;
- `partition_manifest.csv`: immutable block and fold assignment;
- baseline configurations and audited component configurations.

### Required outputs per run

- resolved configuration and software/environment versions;
- input manifests and hashes;
- stage state (`not_started`, `running`, `complete`, or `failed`);
- raw and accepted match counts with rejection reasons;
- trajectory database and deformation comparison table;
- metrics in machine-readable CSV/Parquet/JSON form;
- concise Markdown report and diagnostic figures;
- exact command, start/end times, requested/actual resources, and failure classification.

### Stage commands to implement later

The eventual interface should resemble:

```text
limosat-nice-exp inventory --config experiment.yaml
limosat-nice-exp build-reference --config experiment.yaml
limosat-nice-exp run --arm B0 --fold <fold> --config experiment.yaml
limosat-nice-exp evaluate --arm B0 --fold <fold> --config experiment.yaml
limosat-nice-exp iterate --candidate <name> --config experiment.yaml
limosat-nice-exp report --config experiment.yaml
```

These names specify the execution contract, not a requirement to add six separate scripts. Prefer one reusable CLI/module with stage subcommands.

## 13. Proposed result tables and figures

### Tables

- data availability by deployment, time block, pair gap, and triangle scale;
- 2017 paper-to-LiMOSAT component audit;
- primary and guardrail metrics by arm on common support;
- coverage metrics on full support;
- ablation results;
- errors stratified by scale and deformation regime;
- rejection reasons and resource use.

### Figures

- buoy arrays and satellite coverage through time;
- synthetic operator validation;
- colocated buoy versus satellite tensor-component scatter/hexbin plots;
- error versus scale and deformation magnitude;
- Lagrangian examples of independently observed deformation features;
- common-support maps before and after candidate QA;
- coverage gained by trajectory aggregation;
- examples where high correlation is wrong and where large deformation is real.

## 14. Decisions to resolve in Stage 0

- Which N-ICE deployments provide genuinely nested, contemporaneous geometry suitable for deformation?
- What maximum buoy interpolation gap is defensible?
- What scale bins have adequate independent samples?
- Which exact Sentinel-1 preprocessing should be frozen?
- Which 2017 components are missing or materially different in current LiMOSAT?
- Is local affine fitting or vertex interpolation the more stable primary comparison operator?
- What minimum practical improvement and coverage loss are meaningful given reference uncertainty?
- Which external array can provide the final independent confirmation?
- Should initial work remain in this validation document or be promoted immediately to a GitHub issue and milestone?

## 15. Immediate next action

Implement only Stage 0: produce the input/overlap inventory and the 2017 paper-to-code parity table. Do not start parameter search or deformation-aware filtering until the sample distribution, partitions, reference uncertainty, and comparison operator are frozen.
