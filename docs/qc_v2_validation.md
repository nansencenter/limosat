# QC v2 correctness validation — 2026-09-07

The QC review fixes are covered by 40 passing targeted tests and a comparison
over 253,024 links in six frozen pilot audits. All 611 rejected links are
unchanged from the historical v1 audits. Nine retained links change review
status: three enter review and six leave it. Full and accelerated v2 scoring
agree on every reject, review, and decision reason.

This is regression evidence, not independent truth validation. The three archive
windows omit boundary-crossing links. The full 57-million-row archive has not
been rerun with v2. A partial human spot check is recorded below; it does not
establish rejection precision or archive-wide accuracy.

## Changes

- Bound the absolute sum of affine prediction weights to 2. Fall back to a
  coordinate-median translation for rank-deficient or higher-amplification fits.
- Replace the median-only prescreen with a conservative residual bound. With
  coarse median `m`, query displacement `u`, and maximum neighbour deviation
  `R = max(||u_i - m||)`, the bound is `||u - m|| + 2R`. It covers any clipped
  neighbour subset, every admitted affine fit, and the median fallback. Skip
  only below both the configured prescreen and every local decision floor.
- Refit after the final clipping iteration if it changed the inlier mask.
- Require an existing audit with matching run identity and checkpoint counts
  on resume. Clear stale completion metadata before replay. Compare materialized
  break/duplicate counts against independently stored scan statistics.
- Refuse misplaced source terminal markers during preparation and verify
  chronological terminal positions in the output.
- Consolidate checkpoint writes and index duplicate assignments by row ID.

The speed and residual thresholds are unchanged. The v1 JSON remains intact;
v2 has a distinct protocol ID. Rebuild work directories for v2 rather than
resuming an older checkpoint.

## Frozen-audit comparison

| Audit | Links | Rejects, v1 → v2 | Review, v1 → v2 |
|---|---:|---:|---:|
| 2017 early melt | 47,232 | 235 → 235 | 0 → 0 |
| 2017 freeze-up | 35,103 | 68 → 68 | 0 → 0 |
| 2018 winter | 69,092 | 186 → 186 | 2 → 2 |
| 2020 buoy holdout | 58,331 | 48 → 48 | 11 → 10 |
| 2026 mixed-sensor MIZ | 6,232 | 34 → 34 | 0 → 0 |
| 2024 RADARSAT-2 | 37,034 | 40 → 40 | 28 → 26 |

One local timing pass took 45.7 seconds for full scoring and 20.4 seconds for
accelerated scoring across all six audits. Acceleration is data-dependent: the
2024 case evaluated almost every eligible vector and was slower with the
prescreen. These timings are not a full-archive throughput benchmark.

## Reproduce

From the repository root, using the existing `limosat_scaling` environment:

```bash
python -m pytest -q tests/unit/test_qc_limosat_trajectories.py tests/unit/test_qc_limosat_archive_streaming.py
PYTHONPATH=. python examples/validate_qc_audits.py \
  --input-root /Users/seachu/projects/limosat/results/limosat_qc_20260903 \
  --output-dir /private/tmp/limosat-qc-v2-validation-repeat
```

The output directory must be new. The script checks the six audits against
their manifests, verifies exact full/prescreen decision equality, and records
input, code, protocol, script, and output hashes. Coordinates are float64
EPSG:3413 metres; elapsed time is in days. No dependencies were added.
The local run used Python 3.11.15, NumPy 2.4.3, pandas 3.0.2, and SciPy 1.17.1;
the repository environment specification was not recreated for this check.

The first packet is at `/private/tmp/limosat-qc-validation-20260907/results`.
It includes comparison counts, all changed decisions, a sample key, a blinded
label sheet, stratum counts, hashes, and labelling instructions.

## Independent labelling gate

The fixed packet contains 143 cases: three deterministically ranked cases per
available dataset/decision/gap/interpolation/SIC cell, plus all changed decisions
outside the 2020 buoy holdout. That holdout remains excluded from sample
selection and threshold tuning. Unknown SIC is explicit; dataset-level coverage
does not establish complete sensor-pair or seasonal coverage.

### Partial human spot check

The saved `review_labels.numbers` sheet contains 13 completed labels: 6 valid,
4 invalid, and 3 uncertain. All 13 cases were retained without a QC review flag.
No automatically rejected cases were labelled, so this check does not measure
false rejection. The remaining 130 cases are unlabelled.

The possible misses are QC-0004, QC-0010, QC-0016 and QC-0035. Three comments
question scene suitability (open water or strongly mixed ice/water); this needs
to be distinguished from an incorrect displacement. QC-0016 has no explanatory
note. Two valid labels have low confidence and two have high confidence; the
others have no confidence supplied. Uncertain cases are not counted as valid.

These labels are a partial convenience subset of the targeted sample. They
justify investigating the four retained cases, but do not justify changing
speed, residual or concentration thresholds. No thresholds were changed from
this spot check. Its export and comparison are in
`/private/tmp/limosat-qc-validation-20260907/review/spot_check_summary.md` and
`completed_labels_compared.csv` in the same directory.

### Merge scope and operational promotion

The intended merge scope is an explicitly invoked, post-run analysis tool that
preserves the raw database and records its decisions. It is not a claim that
all retained vectors are valid or that rejection precision is established.
Further scene-screening work and full-archive v2 validation remain follow-up
work; operational promotion retains the independent validation requirement.

An independent reviewer should inspect source/target imagery and record valid,
invalid, or uncertain labels with evidence references before opening the QC
decision key. Report results by stratum, including short-gap hard-speed cases.
The targeted sample cannot establish an unweighted archive-wide false-rejection
rate. Freeze labels before considering any threshold changes; extend coverage
and rerun the full archive before operational promotion.
