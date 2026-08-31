# CryoSat-2 deformation validation pilot

## Frozen question

Does drift-aware SAR shear identify (a) more lead-type CryoSat-2 waveforms and
(b) greater waveform-fit roughness over sea-ice floes? The comparison is an
independent structural validation; it is not a direct measurement of
displacement error or proof that the observed structure formed during the
23-hour SAR pair.

## Frozen inputs

- SAR pair: 2020-03-28 12:13:29 UTC to 2020-03-29 11:16:05 UTC.
- Production ORB and the frozen ALIKED nearest-12, flip-rejected field used in
  the ICESat-2 pilot.
- CryoSat-2 `RDWES1B` Version 1 waveform-fit surface roughness footprints.
- Coordinates and vector calculations use EPSG:3413; roughness is in metres;
  deformation is per day.

## Primary analysis

1. Keep finite, positive roughness values with `norm_res <= 0.5`, following the
   product's published downstream freeboard-quality rule.
2. Apply the SAR-mode waveform classes from Kurtz et al. (2014): leads have
   `peakiness > 0.18` and `stack_sd < 4`; floes have `peakiness < 0.09` and
   `stack_sd > 4`. Ambiguous waveforms are neither class.
3. Move every CryoSat footprint back to its material position at the SAR pair
   start using `x(t) = x0 + alpha * u(x0)` and its exact acquisition time.
4. Restrict ORB and ALIKED comparisons to identical quality-controlled
   footprints supported by both deformation fields.
5. Aggregate in 4 km along-track bins with at least three footprints.
6. Test SAR shear versus lead fraction as the primary CryoSat structural test.
7. Separately test SAR shear versus median floe roughness in bins containing at
   least three floe returns.
8. Use a within-track circular-shift null with a minimum 20 km displacement.

Lead and floe roughness are not pooled for inference. The retrieval fits lead
roughness with a 0-0.1 m bound, so a naive mixed-waveform correlation is retained
only as a diagnostic showing why product-specific classification is necessary.

## Predeclared secondary and sensitivity analyses

- Maximum compression and total deformation versus roughness.
- A 1 km bin-size sensitivity.
- A 4 km sensitivity excluding exact 1.0 m roughness values, without changing
  the primary product-based quality rule.
- A static, no-advection overlay as a control, not a competing tuned method.
- Method coverage and ORB-versus-ALIKED deformation agreement on common bins.

No CryoSat threshold or deformation-field parameter will be selected from the
observed correlation.

## Alignment audit result

The shared selection/alignment ledger and visual audit are complete under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/multisensor_alignment_audit_v1_20260819`.
The original CryoSat run used the 8,523-node pre-final ALIKED field. The
selected 8,520-node fixed-point fold-rejected rerun is under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/cryosat2_validation/results/selected_fold_rejected_v2/pair_10245_10352`.
It removes one common footprint (1,534 to 1,533) but leaves the primary 4 km
shear/lead Spearman values unchanged: 0.361 for ORB and 0.328 for ALIKED, both
with within-track spatial-null `p=0.001`. Floe-only roughness remains null.

The frozen symmetric alignment analysis is under
`/Volumes/KINGSTON/arktalas/experiments/limosat_next_tracking_2020/multisensor_alignment_sensitivity_v1_20260819`.
At 4 km the full envelope stays positive for both methods: 0.276-0.361 for ORB
and 0.243-0.332 for ALIKED. The relationship is therefore much more stable to
the tested registration choices than the ICESat-2 morphology associations.
It still comes from one SAR pair and is structural evidence rather than direct
displacement truth.

The frozen March expansion added no second qualifying CryoSat/SAR-pair event.
All newly selected products were ICESat-2, and their ATL10 lead arms were
insufficient. Consequently the CryoSat shear/lead result is stable under the
declared alignment sensitivity but has not yet replicated across an independent
SAR pair. The next CryoSat action is a time/geometry-selected second pair with
multiple supported tracks, not a threshold or offset sweep on this pair.
