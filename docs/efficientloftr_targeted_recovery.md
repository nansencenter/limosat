# EfficientLoFTR targeted trajectory recovery

## Method

Full-footprint non-consecutive matching is replaced by matching around measured
trajectory losses. The runner first replays the adjacent-only trajectory graph,
selects positions observed at the later pair's source but absent at its target,
buffers them by the existing 6.4 km fold-free interpolation distance, and runs
EfficientLoFTR only for intersecting source-tile cores. Tile geometry and
routing are unchanged. The selected positions and buffer are included in the
pair identity so targeted output cannot be resumed as a full-source run.

The gate was designed on N-ICE and then applied unchanged to the March chain.
Each targeted run was compared with a full-source control using identical
routing and the same shortest-observation trajectory policy.

| Dataset | Adjacent complete | Full non-consecutive | Targeted | Gain recovered | Calls targeted / full | Matching-time reduction | Correct buoy comparisons targeted / full | Folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N-ICE | 1,455 | 1,565 | 1,565 | 100.0% | 104 / 303 | 71.0% | 12 / 12 | 0 |
| March | 1,370 | 2,061 | 2,063 | 100.3% | 122 / 245 | 60.6% | 106 / 106 | 0 |

Buoy availability, within-2-km counts, median error, and P90 error are exactly
unchanged from the routing-matched full controls. Targeted/full trajectory
positions have zero median difference at every image; the largest P90 is
18.4 m. Cumulative trajectory-derived total-deformation Spearman correlation
is at least 0.894 on N-ICE and 0.884 on March. Pair wall time falls by 67.1%
and 56.4%, respectively.

The sparse recovery field has different outer support and triangulation from a
full-source field. It is therefore an internal reconnection measurement, not a
standalone deformation product. The deformation gate applies to the resulting
cumulative trajectory graph; direct sparse-field comparisons remain diagnostic.

## Decision and next action

The targeted policy passes the frozen gate. The next sequence pilot should
schedule non-consecutive matching only after a measured support collapse,
persist the resulting image-pair links, and report matcher calls per recovered
trajectory. Retain a fixed-interval full-source diagnostic pair to detect
selection bias.

Frozen criteria are in
`experiments/configs/efficientloftr_targeted_recovery_gate_v1_20260828.json`.
Full reports are stored on Kingston under
`efficientloftr_targeted_recovery_nice_v1_20260828/gate` and
`efficientloftr_targeted_recovery_march_v1_20260828/gate`.
