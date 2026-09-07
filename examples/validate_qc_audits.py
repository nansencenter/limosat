"""Compare frozen pilot audits and prepare a blinded, fixed validation sample.

Run with the patched checkout on PYTHONPATH and the limosat_scaling interpreter.
Inputs are read-only. Output must be a new directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from limosat.qc.core import QCConfig, PROTOCOL_ID, PROTOCOL_PATH, score_vectors, validate_frozen_protocol


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources, summaries, samples, changes = [], [], [], []
    for path in sorted(args.input_root.glob('*precision_v3/edge_audit.csv')):
        name = path.parent.name
        manifest_path = path.with_name('run_manifest.json')
        manifest = json.loads(manifest_path.read_text())
        config = QCConfig(**manifest['config'])
        validate_frozen_protocol(config)
        source = pd.read_csv(path)
        if len(source) != manifest['summary']['links']:
            raise ValueError(f'Incomplete audit: {path}')
        started = time.perf_counter()
        full = score_vectors(source, config)
        full_seconds = time.perf_counter() - started
        started = time.perf_counter()
        fast = score_vectors(source, config, prescreen_residual_m=1_000.)
        fast_seconds = time.perf_counter() - started
        for column in ('reject', 'review', 'decision_reason'):
            if not full[column].equals(fast[column]):
                raise AssertionError(f'{name}: prescreen mismatch in {column}')
        full['dataset'] = name
        full['old_reject'] = source['reject']
        full['old_review'] = source['review']
        full['old_decision_reason'] = source['decision_reason']
        changed = (source['reject'] != full['reject']) | (source['review'] != full['review'])
        summaries.append(dict(
            dataset=name, links=len(source), old_rejected=int(source.reject.sum()),
            new_rejected=int(full.reject.sum()), old_review=int(source.review.sum()),
            new_review=int(full.review.sum()), changed_decisions=int(changed.sum()),
            newly_rejected=int((~source.reject & full.reject).sum()),
            no_longer_rejected=int((source.reject & ~full.reject).sum()),
            prescreen_mismatches=0, full_evaluated=int(full.local_evaluated.sum()),
            fast_evaluated=int(fast.local_evaluated.sum()),
            full_seconds=full_seconds, fast_seconds=fast_seconds,
        ))
        changes.append(full.loc[changed])
        sources.append(dict(
            dataset=name, audit_path=str(path), audit_sha256=sha256(path),
            manifest_sha256=sha256(manifest_path), source_database=manifest['input_database'],
            source_sha256=manifest['input_sha256'], config=manifest['config'],
            selection=manifest['selection'],
        ))
        # Preserve the existing buoy holdout exclusively for final evaluation.
        if 'buoy_holdout' not in name:
            full['decision_stratum'] = np.select(
                [full.hard_speed_reject, full.reject, full.review],
                ['hard_speed', 'local_only', 'review'], default='retained')
            full['gap_stratum'] = pd.cut(full.elapsed_days * 24,
                [-np.inf, 3, 12, 96, np.inf], right=False,
                labels=['under_3h', '3_to_12h', '12_to_96h', '96h_or_more']).astype(str)
            full['sic_stratum'] = full['sic_regime'].fillna('unknown') if 'sic_regime' in full else 'unknown'
            full['sample_rank'] = [hashlib.sha256(f'20260907:{name}:{int(r)}'.encode()).hexdigest()
                                   for r in full.target_rowid]
            strata = ['decision_stratum', 'gap_stratum', 'interpolated', 'sic_stratum']
            selected = full.sort_values('sample_rank').groupby(strata, observed=True).head(3)
            # Include every changed decision in this diagnostic packet, with
            # selection type exposed in the key, not the blinded label sheet.
            selected = pd.concat([selected, full.loc[changed]]).drop_duplicates('target_rowid')
            selected['changed_decision'] = changed.loc[selected.index].to_numpy()
            samples.append(selected)
        pd.DataFrame(summaries).to_csv(args.output_dir / 'comparison.csv', index=False)
        print(json.dumps(summaries[-1]), flush=True)
    if len(sources) != 6:
        raise ValueError(f'Expected six frozen audits, found {len(sources)}')
    sample = pd.concat(samples, ignore_index=True).sort_values('sample_rank').reset_index(drop=True)
    sample.insert(0, 'case_id', [f'QC-{i:04d}' for i in range(1, len(sample)+1)])
    sample.to_csv(args.output_dir / 'sample_key.csv', index=False)
    pd.concat(changes, ignore_index=True).to_csv(args.output_dir / 'changed_decisions.csv', index=False)
    columns = ['case_id', 'dataset', 'trajectory_id', 'source_rowid', 'target_rowid',
               'source_image_id', 'target_image_id', 'source_time', 'target_time',
               'x0_m', 'y0_m', 'x1_m', 'y1_m']
    labels = sample[columns].copy()
    for column in ['label', 'confidence', 'reviewer', 'evidence_reference', 'notes']:
        labels[column] = ''
    labels.to_csv(args.output_dir / 'labels_blinded.csv', index=False)
    sample.groupby(['dataset','decision_stratum','gap_stratum','interpolated','sic_stratum'],
                   observed=True).size().rename('cases').to_csv(args.output_dir / 'sample_counts.csv')
    code_root = PROTOCOL_PATH.parents[1]
    payload = dict(
        protocol_id=PROTOCOL_ID, protocol_sha256=sha256(PROTOCOL_PATH),
        code_sha256={p.name: sha256(p) for p in code_root.glob('*.py')},
        script_sha256=sha256(Path(__file__)), sources=sources, sample_size=len(sample),
        selection='First 3 SHA256-ranked rows per dataset/decision/gap/interpolation/SIC cell; union all changed decisions. Seed string 20260907.',
        excluded_from_label_packet='full70_2020_buoy_holdout_precision_v3',
        validation_status='unlabelled; no independent false-rejection estimate',
        scope='Six frozen pilot audits; archive windows omit boundary-crossing links. Not a full-archive v2 run.',
        crs='EPSG:3413', coordinate_units='metres', arrays='float64', elapsed_time_units='days',
        versions=dict(numpy=np.__version__, pandas=pd.__version__),
        outputs={p.name:sha256(p) for p in args.output_dir.glob('*.csv')},
    )
    (args.output_dir / 'manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    (args.output_dir / 'README.md').write_text('''# Fixed QC validation packet

Start with `labels_blinded.csv`; keep `sample_key.csv` hidden until labels are frozen.
For each case inspect the original source and target imagery around the coordinates
and, where available, adjacent acquisitions. Label `valid`, `invalid`, or `uncertain`;
record confidence, reviewer, imagery/evidence reference, and notes. Do not infer
truth from neighbour agreement, correlation, SIC, or the QC decision itself.
Coordinates are EPSG:3413 metres; row IDs identify observations in the original
SQLite tables. Source paths, checksums, and selection boundaries are in the manifest.

The sample includes 3 deterministic cases per available dataset/decision/gap/
interpolation/SIC cell plus every changed decision outside the held-out buoy run.
Unknown SIC is explicit. Dataset names identify the available sensor/season cases;
this does not provide complete sensor-pair or seasonal coverage. This is a targeted
validation packet, not a random archive sample: report counts and precision by
stratum; do not quote an unweighted archive-wide false-rejection rate.

After labels are frozen, join case_id to sample_key.csv and report invalid/valid/
uncertain counts for automatic rejects, retained review, and retained background.
Review short-gap hard-speed cases and newly rejected cases separately. Retain
uncertain cases in the denominator and report bounds. The original 2020 buoy
holdout was excluded from sample selection; scoring comparisons there are regression
evidence, not a new independent buoy-accuracy measurement. Do not tune thresholds
on it. A full v2 archive rerun and broader independent labels remain required before
operational promotion. No labels have been supplied or inferred by this script.
''')
    print(json.dumps(dict(sample_size=len(sample),output=str(args.output_dir))), flush=True)


if __name__ == '__main__':
    main()
