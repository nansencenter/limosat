#!/usr/bin/env python3
"""Bounded real-data demonstration using the current, unchanged pair processor.

Recording hooks retain coarse/fine inputs and raw outputs without changing
matching, filtering, consensus, or routing. Only the processing domain is clipped.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType

import numpy as np
from shapely import box

IDS = [6775, 6801, 6901, 6998]
BOUNDS = (768000., 864000., -680000., -584000.)


def modules(checkout):
    package = ModuleType('workflow_core')
    package.__path__ = [str(checkout / 'limosat')]
    sys.modules[package.__name__] = package
    return {n: importlib.import_module('workflow_core.' + n) for n in
            ('config', 'catalog', 'imagery', 'models', 'efficientloftr', 'pairs',
             'field', 'trajectory', 'deformation')}


def save_arrays(path, product):
    np.savez_compressed(path, **{k: v for k, v in vars(product).items()
                                if isinstance(v, np.ndarray)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-checkout', type=Path, required=True)
    ap.add_argument('--model-repository', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--archive-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--first-pair-only', action='store_true')
    args = ap.parse_args()
    c = modules(args.source_checkout)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config = c['config'].RunConfig('workflow-sector', str(out / 'catalogue.json'),
        str(out / 'diagnostic.sqlite'), str(out),
        matcher=c['config'].MatcherConfig(repository=str(args.model_repository),
                                         checkpoint=str(args.checkpoint), device='mps'))
    images = {}
    for a, b in zip(IDS, IDS[1:]):
        meta = json.loads((args.archive_root /
            f'dense_pair_{a}_{b}_aliked1024_direct_evaluation_v1/run_manifest.json').read_text())
        for key, prefix in ((a, 'source'), (b, 'target')):
            path = Path(meta[prefix + '_image_filepath'])
            stamp = re.search(r'_(\d{8}T\d{6})_', path.name)[1]
            images[key] = c['catalog'].ImageRecord(str(key), path,
                datetime.strptime(stamp, '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc),
                footprint=c['imagery'].projected_footprint(path))
    manifest = dict(config=asdict(config), bounds_m=BOUNDS, crs='EPSG:3413',
        scope='Six independently measured pairs in a bounded sector; not a full catalogue run.',
        imagery={str(k): {'path': str(v.path), 'time': v.time_utc.isoformat(),
                         'sha256': hashlib.sha256(v.path.read_bytes()).hexdigest()}
                 for k, v in images.items()},
        source_commit=subprocess.check_output(['git', '-C', str(args.source_checkout),
                                               'rev-parse', 'HEAD'], text=True).strip(),
        official_commit=subprocess.check_output(['git', '-C', str(args.model_repository),
                                                 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (args.source_checkout / 'limosat').glob('*.py')},
        pairs={})
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    matcher = c['efficientloftr'].EfficientLoFTR(config.matcher)
    original = c['pairs'].PairProcessor

    class RecordingProcessor(original):
        def _motion_reachable_domain(self, pair):
            x0, x1, y0, y1 = BOUNDS
            return super()._motion_reachable_domain(pair).intersection(box(x0, y0, x1, y1))

        def _filter_tile_matches(self, pair, tile, raw_matches):
            result = super()._filter_tile_matches(pair, tile, raw_matches)
            folder = out / pair.pair_id / 'tiles'
            folder.mkdir(parents=True, exist_ok=True)
            stage = 'coarse' if self.config.matcher.pixel_size_m == 320 else 'fine'
            base = f'{stage}_{tile.region.tile_id:02d}'
            version = len(list(folder.glob(base + '_*.npz')))
            np.savez_compressed(folder / f'{base}_{version}.npz',
                center_xy_m=tile.region.center_xy_m, shift_m=tile.shift,
                row=tile.region.row, column=tile.region.column,
                pixel_size_m=self.config.matcher.pixel_size_m,
                source=tile.source, target=tile.target,
                source_valid=tile.source_valid, target_valid=tile.target_valid,
                raw_source_px=raw_matches[0], raw_target_px=raw_matches[1],
                raw_score=raw_matches[2], source_xy_m=result[0].source_xy_m,
                target_xy_m=result[0].target_xy_m, score=result[0].score)
            return result

        def _refine_tile_shifts(self, pair, regions, shifts, source_sic, target_sic):
            result = super()._refine_tile_shifts(pair, regions, shifts, source_sic, target_sic)
            folder = out / pair.pair_id
            folder.mkdir(parents=True, exist_ok=True)
            version = len(list(folder.glob('alignment_*.npz')))
            np.savez_compressed(folder / f'alignment_{version}.npz',
                centers_xy_m=np.array([r.center_xy_m for r in regions]),
                initial_shift_m=shifts, refined_shift_m=result.shifts,
                tile_ids=np.array([r.tile_id for r in regions]))
            return result

    # The production coarse pass constructs PairProcessor internally; this
    # substitution makes those instances observable using the same methods.
    c['pairs'].PairProcessor = RecordingProcessor
    processor = RecordingProcessor(config, matcher)
    fields, matches, edges = {}, {}, []
    order = list(zip(IDS, IDS[1:])) + [(IDS[0], IDS[2]), (IDS[1], IDS[3]), (IDS[0], IDS[3])]
    for a, b in order:
        pair = c['catalog'].ImagePair(images[a], images[b])
        folder = out / pair.pair_id
        if (folder / 'complete.json').exists():
            fields[a, b] = c['models'].DisplacementField(pair.pair_id, str(a), str(b),
                pair.source.time_utc, pair.target.time_utc, **dict(np.load(folder / 'field.npz')))
            matches[a, b] = c['models'].MotionMatches(**dict(np.load(folder / 'matches.npz')))
            record = json.loads((folder / 'complete.json').read_text())
        else:
            previous = fields.get((IDS[IDS.index(a)-1], a)) if a != IDS[0] and (a,b) in order[:3] else None
            previous_seconds = ((previous.target_time_utc - previous.source_time_utc).total_seconds()
                                if previous is not None else None)
            result = processor.process(pair, previous_field=previous,
                                       previous_elapsed_seconds=previous_seconds)
            folder.mkdir(parents=True, exist_ok=True)
            save_arrays(folder / 'field.npz', result.field)
            save_arrays(folder / 'matches.npz', result.matches)
            fields[a, b], matches[a, b] = result.field, result.matches
            record = dict(diagnostics=result.diagnostics, runtime=result.runtime_seconds,
                          matches=len(result.matches), supported=int(result.field.available.sum()),
                          field_checksum=result.field.checksum)
            (folder / 'complete.json').write_text(json.dumps(record, indent=2))
        edges.append(c['field'].field_edge_with_fallback(fields[a,b], matches[a,b], config.field))
        manifest['pairs'][pair.pair_id] = record
        (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
        print(pair.pair_id, record['matches'], record['supported'], flush=True)
        if args.first_pair_only:
            return
    batches = tuple(c['trajectory'].iter_global_trajectory_batches(edges, list(images.values()),
        config.field, config.trajectories,
        target_validity_factory=c['imagery'].pass_pixel_validity_factory(list(images.values()))))
    points = [asdict(p) for batch in batches for p in batch.points]
    extensions = [asdict(p) for batch in batches for p in batch.extensions]
    (out / 'trajectories.json').write_text(json.dumps(points, default=str))
    (out / 'extensions.json').write_text(json.dumps(extensions, default=str))
    manifest['complete'] = True
    manifest['trajectory_states'] = {str(k): {state: sum(p['image_id'] == str(k) and p['state'] == state for p in points)
        for state in ('created', 'observed', 'dormant', 'reappeared')} for k in IDS}
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(manifest['trajectory_states'], flush=True)


if __name__ == '__main__':
    main()
