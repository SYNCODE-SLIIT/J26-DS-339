"""Check real token support for split, merge and jitter boundary interventions."""

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import compare_short_objectives as run
from bpdr_input.repair.interventions import corrupt_boundaries
from bpdr_input.preparation.pairs.shared_pair_audit import game_boundary_track


ROOT = run.ROOT.parent / '2026-11-07_boundary_corruption_audit'
KINDS = ('split', 'merge', 'jitter')


def variants(raw):
    times = np.arange(len(raw)) * .02
    peaks, _ = find_peaks(raw[:, 70], height=.5, distance=5)
    edges = times[peaks]
    output = {}
    for kind in KINDS:
        try:
            corrupted = corrupt_boundaries(edges, times, kind, seed=339)
        except ValueError as exc:
            output[kind] = dict(excluded=str(exc))
            continue
        changed = raw.copy()
        changed[:, 70] = game_boundary_track(times, corrupted)
        output[kind] = dict(raw=changed, original_edges=len(edges), corrupted_edges=len(corrupted))
    return output


def main():
    read, write, sha = run.read, run.write, run.sha
    data = run.parent.DATA
    integrity = read(data / 'integrity_receipt.json')
    for name in ('train_pairs', 'train_originals'):
        assert integrity[name + '_sha256'] == sha(data / (name + '.json'))
    assert integrity['input_hashes_sha256'] == sha(data / 'input_hashes.json')
    hashes = read(data / 'input_hashes.json')
    paths = [Path(__file__), Path(corrupt_boundaries.__code__.co_filename),
             Path(game_boundary_track.__code__.co_filename), data / 'integrity_receipt.json']
    ROOT.mkdir(exist_ok=True)
    declaration = dict(source_hashes={str(path): sha(path) for path in paths}, seed=339,
                       kinds=list(KINDS), fits=0, training_roles_only=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == declaration
    else:
        write(path, declaration)
    records = []
    for original, name in [(True, 'train_originals'), (False, 'train_pairs')]:
        for index, row in enumerate(read(data / (name + '.json'))):
            if 'excluded' in row:
                continue
            analysis = row['analysis_A' if original else 'analysis_C']
            assert sha(analysis) == hashes[analysis]
            with np.load(analysis) as archive:
                raw = archive['raw_views'][0].copy()
                masks = {'original': archive['reliable'].copy()} if original else {}
            if not original:
                target = row['target_path']
                assert sha(target) == hashes[target]
                with np.load(target) as archive:
                    masks = dict(positive=archive['mask'] & (archive['target'] != 0),
                                 known_zero=archive['mask'] & (archive['target'] == 0))
            record = dict(id=('A_' + row['source_id']) if original else f'C_{index}', group=row['group'], variants={})
            for kind, changed in variants(raw).items():
                if 'excluded' in changed:
                    record['variants'][kind] = changed
                    continue
                contexts, pad = run.boundary_tokens(raw, changed['raw'], np.arange(len(raw)), 'bpdr')
                control, control_pad = run.boundary_tokens(raw, changed['raw'], np.arange(len(raw)), 'frame_only')
                np.testing.assert_array_equal(contexts[0][:, :, :8], contexts[1][:, :, :8])
                np.testing.assert_array_equal(control[0], control[1])
                np.testing.assert_array_equal(pad, control_pad)
                affected = np.any(contexts[0][:, :, 8] != contexts[1][:, :, 8], axis=1)
                record['variants'][kind] = dict(
                    original_edges=changed['original_edges'], corrupted_edges=changed['corrupted_edges'],
                    queries={label: dict(total=int(mask.sum()), affected=int((mask & affected).sum()))
                             for label, mask in masks.items()},
                )
            records.append(record)
    assert len(records) == 498
    summary = {}
    for kind in KINDS:
        available = [r['variants'][kind] for r in records if 'excluded' not in r['variants'][kind]]
        summary[kind] = dict(assessable=len(available), excluded=len(records) - len(available), queries={
            label: {key: sum(row['queries'].get(label, {}).get(key, 0) for row in available)
                    for key in ('total', 'affected')}
            for label in ('original', 'positive', 'known_zero')
        })
    result = dict(declaration_sha256=sha(path), summary=summary, records=records)
    output = ROOT / 'results.json'
    if output.exists():
        assert read(output) == result
    else:
        write(output, result)
    print('BOUNDARY_CORRUPTION_AUDIT', summary, flush=True)


if __name__ == '__main__':
    main()
