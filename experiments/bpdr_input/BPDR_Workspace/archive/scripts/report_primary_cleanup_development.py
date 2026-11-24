"""Rebuild development aggregates without importing the consumer or study runner."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / 'results/transformer_usability/2026-11-08_primary_cleanup_development'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = ROOT / 'results.json'
    results = json.loads(path.read_text())
    paired = defaultdict(dict)
    for row in results['records']:
        paired[tuple(row['key'])][row['arm']] = row
    assert len(paired) == 189 and all(set(p) == {'original', 'neural'} for p in paired.values())
    slices = {}
    maximum_error = 0.
    for label in ['clean', '10dB', '0dB']:
        cases = [p for key, p in paired.items()
                 if ('clean' if key[1] == 'clean' else f'{key[2]}dB') == label]
        singers = sorted({p['original']['group'] for p in cases})
        def macro(getter, arm):
            return np.array([np.mean([getter(p[arm]) for p in cases if p['original']['group'] == singer]) for singer in singers])
        block = dict(cases=len(cases), singers=len(singers))
        block['rpa50'] = {arm: float(macro(lambda r: r['rpa50'], arm).mean()) for arm in ['original', 'neural']}
        block['estimated_note_count'] = {arm: float(np.mean([p[arm]['notes']['A1']['estimated_notes'] for p in cases])) for arm in ['original', 'neural']}
        for annotator in ['A1', 'A2']:
            for metric in ['onset_pitch', 'with_offset']:
                def f1(row):
                    return row['notes'][annotator]['scores'][metric]['f1']
                original, neural = macro(f1, 'original'), macro(f1, 'neural')
                differences = neural - original
                rng = np.random.default_rng(339)
                bootstrap = [differences[rng.integers(len(singers), size=len(singers))].mean() for _ in range(4000)]
                ci = np.quantile(bootstrap, [.025, .975])
                deltas = [f1(p['neural']) - f1(p['original']) for p in cases]
                harms, benefits = sum(d < -.02 for d in deltas), sum(d > .02 for d in deltas)
                expected = results['summary'][f'{annotator}|{metric}|{label}']
                assert expected['harms'] == harms and expected['benefits'] == benefits
                assert expected['cases'] == len(cases) and expected['singers'] == len(singers)
                maximum_error = max(maximum_error, abs(float(differences.mean()) - expected['paired_singer_macro_gain']),
                                    *abs(ci - np.asarray(expected['paired_bootstrap_95'])))
                block[f'{annotator}|{metric}'] = dict(original=float(original.mean()), neural=float(neural.mean()),
                    gain=float(differences.mean()), ci95=ci.tolist(), harms=harms, benefits=benefits)
        slices[label] = block
    assert maximum_error < 1e-12
    receipt = dict(results_sha256=sha(path), source_sha256=sha(Path(__file__)),
                   cases=189, rows=378, summaries_reconstructed=12,
                   maximum_aggregate_replay_error=maximum_error,
                   scope='Independent saved-metric aggregation only. Consumer inference replay is separate.',
                   slices=slices, production_eligible=False)
    (ROOT / 'aggregate_replay_receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
