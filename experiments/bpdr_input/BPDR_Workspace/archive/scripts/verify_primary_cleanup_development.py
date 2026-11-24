"""Independent input binding and consumer replay for the opened cleanup study.

This verifies the declared development comparison; it selects no policy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


WS = Path(__file__).resolve().parents[1]
BASE = WS / 'results/transformer_usability'
ROOT = BASE / '2026-11-08_primary_cleanup_development'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def inputs():
    audit_path = BASE / '2026-11-07_continuation_audit/evidence_audit.json'
    audit = read(audit_path)['neural_audio']['rows']
    parent = BASE / '2026-11-05_real_noise_neural'
    receipt_path = BASE / '2026-11-07_neural_delivery_encoding_v2/export_receipt.json'
    exports = read(receipt_path)['cases']
    assert len(audit) == len(exports) == 189
    assert {r['file'] for r in audit} == {Path(r['path']).name for r in exports}
    rows = []
    for row in audit:
        for folder, field in [('mix', 'mixed_sha256'), ('enh', 'enhanced_sha256')]:
            path = parent / folder / row['file']
            assert sha(path) == row[field], str(path)
        rows.append(dict(file=row['file'], mixed_sha256=row['mixed_sha256'],
                         enhanced_sha256=row['enhanced_sha256']))
    for row in exports:
        assert sha(Path(row['path'])) == row['sha256']
    result = dict(source_sha256=sha(Path(__file__)), earlier_audit_sha256=sha(audit_path),
                  export_receipt_sha256=sha(receipt_path),
                  declaration_sha256=sha(ROOT / 'declaration.json'), cases=189,
                  mixed_and_float_match_prior_audit=True, delivered_match_export_receipt=True,
                  rows=rows, scope='Input identity only; no inference or qualification.')
    write(ROOT / 'input_integrity_receipt.json', result)
    print('PRIMARY_INPUT_INTEGRITY', len(rows), flush=True)
    return result


def verify():
    import torch
    import evaluate_primary_cleanup_development as study
    torch.set_num_threads(2)
    provenance = inputs()
    result_path = ROOT / 'results.json'
    result = read(result_path)
    declaration_path = ROOT / 'declaration.json'
    declaration = read(declaration_path)
    assert result['declaration_sha256'] == sha(declaration_path)
    assert declaration['source_sha256'] == sha(Path(study.__file__))
    assert declaration['metric_helper_sha256'] == sha(Path(study.compatibility.__file__))
    frozen_path = BASE / '2026-11-07_primary_extractor_freeze/manifest.json'
    assert declaration['consumer_freeze_sha256'] == sha(frozen_path)
    frozen = read(frozen_path)
    for path, digest in frozen['source_hashes'].items():
        assert sha(Path(path)) == digest
    for package, expected in frozen['versions'].items():
        assert study.compatibility.version(package) == expected
    subset_declaration = read(study.compatibility.ROOT / 'declaration.json')
    for path, digest in subset_declaration['metric_hashes'].items():
        assert sha(Path(path)) == digest
    replay_dir = ROOT / 'independent_replay'
    replay_dir.mkdir(exist_ok=True)
    binding = dict(verifier_sha256=sha(Path(__file__)), results_sha256=sha(result_path),
                   declaration_sha256=sha(declaration_path), input_receipt_sha256=sha(ROOT / 'input_integrity_receipt.json'))
    replayed = []
    assert len(result['records']) == 378
    export_cases = read(BASE / '2026-11-07_neural_delivery_encoding_v2/export_receipt.json')['cases']
    export_by_key = {tuple(c['key']): c for c in export_cases}
    expected_pairs = {(key, arm) for key in export_by_key for arm in ('original', 'neural')}
    actual_pairs = {(tuple(row['key']), row['arm']) for row in result['records']}
    assert len(expected_pairs) == 378 and actual_pairs == expected_pairs
    for row in result['records']:
        track, condition, snr = row['key']
        stem = f'{track}_{condition}_{snr}'
        cached = read(ROOT / f"{stem}_{row['arm']}.json")
        assert cached['record'] == row
        assert cached['binding']['declaration_sha256'] == sha(declaration_path)
        if row['arm'] == 'original':
            wave = ROOT / f'{stem}_original.wav'
        else:
            wave = Path(export_by_key[tuple(row['key'])]['path'])
        assert sha(wave) == cached['binding']['waveform_sha256']
        references = [Path(p) for p in cached['binding']['reference_hashes']]
        for path in references:
            assert sha(path) == cached['binding']['reference_hashes'][str(path)]
        replay_path = replay_dir / f"{stem}_{row['arm']}.json"
        if replay_path.exists():
            replay = read(replay_path)
            assert replay['binding'] == binding
        else:
            started = time.perf_counter()
            consumer, _ = study.compatibility.analyze_vocal(wave, trim=False)
            assert sha(wave) == cached['binding']['waveform_sha256']
            notes, rpa = study.metrics(consumer, references, consumer['source_duration_s'], declaration['metrics'])
            assert notes == row['notes']
            assert rpa == row['rpa50']
            assert consumer['preprocessing'] == row['preprocessing']
            replay = dict(binding=binding, notes=notes, rpa50=rpa,
                          consumer=consumer, seconds=time.perf_counter() - started)
            write(replay_path, replay)
        assert replay['notes'] == row['notes'] and replay['rpa50'] == row['rpa50']
        replayed.append(dict(path=str(replay_path), sha256=sha(replay_path)))
        if len(replayed) % 18 == 0:
            print('PRIMARY_INDEPENDENT_REPLAY', len(replayed), flush=True)
    # Reconstruct singer summaries independently of the study's aggregation loop.
    paired = {}
    for row in result['records']:
        paired.setdefault(tuple(row['key']), {})[row['arm']] = row
    maximum_error = 0.
    for name, expected in result['summary'].items():
        annotator, matching, label = name.split('|')
        values = []
        for key, arms in paired.items():
            category = 'clean' if key[1] == 'clean' else f'{key[2]}dB'
            if category != label:
                continue
            delta = arms['neural']['notes'][annotator]['scores'][matching]['f1'] - arms['original']['notes'][annotator]['scores'][matching]['f1']
            values.append((arms['original']['group'], delta))
        singers = sorted(set(group for group, _ in values))
        means = np.array([sum(d for g, d in values if g == s) / sum(g == s for g, _ in values) for s in singers])
        rng = np.random.default_rng(339)
        boot = np.array([means[rng.integers(len(means), size=len(means))].mean() for _ in range(4000)])
        interval = np.quantile(boot, [.025, .975])
        assert expected['cases'] == len(values) and expected['singers'] == len(singers)
        assert expected['harms'] == sum(d < -.02 for _, d in values)
        assert expected['benefits'] == sum(d > .02 for _, d in values)
        errors = [abs(float(means.mean()) - expected['paired_singer_macro_gain'])]
        errors.extend(abs(interval - np.asarray(expected['paired_bootstrap_95'])))
        maximum_error = max(maximum_error, *errors)
    assert maximum_error < 1e-12
    assert sha(result_path) == binding['results_sha256']
    write(ROOT / 'integrity_receipt.json', dict(
        **binding, cases=189, consumer_calls=378, raw_consumer_outputs_retained=True,
        metrics_replayed_exactly=True, maximum_aggregate_replay_error=maximum_error,
        input_cases=provenance['cases'], replay_files=replayed, production_eligible=False))
    print('PRIMARY_DEVELOPMENT_VERIFIED', len(replayed), maximum_error, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['inputs', 'verify'], required=True)
    args = parser.parse_args()
    inputs() if args.mode == 'inputs' else verify()
