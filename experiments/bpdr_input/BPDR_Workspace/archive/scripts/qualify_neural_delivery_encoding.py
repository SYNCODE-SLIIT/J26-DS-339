"""Preserve historical enhancement outputs and test explicit PCM24 delivery."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import replay_noise_audit_subset as replay


WS = Path(__file__).resolve().parents[1]
PARENT = WS / 'results/transformer_usability/2026-11-05_real_noise_neural'
ROOT = PARENT.parent / '2026-11-07_neural_delivery_encoding_v2'
CEILING = 1 - 2 ** -20


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def declare():
    archive = read(PARENT / 'results.json')
    assert archive['declaration_sha256'] == replay.sha(PARENT / 'declaration.json')
    paths = sorted((PARENT / 'enh').glob('*.wav'))
    keys = {(row['track'], row['cond'], row['snr']) for row in archive['records']}
    assert len(paths) == len(keys) == 189
    cases = []
    for path in paths:
        track, condition, snr = path.stem.split('_')
        key = (int(track), condition, int(snr))
        assert key in keys
        audio, rate = sf.read(path, dtype='float64')
        assert audio.ndim == 1 and rate == replay.SR and np.isfinite(audio).all()
        peak = float(np.max(abs(audio)))
        cases.append(dict(key=list(key), path=str(path), sha256=replay.sha(path),
                          frames=len(audio), peak=peak, gain=min(1., CEILING / peak) if peak else 1.))
    assert sum(case['peak'] > 1 for case in cases) == 7
    declaration = dict(
        source_sha256=replay.sha(Path(__file__)), estimator_source_sha256=replay.sha(Path(replay.__file__)),
        results_sha256=replay.sha(PARENT / 'results.json'), cases=cases,
        ceiling=CEILING, output_subtype='PCM_24', sample_rate=replay.SR,
        quantization='floor(scaled*2^23), left aligned int32 samples written as PCM24',
        score_subset='Every historical FLOAT file with peak greater than one.',
        versions={name: version(name) for name in ('numpy', 'soundfile', 'librosa', 'pesto-pitch', 'praat-parselmouth')},
        training=False, production_eligible=False,
    )
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == declaration
    else:
        write(path, declaration)
    return declaration, archive


def export(declaration):
    folder = ROOT / 'pcm24'
    folder.mkdir(exist_ok=True)
    receipts = []
    for case in declaration['cases']:
        source = Path(case['path'])
        assert replay.sha(source) == case['sha256']
        original, rate = sf.read(source, dtype='float64')
        scaled = original * case['gain']
        target = folder / source.name
        integers = np.floor(scaled * 2 ** 23).astype(np.int32)
        if not target.exists():
            sf.write(target, integers * 256, rate, subtype='PCM_24')
        actual, actual_rate = sf.read(target, dtype='float64')
        expected = integers.astype(np.float64) / 2 ** 23
        np.testing.assert_array_equal(actual, expected)
        assert actual_rate == rate and actual.shape == original.shape
        assert sf.info(target).subtype == 'PCM_24' and np.max(abs(actual)) < 1
        receipts.append(dict(key=case['key'], path=str(target), sha256=replay.sha(target),
                             gain=case['gain'], decoded_peak=float(np.max(abs(actual))),
                             original_over_full_scale=case['peak'] > 1,
                             quantization_max_error=float(np.max(abs(actual - scaled)))))
    write(ROOT / 'export_receipt.json', dict(declaration_sha256=replay.sha(ROOT / 'declaration.json'),
                                           cases=receipts, exact_quantized_samples=True))


def score(declaration, archive):
    import torch
    torch.set_num_threads(2)
    receipt = read(ROOT / 'export_receipt.json')
    assert receipt['declaration_sha256'] == replay.sha(ROOT / 'declaration.json')
    outputs = {tuple(case['key']): case for case in receipt['cases']}
    scored = []
    for case in declaration['cases']:
        if case['peak'] <= 1:
            continue
        key = tuple(case['key'])
        track, condition, snr = key
        path = Path(case['path'])
        assert replay.sha(path) == case['sha256']
        target = outputs[key]
        assert replay.sha(Path(target['path'])) == target['sha256']
        original, rate = sf.read(path, dtype='float64')
        delivered, target_rate = sf.read(target['path'], dtype='float64')
        assert rate == target_rate == replay.SR and original.shape == delivered.shape
        reference_path = WS / f'datasets/vocadito/Annotations/F0/vocadito_{track}_f0.csv'
        reference = np.loadtxt(reference_path, delimiter=',')
        reference = reference[reference[:, 0] < len(original) / rate]
        for estimator in ('pyin', 'pesto', 'praat'):
            historical, = [row for row in archive['records'] if
                           (row['track'], row['cond'], row['snr'], row['ext']) == (*key, estimator)]
            values = {}
            for label, audio in (('float', original), ('pcm24', delivered)):
                times, pitch = replay.estimate(estimator, audio)
                values[label] = replay.score(times, pitch, reference, prefer_later_tie=True)
            assert abs(values['float'] - historical['N']) < 1e-12, (key, estimator, values, historical['N'])
            scored.append(dict(key=list(key), estimator=estimator, archived_float=historical['N'],
                               **values, delta=values['pcm24'] - values['float'],
                               historical_gate_accepted=historical['accN'], clean_control=condition == 'clean',
                               reference_sha256=replay.sha(reference_path)))
        write(ROOT / 'subset_progress.json', dict(completed_estimates=len(scored), records=scored))
        print('DELIVERY_ENCODING_SCORED', key, flush=True)
    assert len(scored) == 21
    write(ROOT / 'subset_results.json', dict(
        declaration_sha256=replay.sha(ROOT / 'declaration.json'),
        export_receipt_sha256=replay.sha(ROOT / 'export_receipt.json'), records=scored,
        original_estimates_reproduced=21, delivered_estimates=21, full_study_rescored=False,
        delay_validated=False, production_eligible=False,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score', action='store_true')
    args = parser.parse_args()
    declaration, archive = declare()
    export(declaration)
    print('DELIVERY_EXPORT_VERIFIED', len(declaration['cases']), flush=True)
    if args.score:
        score(declaration, archive)


if __name__ == '__main__':
    main()
