"""Fixed three-case cleanup test using the actual owning pYIN consumer."""

from importlib.metadata import version
from pathlib import Path
import sys
import time

import numpy as np

import qualify_neural_delivery_encoding as export


WS = export.WS
ROOT = export.ROOT.parent / '2026-11-08_primary_cleanup_compatibility'
DEPENDENCIES = export.ROOT.parent / 'evaluation_dependencies/mir_eval_0_8_2'
sys.path.insert(0, str(DEPENDENCIES))
sys.path.insert(0, str(WS.parents[2] / 'src'))
import mir_eval
from melody_understanding.pipeline import analyze_vocal


def main():
    import torch
    torch.set_num_threads(2)
    read, write, sha = export.read, export.write, export.replay.sha
    freeze_path = export.ROOT.parent / '2026-11-07_primary_extractor_freeze/manifest.json'
    frozen = read(freeze_path)
    for path, digest in frozen['source_hashes'].items():
        assert sha(Path(path)) == digest
    for name, expected in frozen['versions'].items():
        assert version(name) == expected
    assert mir_eval.__version__ == '0.8.2'
    receipt_path = export.ROOT / 'export_receipt.json'
    receipt = read(receipt_path)
    assert receipt['declaration_sha256'] == sha(export.ROOT / 'declaration.json')
    selected = [case for case in receipt['cases'] if
                tuple(case['key']) in ((1, 'clean', 0), (1, '10', 10), (1, '10', 0))]
    assert len(selected) == 3
    reference_paths = [WS / f'datasets/vocadito/Annotations/Notes/vocadito_1_notes{r}.csv' for r in ('A1', 'A2')]
    reference_paths.append(WS / 'datasets/vocadito/Annotations/F0/vocadito_1_f0.csv')
    metric_paths = [Path(mir_eval.transcription.__file__), Path(mir_eval.util.__file__)]
    inputs = {}
    for case in selected:
        for path in (Path(case['path']), export.PARENT / 'mix' / Path(case['path']).name):
            inputs[str(path)] = sha(path)
        assert sha(Path(case['path'])) == case['sha256']
    protocol = dict(
        source_sha256=sha(Path(__file__)), consumer_freeze_sha256=sha(freeze_path),
        export_receipt_sha256=sha(receipt_path), input_hashes=inputs,
        reference_hashes={str(p): sha(p) for p in reference_paths},
        metric_hashes={str(p): sha(p) for p in metric_paths},
        cases=[case['key'] for case in selected], mir_eval_version=mir_eval.__version__,
        metrics=dict(onset_tolerance=.05, pitch_tolerance=50., offset_ratio=.2,
                     offset_min_tolerance=.05, strict=False),
        baseline='Saved enhancer input with constant peak-safe gain and explicit floor PCM24 quantization',
        computation_budget='Six owning-consumer calls. No fit or candidate selection.',
        production_eligible=False,
    )
    ROOT.mkdir(exist_ok=True)
    declaration = ROOT / 'declaration.json'
    if declaration.exists():
        assert read(declaration) == protocol
    else:
        write(declaration, protocol)
    records = []
    for case in selected:
        enhanced = Path(case['path'])
        mixed = export.PARENT / 'mix' / enhanced.name
        original, rate = export.sf.read(mixed, dtype='float64')
        peak = float(np.max(abs(original)))
        gain = min(1., export.CEILING / peak) if peak else 1.
        integers = np.floor(original * gain * 2 ** 23).astype(np.int32)
        baseline = ROOT / enhanced.name
        if not baseline.exists():
            export.sf.write(baseline, integers * 256, rate, subtype='PCM_24')
        decoded, baseline_rate = export.sf.read(baseline, dtype='float64')
        assert rate == baseline_rate == export.replay.SR
        np.testing.assert_array_equal(decoded, integers.astype(np.float64) / 2 ** 23)
        duration = len(decoded) / rate
        for arm, path in [('original', baseline), ('neural', enhanced)]:
            expected_hash = sha(path)
            started = time.perf_counter()
            result, _ = analyze_vocal(path, trim=False)
            seconds = time.perf_counter() - started
            assert sha(path) == expected_hash
            assert result['preprocessing']['trim_enabled'] is False
            assert result['preprocessing']['source_offset_s'] == 0
            assert result['source_duration_s'] == duration
            predicted = result['notes']
            estimated_intervals = np.array([[n['onset_s'], n['onset_s'] + n['duration_s']] for n in predicted]).reshape(-1, 2)
            pitches = np.array([440 * 2 ** ((n['pitch_midi'] - 69) / 12) for n in predicted])
            notes = {}
            for reference_path in reference_paths[:2]:
                values = np.loadtxt(reference_path, delimiter=',', ndmin=2)
                starts, hz, lengths = values.T
                eligible = (starts < duration) & (starts + lengths > 0)
                starts, hz, lengths = starts[eligible], hz[eligible], lengths[eligible]
                truncated = int(np.count_nonzero((starts < 0) | (starts + lengths > duration)))
                intervals = np.column_stack((np.maximum(starts, 0), np.minimum(starts + lengths, duration)))
                scores = {}
                for label, offset in [('onset_pitch', None), ('with_offset', .2)]:
                    options = dict(protocol['metrics'], offset_ratio=offset)
                    precision, recall, f1, overlap = mir_eval.transcription.precision_recall_f1_overlap(
                        intervals, hz, estimated_intervals, pitches, **options)
                    scores[label] = dict(precision=precision, recall=recall, f1=f1, overlap=overlap)
                notes[reference_path.stem[-2:]] = dict(reference_notes=len(hz), truncated_reference_notes=truncated,
                                                     estimated_notes=len(predicted), scores=scores)
            reference = np.loadtxt(reference_paths[2], delimiter=',')
            reference = reference[reference[:, 0] < duration]
            times = np.array(result['pitch']['times_s'])
            frequencies = np.array([0 if f is None else f for f in result['pitch']['f0_hz']])
            rpa = export.replay.score(times, frequencies, reference, prefer_later_tie=True)
            record = dict(key=case['key'], arm=arm, waveform_sha256=expected_hash, notes=notes,
                          rpa50=rpa, seconds=seconds, preprocessing=result['preprocessing'])
            records.append(record)
            write(ROOT / f'{enhanced.stem}_{arm}.json', record)
            print('PRIMARY_CONSUMER_CASE', case['key'], arm, rpa, flush=True)
    write(ROOT / 'results.json', dict(declaration_sha256=sha(declaration), records=records,
                                     fresh_confirmation=False, production_eligible=False))


if __name__ == '__main__':
    main()
