"""Actual frozen pYIN consumer, 189 opened cases, two human note annotations."""

import csv
from pathlib import Path
import time

import numpy as np

import check_primary_cleanup_consumer as compatibility


export = compatibility.export
ROOT = export.ROOT.parent / '2026-11-08_primary_cleanup_development'


def metrics(result, reference_paths, duration, options):
    notes = result['notes']
    intervals = np.array([[n['onset_s'], n['onset_s'] + n['duration_s']] for n in notes]).reshape(-1, 2)
    hz = np.array([440 * 2 ** ((n['pitch_midi'] - 69) / 12) for n in notes])
    values = {}
    for path in reference_paths[:2]:
        source = np.loadtxt(path, delimiter=',', ndmin=2)
        start, pitch, length = source.T
        keep = (start < duration) & (start + length > 0)
        start, pitch, length = start[keep], pitch[keep], length[keep]
        reference = np.column_stack((np.maximum(start, 0), np.minimum(start + length, duration)))
        scores = {}
        for name, offset in [('onset_pitch', None), ('with_offset', .2)]:
            precision, recall, f1, overlap = compatibility.mir_eval.transcription.precision_recall_f1_overlap(
                reference, pitch, intervals, hz, **dict(options, offset_ratio=offset))
            scores[name] = dict(precision=precision, recall=recall, f1=f1, overlap=overlap)
        values[path.stem[-2:]] = dict(
            reference_notes=len(pitch), estimated_notes=len(notes),
            truncated_reference_notes=int(np.count_nonzero((start < 0) | (start + length > duration))),
            scores=scores)
    f0 = np.loadtxt(reference_paths[2], delimiter=',')
    f0 = f0[f0[:, 0] < duration]
    times = np.asarray(result['pitch']['times_s'])
    frequencies = np.array([0 if f is None else f for f in result['pitch']['f0_hz']])
    return values, export.replay.score(times, frequencies, f0, prefer_later_tie=True)


def main():
    import torch
    torch.set_num_threads(2)
    read, write, sha = export.read, export.write, export.replay.sha
    frozen_path = export.ROOT.parent / '2026-11-07_primary_extractor_freeze/manifest.json'
    frozen = read(frozen_path)
    for path, digest in frozen['source_hashes'].items():
        assert sha(Path(path)) == digest
    for package, expected in frozen['versions'].items():
        assert compatibility.version(package) == expected
    previous = compatibility.ROOT / 'results.json'
    assert read(compatibility.ROOT / 'integrity_receipt.json')['results_sha256'] == sha(previous)
    protocol_path = compatibility.ROOT / 'declaration.json'
    protocol = read(protocol_path)
    assert protocol['source_sha256'] == sha(Path(compatibility.__file__))
    receipt_path = export.ROOT / 'export_receipt.json'
    receipt = read(receipt_path)
    assert len(receipt['cases']) == 189
    assert receipt['declaration_sha256'] == sha(export.ROOT / 'declaration.json')
    archive = read(export.PARENT / 'results.json')
    assert read(export.ROOT / 'declaration.json')['results_sha256'] == sha(export.PARENT / 'results.json')
    groups = {row['track']: row['group'] for row in archive['records']}
    tracks = {case['key'][0] for case in receipt['cases']}
    allowed = {1,10,13,14,18,20,25,26,27,28,29,3,31,32,35,36,37,38,39,6,8}
    assert tracks == allowed and len(set(groups.values())) == 14
    declaration = dict(
        source_sha256=sha(Path(__file__)), metric_helper_sha256=sha(Path(compatibility.__file__)),
        export_receipt_sha256=sha(receipt_path), consumer_freeze_sha256=sha(frozen_path),
        compatibility_result_sha256=sha(previous), metrics=protocol['metrics'],
        cases=189, consumer_calls=378, annotators=['A1', 'A2'],
        study='Opened development data. No selection or confirmation.',
        descriptive_harm_margin=.02, production_eligible=False,
    )
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == declaration
    else:
        write(path, declaration)
    records = []
    for case in receipt['cases']:
        key = case['key']
        delivered = Path(case['path'])
        assert sha(delivered) == case['sha256']
        mixed = export.PARENT / 'mix' / delivered.name
        audio, rate = export.sf.read(mixed, dtype='float64')
        assert rate == export.replay.SR and audio.ndim == 1 and np.isfinite(audio).all()
        peak = float(np.max(abs(audio)))
        gain = min(1., export.CEILING / peak) if peak else 1.
        integers = np.floor(audio * gain * 2 ** 23).astype(np.int32)
        original = ROOT / f'{delivered.stem}_original.wav'
        if not original.exists():
            export.sf.write(original, integers * 256, rate, subtype='PCM_24')
        decoded, actual_rate = export.sf.read(original, dtype='float64')
        assert actual_rate == rate
        np.testing.assert_array_equal(decoded, integers.astype(np.float64) / 2 ** 23)
        duration = len(decoded) / rate
        references = [export.WS / f'datasets/vocadito/Annotations/Notes/vocadito_{key[0]}_notes{annotator}.csv'
                      for annotator in ('A1', 'A2')]
        references.append(export.WS / f'datasets/vocadito/Annotations/F0/vocadito_{key[0]}_f0.csv')
        for arm, waveform in [('original', original), ('neural', delivered)]:
            binding = dict(declaration_sha256=sha(path), waveform_sha256=sha(waveform),
                           reference_hashes={str(p): sha(p) for p in references})
            cached = ROOT / f'{delivered.stem}_{arm}.json'
            if cached.exists():
                saved = read(cached)
                assert saved['binding'] == binding
                records.append(saved['record'])
                continue
            started = time.perf_counter()
            result, _ = compatibility.analyze_vocal(waveform, trim=False)
            assert sha(waveform) == binding['waveform_sha256']
            assert result['preprocessing']['trim_enabled'] is False
            assert result['source_duration_s'] == duration
            notes, rpa = metrics(result, references, duration, protocol['metrics'])
            record = dict(key=key, arm=arm, group=groups[key[0]], notes=notes, rpa50=rpa,
                          seconds=time.perf_counter() - started, preprocessing=result['preprocessing'])
            write(cached, dict(binding=binding, record=record))
            records.append(record)
            print('PRIMARY_DEVELOPMENT_SCORED', len(records), key, arm, flush=True)
    assert len(records) == 378
    by_case = {}
    for record in records:
        by_case.setdefault(tuple(record['key']), {})[record['arm']] = record
    summary = {}
    for annotator in ('A1', 'A2'):
        for matching in ('onset_pitch', 'with_offset'):
            for label in ('clean', '10dB', '0dB'):
                pairs = [arms for key, arms in by_case.items() if
                         (key[1] == 'clean' if label == 'clean' else key[1] != 'clean' and key[2] == int(label[:-2]))]
                differences = []
                for arms in pairs:
                    difference = (arms['neural']['notes'][annotator]['scores'][matching]['f1'] -
                                  arms['original']['notes'][annotator]['scores'][matching]['f1'])
                    differences.append((arms['original']['group'], difference))
                singers = sorted({group for group, _ in differences})
                per_singer = np.array([np.mean([value for group, value in differences if group == singer])
                                       for singer in singers])
                bootstrap = per_singer[np.random.default_rng(339).integers(
                    0, len(singers), (4000, len(singers)))].mean(axis=1)
                summary[f'{annotator}|{matching}|{label}'] = dict(
                    cases=len(pairs), singers=len(singers), paired_singer_macro_gain=float(per_singer.mean()),
                    paired_bootstrap_95=np.quantile(bootstrap, [.025, .975]).tolist(),
                    harms=sum(value < -.02 for _, value in differences),
                    benefits=sum(value > .02 for _, value in differences),
                    scope='Descriptive development resampling; mixtures and frames are not independent participants.')
    write(ROOT / 'results.json', dict(declaration_sha256=sha(path), records=records, summary=summary,
                                      production_eligible=False))
    print('PRIMARY_DEVELOPMENT_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
