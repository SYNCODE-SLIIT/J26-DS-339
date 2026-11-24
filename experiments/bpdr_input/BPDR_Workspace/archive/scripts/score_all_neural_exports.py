"""Full PCM24 export sensitivity with the historical gate decisions held fixed."""

from pathlib import Path

import numpy as np

import qualify_neural_delivery_encoding as export


ROOT = export.ROOT / 'full_rescore'


def main():
    import torch
    torch.set_num_threads(2)
    read, write, sha = export.read, export.write, export.replay.sha
    declaration = read(export.ROOT / 'declaration.json')
    assert declaration['source_sha256'] == sha(Path(export.__file__))
    assert declaration['estimator_source_sha256'] == sha(Path(export.replay.__file__))
    archive_path = export.PARENT / 'results.json'
    assert declaration['results_sha256'] == sha(archive_path)
    receipt_path = export.ROOT / 'export_receipt.json'
    receipt = read(receipt_path)
    assert receipt['declaration_sha256'] == sha(export.ROOT / 'declaration.json')
    archive = read(archive_path)['records']
    protocol = dict(export_receipt_sha256=sha(receipt_path), script_sha256=sha(Path(__file__)),
                    estimators=['pyin', 'pesto', 'praat'], estimates=567,
                    routing='Historical accN is fixed. No gate recomputation or qualification.',
                    production_eligible=False)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == protocol
    else:
        write(path, protocol)
    records = []
    for case in receipt['cases']:
        key = tuple(case['key'])
        target = Path(case['path'])
        assert sha(target) == case['sha256']
        reference_path = export.WS / f'datasets/vocadito/Annotations/F0/vocadito_{key[0]}_f0.csv'
        binding = dict(export_sha256=case['sha256'], reference_sha256=sha(reference_path),
                       declaration_sha256=sha(path))
        cached = ROOT / f'{target.stem}.json'
        if cached.exists():
            saved = read(cached)
            assert saved['binding'] == binding
            records.extend(saved['records'])
            continue
        audio, rate = export.sf.read(target, dtype='float64')
        assert rate == export.replay.SR and audio.ndim == 1
        reference = np.loadtxt(reference_path, delimiter=',')
        reference = reference[reference[:, 0] < len(audio) / rate]
        scored = []
        for estimator in protocol['estimators']:
            prior, = [row for row in archive if
                      (row['track'], row['cond'], row['snr'], row['ext']) == (*key, estimator)]
            times, pitch = export.replay.estimate(estimator, audio)
            value = export.replay.score(times, pitch, reference, prefer_later_tie=True)
            scored.append(dict(prior, delivered=value,
                               historical_gate_delivered=value if prior['accN'] else prior['C']))
        write(cached, dict(binding=binding, records=scored))
        records.extend(scored)
        print('EXPORTED_CASE_SCORED', len(records) // 3, target.stem, flush=True)
    assert len(records) == 567
    summary = {}
    for estimator in protocol['estimators']:
        for condition, subset in (
            ('clean', [r for r in records if r['ext'] == estimator and r['cond'] == 'clean']),
            ('10dB', [r for r in records if r['ext'] == estimator and r['cond'] != 'clean' and r['snr'] == 10]),
            ('0dB', [r for r in records if r['ext'] == estimator and r['cond'] != 'clean' and r['snr'] == 0]),
        ):
            groups = sorted({r['group'] for r in subset})
            means = {field: float(np.mean([np.mean([r[field] for r in subset if r['group'] == group])
                                          for group in groups]))
                     for field in ('C', 'N', 'delivered', 'historical_gate_delivered')}
            summary[f'{estimator}|{condition}'] = dict(
                **means, cases=len(subset), singers=len(groups),
                always_harms=sum(r['delivered'] < r['C'] - .02 for r in subset),
                routed_harms=sum(r['historical_gate_delivered'] < r['C'] - .02 for r in subset),
                historical_accepted=sum(r['accN'] for r in subset),
                changed_scores=sum(r['delivered'] != r['N'] for r in subset),
            )
    write(ROOT / 'results.json', dict(declaration_sha256=sha(path), records=records,
                                      summary=summary, production_eligible=False))
    print('ALL_EXPORTS_SCORED', len(records), flush=True)


if __name__ == '__main__':
    main()
