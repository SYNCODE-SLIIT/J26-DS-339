"""Inspect waveform alignment in all clean-control enhancer exports."""

from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags

import qualify_neural_delivery_encoding as export


ROOT = export.ROOT / 'alignment'
RATE = export.replay.SR


def alignment(before, after):
    before = before - np.mean(before)
    after = after - np.mean(after)
    scale = float(np.linalg.norm(before) * np.linalg.norm(after))
    if scale <= 1e-12:
        return dict(unassessed='INSUFFICIENT_WINDOW_ENERGY')
    values = correlate(after, before, mode='full', method='fft') / scale
    lags = correlation_lags(len(after), len(before))
    keep = abs(lags) <= round(.1 * RATE)
    lags, values = lags[keep], values[keep]
    index = int(np.argmax(values))
    lag = int(lags[index])
    outside = abs(lags - lag) > round(.002 * RATE)
    return dict(lag_samples=lag, lag_ms=1000 * lag / RATE,
                peak_correlation=float(values[index]), zero_lag_correlation=float(values[lags == 0][0]),
                competing_peak_correlation=float(np.max(values[outside])))


def main():
    read, write, sha = export.read, export.write, export.replay.sha
    rng = np.random.default_rng(339)
    source = rng.normal(size=4000)
    delayed = np.r_[np.zeros(100), source[:-100]]
    assert alignment(source, delayed)['lag_samples'] == 100
    receipt_path = export.ROOT / 'export_receipt.json'
    receipt = read(receipt_path)
    assert receipt['declaration_sha256'] == sha(export.ROOT / 'declaration.json')
    cases = [case for case in receipt['cases'] if case['key'][1] == 'clean']
    assert len(cases) == 21
    protocol = dict(script_sha256=sha(Path(__file__)), export_receipt_sha256=sha(receipt_path),
                    cases=21, window_seconds=1, max_lag_seconds=.1, time_shift_applied=False)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == protocol
    else:
        write(path, protocol)
    reports = []
    parent_cases = {tuple(case['key']): case for case in read(export.ROOT / 'declaration.json')['cases']}
    for case in cases:
        key = tuple(case['key'])
        mixed_path = export.PARENT / 'mix' / Path(case['path']).name
        historical = parent_cases[key]
        assert sha(Path(historical['path'])) == historical['sha256']
        assert sha(Path(case['path'])) == case['sha256']
        mixed, rate = export.sf.read(mixed_path, dtype='float64')
        floating, float_rate = export.sf.read(historical['path'], dtype='float64')
        delivered, target_rate = export.sf.read(case['path'], dtype='float64')
        assert rate == float_rate == target_rate == RATE
        assert mixed.shape == floating.shape == delivered.shape and mixed.ndim == 1
        starts = np.linspace(0, len(mixed) - RATE, 4, dtype=int)
        windows = [('full', slice(None)), *[(str(int(start)), slice(start, start + RATE)) for start in starts]]
        metrics = {label: {name: alignment(mixed[span], audio[span]) for name, span in windows}
                   for label, audio in [('float', floating), ('pcm24', delivered)]}
        reports.append(dict(key=list(key), mix_sha256=sha(mixed_path), metrics=metrics))
    result = dict(declaration_sha256=sha(path), records=reports, internal_delay_configuration_verified=False)
    output = ROOT / 'results.json'
    if output.exists():
        assert read(output) == result
    else:
        write(output, result)
    print('NEURAL_ALIGNMENT_CONTROL_COMPLETE', len(reports), flush=True)


if __name__ == '__main__':
    main()
