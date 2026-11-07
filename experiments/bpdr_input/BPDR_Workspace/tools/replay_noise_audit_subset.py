"""Independent fixed small replay of saved noise-study pitch scores.

Chosen by source ID/condition, not by score: training track 1, clean and rain at
10/0 dB, raw and saved neural output, all three original frozen estimators.
This is a three-case replay, not full validation or a new model experiment.
"""
from pathlib import Path
import hashlib
import json
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

WS = Path(__file__).resolve().parents[1]
ROOT = WS/'results/transformer_usability'
OUT = ROOT/'2026-11-07_continuation_audit'
SR = 22050


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    x, rate = sf.read(path, dtype='float32')
    if x.ndim == 2:
        x = x.mean(axis=1)
    if rate != SR:
        divisor = gcd(rate, SR)
        x = resample_poly(x, SR//divisor, rate//divisor)
    return x.astype(np.float64)


def score(times, frequencies, reference, prefer_later_tie=False):
    reference = reference[reference[:, 1] > 0]
    distances = np.abs(times[:, None] - reference[None, :, 0])
    # Explicit nearest neighbor, independent of the experiment's searchsorted.
    nearest = (len(times)-1-distances[::-1].argmin(axis=0)
               if prefer_later_tie else distances.argmin(axis=0))
    predicted = frequencies[nearest]
    valid = predicted > 0
    error = np.full(len(predicted), np.inf)
    error[valid] = 1200*np.abs(np.log2(predicted[valid]/reference[valid, 1]))
    return float(np.mean(error <= 50))


def estimate(name, audio):
    if name == 'pyin':
        import librosa
        f, v, _ = librosa.pyin(audio.astype(np.float32), sr=SR, fmin=librosa.note_to_hz('C1'),
                               fmax=librosa.note_to_hz('C7'), frame_length=4096,
                               hop_length=256, fill_na=np.nan)
        return np.arange(len(f))*256/SR, np.where(v & np.isfinite(f), f, 0.)
    if name == 'praat':
        import parselmouth
        p = parselmouth.Sound(audio.astype(np.float64), SR).to_pitch_ac(
            time_step=.01, pitch_floor=75, pitch_ceiling=800)
        return p.xs(), p.selected_array['frequency']
    import pesto
    import torch
    t, f, c, _ = pesto.predict(torch.from_numpy(audio.astype(np.float32)), SR)
    t, f, c = [np.asarray(a.detach().cpu(), dtype=float) for a in (t, f, c)]
    return t/1000., np.where((c > .5) & (f > 0), f, 0.)


def main():
    import torch
    torch.set_num_threads(2)
    study = ROOT/'2026-11-05_real_noise_neural'
    archive = json.loads((study/'results.json').read_text())
    voc = WS/'datasets/vocadito'
    x = load(voc/'Audio/vocadito_1.wav')[:8*SR]
    noise = load(WS/'datasets/environmental_noise/1-17367-A-10.wav')
    noise = np.resize(noise, len(x))
    noise /= np.sqrt(np.mean(noise**2))+1e-12
    rms = np.sqrt(np.mean(x**2))+1e-12
    reference = np.loadtxt(voc/'Annotations/F0/vocadito_1_f0.csv', delimiter=',')
    reference = reference[reference[:, 0] < len(x)/SR]
    records = []
    for condition, snr in [('clean', 0), ('10', 10), ('10', 0)]:
        raw = x if condition == 'clean' else x+noise*rms/10**(snr/20)
        path = study/'enh'/f'1_{condition}_{snr}.wav'
        enhanced, rate = sf.read(path, dtype='float64')
        assert rate == SR and enhanced.shape == raw.shape
        for estimator in ('pyin', 'pesto', 'praat'):
            saved = next(r for r in archive['records'] if r['track'] == 1 and
                         r['cond'] == condition and r['snr'] == snr and r['ext'] == estimator)
            for key, signal in [('C', raw), ('N', enhanced)]:
                times, pitch = estimate(estimator, signal)
                measured = score(times, pitch, reference)
                later_tie = score(times, pitch, reference, prefer_later_tie=True)
                records.append(dict(condition=condition, snr=snr, estimator=estimator,
                                    arm=key, replay=measured, archived=saved[key],
                                    absolute_difference=abs(measured-saved[key]),
                                    replay_prefer_later_tie=later_tie,
                                    later_tie_difference=abs(later_tie-saved[key])))
        print('REPLAYED', condition, snr, flush=True)
    report = dict(cases=3, estimates=len(records), source='vocadito_1, training role',
                  selection='Fixed ID 1, clean/rain10/rain0; no score-based choice',
                  results_sha256=sha(study/'results.json'), verifier_sha256=sha(Path(__file__)),
                  maximum_difference=max(r['absolute_difference'] for r in records),
                  maximum_later_tie_difference=max(r['later_tie_difference'] for r in records),
                  records=records, full_study_replayed=False, listening_done=False)
    (OUT/'noise_subset_replay.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('MAXIMUM_DIFFERENCE', report['maximum_difference'], flush=True)


if __name__ == '__main__':
    main()
