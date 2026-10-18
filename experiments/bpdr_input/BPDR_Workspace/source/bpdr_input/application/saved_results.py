"""Verified input/output examples from completed runs; no fitting or inference."""
from bpdr_input import COMPONENT_ROOT
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

COMPONENT = COMPONENT_ROOT


def read(path):
    return Path(path).read_text(encoding='utf-8')


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_candidate(candidate):
    path = Path(candidate['path'])
    if checksum(path) != candidate['sha256']:
        raise ValueError('Saved output checksum mismatch')
    return path


def pitch_example():
    measurement = json.loads(read(COMPONENT / 'runs/presentation_selected_example_measurement.json'))
    records = json.loads(read(COMPONENT / 'runs/presentation_runtime_checks_mixed.json'))
    run = Path(next(r['path'] for r in records if r['label'] == 'mixed_synthetic_detuned_audio_only'))
    report = json.loads(read(run / 'model_report.json'))
    if report['outcome'] != 'VERIFIED_FIXTURE_CANDIDATE':
        raise ValueError('Pitch output was not verified')
    if measurement['checkpoint_sha256'] != report['model_sha256'] or checksum(report['model_path']) != report['model_sha256']:
        raise ValueError('Pitch measurement/model mismatch')
    source = Path(report['source'])
    if checksum(source) != checksum(run / 'original.wav'):
        raise ValueError('Saved pitch input changed')
    return dict(kind='Generated phrase with a known injected local detuning',
        input=source, output=checked_candidate(report['candidate']), reference=source.parent / 'A.wav',
        before_cents=measurement['before_median_cents'], after_cents=measurement['after_median_cents'],
        scope='One controlled generated example; not original-dataset or population accuracy',
        accepted_downstream_unchanged=report['handoff']['status'] == 'ACCEPT_UNCHANGED')


def paired_snr_gain(baseline, before, after):
    arrays = [np.asarray(a, dtype=np.float64) for a in (baseline, before, after)]
    if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 1:
        raise ValueError('Paired audio must have the same mono sample grid')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Non-finite paired samples')
    baseline, before, after = arrays
    return float(10 * np.log10((np.mean((before-baseline)**2)+1e-20) /
                               (np.mean((after-baseline)**2)+1e-20)))


def noise_example():
    saved = json.loads(read(COMPONENT / 'runs/humming_noise_live_check.json'))
    report = saved['report']
    audit = report['noise']['audit']
    if report['outcome'] != 'NOISE_PREVIEW_PASSED' or not audit['passes']:
        raise ValueError('Noise output was not verified')
    source = Path(report['source'])
    if checksum(source) != checksum(Path(saved['path']) / 'original.wav'):
        raise ValueError('Saved noise input changed')
    output = checked_candidate(report['noise_candidate'])
    reference = Path(saved['source_study_case']['directory']) / 'baseline.wav'
    waves = [sf.read(p, dtype='float64') for p in (reference, source, output)]
    if len({sr for _, sr in waves}) != 1:
        raise ValueError('Paired audio sample rates differ')
    gain = paired_snr_gain(*(a for a, _ in waves))
    return dict(kind='Real HumTrans recording mixed with recorded rain', input=source,
        output=output, reference=reference, snr_gain_db=gain,
        reference_attenuation_db=audit['reference_attenuation_db'],
        median_pitch_change_cents=audit['median_pitch_change_cents'],
        scope='Known added-noise mixture; not naturally co-recorded noisy novice validation',
        accepted_downstream_unchanged=report['handoff']['status'] == 'ACCEPT_UNCHANGED')
