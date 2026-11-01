"""Experimental local constant-step renderer, separate from the live demo.

Use only an already permitted constant correction and unchanged support. The
same post-render and encoded-WAV audits still determine whether output survives.
No formant or perceptual-quality claim is made for phase-vocoder resynthesis.
"""
from dataclasses import asdict, dataclass
import importlib.metadata

import numpy as np

from bpdr_input.repair.interventions import envelope
from bpdr_input.repair.repair_controller import check_source


@dataclass(frozen=True)
class PhaseVocoderPolicy:
    version: str = 'phase-vocoder-constant-step-1'
    librosa_version: str = '1.0.0'
    soxr_version: str = '1.1.0'
    n_fft: int = 2048
    hop_length: int = 256
    ramp_seconds: float = .08
    max_cents: float = 30.
    min_support_seconds: float = .86


def render_candidate(audio, sr, times, correction_cents, intervals, policy=PhaseVocoderPolicy()):
    check_source(audio, sr)
    times, correction = np.asarray(times), np.asarray(correction_cents)
    expected = np.arange(len(audio)//441+1)*.02
    if (times.shape != expected.shape or not np.allclose(times, expected, atol=1e-12, rtol=0)
            or correction.shape != times.shape or not np.isfinite(correction).all()
            or np.max(abs(correction)) > policy.max_cents):
        raise ValueError('Invalid phase-vocoder grid or over-limit correction')
    sample_times = np.arange(len(audio))/sr
    covered = np.zeros(len(times), bool)
    all_blend = np.zeros(len(audio))
    supports = []
    previous = 0.
    for start, end in intervals:
        if (not np.isfinite(start+end) or start < previous or not 0 <= start < end <= len(audio)/sr
                or end-start < policy.min_support_seconds-1e-12):
            raise ValueError('Invalid or too-short phase-vocoder support')
        previous = end
        mask = (times >= start) & (times < end)
        values = correction[mask]
        if not len(values) or not np.all(values == values[0]):
            raise ValueError('Phase-vocoder condition requires a constant predicted plateau')
        covered |= mask
        blend = envelope(sample_times, (start, end), policy.ramp_seconds)
        if np.any((all_blend > 0) & (blend > 0)):
            raise ValueError('Overlapping local envelopes')
        all_blend += blend
        supports.append((float(values[0]), blend))
    if np.any((correction != 0) & ~covered):
        raise ValueError('Correction outside permitted support')
    diagnostics = dict(policy=asdict(policy), production_eligible=False,
                       scope='experimental_renderer_only', identity=not bool(np.any(correction)),
                       outside_samples_exact=True, same_sample_count=True,
                       perceptual_quality='not assessed; no formant-preservation claim')
    if not np.any(correction):
        return audio.copy(), diagnostics
    for name, version in [('librosa', policy.librosa_version), ('soxr', policy.soxr_version)]:
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Unpinned {name} version')
    import librosa
    result = audio.astype(np.float64).copy()
    for cents, blend in supports:
        if cents == 0:
            continue
        full = librosa.effects.pitch_shift(audio, sr=sr, n_steps=cents/100.,
                   bins_per_octave=12, res_type='soxr_hq', scale=False,
                   n_fft=policy.n_fft, hop_length=policy.hop_length)
        if full.shape != audio.shape or not np.isfinite(full).all():
            raise ValueError('Pitch-shift returned a different length or nonfinite audio')
        result += (full-audio)*blend
    result = result.astype(np.float32)
    if not np.isfinite(result).all() or np.max(abs(result)) >= 1.:
        raise ValueError('Nonfinite/clipped phase-vocoder output; no normalization')
    if not np.array_equal(result[all_blend == 0], audio[all_blend == 0]):
        raise ValueError('Samples outside support changed')
    return result, diagnostics
