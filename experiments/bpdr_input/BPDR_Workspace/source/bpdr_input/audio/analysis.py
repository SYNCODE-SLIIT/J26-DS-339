"""Measured acoustics only; no intended-note inference or musical correction."""
from functools import lru_cache
from importlib.metadata import version
import os
import hashlib
from importlib.util import find_spec
from pathlib import Path
import numpy as np


@lru_cache(maxsize=4)
def _pitch_model(cache_dir: str):
    # Keep downloaded weights out of the shared repository and raw dataset.
    os.environ['TORCH_HOME'] = cache_dir
    import pesto
    model = pesto.load_model('mir-1k_g7', step_size=20., sampling_rate=22050)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def pesto_pitch(audio, sr, cache_dir):
    import torch
    model = _pitch_model(str(Path(cache_dir).resolve()))
    with torch.inference_mode():
        f0, confidence, amplitude = model(torch.as_tensor(audio, dtype=torch.float32), sr,
                                          convert_to_freq=True, return_activations=False)
    values = [v.detach().cpu().numpy().reshape(-1) for v in (f0, confidence)]
    # PESTO's actual hop is recorded from the model, not assumed from 20 ms.
    hop_seconds = round(model.hop_size * sr / 1000.) / sr
    return np.arange(len(values[0])) * hop_seconds, *values


def extract(audio, sr, cache_dir, pitch_provider=pesto_pitch):
    import librosa
    hop = round(sr * .020)
    spectrum = np.abs(librosa.stft(audio, n_fft=1024, hop_length=hop)) ** 2
    mel = librosa.feature.melspectrogram(S=spectrum, sr=sr, n_mels=64)
    logmel = librosa.power_to_db(mel, ref=1., top_db=80).T
    times = np.arange(len(logmel), dtype=np.float64) * hop / sr
    native_t, native_f, native_c = pitch_provider(audio, sr, cache_dir)
    if not (len(native_t) == len(native_f) == len(native_c)) or len(native_t) == 0:
        raise ValueError('Pitch analyzer returned inconsistent arrays')
    if not all(np.isfinite(a).all() for a in (native_t, native_f, native_c)):
        raise ValueError('Pitch analyzer returned non-finite data')
    if np.any(np.diff(native_t) <= 0) or native_t[0] != 0 or np.any(native_f < 0) or np.any((native_c < 0) | (native_c > 1)):
        raise ValueError('Pitch analyzer returned invalid units, timestamps or confidence')
    # Nearest native frame avoids interpolating across unvoiced gaps.
    idx = np.searchsorted(native_t, times)
    idx = np.clip(idx, 0, len(native_t)-1)
    previous = np.maximum(idx-1, 0)
    idx = np.where(abs(native_t[previous]-times) < abs(native_t[idx]-times), previous, idx)
    confidence = native_c[idx]
    voiced = (confidence >= .8) & (native_f[idx] > 0)
    f0 = np.where(voiced, native_f[idx], np.nan)
    cents = np.full(len(times), np.nan)
    cents[voiced] = 1200 * np.log2(f0[voiced] / 440.)
    delta = np.zeros(len(times))
    valid_delta = np.r_[False, voiced[1:] & voiced[:-1]]
    delta[valid_delta] = np.diff(cents)[valid_delta[1:]]
    rms = librosa.feature.rms(S=np.sqrt(spectrum), frame_length=1024)[0]
    logenergy = 20 * np.log10(np.maximum(rms, 1e-8))
    flux = np.r_[0., np.maximum(np.diff(logmel, axis=0), 0).mean(axis=1)]
    onset = flux / max(float(flux.max()), 1e-8)
    transition = np.r_[0., voiced[1:] != voiced[:-1]].astype(float)
    boundary = np.maximum.reduce([onset, np.minimum(abs(delta)/100, 1), transition])
    # 70 frame-only features; boundary makes 71. Unvoiced pitch uses masked zeros.
    frame = np.column_stack([logmel, np.nan_to_num(cents), confidence, voiced,
                             delta, logenergy, onset]).astype(np.float32)
    features = dict(times_seconds=times, f0_hz=f0, confidence=confidence,
                    voiced_mask=voiced, pitch_change_cents=delta,
                    pitch_change_valid=valid_delta, log_energy_db=logenergy,
                    onset_score=onset, provisional_boundary_score=boundary,
                    log_mel_db=logmel, frame_features_70=frame,
                    boundary_features_71=np.column_stack([frame, boundary]).astype(np.float32),
                    native_pitch_times_seconds=native_t, native_pitch_hz=native_f,
                    native_pitch_confidence=native_c)
    if frame.shape != (len(times), 70) or not np.isfinite(frame).all():
        raise ValueError('Invalid aligned feature matrix')
    weight_path = Path(find_spec('pesto').origin).parent/'weights'/'mir-1k_g7.ckpt'
    with weight_path.open('rb') as stream:
        weight_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    return features, dict(analyzer='PESTO mir-1k_g7', pesto_version=version('pesto-pitch'),
                         pitch_provider='frozen PESTO' if pitch_provider is pesto_pitch else 'injected test provider; not PESTO results',
                         checkpoint_sha256=weight_hash, checkpoint_path=str(weight_path),
                         librosa_version=version('librosa'), frozen=True,
                         sample_rate_hz=sr, hop_samples=hop, window_samples=1024,
                         time_origin_seconds=0, window_centered=True,
                         pitch_alignment='nearest native frame; no interpolation',
                         voiced_confidence_threshold=.8, pitch_units='Hz / cents relative to A4=440 Hz',
                         unvoiced_f0='NaN; zero only in masked model feature',
                         boundary_status='provisional heuristic; not note labels',
                         feature_order=['64 log-mel dB', 'pitch cents', 'confidence', 'voiced',
                                        'pitch change cents', 'log energy dB', 'onset score',
                                        'provisional boundary (71st only)'])
