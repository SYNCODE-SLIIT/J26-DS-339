"""Conservative spectral-gate cleanup candidate and a pitch-preserving do-no-harm selector.

Noise profile policy (declared): the quietest 15% of STFT frames. This is not the proposal's
"voice-free reference region" rule; it is a local, documented alternative and is labelled so.
"""
import librosa
import numpy as np
from scipy.ndimage import uniform_filter

from .evidence import SR, normalize

N_FFT, HOP = 1024, 256


def spectral_gate(x, sr=SR, quiet_fraction=0.15, over=1.5, floor=0.10, noise=None):
    """noise: optional voice-free clip (room check); its mean spectrum is the profile. Otherwise the quietest frames are used."""
    S = librosa.stft(x, n_fft=N_FFT, hop_length=HOP)
    mag, phase = np.abs(S), np.angle(S)
    if noise is not None and len(noise) >= N_FFT:
        profile = np.abs(librosa.stft(noise.astype(np.float32), n_fft=N_FFT, hop_length=HOP)).mean(axis=1, keepdims=True)
    else:
        energy = mag.sum(axis=0)
        k = max(3, int(quiet_fraction * mag.shape[1]))
        profile = mag[:, np.argsort(energy)[:k]].mean(axis=1, keepdims=True)
    gain = np.clip((mag - over * profile) / (mag + 1e-9), floor, 1.0)
    gain = uniform_filter(gain, size=(3, 5), mode="nearest")      # smooth the mask in frequency and time
    y = librosa.istft(gain * mag * np.exp(1j * phase), hop_length=HOP, length=len(x))
    return normalize(y)


def choose(original_track, cleaned_track, min_conf_gain=0.01, max_pitch_shift_c=25.0, max_voicing_loss=0.05):
    """Select the cleaned candidate only when PESTO evidence improves AND pitch/voicing are preserved.

    Uses only input/output acoustics (no clean reference, no labels). Returns (use_cleaned, evidence).
    """
    vo, vc = original_track["voiced"], cleaned_track["voiced"]
    n = min(len(vo), len(vc))
    vo, vc = vo[:n], vc[:n]
    both = vo & vc
    conf_o = float(original_track["conf"][:n][vo].mean()) if vo.any() else 0.0
    conf_c = float(cleaned_track["conf"][:n][vc].mean()) if vc.any() else 0.0
    if both.sum() >= 5:
        shift = np.abs(1200 * np.log2(cleaned_track["f0"][:n][both] / original_track["f0"][:n][both]))
        median_shift = float(np.median(shift))
    else:
        median_shift = float("inf")
    voicing_loss = float(max(0.0, vo.mean() - vc.mean()))
    use = (conf_c - conf_o >= min_conf_gain) and median_shift <= max_pitch_shift_c and voicing_loss <= max_voicing_loss
    return bool(use), {"confidence_original": conf_o, "confidence_cleaned": conf_c,
                       "median_pitch_shift_cents": median_shift, "voicing_loss": voicing_loss,
                       "thresholds": {"min_conf_gain": min_conf_gain, "max_pitch_shift_c": max_pitch_shift_c,
                                      "max_voicing_loss": max_voicing_loss}}
