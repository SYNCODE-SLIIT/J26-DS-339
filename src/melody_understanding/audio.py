"""Read-only audio preparation with explicit source-time bookkeeping."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

@dataclass
class PreparedAudio:
    samples: np.ndarray
    sample_rate: int
    source_duration_s: float
    offset_s: float
    metadata: dict

def prepare_audio(path, sample_rate=22050, trim=False):
    if sample_rate < 8000:
        raise ValueError("sample_rate must be at least 8000 Hz")
    samples, original_sr = sf.read(Path(path), dtype="float32", always_2d=True)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("Audio is empty or contains non-finite samples")
    duration = len(samples) / original_sr
    channels = samples.shape[1]
    mono = samples.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    if peak < 1e-5:
        raise ValueError("Audio is silent or too quiet for this baseline")
    mono = librosa.resample(mono, orig_sr=original_sr, target_sr=sample_rate)
    offset = 0.0
    if trim:
        mono, bounds = librosa.effects.trim(mono, top_db=40)
        offset = float(bounds[0] / sample_rate)
    if len(mono) < sample_rate * .1:
        raise ValueError("Prepared audio must contain at least 0.1 seconds")
    gain = .95 / max(float(np.max(np.abs(mono))), 1e-12)
    mono = mono * gain
    return PreparedAudio(mono, sample_rate, duration, offset, {
        "original_sample_rate": original_sr, "original_channels": channels,
        "original_mono_peak": peak, "normalization_gain": gain,
        "trim_enabled": trim, "trim_threshold_db": 40 if trim else None,
        "source_offset_s": offset, "noise_reduction": "not applied",
        "processed_duration_s": len(mono) / sample_rate,
    })
