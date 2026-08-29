"""A lightweight monophonic F0 baseline using librosa pYIN."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .audio_utils import load_mono
from .config import PipelineConfig
from .models import PitchTrack


def extract_pyin(audio_path: Path, config: PipelineConfig, *, source: str) -> PitchTrack:
    audio, sr = load_mono(audio_path, config.sample_rate)
    f0, voiced, probabilities = librosa.pyin(
        audio,
        fmin=config.fmin_hz,
        fmax=config.fmax_hz,
        sr=sr,
        hop_length=config.hop_length,
        fill_na=np.nan,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=config.hop_length)
    return PitchTrack(
        times_sec=np.asarray(times, dtype=np.float64),
        f0_hz=np.asarray(f0, dtype=np.float64),
        confidence=np.nan_to_num(np.asarray(probabilities, dtype=np.float64)),
        voiced=np.asarray(voiced, dtype=bool),
        source=source,
    )

