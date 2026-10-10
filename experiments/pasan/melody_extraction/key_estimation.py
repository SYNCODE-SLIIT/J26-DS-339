"""Estimate a global musical key from the original full mix."""

from __future__ import annotations

import librosa
import numpy as np

from .models import KeyEstimate


PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def estimate_key_librosa(audio: np.ndarray, sample_rate: int) -> KeyEstimate:
    """Krumhansl-Schmuckler key estimate using a robust median CQT chroma."""
    if len(audio) == 0 or float(np.max(np.abs(audio))) < 1e-7:
        return KeyEstimate("C", "major", 0.0, "librosa_krumhansl", margin=0.0)

    harmonic = librosa.effects.harmonic(audio)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sample_rate)
    weights = np.nanmedian(chroma, axis=1)
    candidates: list[tuple[float, str, str]] = []
    for tonic, name in enumerate(PITCH_CLASSES):
        candidates.append((_correlation(weights, np.roll(MAJOR_PROFILE, tonic)), name, "major"))
        candidates.append((_correlation(weights, np.roll(MINOR_PROFILE, tonic)), name, "minor"))
    candidates.sort(reverse=True)
    best, second = candidates[0], candidates[1]
    return KeyEstimate(
        tonic=best[1],
        mode=best[2],
        strength=best[0],
        method="librosa_krumhansl",
        alternative=f"{second[1]} {second[2]}",
        margin=best[0] - second[0],
    )

