"""Segment-level fusion of a preferred vocal contour and a fallback melody."""

from __future__ import annotations

import numpy as np

from .models import PitchTrack


def fuse_pitch_tracks(
    primary: PitchTrack,
    fallback: PitchTrack,
    *,
    primary_confidence: float = 0.01,
) -> PitchTrack:
    """Use vocal F0 when confident, otherwise retain full-mix melody F0."""
    target_times = fallback.times_sec
    if len(primary.times_sec) == 0:
        return fallback

    primary_f0 = np.interp(
        target_times,
        primary.times_sec,
        np.nan_to_num(primary.f0_hz),
        left=0.0,
        right=0.0,
    )
    primary_conf = np.interp(
        target_times,
        primary.times_sec,
        primary.confidence,
        left=0.0,
        right=0.0,
    )
    primary_voiced = primary_f0 > 0
    use_primary = primary_voiced & (primary_conf >= primary_confidence)

    f0 = np.where(use_primary, primary_f0, fallback.f0_hz)
    confidence = np.where(use_primary, primary_conf, fallback.confidence)
    voiced = np.where(use_primary, True, fallback.voiced).astype(bool)
    return PitchTrack(
        times_sec=target_times.copy(),
        f0_hz=np.asarray(f0, dtype=float),
        confidence=np.asarray(confidence, dtype=float),
        voiced=voiced,
        source="vocal_primary_fullmix_fallback",
    )
