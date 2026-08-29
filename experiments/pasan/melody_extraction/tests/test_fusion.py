from __future__ import annotations

import numpy as np

from experiments.pasan.melody_extraction.fusion import fuse_pitch_tracks
from experiments.pasan.melody_extraction.models import PitchTrack


def _track(f0: list[float], confidence: list[float], source: str) -> PitchTrack:
    values = np.asarray(f0, dtype=float)
    return PitchTrack(
        times_sec=np.arange(len(values), dtype=float),
        f0_hz=values,
        confidence=np.asarray(confidence, dtype=float),
        voiced=values > 0,
        source=source,
    )


def test_fusion_prefers_confident_vocal_and_uses_fallback_elsewhere() -> None:
    vocal = _track([440, 440, 0, 660], [0.9, 0.2, 0, 0.8], "vocal")
    fallback = _track([220, 220, 330, 330], [0.5, 0.5, 0.5, 0.5], "fallback")

    result = fuse_pitch_tracks(vocal, fallback, primary_confidence=0.45)

    assert result.f0_hz.tolist() == [440, 220, 330, 660]
    assert result.source == "vocal_primary_fullmix_fallback"

