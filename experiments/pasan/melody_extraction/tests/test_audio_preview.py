from __future__ import annotations

import numpy as np

from experiments.pasan.melody_extraction.audio_preview import synthesize_notes
from experiments.pasan.melody_extraction.models import NoteEvent


def test_synthesize_notes_preserves_timing_and_is_audible() -> None:
    audio = synthesize_notes(
        [NoteEvent(0.1, 0.5, 69.0, 0.8, "test")],
        sample_rate=8_000,
        duration_sec=1.0,
    )

    assert len(audio) == 8_000
    assert np.max(np.abs(audio[:700])) == 0
    assert np.max(np.abs(audio[800:4_000])) > 0.01
    assert np.max(np.abs(audio[4_100:])) == 0
