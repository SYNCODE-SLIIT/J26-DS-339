from __future__ import annotations

import numpy as np
import pytest

from experiments.pasan.melody_extraction.models import PitchTrack
from experiments.pasan.melody_extraction.pitch_utils import contour_to_notes


def test_contour_to_notes_splits_pitch_changes_and_silence() -> None:
    times = np.arange(8) * 0.1
    track = PitchTrack(
        times_sec=times,
        f0_hz=np.array([440, 440, 440, 0, 493.88, 493.88, 493.88, 0], dtype=float),
        confidence=np.array([0.9, 0.9, 0.9, 0, 0.8, 0.8, 0.8, 0], dtype=float),
        voiced=np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=bool),
        source="test",
    )

    notes = contour_to_notes(track, minimum_note_ms=50)

    assert len(notes) == 2
    assert round(notes[0].midi_pitch) == 69
    assert round(notes[1].midi_pitch) == 71
    assert notes[0].onset_sec == 0
    assert notes[0].offset_sec == pytest.approx(0.3)
