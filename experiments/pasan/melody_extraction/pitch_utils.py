"""Convert frame-level F0 contours into note events."""

from __future__ import annotations

import numpy as np

from .models import NoteEvent, PitchTrack


def hz_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    midi = np.full_like(f0_hz, np.nan, dtype=float)
    valid = np.isfinite(f0_hz) & (f0_hz > 0)
    midi[valid] = 69.0 + 12.0 * np.log2(f0_hz[valid] / 440.0)
    return midi


def contour_to_notes(
    track: PitchTrack,
    *,
    minimum_note_ms: float = 60.0,
    minimum_confidence: float = 0.25,
    semitone_tolerance: float = 0.55,
) -> list[NoteEvent]:
    """Segment a monophonic F0 contour with simple pitch hysteresis."""
    if len(track.times_sec) == 0:
        return []

    midi = hz_to_midi(track.f0_hz)
    valid = (
        track.voiced.astype(bool)
        & np.isfinite(midi)
        & (track.confidence >= minimum_confidence)
    )
    frame_step = (
        float(np.median(np.diff(track.times_sec)))
        if len(track.times_sec) > 1
        else minimum_note_ms / 1_000.0
    )
    minimum_duration = minimum_note_ms / 1_000.0
    notes: list[NoteEvent] = []
    start: int | None = None
    values: list[float] = []

    def finish(end_index: int) -> None:
        nonlocal start, values
        if start is None or not values:
            start, values = None, []
            return
        onset = float(track.times_sec[start])
        offset = float(track.times_sec[end_index] + frame_step)
        if offset - onset >= minimum_duration:
            segment = slice(start, end_index + 1)
            confidence = float(np.clip(np.nanmean(track.confidence[segment]), 0, 1))
            notes.append(
                NoteEvent(
                    onset_sec=onset,
                    offset_sec=offset,
                    midi_pitch=float(np.nanmedian(values)),
                    confidence=confidence,
                    source=track.source,
                )
            )
        start, values = None, []

    for index, value in enumerate(midi):
        if not valid[index]:
            if start is not None:
                finish(index - 1)
            continue
        if start is None:
            start = index
            values = [float(value)]
            continue

        center = float(np.median(values))
        if abs(float(value) - center) <= semitone_tolerance:
            values.append(float(value))
        else:
            finish(index - 1)
            start = index
            values = [float(value)]

    if start is not None:
        finish(len(midi) - 1)
    return notes

