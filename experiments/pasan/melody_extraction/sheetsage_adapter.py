"""Full-mix melody transcription with SheetSage (harmony is disabled)."""

from __future__ import annotations

import io
import os
from pathlib import Path

import pretty_midi

from .errors import ToolUnavailable
from .models import NoteEvent
from .paths import MODEL_CACHE_DIR


def transcribe_sheetsage(audio_path: Path) -> list[NoteEvent]:
    os.environ.setdefault("SHEETSAGE_CACHE_DIR", str(MODEL_CACHE_DIR / "sheetsage"))
    try:
        from sheetsage.align import create_beat_to_time_fn
        from sheetsage.infer import sheetsage
    except ImportError as exc:
        raise ToolUnavailable("SheetSage is missing; run `uv sync` from the repository root") from exc

    try:
        lead_sheet, segment_beats, segment_times = sheetsage(
            str(audio_path),
            use_jukebox=False,
            detect_melody=True,
            detect_harmony=False,
        )
    except Exception as exc:
        if "download failed" in str(exc).lower():
            raise ToolUnavailable(
                "SheetSage is installed, but its public checkpoint host rejected the download. "
                "Place the official assets in MELODY_MODEL_CACHE_DIR/sheetsage or retry later."
            ) from exc
        raise
    midi_bytes = lead_sheet.as_midi(
        pulse_to_time_fn=create_beat_to_time_fn(segment_beats, segment_times)
    )
    midi = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    melodic_instruments = [instrument for instrument in midi.instruments if not instrument.is_drum]
    if not melodic_instruments:
        return []

    # SheetSage MIDI is click, harmony, melody. Harmony is empty because this
    # melody-only adapter calls infer with detect_harmony=False.
    melody_instrument = melodic_instruments[-1]
    return [
        NoteEvent(
            onset_sec=float(note.start),
            offset_sec=float(note.end),
            midi_pitch=float(note.pitch),
            confidence=1.0,
            source="sheetsage",
        )
        for note in sorted(melody_instrument.notes, key=lambda value: (value.start, value.pitch))
    ]
