"""Canonical, tool-independent melody data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class NoteEvent:
    onset_sec: float
    offset_sec: float
    midi_pitch: float
    confidence: float = 1.0
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.onset_sec < 0 or self.offset_sec <= self.onset_sec:
            raise ValueError("A note must have a non-negative onset and a later offset")
        if not 0 <= self.midi_pitch <= 127:
            raise ValueError("MIDI pitch must be between 0 and 127")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass(slots=True)
class PitchTrack:
    times_sec: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray
    voiced: np.ndarray
    source: str

    def __post_init__(self) -> None:
        lengths = {
            len(self.times_sec),
            len(self.f0_hz),
            len(self.confidence),
            len(self.voiced),
        }
        if len(lengths) != 1:
            raise ValueError("Pitch-track arrays must have the same length")


@dataclass(frozen=True, slots=True)
class KeyEstimate:
    tonic: str
    mode: str
    strength: float
    method: str
    alternative: str | None = None
    margin: float | None = None

    @property
    def label(self) -> str:
        return f"{self.tonic} {self.mode}"


@dataclass(slots=True)
class ExtractionResult:
    audio_path: Path
    pipeline: str
    notes: list[NoteEvent] = field(default_factory=list)
    pitch_track: PitchTrack | None = None
    key: KeyEstimate | None = None
    runtime_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

