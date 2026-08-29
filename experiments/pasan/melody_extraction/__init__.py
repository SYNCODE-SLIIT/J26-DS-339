"""Modular melody-extraction experiments for the MTG-Jamendo audio."""

from .config import PipelineConfig
from .models import ExtractionResult, KeyEstimate, NoteEvent, PitchTrack

__all__ = [
    "ExtractionResult",
    "KeyEstimate",
    "NoteEvent",
    "PipelineConfig",
    "PitchTrack",
]
