"""Small filesystem and validation helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import librosa


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def discover_audio_files(audio_dir: Path, limit: int) -> list[Path]:
    if limit < 1:
        raise ValueError("NUMBER_OF_FILES must be at least 1")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")
    files = sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in AUDIO_EXTENSIONS
        and not path.name.startswith("._")
    )
    return files[:limit]


def extract_track_id(audio_path: Path) -> str:
    match = re.match(r"^(\d+)(?:\.low)?$", audio_path.stem)
    if not match:
        raise ValueError(f"Cannot extract numeric track ID from {audio_path.name}")
    return match.group(1)


def audio_duration_seconds(audio_path: Path) -> float:
    return float(librosa.get_duration(path=audio_path))


def consecutive_duplicate_count(events: Iterable[dict]) -> int:
    labels = [event["raw_chord"] for event in events]
    return sum(left == right for left, right in zip(labels, labels[1:]))

