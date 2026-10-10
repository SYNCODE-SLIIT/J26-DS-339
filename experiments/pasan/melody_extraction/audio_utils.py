"""Audio discovery and loading helpers."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def discover_audio_files(
    root: Path,
    *,
    pattern: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    if root.is_file():
        files = [root]
    else:
        candidates = root.rglob(pattern or "*")
        files = sorted(
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
    return files[:limit] if limit is not None else files


def load_mono(audio_path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32), int(sr)


def relative_song_path(audio_path: Path, input_root: Path) -> Path:
    try:
        relative = audio_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        relative = Path(audio_path.name)
    return relative.with_suffix("")

