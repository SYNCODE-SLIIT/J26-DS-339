"""Render extracted note events as small listenable MP3/WAV previews."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .models import NoteEvent


PREVIEW_SAMPLE_RATE = 22_050


def synthesize_notes(
    notes: list[NoteEvent],
    *,
    sample_rate: int = PREVIEW_SAMPLE_RATE,
    duration_sec: float | None = None,
) -> np.ndarray:
    """Create an audible sine/second-harmonic rendering of note events."""
    inferred_duration = max((note.offset_sec for note in notes), default=0.0) + 0.25
    duration = max(float(duration_sec or 0.0), inferred_duration, 0.25)
    audio = np.zeros(max(1, int(np.ceil(duration * sample_rate))), dtype=np.float32)

    for note in notes:
        start = max(0, int(round(note.onset_sec * sample_rate)))
        end = min(len(audio), int(round(note.offset_sec * sample_rate)))
        if end <= start:
            continue
        frame_count = end - start
        time = np.arange(frame_count, dtype=np.float32) / sample_rate
        frequency = 440.0 * 2.0 ** ((note.midi_pitch - 69.0) / 12.0)
        tone = np.sin(2.0 * np.pi * frequency * time)
        tone += 0.2 * np.sin(4.0 * np.pi * frequency * time)

        attack = min(frame_count // 2, max(1, int(0.01 * sample_rate)))
        release = min(frame_count // 2, max(1, int(0.04 * sample_rate)))
        envelope = np.ones(frame_count, dtype=np.float32)
        envelope[:attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
        envelope[-release:] = np.linspace(1.0, 0.0, release, endpoint=True)
        amplitude = 0.12 * (0.35 + 0.65 * note.confidence)
        audio[start:end] += (amplitude * envelope * tone).astype(np.float32)

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.95:
        audio *= 0.95 / peak
    return audio


def _write_compressed(audio: np.ndarray, sample_rate: int, output_stem: Path) -> Path:
    """Prefer MP3 through ffmpeg, with a portable WAV fallback."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        output_path = output_stem.with_suffix(".wav")
        sf.write(output_path, audio, sample_rate, subtype="PCM_16")
        return output_path

    output_path = output_stem.with_suffix(".mp3")
    with tempfile.TemporaryDirectory(prefix="melody-preview-") as directory:
        temporary_wav = Path(directory) / "preview.wav"
        sf.write(temporary_wav, audio, sample_rate, subtype="PCM_16")
        completed = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(temporary_wav),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "96k",
                str(output_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError("ffmpeg preview encoding failed: " + completed.stderr.strip())
    return output_path


def write_audio_previews(
    notes: list[NoteEvent],
    audio_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Write standalone synthesized notes and an alignment-check overlay."""
    original: np.ndarray | None = None
    if audio_path.exists():
        loaded, _ = librosa.load(audio_path, sr=PREVIEW_SAMPLE_RATE, mono=True)
        original = np.asarray(loaded, dtype=np.float32)

    original_duration = len(original) / PREVIEW_SAMPLE_RATE if original is not None else None
    melody = synthesize_notes(notes, duration_sec=original_duration)
    artifacts = {
        "melody_audio": _write_compressed(
            melody,
            PREVIEW_SAMPLE_RATE,
            output_dir / "melody_audio",
        )
    }

    if original is not None:
        if len(original) < len(melody):
            original = np.pad(original, (0, len(melody) - len(original)))
        elif len(melody) < len(original):
            melody = np.pad(melody, (0, len(original) - len(melody)))
        overlay = 0.42 * original + 0.85 * melody
        peak = float(np.max(np.abs(overlay))) if len(overlay) else 0.0
        if peak > 0.98:
            overlay *= 0.98 / peak
        artifacts["melody_overlay"] = _write_compressed(
            overlay.astype(np.float32),
            PREVIEW_SAMPLE_RATE,
            output_dir / "melody_overlay",
        )
    return artifacts
