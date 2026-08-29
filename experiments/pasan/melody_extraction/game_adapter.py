"""Adapter for the official openvpi/GAME singing transcription repository."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pretty_midi

from . import paths
from .errors import ToolUnavailable
from .models import NoteEvent


def transcribe_game(
    audio_path: Path,
    *,
    game_root: Path | None = paths.GAME_ROOT,
    model_path: Path | None = paths.GAME_MODEL,
    python_executable: Path | None = paths.GAME_PYTHON,
) -> list[NoteEvent]:
    if not game_root or not (game_root / "infer.py").exists():
        raise ToolUnavailable("Set MELODY_GAME_ROOT to a clone of openvpi/GAME")
    if not model_path or not model_path.exists():
        raise ToolUnavailable("Set MELODY_GAME_MODEL to a downloaded GAME checkpoint")

    executable = python_executable or Path(sys.executable)
    with tempfile.TemporaryDirectory(prefix="game-") as directory:
        temporary_audio = Path(directory) / ("input" + audio_path.suffix.lower())
        shutil.copy2(audio_path, temporary_audio)
        completed = subprocess.run(
            [
                str(executable),
                str(game_root / "infer.py"),
                "extract",
                str(temporary_audio),
                "-m",
                str(model_path),
                "--output-formats",
                "mid",
            ],
            cwd=game_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError("GAME failed:\n" + (completed.stderr or completed.stdout).strip())
        midi_candidates = list(Path(directory).glob("*.mid")) + list(Path(directory).glob("*.midi"))
        if not midi_candidates:
            raise RuntimeError("GAME completed but did not produce a MIDI file")
        midi = pretty_midi.PrettyMIDI(str(midi_candidates[0]))

    notes: list[NoteEvent] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append(
                NoteEvent(
                    onset_sec=float(note.start),
                    offset_sec=float(note.end),
                    midi_pitch=float(note.pitch),
                    confidence=float(note.velocity / 127.0),
                    source="game",
                )
            )
    return sorted(notes, key=lambda note: (note.onset_sec, note.midi_pitch))

