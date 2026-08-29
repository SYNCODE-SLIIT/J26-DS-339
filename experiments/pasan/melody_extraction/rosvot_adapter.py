"""Adapter for the official ROSVOT singing-voice transcription repository."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pretty_midi

from . import paths
from .errors import ToolUnavailable
from .models import NoteEvent


def transcribe_rosvot(
    audio_path: Path,
    *,
    rosvot_root: Path | None = paths.ROSVOT_ROOT,
    python_executable: Path | None = paths.ROSVOT_PYTHON,
) -> list[NoteEvent]:
    script = rosvot_root / "inference" / "rosvot.py" if rosvot_root else None
    if not script or not script.exists():
        raise ToolUnavailable("Set MELODY_ROSVOT_ROOT to a clone of RickyL-2000/ROSVOT")
    executable = python_executable or Path(sys.executable)

    with tempfile.TemporaryDirectory(prefix="rosvot-") as directory:
        output_dir = Path(directory) / "output"
        completed = subprocess.run(
            [str(executable), str(script), "-o", str(output_dir), "-p", str(audio_path)],
            cwd=rosvot_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError("ROSVOT failed:\n" + (completed.stderr or completed.stdout).strip())
        midi_files = sorted(output_dir.rglob("*.mid")) + sorted(output_dir.rglob("*.midi"))
        if not midi_files:
            raise RuntimeError("ROSVOT completed but did not produce a MIDI file")
        midi = pretty_midi.PrettyMIDI(str(midi_files[0]))

    notes = [
        NoteEvent(
            onset_sec=float(note.start),
            offset_sec=float(note.end),
            midi_pitch=float(note.pitch),
            confidence=float(note.velocity / 127.0),
            source="rosvot",
        )
        for instrument in midi.instruments
        for note in instrument.notes
    ]
    return sorted(notes, key=lambda note: (note.onset_sec, note.midi_pitch))

