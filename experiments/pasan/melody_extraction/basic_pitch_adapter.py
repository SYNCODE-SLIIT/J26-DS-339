"""Basic Pitch adapter supporting either direct import or a Python 3.10 worker."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from . import paths
from .errors import ToolUnavailable
from .models import NoteEvent


def _convert_events(events: Iterable[object]) -> list[NoteEvent]:
    notes: list[NoteEvent] = []
    for event in events:
        values = list(event)  # Basic Pitch returns tuple-like note events.
        if len(values) < 4:
            continue
        onset, offset, pitch, amplitude = values[:4]
        notes.append(
            NoteEvent(
                onset_sec=float(onset),
                offset_sec=float(offset),
                midi_pitch=float(pitch),
                confidence=max(0.0, min(1.0, float(amplitude))),
                source="basic_pitch",
            )
        )
    return notes


def transcribe_basic_pitch(
    audio_path: Path,
    *,
    python_executable: Path | None = paths.BASIC_PITCH_PYTHON,
) -> list[NoteEvent]:
    if python_executable and python_executable.resolve() != Path(sys.executable).resolve():
        worker = Path(__file__).with_name("workers") / "basic_pitch_worker.py"
        with tempfile.TemporaryDirectory(prefix="basic-pitch-") as directory:
            output_json = Path(directory) / "notes.json"
            completed = subprocess.run(
                [str(python_executable), str(worker), str(audio_path), str(output_json)],
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    "Basic Pitch worker failed:\n"
                    + (completed.stderr or completed.stdout).strip()
                )
            rows = json.loads(output_json.read_text())
        return [NoteEvent(**row) for row in rows]

    try:
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise ToolUnavailable(
            "Basic Pitch needs a Python <=3.11 environment. Set "
            "MELODY_BASIC_PITCH_PYTHON; see this folder's README."
        ) from exc
    _, _, events = predict(str(audio_path))
    return _convert_events(events)

