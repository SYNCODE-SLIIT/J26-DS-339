"""Essentia Melodia adapter supporting an isolated Python environment."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import paths
from .errors import ToolUnavailable
from .models import PitchTrack


def _extract_in_process(audio_path: Path, *, sample_rate: int, hop_length: int) -> PitchTrack:
    try:
        import essentia.standard as es
    except ImportError as exc:
        raise ToolUnavailable(
            "Essentia is unavailable in the root Python 3.12 environment. Set "
            "MELODY_ESSENTIA_PYTHON; see this folder's README."
        ) from exc
    audio = es.EqloudLoader(filename=str(audio_path), sampleRate=sample_rate)()
    pitch, confidence = es.PredominantPitchMelodia(
        frameSize=2048,
        hopSize=hop_length,
    )(audio)
    times = np.arange(len(pitch), dtype=float) * hop_length / sample_rate
    pitch = np.asarray(pitch, dtype=float)
    confidence = np.clip(np.asarray(confidence, dtype=float), 0, 1)
    return PitchTrack(times, pitch, confidence, pitch > 0, "essentia_melodia")


def extract_melodia(
    audio_path: Path,
    *,
    sample_rate: int = 44_100,
    hop_length: int = 128,
    python_executable: Path | None = paths.ESSENTIA_PYTHON,
) -> PitchTrack:
    if not python_executable or python_executable.resolve() == Path(sys.executable).resolve():
        return _extract_in_process(
            audio_path,
            sample_rate=sample_rate,
            hop_length=hop_length,
        )

    worker = Path(__file__).with_name("workers") / "essentia_worker.py"
    with tempfile.TemporaryDirectory(prefix="essentia-") as directory:
        output_path = Path(directory) / "f0.npz"
        completed = subprocess.run(
            [
                str(python_executable),
                str(worker),
                str(audio_path),
                str(output_path),
                str(sample_rate),
                str(hop_length),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                "Essentia worker failed:\n" + (completed.stderr or completed.stdout).strip()
            )
        arrays = np.load(output_path)
        return PitchTrack(
            arrays["times_sec"],
            arrays["f0_hz"],
            arrays["confidence"],
            arrays["voiced"],
            "essentia_melodia",
        )

