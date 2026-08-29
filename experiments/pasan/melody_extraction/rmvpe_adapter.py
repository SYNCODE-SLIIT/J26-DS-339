"""RMVPE vocal-F0 adapter using an isolated, API-compatible environment."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import paths
from .errors import ToolUnavailable
from .models import PitchTrack


def extract_rmvpe(
    audio_path: Path,
    *,
    python_executable: Path | None = paths.RMVPE_PYTHON,
    model_path: Path | None = paths.RMVPE_MODEL,
) -> PitchTrack:
    if not python_executable:
        raise ToolUnavailable(
            "Set MELODY_RMVPE_PYTHON to an environment exposing `from rmvpe import RMVPE`"
        )
    worker = Path(__file__).with_name("workers") / "rmvpe_worker.py"
    with tempfile.TemporaryDirectory(prefix="rmvpe-") as directory:
        output_path = Path(directory) / "f0.npz"
        command = [str(python_executable), str(worker), str(audio_path), str(output_path)]
        if model_path:
            command.append(str(model_path))
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError("RMVPE worker failed:\n" + (completed.stderr or completed.stdout).strip())
        arrays = np.load(output_path)
        return PitchTrack(
            times_sec=arrays["times_sec"],
            f0_hz=arrays["f0_hz"],
            confidence=arrays["confidence"],
            voiced=arrays["voiced"],
            source="rmvpe",
        )
