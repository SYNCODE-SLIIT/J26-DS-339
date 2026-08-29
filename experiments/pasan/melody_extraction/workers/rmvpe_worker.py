"""Standalone worker for environments exposing the common RMVPE Python API."""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
from rmvpe import RMVPE


def main() -> None:
    audio_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    model_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    audio, _ = librosa.load(audio_path, sr=16_000, mono=True)
    kwargs = {"model_path": str(model_path)} if model_path else {}
    model = RMVPE(**kwargs)
    f0 = np.asarray(model.infer_from_audio(audio, thred=0.03), dtype=float)
    voiced = f0 > 0
    np.savez_compressed(
        output_path,
        times_sec=np.arange(len(f0), dtype=float) * 0.01,
        f0_hz=f0,
        confidence=voiced.astype(float),
        voiced=voiced,
    )


if __name__ == "__main__":
    main()

