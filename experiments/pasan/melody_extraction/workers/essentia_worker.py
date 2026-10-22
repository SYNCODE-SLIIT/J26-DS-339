"""Standalone worker executed by an optional Essentia environment."""

from __future__ import annotations

import sys
from pathlib import Path

import essentia.standard as es
import numpy as np


def main() -> None:
    audio_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    sample_rate = int(sys.argv[3])
    hop_length = int(sys.argv[4])
    audio = es.EqloudLoader(filename=str(audio_path), sampleRate=sample_rate)()
    pitch, confidence = es.PredominantPitchMelodia(
        frameSize=2048,
        hopSize=hop_length,
    )(audio)
    pitch = np.asarray(pitch, dtype=float)
    confidence = np.clip(np.asarray(confidence, dtype=float), 0, 1)
    np.savez_compressed(
        output_path,
        times_sec=np.arange(len(pitch), dtype=float) * hop_length / sample_rate,
        f0_hz=pitch,
        confidence=confidence,
        voiced=pitch > 0,
    )


if __name__ == "__main__":
    main()

