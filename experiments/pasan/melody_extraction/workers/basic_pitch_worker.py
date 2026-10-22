"""Standalone worker executed by the optional Basic Pitch environment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from basic_pitch.inference import predict


def main() -> None:
    audio_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    _, _, events = predict(str(audio_path))
    rows = []
    for event in events:
        values = list(event)
        if len(values) < 4:
            continue
        onset, offset, pitch, confidence = values[:4]
        rows.append(
            {
                "onset_sec": float(onset),
                "offset_sec": float(offset),
                "midi_pitch": float(pitch),
                "confidence": max(0.0, min(1.0, float(confidence))),
                "source": "basic_pitch",
            }
        )
    rows.sort(key=lambda row: (row["onset_sec"], row["midi_pitch"]))
    output_path.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
