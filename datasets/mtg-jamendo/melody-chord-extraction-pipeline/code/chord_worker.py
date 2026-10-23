"""Run one LV-Chordia vocabulary in an isolated process."""

from __future__ import annotations

import json
import sys

from chords import save
from common import DATASET


def main() -> None:
    from lv_chordia import chord_recognition

    request = json.load(sys.stdin)
    track = request["track"]
    vocabulary = request["vocabulary"]
    audio_path = DATASET / track["source_mp3"]
    save(track, vocabulary, float(request["duration_sec"]),
         chord_recognition(str(audio_path), chord_dict_name=vocabulary))


if __name__ == "__main__":
    main()
