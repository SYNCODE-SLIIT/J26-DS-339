"""Run vocal separation and GAME for one song in an isolated process."""

from __future__ import annotations

import json
import sys

from melody import GameExtractor


def main() -> None:
    request = json.load(sys.stdin)
    extractor = GameExtractor()
    if request.get("mode") == "stem_only":
        extractor.stem(request["track"], float(request["duration_sec"]), bpdr_verified=True)
    else:
        extractor.extract(request["track"], float(request["duration_sec"]),
                          keep_stem=bool(request.get("keep_stem", False)))


if __name__ == "__main__":
    main()
