"""Run vocal separation and GAME for one song in an isolated process."""

from __future__ import annotations

import json
import sys

from melody import GameExtractor


def main() -> None:
    request = json.load(sys.stdin)
    GameExtractor().extract(request["track"], float(request["duration_sec"]))


if __name__ == "__main__":
    main()
