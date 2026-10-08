"""Configuration for the isolated chord-refinement data pipeline."""

from __future__ import annotations

import os
from pathlib import Path


COMPONENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = COMPONENT_DIR.parent.parent


def _configured_path(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()


AUDIO_DIR = _configured_path(
    "CHORD_AUDIO_DIR",
    REPOSITORY_ROOT / "datasets" / "mtg-jamendo" / "vocal-tagged-audio",
)
OUTPUT_DIR = _configured_path("CHORD_OUTPUT_DIR", COMPONENT_DIR / "outputs")
CHORD_VOCABULARY = os.environ.get("CHORD_VOCABULARY", "submission")
NUMBER_OF_FILES = int(os.environ.get("CHORD_NUMBER_OF_FILES", "1"))

SUPPORTED_VOCABULARIES = {"submission", "ismir2017", "full", "extended"}

