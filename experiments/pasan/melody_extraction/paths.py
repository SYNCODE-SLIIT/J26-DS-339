"""All filesystem defaults and environment-variable overrides live here."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PASAN_DIR = PACKAGE_DIR.parent
REPOSITORY_ROOT = PASAN_DIR.parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


def _optional_path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


# Change these with environment variables or the CLI flags; code elsewhere should
# import them instead of embedding dataset paths.
DATASET_DIR = _path_from_env("MELODY_DATASET_DIR", PASAN_DIR / "dataset")
INPUT_AUDIO_DIR = _path_from_env(
    "MELODY_INPUT_DIR",
    DATASET_DIR / "mtg-jamendo" / "vocal-tagged-audio",
)
OUTPUT_DIR = _path_from_env("MELODY_OUTPUT_DIR", PACKAGE_DIR / "outputs")
MODEL_CACHE_DIR = _path_from_env("MELODY_MODEL_CACHE_DIR", PACKAGE_DIR / ".models")

# Optional tools that are intentionally kept in their own environments.
BASIC_PITCH_PYTHON = _optional_path_from_env("MELODY_BASIC_PITCH_PYTHON")
ESSENTIA_PYTHON = _optional_path_from_env("MELODY_ESSENTIA_PYTHON")
GAME_ROOT = _optional_path_from_env("MELODY_GAME_ROOT")
GAME_MODEL = _optional_path_from_env("MELODY_GAME_MODEL")
GAME_PYTHON = _optional_path_from_env("MELODY_GAME_PYTHON")
RMVPE_PYTHON = _optional_path_from_env("MELODY_RMVPE_PYTHON")
RMVPE_MODEL = _optional_path_from_env("MELODY_RMVPE_MODEL")
ROSVOT_ROOT = _optional_path_from_env("MELODY_ROSVOT_ROOT")
ROSVOT_PYTHON = _optional_path_from_env("MELODY_ROSVOT_PYTHON")
