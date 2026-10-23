"""MLX-native vocal separation, isolated from transcription logic."""

from __future__ import annotations

import logging
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from .config import PipelineConfig
from .errors import ToolUnavailable


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


@contextmanager
def _stereo_input(audio_path: Path) -> Iterator[Path]:
    """Work around MLX RoFormer's current mono-input channel mismatch."""
    try:
        if sf.info(audio_path).channels >= 2:
            yield audio_path
            return
    except RuntimeError:
        # Let the separator's own decoder handle formats libsndfile cannot inspect.
        yield audio_path
        return

    with tempfile.TemporaryDirectory(prefix="mlx-stereo-") as directory:
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        stereo = np.repeat(audio, 2, axis=1) if audio.shape[1] == 1 else audio
        temporary_path = Path(directory) / "input_stereo.wav"
        sf.write(temporary_path, stereo, sample_rate, subtype="PCM_16")
        yield temporary_path


class MlxVocalSeparator:
    """Lazily load one MLX model and reuse it across a sequential batch."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._separator = None

    @property
    def model_slug(self) -> str:
        return _slug(Path(self.config.separation_model).stem)

    def _load(self, output_dir: Path):
        try:
            from mlx_audio_separator import Separator
        except ImportError as exc:
            raise ToolUnavailable(
                "mlx-audio-separator is missing; run `uv sync` from the repository root"
            ) from exc

        # A Separator keeps its output directory, so a new lightweight wrapper is
        # created per song while the converted checkpoint remains cached on disk.
        separator = Separator(
            log_level=logging.INFO,
            model_file_dir=str(self.config.model_cache_dir / "mlx-audio-separator"),
            output_dir=str(output_dir),
            output_format="WAV",
            output_single_stem="Vocals",
            save_converted_safetensors=True,
        )
        separator.load_model(self.config.separation_model)
        return separator

    def _set_output_dir(self, output_dir: Path) -> None:
        if self._separator is None:
            self._separator = self._load(output_dir)
            return
        self._separator.output_dir = str(output_dir)
        if self._separator.model_instance is not None:
            self._separator.model_instance.output_dir = str(output_dir)

    def separate_vocals(self, audio_path: Path, stem_dir: Path) -> Path:
        stem_dir.mkdir(parents=True, exist_ok=True)
        expected = stem_dir / "vocals.wav"
        if expected.exists() and not self.config.overwrite:
            return expected

        self._set_output_dir(stem_dir)
        with _stereo_input(audio_path) as separator_input:
            outputs = self._separator.separate(
                str(separator_input),
                custom_output_names={"Vocals": "vocals"},
            )
        candidates = [Path(value) for value in outputs]
        for candidate in candidates:
            path = candidate if candidate.is_absolute() else stem_dir / candidate
            if path.exists() and "vocal" in path.stem.lower():
                if path != expected:
                    path.replace(expected)
                return expected
        if expected.exists():
            return expected
        raise RuntimeError(f"MLX separation produced no vocal stem; outputs={outputs!r}")
