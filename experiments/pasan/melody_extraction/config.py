"""Shared settings for every extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from . import paths


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    input_dir: Path = paths.INPUT_AUDIO_DIR
    output_dir: Path = paths.OUTPUT_DIR
    model_cache_dir: Path = paths.MODEL_CACHE_DIR

    sample_rate: int = 22_050
    hop_length: int = 256
    fmin_hz: float = 65.406  # C2
    fmax_hz: float = 2_093.005  # C7
    minimum_note_ms: float = 60.0
    minimum_confidence: float = 0.25

    separation_model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    overwrite: bool = False

    def with_paths(
        self,
        *,
        input_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> "PipelineConfig":
        return replace(
            self,
            input_dir=(input_dir or self.input_dir).expanduser().resolve(),
            output_dir=(output_dir or self.output_dir).expanduser().resolve(),
        )

