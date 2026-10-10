"""Composable melody pipelines and their shared runner."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from .artifacts import write_result
from .audio_utils import load_mono, relative_song_path
from .basic_pitch_adapter import transcribe_basic_pitch
from .config import PipelineConfig
from .essentia_adapter import extract_melodia
from .fusion import fuse_pitch_tracks
from .game_adapter import transcribe_game
from .key_estimation import estimate_key_librosa
from .librosa_melody import extract_pyin
from .mlx_separator import MlxVocalSeparator
from .models import ExtractionResult, KeyEstimate, NoteEvent, PitchTrack
from .pitch_utils import contour_to_notes
from .rmvpe_adapter import extract_rmvpe
from .rosvot_adapter import transcribe_rosvot
from .sheetsage_adapter import transcribe_sheetsage


@dataclass(frozen=True, slots=True)
class PipelineInfo:
    name: str
    description: str
    optional_tools: tuple[str, ...] = ()


PIPELINES = {
    "pyin_fullmix": PipelineInfo(
        "pyin_fullmix",
        "Fast baseline: librosa pYIN directly on the original mix.",
    ),
    "mlx_pyin_vocals": PipelineInfo(
        "mlx_pyin_vocals",
        "MLX vocal separation, then librosa pYIN.",
        ("MLX separator checkpoint",),
    ),
    "basic_pitch_fullmix": PipelineInfo(
        "basic_pitch_fullmix",
        "Spotify Basic Pitch directly on the original mix.",
        ("Basic Pitch Python environment",),
    ),
    "mlx_basic_pitch_vocals": PipelineInfo(
        "mlx_basic_pitch_vocals",
        "MLX vocal separation, then Spotify Basic Pitch.",
        ("MLX separator checkpoint", "Basic Pitch Python environment"),
    ),
    "mlx_game_vocals": PipelineInfo(
        "mlx_game_vocals",
        "MLX vocal separation, then GAME singing transcription.",
        ("MLX separator checkpoint", "GAME repository and checkpoint"),
    ),
    "rmvpe_fullmix": PipelineInfo(
        "rmvpe_fullmix",
        "RMVPE vocal F0 directly on the polyphonic original mix.",
        ("RMVPE Python environment and checkpoint",),
    ),
    "mlx_rmvpe_vocals": PipelineInfo(
        "mlx_rmvpe_vocals",
        "MLX vocal separation, then RMVPE vocal F0.",
        ("MLX separator checkpoint", "RMVPE Python environment and checkpoint"),
    ),
    "mlx_rosvot_vocals": PipelineInfo(
        "mlx_rosvot_vocals",
        "MLX vocal separation, then ROSVOT singing transcription.",
        ("MLX separator checkpoint", "ROSVOT repository/checkpoints/environment"),
    ),
    "essentia_melodia_fullmix": PipelineInfo(
        "essentia_melodia_fullmix",
        "Essentia PredominantPitchMelodia on the original mix.",
        ("Essentia Python environment",),
    ),
    "sheetsage_fullmix": PipelineInfo(
        "sheetsage_fullmix",
        "SheetSage lead-melody transcription on the original mix (harmony disabled).",
        ("SheetSage checkpoint on first run",),
    ),
    "hybrid_vocal_melodia": PipelineInfo(
        "hybrid_vocal_melodia",
        "Confident MLX+pYIN vocal F0, with Essentia full-mix melody as fallback.",
        ("MLX separator checkpoint", "Essentia Python environment"),
    ),
}


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.separator = MlxVocalSeparator(config)
        self._key_cache: dict[Path, KeyEstimate] = {}

    def output_dir_for(self, pipeline: str, audio_path: Path) -> Path:
        relative = relative_song_path(audio_path, self.config.input_dir)
        return self.config.output_dir / pipeline / relative

    def vocal_stem_for(self, audio_path: Path) -> Path:
        relative = relative_song_path(audio_path, self.config.input_dir)
        stem_dir = (
            self.config.output_dir
            / "_stems"
            / self.separator.model_slug
            / relative
        )
        return self.separator.separate_vocals(audio_path, stem_dir)

    def estimate_key(self, audio_path: Path) -> KeyEstimate:
        cache_key = audio_path.resolve()
        if cache_key not in self._key_cache:
            audio, sr = load_mono(audio_path, self.config.sample_rate)
            self._key_cache[cache_key] = estimate_key_librosa(audio, sr)
        return self._key_cache[cache_key]

    def _confidence_floor(self, track: PitchTrack) -> float:
        # Confidence values are model-specific, not calibrated probabilities.
        # pYIN's own voiced mask is the useful gate; its probabilities are often
        # near 0.01 even for frames it classifies as voiced.
        if "pyin" in track.source:
            return 0.0
        if track.source == "essentia_melodia":
            return self.config.essentia_minimum_confidence
        if track.source == "vocal_primary_fullmix_fallback":
            return self.config.fusion_minimum_confidence
        return self.config.minimum_confidence

    def _notes_from_track(self, track: PitchTrack) -> list[NoteEvent]:
        return contour_to_notes(
            track,
            minimum_note_ms=self.config.minimum_note_ms,
            minimum_confidence=self._confidence_floor(track),
        )

    def _extract(self, pipeline: str, audio_path: Path) -> tuple[list[NoteEvent], PitchTrack | None, dict]:
        if pipeline == "pyin_fullmix":
            track = extract_pyin(audio_path, self.config, source="pyin_fullmix")
            return self._notes_from_track(track), track, {"transcription_input": str(audio_path)}

        if pipeline == "mlx_pyin_vocals":
            vocals = self.vocal_stem_for(audio_path)
            track = extract_pyin(vocals, self.config, source="mlx_pyin_vocals")
            return self._notes_from_track(track), track, {"transcription_input": str(vocals)}

        if pipeline == "basic_pitch_fullmix":
            notes = transcribe_basic_pitch(audio_path)
            return notes, None, {"transcription_input": str(audio_path)}

        if pipeline == "mlx_basic_pitch_vocals":
            vocals = self.vocal_stem_for(audio_path)
            notes = transcribe_basic_pitch(vocals)
            return notes, None, {"transcription_input": str(vocals)}

        if pipeline == "mlx_game_vocals":
            vocals = self.vocal_stem_for(audio_path)
            notes = transcribe_game(vocals)
            return notes, None, {"transcription_input": str(vocals)}

        if pipeline == "rmvpe_fullmix":
            track = extract_rmvpe(audio_path)
            return self._notes_from_track(track), track, {"transcription_input": str(audio_path)}

        if pipeline == "mlx_rmvpe_vocals":
            vocals = self.vocal_stem_for(audio_path)
            track = extract_rmvpe(vocals)
            return self._notes_from_track(track), track, {"transcription_input": str(vocals)}

        if pipeline == "mlx_rosvot_vocals":
            vocals = self.vocal_stem_for(audio_path)
            notes = transcribe_rosvot(vocals)
            return notes, None, {"transcription_input": str(vocals)}

        if pipeline == "essentia_melodia_fullmix":
            track = extract_melodia(audio_path)
            return self._notes_from_track(track), track, {"transcription_input": str(audio_path)}

        if pipeline == "sheetsage_fullmix":
            notes = transcribe_sheetsage(audio_path)
            return notes, None, {
                "transcription_input": str(audio_path),
                "sheetsage_features": "handcrafted",
                "harmony_detection": False,
            }

        if pipeline == "hybrid_vocal_melodia":
            vocals = self.vocal_stem_for(audio_path)
            vocal_track = extract_pyin(vocals, self.config, source="mlx_pyin_vocals")
            fullmix_track = extract_melodia(audio_path)
            track = fuse_pitch_tracks(
                vocal_track,
                fullmix_track,
                primary_confidence=self.config.fusion_primary_confidence,
            )
            return self._notes_from_track(track), track, {
                "transcription_input": [str(vocals), str(audio_path)],
                "fusion": (
                    "voiced vocal confidence >= "
                    f"{self.config.fusion_primary_confidence:.2f}, otherwise full-mix Melodia"
                ),
            }

        raise ValueError(f"Unknown pipeline: {pipeline}")

    def run(self, pipeline: str, audio_path: Path) -> tuple[ExtractionResult, str]:
        if pipeline not in PIPELINES:
            raise ValueError(f"Unknown pipeline {pipeline!r}. Choose from {sorted(PIPELINES)}")
        output_dir = self.output_dir_for(pipeline, audio_path)
        metadata_path = output_dir / "metadata.json"
        if metadata_path.exists() and not self.config.overwrite:
            metadata = json.loads(metadata_path.read_text())
            notes_path = output_dir / "notes.parquet"
            note_rows = pd.read_parquet(notes_path).to_dict(orient="records")
            cached_key = KeyEstimate(**metadata["key"]) if metadata.get("key") else None
            result = ExtractionResult(
                audio_path=audio_path,
                pipeline=pipeline,
                notes=[NoteEvent(**row) for row in note_rows],
                key=cached_key,
                runtime_sec=float(metadata.get("runtime_sec", 0)),
                metadata={"cached_notes_path": str(notes_path)},
            )
            return result, "cached"

        started = time.perf_counter()
        notes, track, metadata = self._extract(pipeline, audio_path)
        result = ExtractionResult(
            audio_path=audio_path,
            pipeline=pipeline,
            notes=notes,
            pitch_track=track,
            key=self.estimate_key(audio_path),
            runtime_sec=time.perf_counter() - started,
            metadata={
                **metadata,
                "versions": {
                    "librosa": _package_version("librosa"),
                    "mlx_audio_separator": _package_version("mlx-audio-separator"),
                    "basic_pitch": _package_version("basic-pitch"),
                    "essentia": _package_version("essentia"),
                    "sheetsage_infer": _package_version("sheetsage-infer"),
                },
                "settings": {
                    "sample_rate": self.config.sample_rate,
                    "hop_length": self.config.hop_length,
                    "minimum_note_ms": self.config.minimum_note_ms,
                    "minimum_confidence": self.config.minimum_confidence,
                    "essentia_minimum_confidence": self.config.essentia_minimum_confidence,
                    "fusion_minimum_confidence": self.config.fusion_minimum_confidence,
                    "fusion_primary_confidence": self.config.fusion_primary_confidence,
                    "effective_minimum_confidence": (
                        self._confidence_floor(track) if track is not None else None
                    ),
                    "separation_model": self.config.separation_model,
                },
            },
        )
        write_result(result, output_dir)
        return result, "completed"
