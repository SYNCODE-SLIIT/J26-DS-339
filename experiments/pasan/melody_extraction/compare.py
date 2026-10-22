"""Run several tools on the same files and write a compact comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .audio_utils import discover_audio_files
from .config import PipelineConfig
from .errors import ToolUnavailable
from .pipelines import PipelineRunner


def compare_pipelines(
    config: PipelineConfig,
    pipelines: list[str],
    *,
    limit: int | None = None,
    pattern: str | None = None,
    skip_unavailable: bool = True,
) -> pd.DataFrame:
    runner = PipelineRunner(config)
    files = discover_audio_files(config.input_dir, pattern=pattern, limit=limit)
    rows: list[dict] = []
    unavailable: dict[str, str] = {}
    for audio_path in files:
        for pipeline in pipelines:
            if pipeline in unavailable:
                rows.append(
                    {
                        "audio_path": str(audio_path),
                        "pipeline": pipeline,
                        "status": "unavailable",
                        "error": unavailable[pipeline],
                    }
                )
                continue
            try:
                result, status = runner.run(pipeline, audio_path)
                pitches = [note.midi_pitch for note in result.notes]
                rows.append(
                    {
                        "audio_path": str(audio_path),
                        "pipeline": pipeline,
                        "status": status,
                        "note_count": len(result.notes),
                        "mean_midi_pitch": float(np.mean(pitches)) if pitches else np.nan,
                        "key": result.key.label if result.key else None,
                        "key_strength": result.key.strength if result.key else np.nan,
                        "runtime_sec": result.runtime_sec,
                        "error": None,
                    }
                )
            except ToolUnavailable as exc:
                if not skip_unavailable:
                    raise
                unavailable[pipeline] = str(exc)
                rows.append(
                    {
                        "audio_path": str(audio_path),
                        "pipeline": pipeline,
                        "status": "unavailable",
                        "error": str(exc),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "audio_path": str(audio_path),
                        "pipeline": pipeline,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    frame = pd.DataFrame(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.output_dir / "comparison.csv", index=False)
    json_rows = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    (config.output_dir / "comparison.json").write_text(json.dumps(json_rows, indent=2) + "\n")
    return frame
