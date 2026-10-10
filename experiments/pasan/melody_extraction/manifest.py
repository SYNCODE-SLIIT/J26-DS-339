"""Build one ML dataset index from per-song extraction artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_manifest(output_dir: Path, pipelines: list[str] | None = None) -> pd.DataFrame:
    selected = set(pipelines or [])
    rows: list[dict] = []
    for metadata_path in sorted(output_dir.glob("*/**/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        pipeline = metadata.get("pipeline")
        if selected and pipeline not in selected:
            continue
        key = metadata.get("key") or {}
        artifacts = metadata.get("artifacts") or {}
        rows.append(
            {
                "audio_path": metadata.get("audio_path"),
                "pipeline": pipeline,
                "notes_parquet": artifacts.get("notes_parquet"),
                "midi_path": artifacts.get("midi"),
                "f0_path": artifacts.get("f0"),
                "melody_audio_path": artifacts.get("melody_audio"),
                "melody_overlay_path": artifacts.get("melody_overlay"),
                "note_count": metadata.get("note_count"),
                "key_tonic": key.get("tonic"),
                "key_mode": key.get("mode"),
                "key_strength": key.get("strength"),
                "runtime_sec": metadata.get("runtime_sec"),
                "metadata_path": str(metadata_path),
            }
        )

    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "manifest.parquet", index=False)
    frame.to_csv(output_dir / "manifest.csv", index=False)
    return frame
