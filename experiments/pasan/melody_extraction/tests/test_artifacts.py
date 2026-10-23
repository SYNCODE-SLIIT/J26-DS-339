from __future__ import annotations

import json

import pandas as pd

from experiments.pasan.melody_extraction.artifacts import write_result
from experiments.pasan.melody_extraction.manifest import build_manifest
from experiments.pasan.melody_extraction.models import ExtractionResult, KeyEstimate, NoteEvent


def test_result_writes_canonical_artifacts(tmp_path) -> None:
    result = ExtractionResult(
        audio_path=tmp_path / "song.mp3",
        pipeline="test",
        notes=[NoteEvent(0.1, 0.5, 69, 0.75, "test")],
        key=KeyEstimate("A", "minor", 0.8, "test"),
    )

    artifacts = write_result(result, tmp_path / "result")

    assert artifacts["midi"].exists()
    assert pd.read_parquet(artifacts["notes_parquet"]).iloc[0]["midi_pitch"] == 69
    metadata = json.loads(artifacts["metadata"].read_text())
    assert metadata["key"]["tonic"] == "A"
    assert metadata["note_count"] == 1
    assert artifacts["melody_audio"].exists()

    manifest = build_manifest(tmp_path)
    assert len(manifest) == 1
    assert manifest.iloc[0]["key_mode"] == "minor"
    assert manifest.iloc[0]["melody_audio_path"] == str(artifacts["melody_audio"])
