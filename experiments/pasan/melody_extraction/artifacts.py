"""Write a consistent set of ML-ready artifacts for every tool."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

from .models import ExtractionResult, NoteEvent


def write_midi(notes: list[NoteEvent], output_path: Path) -> None:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="melody")
    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                # Confidence is saved losslessly in Parquet/CSV. Keep MIDI
                # velocity audible even for tools whose confidence scale is low.
                velocity=max(32, min(127, round(64 + note.confidence * 63))),
                pitch=max(0, min(127, round(note.midi_pitch))),
                start=note.onset_sec,
                end=note.offset_sec,
            )
        )
    midi.instruments.append(instrument)
    midi.write(str(output_path))


def write_result(result: ExtractionResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(note) for note in result.notes]
    columns = ["onset_sec", "offset_sec", "midi_pitch", "confidence", "source"]
    note_frame = pd.DataFrame(rows, columns=columns)

    notes_parquet = output_dir / "notes.parquet"
    notes_csv = output_dir / "notes.csv"
    midi_path = output_dir / "melody.mid"
    metadata_path = output_dir / "metadata.json"
    note_frame.to_parquet(notes_parquet, index=False)
    note_frame.to_csv(notes_csv, index=False)
    write_midi(result.notes, midi_path)

    artifacts = {
        "notes_parquet": notes_parquet,
        "notes_csv": notes_csv,
        "midi": midi_path,
        "metadata": metadata_path,
    }
    if result.pitch_track is not None:
        pitch_path = output_dir / "f0.npz"
        np.savez_compressed(
            pitch_path,
            times_sec=result.pitch_track.times_sec,
            f0_hz=result.pitch_track.f0_hz,
            confidence=result.pitch_track.confidence,
            voiced=result.pitch_track.voiced,
            source=np.array(result.pitch_track.source),
        )
        artifacts["f0"] = pitch_path

    metadata = {
        "audio_path": str(result.audio_path),
        "pipeline": result.pipeline,
        "runtime_sec": result.runtime_sec,
        "note_count": len(result.notes),
        "key": asdict(result.key) if result.key else None,
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        **result.metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return artifacts
