"""Paths, MTG-Jamendo joins, and the shared note-file format."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent
ASSETS = Path(os.environ.get("MELODY_MODEL_ASSETS", DATASET / "melody-extraction")).resolve()
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work"
METHODS = ("game_vocal", "sheetsage2_instrumental", "combined")
CSV_COLUMNS = ("track_id", "onset_sec", "offset_sec", "midi_pitch",
               "midi_pitch_rounded", "velocity", "source", "model_midi_pitch")


def track_id_from_annotation(value: str) -> str:
    return str(int(value.removeprefix("track_")))


def inventory() -> list[dict[str, str]]:
    """Join local MP3s to MTG IDs and all five official autotagging splits."""
    annotations: dict[str, dict] = {}
    with (DATASET / "annotations" / "autotagging.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            annotations[track_id_from_annotation(row["TRACK_ID"])] = row
    splits: dict[str, dict[str, str]] = {}
    for index in range(5):
        for split in ("train", "validation", "test"):
            path = DATASET / "splits" / f"split-{index}" / f"autotagging-{split}.tsv"
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    splits.setdefault(track_id_from_annotation(row["TRACK_ID"]), {})[
                        f"split_{index}"] = split
    result = []
    for audio in (DATASET / "vocal-tagged-audio").rglob("*.low.mp3"):
        track_id = audio.name.split(".")[0]
        annotation = annotations.get(track_id, {})
        result.append({
            "track_id": track_id,
            "mtg_track_id": annotation.get("TRACK_ID", f"track_{int(track_id):07d}"),
            "artist_id": annotation.get("ARTIST_ID", ""),
            "album_id": annotation.get("ALBUM_ID", ""),
            "annotation_path": annotation.get("PATH", ""),
            "source_mp3": audio.relative_to(DATASET).as_posix(),
            "duration_annotation_sec": annotation.get("DURATION", ""),
            "tags": ";".join(tag for tag in
                             [annotation.get("TAGS", ""), *(annotation.get(None) or [])] if tag),
            **{f"split_{i}": splits.get(track_id, {}).get(f"split_{i}", "") for i in range(5)},
        })
    result.sort(key=lambda row: int(row["track_id"]))
    if len({row["track_id"] for row in result}) != len(result):
        raise ValueError("Duplicate local audio track ID")
    return result


def paths(method: str, track_id: str) -> dict[str, Path]:
    if method not in METHODS:
        raise ValueError(method)
    directory = OUTPUTS / method
    return {suffix: directory / f"{track_id}.{suffix}" for suffix in ("mid", "json", "csv")}


def complete(method: str, track_id: str) -> bool:
    files = paths(method, track_id)
    if not all(path.exists() and path.stat().st_size for path in files.values()):
        return False
    try:
        data = json.loads(files["json"].read_text())
        return data["track_id"] == track_id and data["method"] == method and isinstance(data["notes"], list)
    except (ValueError, KeyError, OSError):
        return False


def read_notes(method: str, track_id: str) -> list[dict]:
    return json.loads(paths(method, track_id)["json"].read_text())["notes"]


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    os.replace(temporary, path)


def write_result(method: str, track_id: str, notes: list[dict], *, metadata: dict | None = None) -> None:
    """Atomically publish the same notes in MIDI, JSON, and flat CSV."""
    import pretty_midi

    files = paths(method, track_id)
    files["mid"].parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(notes, key=lambda n: (float(n["onset_sec"]), float(n["offset_sec"])))
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name=method)
    for note in ordered:
        onset = float(note["onset_sec"])
        offset = float(note["offset_sec"])
        pitch = int(note["midi_pitch_rounded"])
        if not (0 <= pitch <= 127 and 0 <= onset < offset):
            raise ValueError(f"Invalid note for {track_id}/{method}: {note}")
        instrument.notes.append(pretty_midi.Note(
            velocity=int(note.get("velocity", 100)), pitch=pitch, start=onset, end=offset))
    midi.instruments.append(instrument)
    temporary_midi = files["mid"].with_name(f".{track_id}.{os.getpid()}.tmp.mid")
    midi.write(str(temporary_midi))
    os.replace(temporary_midi, files["mid"])
    payload = {"schema_version": 1, "track_id": track_id, "method": method,
               "time_origin": "full_song_start", "pitch_unit": "MIDI semitones",
               "notes": ordered}
    if metadata:
        payload.update(metadata)
    atomic_text(files["json"], json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary_csv = files["csv"].with_name(f".{track_id}.{os.getpid()}.tmp.csv")
    with temporary_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for note in ordered:
            writer.writerow({"track_id": track_id, **note})
    os.replace(temporary_csv, files["csv"])
