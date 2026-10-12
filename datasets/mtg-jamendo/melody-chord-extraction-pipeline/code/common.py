"""One dataset inventory and durable, per-track output paths for every stage."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent
ASSETS = Path(os.environ.get("MELODY_MODEL_ASSETS", DATASET / "melody-extraction")).resolve()
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work"
MELODIES = ("game_vocal", "sheetsage2_instrumental", "combined")
CHORDS = ("submission", "full")
ROMAN = ("roman_submission", "roman_full")
STAGES = (*MELODIES, "key", *CHORDS, "key_comparison", *ROMAN)
CSV_COLUMNS = ("track_id", "onset_sec", "offset_sec", "midi_pitch",
               "midi_pitch_rounded", "velocity", "source", "model_midi_pitch")


def inventory() -> list[dict[str, str]]:
    """Local full MP3s joined to MTG annotations and all five official splits."""
    annotations = {}
    with (DATASET / "annotations" / "autotagging.tsv").open(newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            annotations[str(int(row["TRACK_ID"].removeprefix("track_")))] = row
    splits: dict[str, dict[str, str]] = {}
    for number in range(5):
        for split in ("train", "validation", "test"):
            path = DATASET / "splits" / f"split-{number}" / f"autotagging-{split}.tsv"
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream, delimiter="\t"):
                    track_id = str(int(row["TRACK_ID"].removeprefix("track_")))
                    splits.setdefault(track_id, {})[f"split_{number}"] = split
    found = []
    for mp3 in (DATASET / "vocal-tagged-audio").rglob("*.low.mp3"):
        track_id = str(int(mp3.name.split(".")[0]))
        annotation = annotations.get(track_id, {})
        found.append({
            "track_id": track_id,
            "mtg_track_id": annotation.get("TRACK_ID", f"track_{int(track_id):07d}"),
            "artist_id": annotation.get("ARTIST_ID", ""),
            "album_id": annotation.get("ALBUM_ID", ""),
            "annotation_path": annotation.get("PATH", ""),
            "source_mp3": mp3.relative_to(DATASET).as_posix(),
            "duration_annotation_sec": annotation.get("DURATION", ""),
            "tags": ";".join(filter(None, [annotation.get("TAGS", ""), *(annotation.get(None) or [])])),
            **{f"split_{i}": splits.get(track_id, {}).get(f"split_{i}", "") for i in range(5)},
        })
    found.sort(key=lambda row: int(row["track_id"]))
    if len(found) != len({row["track_id"] for row in found}):
        raise ValueError("Duplicate numeric track ID in local MP3s")
    return found


def fingerprint(track: dict[str, str]) -> dict[str, int]:
    stat = (DATASET / track["source_mp3"]).stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def track_dir(track: dict[str, str]) -> Path:
    return OUTPUTS / track["mtg_track_id"]


def melody_paths(method: str, track: dict[str, str]) -> dict[str, Path]:
    if method not in MELODIES:
        raise ValueError(method)
    directory = track_dir(track) / "melody"
    return {suffix: directory / f"{method}.{suffix}" for suffix in ("mid", "json", "csv")}


def chord_paths(vocabulary: str, track: dict[str, str]) -> dict[str, Path]:
    if vocabulary not in CHORDS:
        raise ValueError(vocabulary)
    directory = track_dir(track) / "chords"
    return {suffix: directory / f"{vocabulary}.{suffix}" for suffix in ("json", "csv")}


def key_path(track: dict[str, str]) -> Path:
    return track_dir(track) / "key.json"


def comparison_path(track: dict[str, str]) -> Path:
    return track_dir(track) / "key_comparison.json"


def roman_paths(vocabulary: str, track: dict[str, str]) -> dict[str, Path]:
    if vocabulary not in CHORDS:
        raise ValueError(vocabulary)
    directory = track_dir(track) / "roman"
    return {suffix: directory / f"{vocabulary}.{suffix}" for suffix in ("json", "csv")}


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_notes(notes: list[dict], duration_sec: float) -> bool:
    if not isinstance(notes, list):
        return False
    for note in notes:
        try:
            start, end = float(note["onset_sec"]), float(note["offset_sec"])
            exact_pitch = float(note["midi_pitch"])
            pitch = int(note["midi_pitch_rounded"])
            velocity = int(note["velocity"])
            if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or
                    end <= start or end > duration_sec + 1.0 or
                    not math.isfinite(exact_pitch) or not 0 <= exact_pitch <= 127 or
                    not 0 <= pitch <= 127 or not 0 <= velocity <= 127 or
                    note["source"] not in MELODIES):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def melody_complete(method: str, track: dict[str, str]) -> dict | None:
    files = melody_paths(method, track)
    if not all(path.is_file() and path.stat().st_size for path in files.values()):
        return None
    try:
        data = json.loads(files["json"].read_text(encoding="utf-8"))
        if (data.get("schema_version") != 1 or data.get("track_id") != track["track_id"]
                or data.get("mtg_track_id") != track["mtg_track_id"]
                or data.get("method") != method
                or data.get("source_mp3") != track["source_mp3"]
                or data.get("source_fingerprint") != fingerprint(track)):
            return None
        notes = data["notes"]
        if not valid_notes(notes, float(data["duration_sec"])):
            return None
        with files["csv"].open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                return None
            rows = list(reader)
        if len(rows) != len(notes):
            return None
        for note, row in zip(notes, rows):
            if (row["track_id"] != track["track_id"]
                    or row["source"] != note["source"]
                    or float(row["onset_sec"]) != float(note["onset_sec"])
                    or float(row["offset_sec"]) != float(note["offset_sec"])
                    or float(row["midi_pitch"]) != float(note["midi_pitch"])
                    or int(row["midi_pitch_rounded"]) != int(note["midi_pitch_rounded"])):
                return None
        # A parseable MIDI is required, not just a nonempty file.
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(str(files["mid"]))
        if sum(len(instrument.notes) for instrument in midi.instruments) != len(notes):
            return None
        return data
    except (OSError, ValueError, TypeError, KeyError, EOFError):
        return None


def read_notes(method: str, track: dict[str, str]) -> list[dict]:
    return json.loads(melody_paths(method, track)["json"].read_text(encoding="utf-8"))["notes"]


def write_notes(method: str, track: dict[str, str], notes: list[dict],
                *, duration_sec: float, metadata: dict | None = None) -> dict:
    """Publish matching MIDI, JSON, CSV; JSON is the final commit marker."""
    import pretty_midi

    if not valid_notes(notes, duration_sec):
        raise ValueError(f"Invalid {method} notes for {track['track_id']}")
    ordered = sorted(notes, key=lambda n: (float(n["onset_sec"]), float(n["offset_sec"])))
    files = melody_paths(method, track)
    files["json"].parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name=method)
    for note in ordered:
        instrument.notes.append(pretty_midi.Note(
            velocity=int(note.get("velocity", 100)), pitch=int(note["midi_pitch_rounded"]),
            start=float(note["onset_sec"]), end=float(note["offset_sec"])))
    midi.instruments.append(instrument)
    temporary = files["mid"].with_name(f".{method}.{os.getpid()}.tmp.mid")
    try:
        midi.write(str(temporary))
        os.replace(temporary, files["mid"])
    finally:
        temporary.unlink(missing_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for note in ordered:
        writer.writerow({"track_id": track["track_id"], **note})
    atomic_text(files["csv"], buffer.getvalue())
    data = {"schema_version": 1, "track_id": track["track_id"],
            "mtg_track_id": track["mtg_track_id"], "method": method,
            "source_mp3": track["source_mp3"], "source_fingerprint": fingerprint(track),
            "time_origin": "full_song_start", "pitch_unit": "MIDI semitones",
            "duration_sec": duration_sec, "notes": ordered}
    if metadata:
        data.update(metadata)
    atomic_text(files["json"], json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    if melody_complete(method, track) is None:
        raise ValueError(f"Written {method} files failed validation")
    return data
