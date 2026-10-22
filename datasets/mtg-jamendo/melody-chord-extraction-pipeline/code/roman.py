"""Convert saved absolute chords to the proposal's relative-root/family plan."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from common import (atomic_text, chord_paths, fingerprint, key_path, roman_paths, sha256)


VERSION = 1
PITCH_CLASS = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
               "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6,
               "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
               "B": 11, "Cb": 11, "B#": 0}
NAMES = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
DEGREES = ("I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII")
MINOR_FAMILIES = {"MIN", "DIM", "HDIM7", "DIM7"}
CSV_COLUMNS = ("track_id", "start_sec", "end_sec", "original_chord_label",
               "relative_root_semitones", "relative_root", "family", "basic_label",
               "mapping_status", "bass_degree")


def family_for_quality(quality: str | None) -> str | None:
    """Keep only the proposed families; do not guess for rootless/power chords."""
    if not quality:
        return None
    if quality.startswith("hdim7") or quality.startswith("min7(b5"):
        return "HDIM7"
    # Omitted or altered chord-defining tones do not safely imply a family.
    if "*" in quality or "b5" in quality or "#5" in quality:
        return None
    if quality.startswith("dim7"):
        return "DIM7"
    if quality.startswith("dim"):
        return "DIM"
    if quality.startswith("aug"):
        return "AUG"
    if quality.startswith("sus2"):
        return "SUS2"
    if quality.startswith("sus4"):
        return "SUS4"
    if quality.startswith("min"):
        return "MIN"
    if quality.startswith("maj"):
        return "MAJ"
    if quality[0].isdigit() and quality.startswith(("7", "9", "11", "13")):
        return "DOM7"
    return None


def normalize_event(event: dict, track_id: str, tonic_pc: int | None) -> dict:
    root = event["root"]
    quality = event["quality"]
    base = {"track_id": track_id, "start_sec": event["start_sec"],
            "end_sec": event["end_sec"], "original_chord_label": event["chord_label"],
            "relative_root_semitones": None, "relative_root": None, "family": None,
            "basic_label": None, "mapping_status": "unsupported_quality",
            "bass_degree": event["bass_degree"]}
    if event["chord_label"] == "N":
        base.update(family="N", basic_label="N", mapping_status="no_chord")
        return base
    if root not in PITCH_CLASS:
        base["mapping_status"] = "unsupported_root"
        return base
    family = family_for_quality(quality)
    base["family"] = family
    if family is None:
        return base
    if tonic_pc is None:
        base["mapping_status"] = "key_unavailable"
        return base
    interval = (PITCH_CLASS[root] - tonic_pc) % 12
    degree = DEGREES[interval]
    display = degree.lower() if family in MINOR_FAMILIES else degree
    base.update(relative_root_semitones=interval, relative_root=degree,
                basic_label=f"{display}:{family}", mapping_status="mapped")
    return base


def build(track: dict[str, str], vocabulary: str) -> dict:
    source_path = chord_paths(vocabulary, track)["json"]
    key_file = key_path(track)
    chords = json.loads(source_path.read_text(encoding="utf-8"))
    key = json.loads(key_file.read_text(encoding="utf-8"))
    selected = key["melody_key"]["selected"]
    tonic_pc = selected["tonic_pc"] if selected else None
    events = [normalize_event(event, track["track_id"], tonic_pc)
              for event in chords["events"]]
    counts = {status: sum(event["mapping_status"] == status for event in events)
              for status in ("mapped", "no_chord", "key_unavailable", "unsupported_root",
                             "unsupported_quality")}
    return {"schema_version": 1, "mapping_version": VERSION,
            "track_id": track["track_id"], "mtg_track_id": track["mtg_track_id"],
            "source_mp3": track["source_mp3"], "source_fingerprint": fingerprint(track),
            "time_origin": "full_song_start", "duration_sec": chords["duration_sec"],
            "vocabulary": vocabulary, "chord_json_sha256": sha256(source_path),
            "key_json_sha256": sha256(key_file), "key_source": "combined_melody_only",
            "key": selected, "mode_is_separate_from_relative_root": True,
            "mapping_counts": counts, "events": events}


def completed(track: dict[str, str], vocabulary: str) -> dict | None:
    paths = roman_paths(vocabulary, track)
    if not all(path.is_file() and path.stat().st_size for path in paths.values()):
        return None
    try:
        data = json.loads(paths["json"].read_text(encoding="utf-8"))
        if data != build(track, vocabulary):
            return None
        with paths["csv"].open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                return None
            rows = list(reader)
        if len(rows) != len(data["events"]):
            return None
        for expected, row in zip(data["events"], rows):
            if any(row[column] != ("" if expected[column] is None else str(expected[column]))
                   for column in CSV_COLUMNS):
                return None
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save(track: dict[str, str], vocabulary: str) -> dict:
    data = build(track, vocabulary)
    paths = roman_paths(vocabulary, track)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(data["events"])
    atomic_text(paths["csv"], buffer.getvalue())
    atomic_text(paths["json"], json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    if completed(track, vocabulary) is None:
        raise ValueError("Written Roman-numeral files failed resume validation")
    return data
