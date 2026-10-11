"""Lossless LV-Chordia labels, checkpoint validation, and durable result files."""

from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import math
import re
from pathlib import Path

from common import CHORDS, atomic_text, chord_paths, fingerprint

MODEL = "lv-chordia"
MODEL_VERSION = "1.1.0"
CSV_COLUMNS = ("track_id", "start_sec", "end_sec", "chord_label", "root", "quality", "bass_degree")
CHORD = re.compile(r"^([A-G](?:#|b)?):(.+?)(?:/([#b]?\d+))?$")


def verify_model() -> None:
    """Prevent a silent random-weight run if bundled checkpoints are absent."""
    import lv_chordia
    from lv_chordia.chord_recognition import MODEL_NAMES
    from lv_chordia.mir.common import CACHE_DATA_PATH

    version = importlib.metadata.version("lv-chordia")
    if version != MODEL_VERSION or lv_chordia.__version__ != MODEL_VERSION:
        raise RuntimeError(f"Expected lv-chordia {MODEL_VERSION}; found {version}")
    missing = [name for name in MODEL_NAMES if not (Path(CACHE_DATA_PATH) / f"{name}.sdict").is_file()]
    if missing:
        raise RuntimeError(f"LV-Chordia weights missing in {CACHE_DATA_PATH}: {missing}")


def parse_label(label: str) -> tuple[str | None, str | None, str | None]:
    if label == "N":
        return None, None, None
    match = CHORD.fullmatch(label)
    return match.groups() if match else (None, None, None)


def format_events(raw: list[dict], duration_sec: float) -> tuple[list[dict], list[str], float]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("Model returned no chord segments")
    events = []
    for index, segment in enumerate(raw):
        start = float(segment["start_time"])
        end = float(segment["end_time"])
        label = segment["chord"]
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError(f"Invalid times in segment {index}: {segment}")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Missing chord label in segment {index}")
        root, quality, bass = parse_label(label)
        events.append({"start_sec": start, "end_sec": end, "chord_label": label,
                       "root": root, "quality": quality, "bass_degree": bass})
    # Times are rounded to 0.01 seconds by LV-Chordia. Preserve them verbatim.
    if events[0]["start_sec"] > 0.06:
        raise ValueError(f"First chord starts at {events[0]['start_sec']} seconds")
    for left, right in zip(events, events[1:]):
        delta = right["start_sec"] - left["end_sec"]
        if abs(delta) > 0.06:
            raise ValueError(f"Gap or overlap of {delta:.3f} seconds between chords")
    if abs(events[-1]["end_sec"] - duration_sec) > 1.0:
        raise ValueError(f"Chord timeline ends at {events[-1]['end_sec']}, audio at {duration_sec}")
    no_chord = sum(event["end_sec"] - event["start_sec"] for event in events
                   if event["chord_label"] == "N")
    fraction = round(no_chord / max(events[-1]["end_sec"], 0.01), 5)
    flags = []
    if fraction >= 0.99:
        flags.append("all_no_chord")
    if any(event["chord_label"] != "N" and event["root"] is None for event in events):
        flags.append("unparsed_chord_label")
    return events, flags, fraction


def completed(track: dict[str, str], vocabulary: str) -> dict | None:
    if vocabulary not in CHORDS:
        raise ValueError(vocabulary)
    files = chord_paths(vocabulary, track)
    json_path, csv_path = files["json"], files["csv"]
    if not json_path.is_file() or not csv_path.is_file():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if (data.get("schema_version") != 1 or data.get("track_id") != track["track_id"]
                or data.get("source_mp3") != track["source_mp3"]
                or data.get("source_fingerprint") != fingerprint(track)
                or data.get("mtg_track_id") != track["mtg_track_id"]
                or data.get("model") != {"name": MODEL, "version": MODEL_VERSION,
                                          "vocabulary": vocabulary}):
            return None
        events = data["events"]
        validated, flags, fraction = format_events(
            [{"start_time": row["start_sec"], "end_time": row["end_sec"],
              "chord": row["chord_label"]} for row in events], float(data["duration_sec"]))
        if events != validated or data.get("quality_flags") != flags or data.get("no_chord_fraction") != fraction:
            return None
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                return None
            rows = list(reader)
        if len(rows) != len(events):
            return None
        for event, row in zip(events, rows):
            if (row["track_id"] != track["track_id"] or row["chord_label"] != event["chord_label"]
                    or float(row["start_sec"]) != event["start_sec"]
                    or float(row["end_sec"]) != event["end_sec"]
                    or row["root"] != (event["root"] or "")
                    or row["quality"] != (event["quality"] or "")
                    or row["bass_degree"] != (event["bass_degree"] or "")):
                return None
        return data
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save(track: dict[str, str], vocabulary: str, duration_sec: float, raw: list[dict]) -> dict:
    events, flags, fraction = format_events(raw, duration_sec)
    data = {"schema_version": 1, "track_id": track["track_id"],
            "mtg_track_id": track["mtg_track_id"], "source_mp3": track["source_mp3"],
            "source_fingerprint": fingerprint(track), "time_origin": "full_song_start",
            "duration_sec": duration_sec,
            "model": {"name": MODEL, "version": MODEL_VERSION, "vocabulary": vocabulary},
            "quality_flags": flags, "no_chord_fraction": fraction, "events": events}
    files = chord_paths(vocabulary, track)
    json_path, csv_path = files["json"], files["csv"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for event in events:
        writer.writerow({"track_id": track["track_id"], **event})
    # JSON is the commit marker. A stopped write leaves the song incomplete.
    atomic_text(csv_path, buffer.getvalue())
    atomic_text(json_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    if completed(track, vocabulary) is None:
        raise ValueError("Written chord files failed the resume validation")
    return data
