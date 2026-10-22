"""Diagnostic comparison; never supplies the key used for Roman labels."""

from __future__ import annotations

import json

from common import (atomic_text, chord_paths, comparison_path, fingerprint,
                    key_path, sha256)
from key import rank
from roman import PITCH_CLASS, family_for_quality


VERSION = 1
CHORD_INTERVALS = {
    "MAJ": (0, 4, 7), "MIN": (0, 3, 7), "DOM7": (0, 4, 7, 10),
    "DIM": (0, 3, 6), "HDIM7": (0, 3, 6, 10), "DIM7": (0, 3, 6, 9),
    "AUG": (0, 4, 8), "SUS2": (0, 2, 7), "SUS4": (0, 5, 7),
}


def from_chords(events: list[dict]) -> dict:
    histogram = [0.0] * 12
    usable = 0
    for event in events:
        root, family = event["root"], family_for_quality(event["quality"])
        if root not in PITCH_CLASS or family not in CHORD_INTERVALS:
            continue
        usable += 1
        weight = min(float(event["end_sec"]) - float(event["start_sec"]), 8.0)
        for interval in CHORD_INTERVALS[family]:
            histogram[(PITCH_CLASS[root] + interval) % 12] += weight * (1.5 if interval == 0 else 1.0)
    return {"method": "duration-weighted chord-tone Krumhansl-Kessler profile",
            "usable_chord_segments": usable, **rank(histogram)}


def build(track: dict[str, str]) -> dict:
    melody_key_file = key_path(track)
    chord_file = chord_paths("submission", track)["json"]
    melody_key = json.loads(melody_key_file.read_text(encoding="utf-8"))["melody_key"]
    chords = json.loads(chord_file.read_text(encoding="utf-8"))
    diagnostic = from_chords(chords["events"])
    selected, other = melody_key["selected"], diagnostic["selected"]
    if selected is None or other is None:
        agreement = "insufficient_evidence"
    elif selected["tonic_pc"] == other["tonic_pc"] and selected["mode"] == other["mode"]:
        agreement = "agree"
    else:
        agreement = "disagree"
    review_reasons = []
    if agreement != "agree":
        review_reasons.append("key_disagreement" if agreement == "disagree" else "insufficient_evidence")
    if melody_key["confidence"] == "low":
        review_reasons.append("low_melody_key_confidence")
    if diagnostic["confidence"] == "low":
        review_reasons.append("low_chord_key_confidence")
    return {"schema_version": 1, "comparison_version": VERSION,
            "track_id": track["track_id"], "mtg_track_id": track["mtg_track_id"],
            "source_mp3": track["source_mp3"], "source_fingerprint": fingerprint(track),
            "melody_key_json_sha256": sha256(melody_key_file),
            "submission_chord_json_sha256": sha256(chord_file),
            "melody_key": selected, "chord_key": diagnostic,
            "agreement": agreement, "review_flag": bool(review_reasons),
            "review_reasons": review_reasons,
            "usage": "diagnostic_only; never used to choose the melody key or Roman labels"}


def completed(track: dict[str, str]) -> dict | None:
    path = comparison_path(track)
    if not path.is_file() or not path.stat().st_size:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data == build(track) else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save(track: dict[str, str]) -> dict:
    data = build(track)
    atomic_text(comparison_path(track), json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    if completed(track) is None:
        raise ValueError("Written key comparison failed resume validation")
    return data
