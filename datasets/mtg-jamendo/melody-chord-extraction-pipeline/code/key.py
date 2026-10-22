"""Melody-only key estimate, independent of chord extraction."""

from __future__ import annotations

import json
import math
import statistics

from common import atomic_text, fingerprint, key_path, melody_paths, sha256
from roman import NAMES


VERSION = 1
# Krumhansl-Kessler probe-tone profiles, indexed by semitones above the tonic.
MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def correlation(a: list[float], b: tuple[float, ...]) -> float:
    mean_a, mean_b = sum(a) / 12, sum(b) / 12
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    norm_a = sum((x - mean_a) ** 2 for x in a)
    norm_b = sum((y - mean_b) ** 2 for y in b)
    return numerator / math.sqrt(norm_a * norm_b) if norm_a and norm_b else 0.0


def rank(histogram: list[float], endings: list[int] | None = None) -> dict:
    """Rank 24 keys by tonal profile; melodic phrase arrivals are a small tie-break."""
    if sum(value > 0 for value in histogram) < 3:
        return {"selected": None, "candidates": [], "confidence": "insufficient_evidence",
                "top_score_gap": None}
    candidates = []
    for tonic_pc in range(12):
        shifted = [histogram[(tonic_pc + interval) % 12] for interval in range(12)]
        for mode, profile in (("major", MAJOR), ("minor", MINOR)):
            profile_score = correlation(shifted, profile)
            tonic_ending_fraction = (sum(pc == tonic_pc for pc in endings) / len(endings)
                                     if endings else 0.0)
            score = profile_score + 0.08 * tonic_ending_fraction
            candidates.append({"tonic": NAMES[tonic_pc], "tonic_pc": tonic_pc,
                               "mode": mode, "score": round(score, 5),
                               "profile_correlation": round(profile_score, 5),
                               "tonic_ending_fraction": round(tonic_ending_fraction, 5)})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    gap = round(candidates[0]["score"] - candidates[1]["score"], 5)
    best = candidates[0]
    confidence = ("high" if gap >= 0.1 and best["profile_correlation"] >= 0.65 else
                  "medium" if gap >= 0.05 and best["profile_correlation"] >= 0.5 else "low")
    return {"selected": {"tonic": best["tonic"], "tonic_pc": best["tonic_pc"],
                         "mode": best["mode"], "confidence": confidence},
            "candidates": candidates[:3], "confidence": confidence,
            "top_score_gap": gap}


def melody_endings(notes: list[dict]) -> list[int]:
    if not notes:
        return []
    ordered = sorted(notes, key=lambda row: row["onset_sec"])
    gaps = [float(right["onset_sec"]) - float(left["offset_sec"])
            for left, right in zip(ordered, ordered[1:])]
    positive = [gap for gap in gaps if gap > 0]
    cutoff = max(0.6, 2 * statistics.median(positive)) if positive else 0.6
    endings = [int(note["midi_pitch_rounded"]) % 12
               for note, gap in zip(ordered, gaps) if gap >= cutoff]
    endings.append(int(ordered[-1]["midi_pitch_rounded"]) % 12)
    return endings


def from_melody(notes: list[dict]) -> dict:
    histogram = [0.0] * 12
    for note in notes:
        duration = float(note["offset_sec"]) - float(note["onset_sec"])
        histogram[int(note["midi_pitch_rounded"]) % 12] += min(duration, 2.0)
    endings = melody_endings(notes)
    result = rank(histogram, endings)
    return {"method": "Krumhansl-Kessler profile correlation plus phrase-end tonic tie-break",
            "note_count": len(notes), "phrase_end_count": len(endings),
            "pitch_class_duration_sec_capped_at_2": [round(x, 4) for x in histogram],
            **result}


def build(track: dict[str, str]) -> dict:
    melody_file = melody_paths("combined", track)["json"]
    melody = json.loads(melody_file.read_text(encoding="utf-8"))
    melodic = from_melody(melody["notes"])
    return {"schema_version": 1, "estimator_version": VERSION,
            "track_id": track["track_id"], "mtg_track_id": track["mtg_track_id"],
            "source_mp3": track["source_mp3"], "source_fingerprint": fingerprint(track),
            "melody_json_sha256": sha256(melody_file),
            "melody_key": melodic}


def completed(track: dict[str, str]) -> dict | None:
    path = key_path(track)
    if not path.is_file() or not path.stat().st_size:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if data == build(track) else None
    except (OSError, ValueError, TypeError, KeyError):
        return None


def save(track: dict[str, str]) -> dict:
    data = build(track)
    atomic_text(key_path(track), json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    if completed(track) is None:
        raise ValueError("Written key file failed resume validation")
    return data
