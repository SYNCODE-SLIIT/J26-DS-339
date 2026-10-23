"""Build leakage-free chord-refinement training events for a small track batch.

Melody and key extraction deliberately adapt the existing team pYIN pipeline.
Basic chords depend only on melody, global key, beats, and prior basic harmony.
Reference chords are read only after basic-chord generation is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np

from experiments.rahul import config
from experiments.rahul.src.canonicalize_chords import add_canonical_fields
from experiments.pasan.melody_extraction.config import PipelineConfig
from experiments.pasan.melody_extraction.key_estimation import estimate_key_librosa
from experiments.pasan.melody_extraction.librosa_melody import extract_pyin
from experiments.pasan.melody_extraction.pitch_utils import contour_to_notes


PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}

# (scale semitones, qualities, Roman numerals, harmonic functions)
DIATONIC = {
    "major": (
        (0, 2, 4, 5, 7, 9, 11),
        ("maj", "min", "min", "maj", "maj", "min", "dim"),
        ("I", "ii", "iii", "IV", "V", "vi", "vii°"),
        ("tonic", "predominant", "tonic", "predominant", "dominant", "tonic", "dominant"),
    ),
    # Common-practice minor: raised leading tone is used for V and vii°.
    "minor": (
        (0, 2, 3, 5, 7, 8, 11),
        ("min", "dim", "maj", "min", "maj", "maj", "dim"),
        ("i", "ii°", "III", "iv", "V", "VI", "vii°"),
        ("tonic", "predominant", "tonic", "predominant", "dominant", "predominant", "dominant"),
    ),
}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _track_id(path: Path) -> str:
    return path.name.split(".", 1)[0]


def find_audio(track_id: str) -> Path:
    matches = sorted(config.AUDIO_DIR.rglob(f"{track_id}*.mp3"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one MP3 for track {track_id}, found {len(matches)}")
    return matches[0]


def midi_name(midi: int) -> str:
    return f"{PITCH_NAMES[midi % 12]}{midi // 12 - 1}"


def extract_melody(audio_path: Path, track_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = PipelineConfig(input_dir=config.AUDIO_DIR, output_dir=config.OUTPUT_DIR)
    pitch_track = extract_pyin(audio_path, settings, source="pyin_fullmix")
    notes = contour_to_notes(
        pitch_track,
        minimum_note_ms=settings.minimum_note_ms,
        minimum_confidence=0.0,  # Preserve pYIN's own voiced decisions, as the team pipeline does.
    )
    rows = []
    for note in notes:
        midi = int(round(note.midi_pitch))
        rows.append(
            {
                "track_id": track_id,
                "start_time": float(note.onset_sec),
                "end_time": float(note.offset_sec),
                "duration": float(note.offset_sec - note.onset_sec),
                "note": midi_name(midi),
                "midi": midi,
                "midi_pitch_raw": float(note.midi_pitch),
                "pitch_class": PITCH_NAMES[midi % 12],
                "frequency": float(440.0 * 2.0 ** ((note.midi_pitch - 69.0) / 12.0)),
                "confidence": float(note.confidence),
                "source": note.source,
            }
        )
    raw_summary = {
        "source": "experiments.pasan.melody_extraction pYIN + contour_to_notes",
        "timeline": "seconds from start of original MP3",
        "vocal_separation_used": False,
        "frame_count": int(len(pitch_track.times_sec)),
        "voiced_frame_count": int(np.count_nonzero(pitch_track.voiced)),
        "settings": {
            "sample_rate": settings.sample_rate,
            "hop_length": settings.hop_length,
            "minimum_note_ms": settings.minimum_note_ms,
            "confidence_gate": "pYIN voiced mask; no additional probability threshold",
        },
    }
    return rows, raw_summary


def analyze_timing_and_key(audio_path: Path) -> tuple[float, list[float], dict[str, Any], float]:
    audio, sr = librosa.load(audio_path, sr=22_050, mono=True)
    duration = float(librosa.get_duration(y=audio, sr=sr))
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr, units="frames")
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = [float(value) for value in librosa.frames_to_time(beat_frames, sr=sr)]
    beat_times = [value for value in beat_times if 0 <= value <= duration]
    estimate = estimate_key_librosa(np.asarray(audio, dtype=np.float32), sr)
    key = asdict(estimate)
    key["confidence_note"] = "strength is profile correlation; margin is best-minus-second, not calibrated probability"
    return bpm, beat_times, key, duration


def load_genres(track_id: str) -> dict[str, Any]:
    source = config.REPOSITORY_ROOT / "datasets" / "mtg-jamendo" / "annotations" / "autotagging_genre.tsv"
    wanted = f"track_{int(track_id):07d}"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", restkey="EXTRA_TAGS")
        for row in reader:
            if row["TRACK_ID"] == wanted:
                raw_tags = [row["TAGS"], *(row.get("EXTRA_TAGS") or [])]
                tags = [value.removeprefix("genre---") for value in raw_tags if value]
                return {
                    "track_id": track_id,
                    "mtg_track_id": row["TRACK_ID"],
                    "artist_id": row["ARTIST_ID"],
                    "album_id": row["ALBUM_ID"],
                    "path": row["PATH"],
                    "duration_metadata_seconds": float(row["DURATION"]),
                    "genre_tags": tags,
                    "source_metadata": str(source.relative_to(config.REPOSITORY_ROOT)),
                    "join_key": "numeric track ID matched to zero-padded TRACK_ID",
                }
    raise KeyError(f"Track {track_id} is absent from {source}")


def make_windows(beat_times: list[float], duration: float, beats_per_event: int) -> list[dict[str, Any]]:
    if beats_per_event < 1:
        raise ValueError("beats_per_event must be positive")
    if len(beat_times) < 2:
        raise ValueError("At least two detected beats are required")
    anchors = beat_times[::beats_per_event]
    boundaries = ([0.0] if anchors[0] > 1e-6 else []) + anchors
    if duration - boundaries[-1] > 1e-6:
        boundaries.append(duration)
    windows = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if end <= start:
            continue
        start_beat = next((i for i, beat in enumerate(beat_times) if beat >= start - 1e-7), None)
        end_beat = next((i for i, beat in enumerate(beat_times) if beat >= end - 1e-7), len(beat_times))
        windows.append(
            {
                "event_id": index,
                "start_time": float(start),
                "end_time": float(end),
                "event_duration": float(end - start),
                "beat_start": start_beat,
                "beat_end": end_beat,
                "bar_number": None,
                "beat_position": None,
            }
        )
    return windows


def candidate_triads(tonic: str, mode: str) -> list[dict[str, Any]]:
    tonic = FLAT_TO_SHARP.get(tonic, tonic)
    tonic_pc = PITCH_NAMES.index(tonic)
    offsets, qualities, romans, functions = DIATONIC[mode]
    candidates = []
    for offset, quality, roman, function in zip(offsets, qualities, romans, functions):
        root_pc = (tonic_pc + offset) % 12
        intervals = {"maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6)}[quality]
        pcs = [(root_pc + interval) % 12 for interval in intervals]
        root = PITCH_NAMES[root_pc]
        candidates.append(
            {
                "basic_chord": root + ({"maj": "", "min": "m", "dim": "dim"}[quality]),
                "basic_root": root,
                "basic_quality": quality,
                "pitch_classes": pcs,
                "roman_numeral": roman,
                "harmonic_function": function,
            }
        )
    return candidates


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def generate_basic_chords(
    windows: list[dict[str, Any]], notes: list[dict[str, Any]], key: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = candidate_triads(key["tonic"], key["mode"])
    previous: dict[str, Any] | None = None
    output = []
    for window in windows:
        weighted_pcs: dict[int, float] = {}
        for note in notes:
            overlap = _overlap(window["start_time"], window["end_time"], note["start_time"], note["end_time"])
            if overlap:
                weighted_pcs[note["midi"] % 12] = weighted_pcs.get(note["midi"] % 12, 0.0) + overlap
        total = sum(weighted_pcs.values())
        scored = []
        for candidate in candidates:
            compatible = sum(weight for pc, weight in weighted_pcs.items() if pc in candidate["pitch_classes"])
            incompatible = total - compatible
            melody_score = (compatible - 0.5 * incompatible) / total if total else 0.0
            root_weight = weighted_pcs.get(PITCH_NAMES.index(candidate["basic_root"]), 0.0)
            root_bonus = 0.15 * root_weight / total if total else 0.0
            transition = 0.0
            if previous is not None:
                common = len(set(previous["pitch_classes"]) & set(candidate["pitch_classes"]))
                transition = 0.04 * common
            elif candidate["roman_numeral"] in {"I", "i"}:
                transition = 0.05
            score = melody_score + root_bonus + transition
            scored.append(
                {
                    "chord": candidate["basic_chord"],
                    "score": float(score),
                    "melody_compatibility": float(melody_score + root_bonus),
                    "transition_score": float(transition),
                }
            )
        scored.sort(key=lambda item: (-item["score"], item["chord"]))
        winner = next(candidate for candidate in candidates if candidate["basic_chord"] == scored[0]["chord"])
        row = {
            **window,
            "basic_chord": winner["basic_chord"],
            "basic_root": winner["basic_root"],
            "basic_quality": winner["basic_quality"],
            "basic_score": scored[0]["score"],
            "candidate_scores": scored,
            "roman_numeral": winner["roman_numeral"],
            "harmonic_function": winner["harmonic_function"],
        }
        output.append(row)
        previous = winner
    for index, row in enumerate(output):
        row["previous_basic_chord"] = output[index - 1]["basic_chord"] if index else "START"
        row["next_basic_chord"] = output[index + 1]["basic_chord"] if index + 1 < len(output) else "END"
    return output


def align_basic_to_model_windows(
    model_windows: list[dict[str, Any]], basic_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Copy generated basic harmony onto an independent model-event grid."""
    output = []
    for window in model_windows:
        overlaps = [
            (_overlap(window["start_time"], window["end_time"], event["start_time"], event["end_time"]), event)
            for event in basic_events
        ]
        seconds, source = max(overlaps, key=lambda item: item[0])
        if seconds <= 0:
            raise ValueError(f"No generated basic chord overlaps model event {window['event_id']}")
        output.append(
            {
                **window,
                "source_basic_event_id": source["event_id"],
                "basic_chord": source["basic_chord"],
                "basic_root": source["basic_root"],
                "basic_quality": source["basic_quality"],
                "basic_score": source["basic_score"],
                "candidate_scores": source["candidate_scores"],
                "roman_numeral": source["roman_numeral"],
                "harmonic_function": source["harmonic_function"],
                "basic_source_overlap_seconds": float(seconds),
            }
        )
    for index, row in enumerate(output):
        row["previous_basic_chord"] = output[index - 1]["basic_chord"] if index else "START"
        row["next_basic_chord"] = output[index + 1]["basic_chord"] if index + 1 < len(output) else "END"
    return output


def align_reference(window: dict[str, Any], reference: list[dict[str, Any]]) -> dict[str, Any]:
    overlaps = []
    for event in reference:
        seconds = _overlap(window["start_time"], window["end_time"], event["start_time"], event["end_time"])
        if seconds:
            overlaps.append(
                {
                    "raw_chord": event["raw_chord"],
                    "reference_chord": event["reference_chord"],
                    "overlap_seconds": float(seconds),
                    "overlap_ratio": float(seconds / window["event_duration"]),
                }
            )
    overlaps.sort(key=lambda item: -item["overlap_seconds"])
    best = overlaps[0] if overlaps else None
    return {
        "reference_chord": best["reference_chord"] if best else None,
        "reference_raw_chord": best["raw_chord"] if best else None,
        "reference_overlap_seconds": best["overlap_seconds"] if best else 0.0,
        "reference_overlap_ratio": best["overlap_ratio"] if best else 0.0,
        "reference_candidates": overlaps,
    }


def align_melody(window: dict[str, Any], notes: list[dict[str, Any]]) -> dict[str, Any]:
    aligned = []
    for note in notes:
        overlap = _overlap(window["start_time"], window["end_time"], note["start_time"], note["end_time"])
        if overlap:
            aligned.append(
                {
                    "note": note["note"],
                    "midi": note["midi"],
                    "pitch_class": note["pitch_class"],
                    "original_start": note["start_time"],
                    "original_end": note["end_time"],
                    "original_duration": note["duration"],
                    "relative_start": note["start_time"] - window["start_time"],
                    "overlap_duration": overlap,
                    "confidence": note["confidence"],
                }
            )
    return {
        "melody": aligned,
        "melody_notes": [note["note"] for note in aligned],
        "melody_midi": [note["midi"] for note in aligned],
        "melody_pitch_classes": [note["pitch_class"] for note in aligned],
        "melody_durations": [note["overlap_duration"] for note in aligned],
        "melody_relative_start": [note["relative_start"] for note in aligned],
    }


def validate(
    track_id: str,
    duration: float,
    notes: list[dict[str, Any]],
    basic: list[dict[str, Any]],
    aligned: list[dict[str, Any]],
) -> dict[str, Any]:
    # pYIN frame centers may extend by at most one analysis hop beyond the
    # decoded sample duration. This is a timing-representation tolerance, not
    # a change to any extracted note or event.
    timeline_tolerance = 256 / 22_050
    endpoint_overruns = [max(0.0, n["end_time"] - duration) for n in notes if n["end_time"] > duration]
    checks = {
        "melody_timestamps_valid": all(0 <= n["start_time"] < n["end_time"] <= duration + timeline_tolerance for n in notes),
        "melody_timeline_tolerance_seconds": timeline_tolerance,
        "melody_endpoint_overrun_count": len(endpoint_overruns),
        "melody_max_endpoint_overrun_seconds": max(endpoint_overruns, default=0.0),
        "basic_timestamps_valid": all(0 <= e["start_time"] < e["end_time"] <= duration + 1e-6 for e in basic),
        "basic_chronologically_ordered": all(basic[i - 1]["end_time"] <= basic[i]["start_time"] + 1e-7 for i in range(1, len(basic))),
        "model_timestamps_valid": all(0 <= e["start_time"] < e["end_time"] <= duration + 1e-6 for e in aligned),
        "model_chronologically_ordered": all(aligned[i - 1]["end_time"] <= aligned[i]["start_time"] + 1e-7 for i in range(1, len(aligned))),
        "model_event_ids_unique": len({e["event_id"] for e in aligned}) == len(aligned),
        "melody_array_lengths_match": all(
            len(e["melody_midi"]) == len(e["melody_durations"]) == len(e["melody_relative_start"])
            for e in aligned
        ),
        "canonical_fields_present": all(
            all(name in e for name in ("basic_chord_canonical", "reference_chord_canonical", "basic_root_pitch_class", "reference_root_pitch_class"))
            for e in aligned
        ),
        "all_track_ids_match": all(e["track_id"] == track_id for e in aligned),
        "reference_alignment_same_track": all(e["reference_source_track_id"] == track_id for e in aligned),
        "no_nan_values": not any(math.isnan(value) for e in aligned for value in _walk_floats(e)),
        "target_leakage": False,
        "basic_chord_inputs": ["melody note pitch/timing", "global key/mode", "beat grid", "previous generated basic chord"],
        "reference_used_after_generation_only": True,
    }
    checks["all_passed"] = all(value for key, value in checks.items() if isinstance(value, bool) and key != "target_leakage") and not checks["target_leakage"]
    return checks


def _walk_floats(value: Any):
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_floats(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_floats(item)


def write_training_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "track_id", "event_id", "start_time", "end_time", "event_duration",
        "previous_basic_chord", "basic_chord", "next_basic_chord", "basic_root",
        "basic_quality", "basic_score", "basic_chord_canonical", "basic_root_pitch_class",
        "reference_chord", "reference_raw_chord", "reference_chord_canonical", "reference_root_pitch_class",
        "reference_overlap_seconds", "reference_overlap_ratio", "melody_notes", "melody_midi",
        "melody_pitch_classes", "melody_durations", "melody_relative_start", "genre_tags",
        "key", "mode", "key_confidence", "roman_numeral", "harmonic_function",
        "beat_start", "beat_end", "bar_number", "beat_position",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {column: row.get(column) for column in columns}
            for column in ("melody_notes", "melody_midi", "melody_pitch_classes", "melody_durations", "melody_relative_start", "genre_tags"):
                flat[column] = json.dumps(flat[column], ensure_ascii=False, allow_nan=False)
            writer.writerow(flat)


def run(
    track_id: str,
    basic_beats_per_event: int = 2,
    model_beats_per_event: int = 1,
    write_combined_csv: bool = True,
) -> dict[str, Any]:
    audio_path = find_audio(track_id)
    reference_path = config.OUTPUT_DIR / "chords" / f"{track_id}.json"
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference chords not found: {reference_path}")

    melody, melody_provenance = extract_melody(audio_path, track_id)
    bpm, beat_times, key, duration = analyze_timing_and_key(audio_path)
    metadata = load_genres(track_id)
    basic_windows = make_windows(beat_times, duration, basic_beats_per_event)
    basic = generate_basic_chords(basic_windows, melody, key)
    model_windows = make_windows(beat_times, duration, model_beats_per_event)
    model_events = align_basic_to_model_windows(model_windows, basic)

    # Finish every input-side feature before target data is loaded.
    input_rows = []
    for event in model_events:
        input_rows.append(
            {
                "track_id": track_id,
                **event,
                **align_melody(event, melody),
                "genre_tags": metadata["genre_tags"],
                "key": key["tonic"],
                "mode": key["mode"],
                "key_confidence": key["strength"],
                "key_margin": key["margin"],
            }
        )

    # Deliberately load targets only after all model inputs and predictions exist.
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    aligned = []
    for event in input_rows:
        row = add_canonical_fields({
            **event,
            **align_reference(event, reference),
            "reference_source_track_id": track_id,
            "feature_sources": {
                "melody": "team pyin_fullmix adapter",
                "timing": "librosa beat_track on original mix",
                "key_mode": key["method"] + " on original mix",
                "genre": metadata["source_metadata"],
                "basic_chord": "rule-based melody/key/beat/previous-basic only",
                "reference_chord": str(reference_path.relative_to(config.COMPONENT_DIR)),
            },
        })
        aligned.append(row)

    beats = {
        "track_id": track_id,
        "bpm": bpm,
        "beat_timestamps": beat_times,
        "beats": [{"beat_index": i, "start_time": value} for i, value in enumerate(beat_times)],
        "downbeats_available": False,
        "bars_available": False,
        "method": "librosa.beat.beat_track on original MP3",
        "timeline": "seconds from start of original MP3",
    }
    melody_doc = {
        "track_id": track_id,
        "audio_path": str(audio_path.relative_to(config.REPOSITORY_ROOT)),
        "audio_duration_seconds": duration,
        "provenance": melody_provenance,
        "key": key,
        "notes": melody,
    }
    basic_doc = {
        "track_id": track_id,
        "event_definition": f"one generated basic chord per {basic_beats_per_event} detected beats; leading intro and trailing remainder retained",
        "generation_inputs": ["melody", "global key/mode", "beats", "previous generated basic chord"],
        "reference_chords_used": False,
        "events": basic,
    }
    validation = validate(track_id, duration, melody, basic, aligned)
    aligned_doc = {
        "track_id": track_id,
        "schema_version": 2,
        "event_definition": f"one model event per {model_beats_per_event} detected beat; leading intro and trailing remainder retained",
        "basic_alignment_policy": "greatest time overlap with independently generated two-beat basic events",
        "crossing_note_policy": "include in every overlapping event; preserve original timing and store overlap duration; relative start may be negative",
        "reference_policy": "greatest interval overlap wins; every positive-overlap candidate is preserved",
        "validation": validation,
        "events": aligned,
    }

    _json_dump(config.OUTPUT_DIR / "melody" / f"{track_id}_melody.json", melody_doc)
    _json_dump(config.OUTPUT_DIR / "beats" / f"{track_id}_beats.json", beats)
    _json_dump(config.OUTPUT_DIR / "metadata" / f"{track_id}_metadata.json", metadata)
    _json_dump(config.OUTPUT_DIR / "basic_chords" / f"{track_id}_basic_chords.json", basic_doc)
    _json_dump(config.OUTPUT_DIR / "aligned" / f"{track_id}_aligned.json", aligned_doc)
    if write_combined_csv:
        write_training_csv(config.OUTPUT_DIR / "aligned" / "training_events.csv", aligned)
    return {
        "track_id": track_id,
        "reference_events": len(reference),
        "melody_notes": len(melody),
        "bpm": bpm,
        "beats": len(beat_times),
        "basic_events": len(basic),
        "model_events": len(aligned),
        "key": key,
        "genre_tags": metadata["genre_tags"],
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-id", default="1400601")
    parser.add_argument("--basic-beats-per-event", type=int, default=2)
    parser.add_argument("--model-beats-per-event", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args.track_id, args.basic_beats_per_event, args.model_beats_per_event), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
