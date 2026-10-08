"""Extract time-aligned reference chords from original mixed audio."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

from experiments.rahul import config
from experiments.rahul.src.normalize_chords import parse_chord_label
from experiments.rahul.src.utils import (
    audio_duration_seconds,
    consecutive_duplicate_count,
    discover_audio_files,
    extract_track_id,
)

FIELDS = [
    "track_id", "start_time", "end_time", "raw_chord", "root", "quality",
    "bass_degree", "reference_chord",
]


def configure_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("experiments.rahul")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(log_dir / "extraction.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def recognize(audio_path: Path, vocabulary: str) -> list[dict]:
    """Call the verified LV-Chordia API and preserve every raw label."""
    try:
        from lv_chordia.chord_recognition import chord_recognition
    except ImportError as exc:
        raise RuntimeError(
            "LV-Chordia is unavailable. Install experiments/rahul/requirements.txt; "
            "do not substitute a major/minor-only detector."
        ) from exc

    track_id = extract_track_id(audio_path)
    events = []
    for raw_event in chord_recognition(str(audio_path), chord_dict_name=vocabulary):
        raw_chord = str(raw_event["chord"])
        parsed = parse_chord_label(raw_chord)
        events.append({
            "track_id": track_id,
            "start_time": float(raw_event["start_time"]),
            "end_time": float(raw_event["end_time"]),
            "raw_chord": raw_chord,
            "root": parsed.root,
            "quality": parsed.quality,
            "bass_degree": parsed.bass_degree,
            "reference_chord": parsed.reference_chord,
        })
    return events


def validate_events(events: list[dict], audio_duration: float) -> dict:
    invalid_order = sum(event["start_time"] >= event["end_time"] for event in events)
    negative = sum(event["start_time"] < 0 for event in events)
    beyond_audio = sum(event["end_time"] > audio_duration + 0.05 for event in events)
    chronological = all(
        left["start_time"] <= right["start_time"]
        for left, right in zip(events, events[1:])
    )
    gaps = [
        right["start_time"] - left["end_time"]
        for left, right in zip(events, events[1:])
        if right["start_time"] - left["end_time"] > 0.01
    ]
    overlaps = [
        left["end_time"] - right["start_time"]
        for left, right in zip(events, events[1:])
        if left["end_time"] - right["start_time"] > 0.01
    ]
    if invalid_order or negative or beyond_audio or not chronological:
        raise ValueError(
            "Invalid timestamps: "
            f"order={invalid_order}, negative={negative}, beyond_audio={beyond_audio}, "
            f"chronological={chronological}"
        )
    durations = [event["end_time"] - event["start_time"] for event in events]
    return {
        "event_count": len(events),
        "no_chord_count": sum(event["raw_chord"] == "N" for event in events),
        "minimum_duration": min(durations, default=None),
        "maximum_duration": max(durations, default=None),
        "average_duration": sum(durations) / len(durations) if durations else None,
        "consecutive_duplicate_count": consecutive_duplicate_count(events),
        "chronologically_ordered": chronological,
        "gap_count": len(gaps),
        "overlap_count": len(overlaps),
        "maximum_gap": max(gaps, default=0.0),
        "maximum_overlap": max(overlaps, default=0.0),
    }


def quality_category(raw_chord: str) -> str:
    """Classify one raw label; every event receives exactly one category."""
    if raw_chord == "N":
        return "N"
    quality = parse_chord_label(raw_chord).quality
    direct = {
        "maj": "major", "min": "minor", "7": "dominant7", "maj7": "major7",
        "min7": "minor7", "sus2": "sus2", "sus4": "sus4", "dim": "diminished",
        "dim7": "diminished", "aug": "augmented", "add9": "add9",
    }
    if quality in direct:
        return direct[quality]
    if quality and (
        quality in {"6", "min6", "9", "maj9", "min9", "11", "13", "hdim7"}
        or quality.startswith("add")
    ):
        return "other_extended"
    return "unknown/other"


def build_vocabulary_report(
    events: list[dict], validations: dict[str, dict], audio_durations: dict[str, float]
) -> dict:
    raw_counts = Counter(event["raw_chord"] for event in events)
    quality_counts = Counter(event["quality"] for event in events if event["quality"])
    category_counts = Counter(quality_category(event["raw_chord"]) for event in events)
    total_event_duration = sum(event["end_time"] - event["start_time"] for event in events)
    no_chord_duration = sum(
        event["end_time"] - event["start_time"]
        for event in events if event["raw_chord"] == "N"
    )
    categories = [
        "major", "minor", "dominant7", "major7", "minor7", "sus2",
        "sus4", "diminished", "augmented", "add9", "other_extended",
        "unknown/other",
    ]
    classified_non_n = sum(category_counts[name] for name in categories)
    no_chord_events = category_counts["N"]
    return {
        "total_songs_processed": len(audio_durations),
        "track_ids": list(audio_durations),
        "total_audio_duration_seconds": sum(audio_durations.values()),
        "total_chord_events": len(events),
        "unique_raw_chord_labels": sorted(raw_counts),
        "unique_chord_qualities": sorted(quality_counts),
        "chord_quality_frequencies": dict(quality_counts.most_common()),
        "requested_category_counts": {name: category_counts.get(name, 0) for name in categories},
        "no_chord_event_count": no_chord_events,
        "no_chord_duration_seconds": no_chord_duration,
        "no_chord_duration_percentage": (
            100.0 * no_chord_duration / total_event_duration if total_event_duration else 0.0
        ),
        "inversion_slash_chord_count": sum(event["bass_degree"] is not None for event in events),
        "thirty_most_frequent_raw_chord_labels": raw_counts.most_common(30),
        "labels_parser_could_not_normalize": sorted({
            event["raw_chord"] for event in events
            if not parse_chord_label(event["raw_chord"]).normalized
        }),
        "uncategorized_raw_chord_labels": sorted({
            event["raw_chord"] for event in events
            if quality_category(event["raw_chord"]) == "unknown/other"
        }),
        "classification_consistency": {
            "classified_non_n_events": classified_non_n,
            "no_chord_events": no_chord_events,
            "classified_plus_n_events": classified_non_n + no_chord_events,
            "total_chord_events": len(events),
            "is_consistent": classified_non_n + no_chord_events == len(events),
        },
        "per_track_validation": validations,
    }


def write_outputs(
    track_events: list[dict], combined_events: list[dict], output_dir: Path, track_id: str
) -> tuple[Path, Path]:
    chord_dir = output_dir / "chords"
    chord_dir.mkdir(parents=True, exist_ok=True)
    json_path = chord_dir / f"{track_id}.json"
    csv_path = chord_dir / "reference_chord_events.csv"
    json_path.write_text(json.dumps(track_events, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(combined_events)
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=config.AUDIO_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--vocabulary", default=config.CHORD_VOCABULARY)
    parser.add_argument("--limit", type=int, default=config.NUMBER_OF_FILES)
    parser.add_argument("--track-id", help="Optionally select one exact numeric track ID")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.vocabulary not in config.SUPPORTED_VOCABULARIES:
        raise ValueError(f"Unsupported vocabulary: {args.vocabulary!r}")
    logger = configure_logging(args.output_dir)
    files = discover_audio_files(args.audio_dir.expanduser().resolve(), args.limit)
    if args.track_id:
        files = [
            path for path in discover_audio_files(args.audio_dir, 100_000)
            if extract_track_id(path) == args.track_id
        ][:1]
    if not files:
        raise FileNotFoundError("No matching audio files found")

    all_events: list[dict] = []
    validations: dict[str, dict] = {}
    audio_durations: dict[str, float] = {}
    for index, audio_path in enumerate(files, start=1):
        track_id = extract_track_id(audio_path)
        logger.info("[%d/%d] Recognizing track %s from %s", index, len(files), track_id, audio_path)
        duration = audio_duration_seconds(audio_path)
        events = recognize(audio_path, args.vocabulary)
        validation = validate_events(events, duration)
        validations[track_id] = validation
        audio_durations[track_id] = duration
        all_events.extend(events)
        json_path, csv_path = write_outputs(events, all_events, args.output_dir, track_id)
        logger.info(
            "Completed track=%s duration=%.3fs events=%d json=%s csv=%s validation=%s",
            track_id, duration, len(events), json_path, csv_path, validation,
        )

    report = build_vocabulary_report(all_events, validations, audio_durations)
    report_path = args.output_dir / "chords" / "vocabulary_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved vocabulary report to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

