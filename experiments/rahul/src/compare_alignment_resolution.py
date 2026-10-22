"""Compare production two-beat targets with an isolated one-beat alignment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.rahul import config
from experiments.rahul.src.build_training_events import align_melody, align_reference, make_windows
from experiments.rahul.src.canonicalize_chords import canonicalize_chord
from experiments.rahul.src.normalize_chords import parse_chord_label


TRACK_IDS = (
    "1012600", "1012601", "1012602", "1028800", "1028801",
    "1028802", "1030101", "1030102", "1051200", "1400601",
)
QUALITY_CATEGORIES = (
    "major_triads", "minor_triads", "diminished_triads", "dominant7",
    "major7", "minor7", "sus2", "sus4", "add9", "other_extended_chords", "N",
)


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def category(raw_chord: str) -> str:
    parsed = parse_chord_label(raw_chord)
    if raw_chord == "N":
        return "N"
    return {
        "maj": "major_triads",
        "min": "minor_triads",
        "dim": "diminished_triads",
        "7": "dominant7",
        "maj7": "major7",
        "min7": "minor7",
        "sus2": "sus2",
        "sus4": "sus4",
        "add9": "add9",
    }.get(parsed.quality, "other_extended_chords")


def is_refined(raw_chord: str) -> bool:
    parsed = parse_chord_label(raw_chord)
    return raw_chord != "N" and (
        parsed.bass_degree is not None or parsed.quality not in {"maj", "min", "dim"}
    )


def canonical_reference(raw_chord: str) -> str:
    display = parse_chord_label(raw_chord).reference_chord
    return canonicalize_chord(display)[0]


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(category(event["reference_raw_chord"]) for event in events)
    inversions = sum(
        parse_chord_label(event["reference_raw_chord"]).bass_degree is not None
        for event in events
    )
    labels = sorted({event["reference_chord_canonical"] for event in events})
    return {
        "event_count": len(events),
        "category_counts": {name: counts[name] for name in QUALITY_CATEGORIES},
        "inversions": inversions,
        "unique_canonical_target_count": len(labels),
        "unique_canonical_targets": labels,
    }


def basic_for_window(window: dict[str, Any], basic_events: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        (overlap(window["start_time"], window["end_time"], event["start_time"], event["end_time"]), event)
        for event in basic_events
    ]
    amount, winner = max(candidates, key=lambda item: (item[0], -item[1]["event_id"]))
    if amount <= 0:
        raise RuntimeError(f"No two-beat basic event overlaps {window}")
    return winner


def build_one_beat_track(track_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    outputs = config.OUTPUT_DIR
    beats = json.loads((outputs / "beats" / f"{track_id}_beats.json").read_text(encoding="utf-8"))
    melody_doc = json.loads((outputs / "melody" / f"{track_id}_melody.json").read_text(encoding="utf-8"))
    basic_doc = json.loads((outputs / "basic_chords" / f"{track_id}_basic_chords.json").read_text(encoding="utf-8"))
    production_doc = json.loads((outputs / "aligned" / f"{track_id}_aligned.json").read_text(encoding="utf-8"))
    reference = json.loads((outputs / "chords" / f"{track_id}.json").read_text(encoding="utf-8"))

    windows = make_windows(
        beats["beat_timestamps"], melody_doc["audio_duration_seconds"], beats_per_event=1
    )
    one_beat = []
    for window in windows:
        basic = basic_for_window(window, basic_doc["events"])
        target = align_reference(window, reference)
        canonical, reference_pc = canonicalize_chord(target["reference_chord"])
        basic_canonical, basic_pc = canonicalize_chord(basic["basic_chord"])
        one_beat.append(
            {
                "track_id": track_id,
                **window,
                "source_two_beat_basic_event_id": basic["event_id"],
                "basic_chord": basic["basic_chord"],
                "basic_chord_canonical": basic_canonical,
                "basic_root_pitch_class": basic_pc,
                "basic_root": basic["basic_root"],
                "basic_quality": basic["basic_quality"],
                "basic_score": basic["basic_score"],
                "roman_numeral": basic["roman_numeral"],
                "harmonic_function": basic["harmonic_function"],
                **target,
                "reference_chord_canonical": canonical,
                "reference_root_pitch_class": reference_pc,
                **align_melody(window, melody_doc["notes"]),
            }
        )
    return one_beat, production_doc["events"], reference


def raw_sequence(reference: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [
        {
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "raw_chord": event["raw_chord"],
            "reference_chord": event["reference_chord"],
            "overlap_seconds": overlap(start, end, event["start_time"], event["end_time"]),
        }
        for event in reference
        if overlap(start, end, event["start_time"], event["end_time"]) > 0
    ]


def compare() -> dict[str, Any]:
    all_one: list[dict[str, Any]] = []
    all_two: list[dict[str, Any]] = []
    all_raw: list[dict[str, Any]] = []
    lost_examples = []
    lost_after_one = []

    experiment_dir = config.OUTPUT_DIR / "experiments" / "one_beat_alignment"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for track_id in TRACK_IDS:
        one_beat, two_beat, reference = build_one_beat_track(track_id)
        all_one.extend(one_beat)
        all_two.extend(two_beat)
        all_raw.extend(reference)
        (experiment_dir / f"{track_id}_one_beat.json").write_text(
            json.dumps(one_beat, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        for raw_event in reference:
            if not is_refined(raw_event["raw_chord"]):
                continue
            overlapping_two = [
                event for event in two_beat
                if overlap(event["start_time"], event["end_time"], raw_event["start_time"], raw_event["end_time"]) > 0
            ]
            retained_two = any(event["reference_raw_chord"] == raw_event["raw_chord"] for event in overlapping_two)
            overlapping_one = [
                event for event in one_beat
                if overlap(event["start_time"], event["end_time"], raw_event["start_time"], raw_event["end_time"]) > 0
            ]
            retained_one = any(event["reference_raw_chord"] == raw_event["raw_chord"] for event in overlapping_one)
            if not retained_two:
                context = {
                    "track_id": track_id,
                    "refined_raw_event": raw_event,
                    "duration": raw_event["end_time"] - raw_event["start_time"],
                    "overlapping_two_beat_events": [
                        {
                            "time_range": [event["start_time"], event["end_time"]],
                            "basic_chord": event["basic_chord"],
                            "raw_reference_sequence": raw_sequence(reference, event["start_time"], event["end_time"]),
                            "chosen_target": event["reference_chord_canonical"],
                        }
                        for event in overlapping_two
                    ],
                    "one_beat_chosen_targets": [
                        {
                            "time_range": [event["start_time"], event["end_time"]],
                            "basic_chord": event["basic_chord"],
                            "chosen_target": event["reference_chord_canonical"],
                            "chosen_raw_chord": event["reference_raw_chord"],
                        }
                        for event in overlapping_one
                    ],
                    "recovered_at_one_beat": retained_one,
                }
                lost_examples.append(context)
                if not retained_one:
                    lost_after_one.append(context)

    raw_counts = Counter(category(event["raw_chord"]) for event in all_raw)
    raw_canonical = sorted({canonical_reference(event["raw_chord"]) for event in all_raw})
    raw_refined = [event for event in all_raw if is_refined(event["raw_chord"])]
    report = {
        "track_ids": list(TRACK_IDS),
        "method": {
            "one_beat_boundaries": "detected beat grid only",
            "basic_chords": "unchanged two-beat basic chord repeated by greatest time overlap",
            "reference_alignment": "greatest interval overlap",
            "canonicalization": "root spelling only; full quality and inversion retained",
            "production_files_modified": False,
        },
        "raw_reference": {
            "event_count": len(all_raw),
            "category_counts": {name: raw_counts[name] for name in QUALITY_CATEGORIES},
            "inversions": sum(parse_chord_label(event["raw_chord"]).bass_degree is not None for event in all_raw),
            "unique_canonical_label_count": len(raw_canonical),
            "unique_canonical_labels": raw_canonical,
            "refined_event_count": len(raw_refined),
            "unique_refined_canonical_labels": sorted({canonical_reference(event["raw_chord"]) for event in raw_refined}),
        },
        "two_beat": summarize(all_two),
        "one_beat": summarize(all_one),
        "increase": {
            "events": len(all_one) - len(all_two),
            "ratio": len(all_one) / len(all_two),
            "percent": (len(all_one) / len(all_two) - 1.0) * 100.0,
        },
        "refined_raw_events_not_retained_at_two_beats": len(lost_examples),
        "of_those_recovered_at_one_beat": sum(item["recovered_at_one_beat"] for item in lost_examples),
        "refined_raw_events_still_not_retained_at_one_beat": len(lost_after_one),
        "lost_refinement_examples": lost_examples,
    }
    (experiment_dir / "comparison_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2, ensure_ascii=False, allow_nan=False))
