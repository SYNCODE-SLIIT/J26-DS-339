"""Targeted 2-beat versus 1-beat survival test for rare LV-Chordia labels."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.rahul import config
from experiments.rahul.src.build_training_events import (
    align_reference,
    analyze_timing_and_key,
    extract_melody,
    find_audio,
    generate_basic_chords,
    make_windows,
)
from experiments.rahul.src.canonicalize_chords import canonicalize_chord
from experiments.rahul.src.compare_alignment_resolution import basic_for_window, overlap


RARE_QUALITIES = {"sus2", "sus4", "add9", "dim", "dim7", "hdim7", "9", "maj9", "min9", "11", "13"}


def discover_rare_events() -> tuple[dict[str, list[dict[str, Any]]], int]:
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files = [path for path in (config.OUTPUT_DIR / "chords").glob("*.json") if path.stem.isdigit()]
    for path in sorted(files):
        for event in json.loads(path.read_text(encoding="utf-8")):
            if event.get("quality") in RARE_QUALITIES:
                by_track[event["track_id"]].append(event)
    return dict(by_track), len(files)


def selected_targets(events: list[dict[str, Any]], rare: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "raw_chord": event["reference_raw_chord"],
            "canonical_chord": event["reference_chord_canonical"],
            "overlap_seconds": overlap(event["start_time"], event["end_time"], rare["start_time"], rare["end_time"]),
        }
        for event in events
        if overlap(event["start_time"], event["end_time"], rare["start_time"], rare["end_time"]) > 0
    ]


def build_targets(windows: list[dict[str, Any]], reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for window in windows:
        aligned = align_reference(window, reference)
        canonical = canonicalize_chord(aligned["reference_chord"])[0]
        output.append({**window, **aligned, "reference_chord_canonical": canonical})
    return output


def run() -> dict[str, Any]:
    rare_by_track, searched_files = discover_rare_events()
    experiment_dir = config.OUTPUT_DIR / "experiments" / "rare_quality_alignment"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    survival_rows = []
    track_summaries = []

    for index, (track_id, rare_events) in enumerate(sorted(rare_by_track.items()), start=1):
        print(f"[{index}/{len(rare_by_track)}] preparing {track_id}", flush=True)
        audio_path = find_audio(track_id)
        melody, melody_provenance = extract_melody(audio_path, track_id)
        bpm, beat_times, key, duration = analyze_timing_and_key(audio_path)
        two_windows = make_windows(beat_times, duration, beats_per_event=2)
        two_basic = generate_basic_chords(two_windows, melody, key)

        # Load targets only after the independent two-beat basic harmony exists.
        reference_path = config.OUTPUT_DIR / "chords" / f"{track_id}.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        two_targets = build_targets(two_basic, reference)

        one_windows = make_windows(beat_times, duration, beats_per_event=1)
        one_with_basic = []
        for window in one_windows:
            source_basic = basic_for_window(window, two_basic)
            one_with_basic.append(
                {
                    **window,
                    "basic_chord": source_basic["basic_chord"],
                    "source_two_beat_basic_event_id": source_basic["event_id"],
                }
            )
        one_targets = build_targets(one_with_basic, reference)

        for rare in rare_events:
            canonical = canonicalize_chord(rare["reference_chord"])[0]
            two_selected = selected_targets(two_targets, rare)
            one_selected = selected_targets(one_targets, rare)
            survival_rows.append(
                {
                    "track_id": track_id,
                    "start_time": rare["start_time"],
                    "end_time": rare["end_time"],
                    "duration": rare["end_time"] - rare["start_time"],
                    "quality": rare["quality"],
                    "raw_chord": rare["raw_chord"],
                    "reference_chord": rare["reference_chord"],
                    "canonical_chord": canonical,
                    "two_beat_chosen_targets": two_selected,
                    "one_beat_chosen_targets": one_selected,
                    "two_beat_survived": any(item["raw_chord"] == rare["raw_chord"] for item in two_selected),
                    "one_beat_survived": any(item["raw_chord"] == rare["raw_chord"] for item in one_selected),
                }
            )

        track_summary = {
            "track_id": track_id,
            "audio_path": str(audio_path.relative_to(config.REPOSITORY_ROOT)),
            "audio_duration_seconds": duration,
            "bpm": bpm,
            "beat_count": len(beat_times),
            "key": key,
            "melody_note_count": len(melody),
            "melody_provenance": melody_provenance,
            "two_beat_event_count": len(two_targets),
            "one_beat_event_count": len(one_targets),
            "rare_raw_event_count": len(rare_events),
        }
        track_summaries.append(track_summary)
        (experiment_dir / f"{track_id}_comparison.json").write_text(
            json.dumps(
                {
                    "summary": track_summary,
                    "two_beat_targets": two_targets,
                    "one_beat_targets": one_targets,
                    "rare_survival": [row for row in survival_rows if row["track_id"] == track_id],
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
        print(
            f"completed {track_id}: rare={len(rare_events)} "
            f"two={len(two_targets)} one={len(one_targets)}",
            flush=True,
        )

    quality_summary = []
    for quality in sorted(RARE_QUALITIES):
        rows = [row for row in survival_rows if row["quality"] == quality]
        quality_summary.append(
            {
                "quality": quality,
                "track_ids": sorted({row["track_id"] for row in rows}),
                "raw_events": len(rows),
                "survived_two_beats": sum(row["two_beat_survived"] for row in rows),
                "survived_one_beat": sum(row["one_beat_survived"] for row in rows),
            }
        )

    report = {
        "searched_reference_json_files": searched_files,
        "rare_track_count": len(rare_by_track),
        "rare_track_ids": sorted(rare_by_track),
        "method": {
            "production_files_modified": False,
            "boundaries": "detected beat grid only",
            "basic_generation": "existing melody/key/beat/previous-basic algorithm unchanged",
            "one_beat_basic": "copied from independently generated two-beat basic event by overlap",
            "reference_loaded_after_basic_generation": True,
            "reference_selection": "greatest interval overlap",
            "canonicalization": "root spelling only",
        },
        "quality_summary": quality_summary,
        "track_summaries": track_summaries,
        "rare_event_survival": survival_rows,
        "totals": {
            "rare_raw_events": len(survival_rows),
            "survived_two_beats": sum(row["two_beat_survived"] for row in survival_rows),
            "survived_one_beat": sum(row["one_beat_survived"] for row in survival_rows),
            "lost_two_beats": sum(not row["two_beat_survived"] for row in survival_rows),
            "lost_one_beat": sum(not row["one_beat_survived"] for row in survival_rows),
            "raw_events_by_quality": dict(sorted(Counter(row["quality"] for row in survival_rows).items())),
        },
    }
    (experiment_dir / "rare_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"quality_summary": result["quality_summary"], "totals": result["totals"]}, indent=2))
