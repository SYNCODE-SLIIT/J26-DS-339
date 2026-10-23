"""Build and validate the production one-beat pilot before atomic CSV replacement."""

from __future__ import annotations

import argparse
import csv
import json
import math
from typing import Any

from experiments.rahul import config
from experiments.rahul.src.build_training_events import run, write_training_csv
from experiments.rahul.src.canonicalize_chords import canonicalize_chord
from experiments.rahul.src.normalize_chords import parse_chord_label


DEFAULT_TRACK_IDS = (
    "1012600", "1012601", "1012602", "1028800", "1028801",
    "1028802", "1030101", "1030102", "1051200", "1400601",
)


def _csv_row_count(path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _has_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_nonfinite(item) for item in value)
    return False


def _refined_classes(rows: list[dict[str, Any]]) -> set[str]:
    labels = set()
    for row in rows:
        parsed = parse_chord_label(row["reference_raw_chord"])
        if row["reference_chord_canonical"] != "N" and (
            parsed.bass_degree is not None or parsed.quality not in {"maj", "min", "dim"}
        ):
            labels.add(row["reference_chord_canonical"])
    return labels


def validate_candidate(rows, track_ids, successes, failures) -> dict[str, Any]:
    by_track = {track_id: [row for row in rows if row["track_id"] == track_id] for track_id in track_ids}
    checks = {
        "exactly_10_requested_tracks": len(track_ids) == 10 and len(set(track_ids)) == 10,
        "all_10_tracks_succeeded": len(successes) == 10 and not failures,
        "exactly_10_output_tracks": set(by_track) == {row["track_id"] for row in rows} and all(by_track.values()),
        "no_duplicate_event_ids_within_tracks": all(
            len({row["event_id"] for row in events}) == len(events) for events in by_track.values()
        ),
        "chronological_ordering": all(
            all(events[i - 1]["end_time"] <= events[i]["start_time"] + 1e-7 for i in range(1, len(events)))
            for events in by_track.values()
        ),
        "no_event_crosses_tracks": all(all(row["track_id"] == track_id for row in events) for track_id, events in by_track.items()),
        "all_per_track_validations_passed": all(item["validation"]["all_passed"] for item in successes),
        "basic_chords_independently_generated": all(
            item["validation"]["reference_used_after_generation_only"] and not item["validation"]["target_leakage"]
            for item in successes
        ),
        "canonical_fields_present": all(
            all(name in row for name in ("basic_chord_canonical", "reference_chord_canonical", "basic_root_pitch_class", "reference_root_pitch_class"))
            for row in rows
        ),
        "original_chord_fields_preserved": all(
            canonicalize_chord(row["basic_chord"])[0] == row["basic_chord_canonical"]
            and canonicalize_chord(row["reference_chord"])[0] == row["reference_chord_canonical"]
            for row in rows
        ),
        "melody_array_lengths_consistent": all(
            len(row["melody_midi"]) == len(row["melody_durations"]) == len(row["melody_relative_start"])
            for row in rows
        ),
        "genre_key_mode_consistent_within_track": all(
            len({tuple(row["genre_tags"]) for row in events}) == 1
            and len({row["key"] for row in events}) == 1
            and len({row["mode"] for row in events}) == 1
            for events in by_track.values() if events
        ),
        "no_nan_or_infinite_values": not _has_nonfinite(rows),
        "reproduces_expected_event_count_2964": len(rows) == 2964,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def build_pilot(track_ids: list[str]) -> dict[str, Any]:
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("Pilot track IDs must be unique")

    csv_path = config.OUTPUT_DIR / "aligned" / "training_events.csv"
    old_event_count = _csv_row_count(csv_path)
    successes, failures, combined_events = [], [], []
    for index, track_id in enumerate(track_ids, start=1):
        print(f"[{index}/{len(track_ids)}] processing track {track_id}", flush=True)
        try:
            summary = run(track_id, basic_beats_per_event=2, model_beats_per_event=1, write_combined_csv=False)
            aligned_path = config.OUTPUT_DIR / "aligned" / f"{track_id}_aligned.json"
            combined_events.extend(json.loads(aligned_path.read_text(encoding="utf-8"))["events"])
            successes.append(summary)
            print(
                f"completed {track_id}: genres={summary['genre_tags']} key={summary['key']['tonic']} "
                f"mode={summary['key']['mode']} basic_events={summary['basic_events']} "
                f"model_events={summary['model_events']}", flush=True,
            )
        except Exception as exc:
            failure = {"track_id": track_id, "error_type": type(exc).__name__, "reason": str(exc)}
            failures.append(failure)
            print(f"failed {track_id}: {failure['error_type']}: {failure['reason']}", flush=True)

    combined_events.sort(key=lambda row: (row["track_id"], row["start_time"], row["event_id"]))
    validation = validate_candidate(combined_events, track_ids, successes, failures)
    canonical_targets = {row["reference_chord_canonical"] for row in combined_events}
    refined_targets = _refined_classes(combined_events)
    experiment_path = config.OUTPUT_DIR / "experiments" / "one_beat_alignment" / "comparison_report.json"
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))["one_beat"]
    expected_targets = set(experiment["unique_canonical_targets"])
    experiment_comparison = {
        "event_count_matches": len(combined_events) == experiment["event_count"],
        "canonical_target_count_matches": len(canonical_targets) == experiment["unique_canonical_target_count"],
        "canonical_target_set_matches": canonical_targets == expected_targets,
        "missing_from_production": sorted(expected_targets - canonical_targets),
        "additional_in_production": sorted(canonical_targets - expected_targets),
        "refined_class_count_matches_32": len(refined_targets) == 32,
        "recovered_classes_present": all(label in canonical_targets for label in ("A#m/b7", "Bm/b7", "D/b7")),
    }
    experiment_comparison["all_passed"] = all(
        value for key, value in experiment_comparison.items() if isinstance(value, bool) and key != "all_passed"
    )
    validation["matches_one_beat_experiment"] = experiment_comparison["all_passed"]
    validation["all_passed"] = all(
        value for key, value in validation.items() if isinstance(value, bool) and key != "all_passed"
    )

    report = {
        "migration": "two-beat model events to one-beat model events",
        "basic_generation_resolution_beats": 2,
        "model_event_resolution_beats": 1,
        "old_event_count": old_event_count,
        "new_event_count": len(combined_events),
        "track_count": len({row["track_id"] for row in combined_events}),
        "basic_vocabulary_count": len({row["basic_chord"] for row in combined_events}),
        "reference_vocabulary_count": len({row["reference_chord"] for row in combined_events}),
        "canonical_target_vocabulary_count": len(canonical_targets),
        "refined_target_class_count": len(refined_targets),
        "refined_target_classes": sorted(refined_targets),
        "N_event_count": sum(row["reference_chord_canonical"] == "N" for row in combined_events),
        "empty_melody_event_count": sum(not row["melody_midi"] for row in combined_events),
        "event_count_per_track": {track_id: sum(row["track_id"] == track_id for row in combined_events) for track_id in track_ids},
        "successful_tracks": successes,
        "failed_tracks": failures,
        "validation": validation,
        "experiment_comparison": experiment_comparison,
        "combined_csv": str(csv_path.relative_to(config.COMPONENT_DIR)),
        "production_csv_replaced": False,
    }
    if not validation["all_passed"]:
        raise RuntimeError("Candidate validation failed; production CSV was not replaced: " + json.dumps(report))

    candidate_path = csv_path.with_name("training_events.candidate.csv")
    write_training_csv(candidate_path, combined_events)
    if _csv_row_count(candidate_path) != len(combined_events):
        raise RuntimeError("Candidate CSV row count changed during serialization; production CSV was not replaced")
    candidate_path.replace(csv_path)
    report["production_csv_replaced"] = True
    report_path = config.OUTPUT_DIR / "aligned" / "production_migration_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-ids", nargs="+", default=list(DEFAULT_TRACK_IDS))
    args = parser.parse_args()
    print(json.dumps(build_pilot(args.track_ids), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
