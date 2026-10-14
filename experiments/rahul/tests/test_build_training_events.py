from experiments.rahul.src.build_training_events import (
    align_melody,
    align_basic_to_model_windows,
    align_reference,
    candidate_triads,
    generate_basic_chords,
    make_windows,
)


def test_c_major_candidates_are_diatonic_triads():
    candidates = candidate_triads("C", "major")
    assert [item["basic_chord"] for item in candidates] == [
        "C", "Dm", "Em", "F", "G", "Am", "Bdim"
    ]
    assert [item["roman_numeral"] for item in candidates] == [
        "I", "ii", "iii", "IV", "V", "vi", "vii°"
    ]


def test_basic_generation_uses_melody_and_key():
    windows = [{"event_id": 0, "start_time": 0.0, "end_time": 2.0, "event_duration": 2.0}]
    notes = [
        {"start_time": 0.0, "end_time": 2.0, "midi": 60},
        {"start_time": 0.0, "end_time": 2.0, "midi": 64},
        {"start_time": 0.0, "end_time": 2.0, "midi": 67},
    ]
    result = generate_basic_chords(windows, notes, {"tonic": "C", "mode": "major"})
    assert result[0]["basic_chord"] == "C"
    assert result[0]["previous_basic_chord"] == "START"
    assert result[0]["next_basic_chord"] == "END"


def test_alignment_uses_interval_overlap_and_preserves_ambiguity():
    window = {"start_time": 2.0, "end_time": 4.0, "event_duration": 2.0}
    reference = [
        {"start_time": 1.0, "end_time": 3.25, "raw_chord": "C:maj7", "reference_chord": "Cmaj7"},
        {"start_time": 3.25, "end_time": 5.0, "raw_chord": "G:7", "reference_chord": "G7"},
    ]
    result = align_reference(window, reference)
    assert result["reference_chord"] == "Cmaj7"
    assert result["reference_overlap_seconds"] == 1.25
    assert len(result["reference_candidates"]) == 2


def test_crossing_note_is_included_with_original_and_overlap_timing():
    window = {"start_time": 2.0, "end_time": 4.0}
    notes = [{
        "note": "C4", "midi": 60, "pitch_class": "C", "start_time": 1.5,
        "end_time": 2.5, "duration": 1.0, "confidence": 0.8,
    }]
    result = align_melody(window, notes)
    assert result["melody"][0]["relative_start"] == -0.5
    assert result["melody"][0]["overlap_duration"] == 0.5


def test_two_beat_windows_preserve_intro_and_tail():
    windows = make_windows([0.5, 1.0, 1.5, 2.0, 2.5], 3.0, 2)
    assert [(row["start_time"], row["end_time"]) for row in windows] == [
        (0.0, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.0)
    ]


def test_two_beat_basic_chords_are_copied_to_one_beat_model_grid():
    model = make_windows([0.5, 1.0, 1.5, 2.0], 2.5, 1)
    basic = []
    for event_id, (start, end, chord) in enumerate(((0.0, 0.5, "C"), (0.5, 1.5, "Dm"), (1.5, 2.5, "G"))):
        basic.append({
            "event_id": event_id, "start_time": start, "end_time": end,
            "basic_chord": chord, "basic_root": chord.rstrip("m"),
            "basic_quality": "min" if chord.endswith("m") else "maj",
            "basic_score": 1.0, "candidate_scores": [], "roman_numeral": "I",
            "harmonic_function": "tonic",
        })
    result = align_basic_to_model_windows(model, basic)
    assert [row["basic_chord"] for row in result] == ["C", "Dm", "Dm", "G", "G"]
    assert [row["source_basic_event_id"] for row in result] == [0, 1, 1, 2, 2]
    assert result[0]["previous_basic_chord"] == "START"
    assert result[1]["next_basic_chord"] == "Dm"
    assert result[-1]["next_basic_chord"] == "END"
