"""Make one melody from GAME vocals and SheetSage2 instrumental notes.

Instrument notes are admitted only during vocal-free gaps lasting at least two
SheetSage2 beats (and at least 0.8 seconds). Short rests inside a vocal phrase
stay empty. The exported MIDI has one track and at most one note at a time.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pretty_midi

from pipeline import RESULTS, STATUS, preview, rows


MIN_NOTE_SECONDS = 0.08


def beat_period(path: Path) -> float:
    if path.exists():
        times = []
        for line in path.read_text().splitlines():
            try:
                times.append(float(line.split()[0]))
            except (IndexError, ValueError):
                continue
        intervals = [b - a for a, b in zip(times, times[1:]) if 0.2 <= b - a <= 2.0]
        if len(intervals) >= 3:
            return statistics.median(intervals)
    return 0.5


def one_note_at_a_time(notes: list[dict], duration: float) -> list[dict]:
    """Resolve rare within-source overlaps without adding harmony."""
    ordered = sorted(notes, key=lambda n: (float(n["onset_sec"]), -float(n["offset_sec"])))
    result = []
    for source in ordered:
        note = dict(source)
        note["onset_sec"] = max(0.0, float(note["onset_sec"]))
        note["offset_sec"] = min(duration, float(note["offset_sec"]))
        if note["offset_sec"] - note["onset_sec"] < MIN_NOTE_SECONDS:
            continue
        if result and note["onset_sec"] < result[-1]["offset_sec"]:
            previous = result[-1]
            if note["onset_sec"] - previous["onset_sec"] < MIN_NOTE_SECONDS:
                if note["offset_sec"] > previous["offset_sec"]:
                    result[-1] = note
                continue
            previous["offset_sec"] = note["onset_sec"]
            if previous["offset_sec"] - previous["onset_sec"] < MIN_NOTE_SECONDS:
                result.pop()
        result.append(note)
    return result


def instrumental_gaps(vocals: list[dict], duration: float, minimum: float) -> list[tuple[float, float]]:
    blocks: list[list[float]] = []
    for note in vocals:
        start, end = note["onset_sec"], note["offset_sec"]
        if blocks and start - blocks[-1][1] < minimum:
            blocks[-1][1] = max(blocks[-1][1], end)
        else:
            blocks.append([start, end])
    gaps = []
    cursor = 0.0
    for start, end in blocks:
        if start - cursor >= minimum:
            gaps.append((cursor, start))
        cursor = end
    if duration - cursor >= minimum:
        gaps.append((cursor, duration))
    return gaps


def combine(track_id: str, duration: float) -> dict:
    vocal_file = RESULTS / "game_vocal" / f"{track_id}.json"
    instrumental_file = RESULTS / "sheetsage2" / f"{track_id}.instrumental.json"
    vocals = one_note_at_a_time(json.loads(vocal_file.read_text())["notes"], duration)
    instrumental = one_note_at_a_time(json.loads(instrumental_file.read_text())["notes"], duration)
    period = beat_period(RESULTS / "sheetsage2" / f"{track_id}.beat.lab")
    minimum_gap = max(0.8, 2.0 * period)
    gaps = instrumental_gaps(vocals, duration, minimum_gap)

    selected = []
    for note in vocals:
        selected.append({"onset_sec": note["onset_sec"], "offset_sec": note["offset_sec"],
                         "midi_pitch": note["midi_pitch"], "source": "game_vocal"})
    for left, right in gaps:
        for note in instrumental:
            start = max(left, note["onset_sec"])
            end = min(right, note["offset_sec"])
            if end - start >= MIN_NOTE_SECONDS:
                selected.append({"onset_sec": start, "offset_sec": end,
                                 "midi_pitch": note["midi_pitch"],
                                 "source": "sheetsage2_instrumental"})
    selected.sort(key=lambda n: (n["onset_sec"], n["offset_sec"]))
    for current, following in zip(selected, selected[1:]):
        if current["offset_sec"] > following["onset_sec"] + 0.0001:
            raise ValueError(f"Combined notes overlap in {track_id}")

    destination = RESULTS / "combined"
    destination.mkdir(parents=True, exist_ok=True)
    midi_path = destination / f"{track_id}.mid"
    midi = pretty_midi.PrettyMIDI()
    lead = pretty_midi.Instrument(program=0, name="combined_melody")
    for note in selected:
        pitch = max(0, min(127, round(float(note["midi_pitch"]))))
        lead.notes.append(pretty_midi.Note(velocity=100, pitch=pitch,
                                          start=note["onset_sec"], end=note["offset_sec"]))
        note["midi_pitch_rounded"] = pitch
        note["onset_sec"] = round(note["onset_sec"], 5)
        note["offset_sec"] = round(note["offset_sec"], 5)
    midi.instruments.append(lead)
    midi.write(str(midi_path))
    preview_path = midi_path.with_suffix(".preview.wav")
    preview_path.unlink(missing_ok=True)
    preview(midi_path, preview_path, duration)
    payload = {"track_id": track_id, "method": "game_vocal_plus_sheetsage2_instrumental",
               "time_origin": "clip_start", "notes": selected,
               "selection": {"minimum_vocal_gap_sec": round(minimum_gap, 5),
                             "median_beat_sec": round(period, 5),
                             "instrumental_gaps_sec": [[round(a, 5), round(b, 5)] for a, b in gaps],
                             "vocal_notes": len(vocals),
                             "instrumental_candidates": len(instrumental)}}
    midi_path.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    return {"status": "ok", "notes": len(selected),
            "vocal_notes": len(vocals),
            "instrumental_notes": sum(n["source"] == "sheetsage2_instrumental" for n in selected)}


def main() -> None:
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    selected = rows()
    for index, row in enumerate(selected, 1):
        track_id = row["track_id"]
        try:
            result = combine(track_id, float(row["clip_duration_sec"]))
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        status.setdefault(track_id, {})["combined"] = result
        STATUS.write_text(json.dumps(status, indent=2) + "\n")
        print(f"[{index}/{len(selected)}] combined {track_id}: {result}", flush=True)


if __name__ == "__main__":
    main()
