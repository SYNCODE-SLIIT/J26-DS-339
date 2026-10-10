"""Use GAME vocals; fill only long vocal gaps with SheetSage2's melody."""

from __future__ import annotations

import json

from common import paths, read_notes, write_result


MIN_NOTE_SECONDS = 0.08


def monophonic(notes: list[dict], duration: float) -> list[dict]:
    result: list[dict] = []
    for source in sorted(notes, key=lambda n: (float(n["onset_sec"]), -float(n["offset_sec"]))):
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


def vocal_gaps(vocals: list[dict], duration: float, minimum: float) -> list[tuple[float, float]]:
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
        cursor = max(cursor, end)
    if duration - cursor >= minimum:
        gaps.append((cursor, duration))
    return gaps


def combine(track_id: str, duration: float) -> dict[str, int]:
    vocals = monophonic(read_notes("game_vocal", track_id), duration)
    sheet_data = json.loads(paths("sheetsage2_instrumental", track_id)["json"].read_text())
    instrumental = monophonic(sheet_data["notes"], duration)
    beat = float(sheet_data.get("median_beat_sec") or 0.5)
    minimum = max(0.8, 2 * beat)
    gaps = vocal_gaps(vocals, duration, minimum)
    selected = [dict(note) for note in vocals]
    for left, right in gaps:
        for source in instrumental:
            start = max(left, source["onset_sec"])
            end = min(right, source["offset_sec"])
            if end - start >= MIN_NOTE_SECONDS:
                note = dict(source)
                note["onset_sec"] = start
                note["offset_sec"] = end
                selected.append(note)
    selected.sort(key=lambda n: (n["onset_sec"], n["offset_sec"]))
    if any(a["offset_sec"] > b["onset_sec"] + 1e-6 for a, b in zip(selected, selected[1:])):
        raise ValueError(f"Combined notes overlap for track {track_id}")
    write_result("combined", track_id, selected, metadata={
        "duration_sec": duration,
        "selection": {"priority": "game_vocal", "fallback": "sheetsage2_instrumental",
                      "median_beat_sec": beat, "minimum_vocal_gap_sec": minimum,
                      "instrumental_gaps_sec": [[a, b] for a, b in gaps]},
    })
    return {"game_vocal": sum(n["source"] == "game_vocal" for n in selected),
            "sheetsage2_instrumental": sum(n["source"] == "sheetsage2_instrumental" for n in selected),
            "combined": len(selected)}
