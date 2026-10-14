"""Deterministic, rest-aligned short crop selection from unshifted full-song GAME notes."""

from __future__ import annotations

from dataclasses import dataclass


CROP_POLICY_VERSION = "vocal-rest-crop-v1"


@dataclass(frozen=True)
class Crop:
    start_sec: float
    end_sec: float
    sung_sec: float
    note_count: int
    warnings: tuple[str, ...]


def choose(notes: list[dict], duration: float, maximum: float = 20.0) -> Crop | None:
    valid = sorted((max(0.0, float(n["onset_sec"])), min(duration, float(n["offset_sec"])))
                   for n in notes if float(n["offset_sec"]) > float(n["onset_sec"]))
    valid = [(a, b) for a, b in valid if b > a]
    if not valid:
        return None
    # Candidate edges are track boundaries and midpoints of genuine predicted rests.
    pauses = [0.0]
    pauses += [(b + c) / 2 for (_, b), (c, _) in zip(valid, valid[1:]) if c - b >= .24]
    pauses += [duration]
    pauses = sorted(set(pauses))
    candidates: list[tuple[float, float, float, int, tuple[str, ...]]] = []
    for start in pauses[:-1]:
        for end in pauses[1:]:
            if end <= start or end - start > maximum + 1e-9:
                continue
            selected = [(a, b) for a, b in valid if a >= start and b <= end]
            if not selected:
                continue
            sung = sum(b - a for a, b in selected)
            if sung < 5.0:
                continue
            warnings = tuple("predicted_note_crosses_crop_edge" for a, b in valid
                             if (a < start < b) or (a < end < b))
            candidates.append((sung, start, end, len(selected), warnings))
    if not candidates:
        # Continuous singing can leave no 20-second pair of pause midpoints.
        # Use complete predicted note edges and flag the absent rest context.
        for first, (start, _) in enumerate(valid):
            for last in range(first, len(valid)):
                end = valid[last][1]
                if end - start > maximum:
                    break
                selected = valid[first:last + 1]
                sung = sum(b - a for a, b in selected)
                if sung >= 5.0:
                    candidates.append((sung, start, end, len(selected),
                                       ('no_natural_pause_at_crop_edge',)))
    if not candidates:
        return None
    sung, start, end, count, warnings = max(candidates, key=lambda row: (row[0], row[3], -row[1]))
    return Crop(start, end, sung, count, warnings)
