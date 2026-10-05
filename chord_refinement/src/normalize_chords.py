"""Conservative parsing and normalization of LV-Chordia labels."""

from __future__ import annotations

import re
from dataclasses import dataclass


QUALITY_SUFFIXES = {
    "maj": "", "min": "m", "maj7": "maj7", "min7": "m7", "7": "7",
    "sus2": "sus2", "sus4": "sus4", "dim": "dim", "dim7": "dim7",
    "hdim7": "m7b5", "aug": "aug", "6": "6", "min6": "m6", "9": "9",
    "maj9": "maj9", "min9": "m9", "11": "11", "13": "13", "add9": "add9",
}

CHORD_PATTERN = re.compile(
    r"^(?P<root>[A-G](?:#|b)?):(?P<quality>[^/\s]+)(?:/(?P<bass_degree>[#b]?\d+))?$"
)


@dataclass(frozen=True, slots=True)
class ParsedChord:
    root: str | None
    quality: str | None
    bass_degree: str | None
    reference_chord: str
    parsed: bool
    normalized: bool


def parse_chord_label(raw_chord: str) -> ParsedChord:
    """Parse known JAMS-like structure without discarding unknown syntax."""
    raw = raw_chord.strip()
    if raw == "N":
        return ParsedChord(None, None, None, "N", True, True)
    match = CHORD_PATTERN.fullmatch(raw)
    if not match:
        return ParsedChord(None, None, None, raw, False, False)

    root = match.group("root")
    quality = match.group("quality")
    bass_degree = match.group("bass_degree")
    suffix = QUALITY_SUFFIXES.get(quality)
    if suffix is None:
        return ParsedChord(root, quality, bass_degree, raw, True, False)

    reference = f"{root}{suffix}"
    if bass_degree is not None:
        reference += f"/{bass_degree}"
    return ParsedChord(root, quality, bass_degree, reference, True, True)


def normalize_chord_label(raw_chord: str) -> str:
    return parse_chord_label(raw_chord).reference_chord

