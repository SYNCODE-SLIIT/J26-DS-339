"""Add sharp-root canonical chord columns without changing source labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from experiments.rahul import config


ROOT_TO_PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
PITCH_CLASS_TO_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
CHORD_PATTERN = re.compile(r"^(?P<root>[A-G](?:#|b)?)(?P<remainder>.*)$")
DERIVED_COLUMNS = (
    "basic_chord_canonical",
    "reference_chord_canonical",
    "basic_root_pitch_class",
    "reference_root_pitch_class",
)


def canonicalize_chord(label: str) -> tuple[str, int | None]:
    """Return a sharp-root chord and integer pitch class; preserve all suffix text."""
    if label == "N":
        return "N", None
    match = CHORD_PATTERN.fullmatch(label)
    if not match or match.group("root") not in ROOT_TO_PITCH_CLASS:
        raise ValueError(f"Unsupported chord label: {label!r}")
    pitch_class = ROOT_TO_PITCH_CLASS[match.group("root")]
    return PITCH_CLASS_TO_SHARP[pitch_class] + match.group("remainder"), pitch_class


def add_canonical_fields(row: dict) -> dict:
    basic, basic_pc = canonicalize_chord(str(row["basic_chord"]))
    reference, reference_pc = canonicalize_chord(str(row["reference_chord"]))
    return {
        **row,
        "basic_chord_canonical": basic,
        "reference_chord_canonical": reference,
        "basic_root_pitch_class": basic_pc,
        "reference_root_pitch_class": reference_pc,
    }


def normalize_csv(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        original_columns = list(reader.fieldnames or [])
        original_rows = list(reader)
    if not original_rows:
        raise ValueError(f"Dataset has no rows: {path}")

    protected = ("basic_chord", "reference_chord", "reference_raw_chord")
    protected_before = [[row[column] for column in protected] for row in original_rows]
    normalized_rows = [add_canonical_fields(row) for row in original_rows]
    output_columns = original_columns + [name for name in DERIVED_COLUMNS if name not in original_columns]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(normalized_rows)

    with path.open(encoding="utf-8", newline="") as handle:
        written_rows = list(csv.DictReader(handle))
    protected_after = [[row[column] for column in protected] for row in written_rows]
    if len(written_rows) != len(original_rows) or protected_after != protected_before:
        raise RuntimeError("Row count or protected chord columns changed during normalization")

    return {
        "row_count": len(written_rows),
        "protected_columns_unchanged": True,
        "derived_columns": list(DERIVED_COLUMNS),
    }


def normalize_aligned_jsons(directory: Path, track_ids: set[str]) -> int:
    updated = 0
    for track_id in sorted(track_ids):
        path = directory / f"{track_id}_aligned.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        original = [
            (event["basic_chord"], event["reference_chord"], event["reference_raw_chord"])
            for event in document["events"]
        ]
        document["events"] = [add_canonical_fields(event) for event in document["events"]]
        protected_after = [
            (event["basic_chord"], event["reference_chord"], event["reference_raw_chord"])
            for event in document["events"]
        ]
        if protected_after != original:
            raise RuntimeError(f"Protected chord labels changed in {path}")
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        updated += 1
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=config.OUTPUT_DIR / "aligned" / "training_events.csv",
    )
    parser.add_argument(
        "--skip-aligned-json",
        action="store_true",
        help="Add columns to the CSV only.",
    )
    args = parser.parse_args()
    csv_path = args.csv.expanduser().resolve()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        track_ids = {row["track_id"] for row in csv.DictReader(handle)}
    result = normalize_csv(csv_path)
    result["aligned_json_files_updated"] = (
        0
        if args.skip_aligned_json
        else normalize_aligned_jsons(csv_path.parent, track_ids)
    )
    result["csv"] = str(csv_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
