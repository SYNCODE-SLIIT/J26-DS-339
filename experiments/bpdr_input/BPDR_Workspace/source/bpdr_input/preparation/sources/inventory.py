"""Read-only local vocadito inventory. This does not create repair targets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import wave


def audit_vocadito(dataset_root: Path) -> dict:
    root = dataset_root.resolve()
    metadata_path = root / "vocadito_metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as stream:
        metadata = list(csv.DictReader(stream))
    if not metadata or not {"track_id", "singer_id", "language"}.issubset(metadata[0]):
        raise ValueError("Expected vocadito track_id, singer_id and language metadata")
    track_ids = [row["track_id"] for row in metadata]
    if len(set(track_ids)) != len(track_ids):
        raise ValueError("Duplicate metadata track identifiers")
    by_track = {row["track_id"]: row for row in metadata}
    paths = sorted((root / "Audio").glob("*.wav"))
    paths = [path for path in paths if not path.name.startswith("._")]
    if not paths:
        raise ValueError("No WAV recordings found in Audio")

    recordings = []
    for path in paths:
        track_id = path.stem.removeprefix("vocadito_")
        errors = []
        duration = None
        rate = channels = frames = None
        try:
            with wave.open(str(path), "rb") as stream:
                rate, channels, frames = stream.getframerate(), stream.getnchannels(), stream.getnframes()
                if rate <= 0 or frames <= 0:
                    raise ValueError("Empty recording or invalid sample rate")
                expected_bytes = frames * channels * stream.getsampwidth()
                actual_bytes = 0
                while block := stream.readframes(65536):
                    actual_bytes += len(block)
                if actual_bytes != expected_bytes:
                    errors.append("TRUNCATED_AUDIO_DATA")
                duration = frames / rate
        except (wave.Error, EOFError, ValueError) as error:
            errors.append(f"AUDIO_READ_FAILED: {error}")

        annotations = {}
        for kind, directory, suffix in (
            ("f0", "F0", "_f0.csv"),
            ("notes_a1", "Notes", "_notesA1.csv"),
            ("notes_a2", "Notes", "_notesA2.csv"),
        ):
            annotation = root / "Annotations" / directory / (path.stem + suffix)
            annotations[kind] = str(annotation) if annotation.is_file() else None
            if not annotation.is_file():
                errors.append(f"MISSING_{kind.upper()}")
        row = by_track.get(track_id)
        if row is None:
            errors.append("MISSING_TRACK_METADATA")
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        recordings.append({
            "source_id": path.stem,
            "singer_id": row["singer_id"] if row else None,
            "language": row["language"] if row else None,
            "audio_path": str(path),
            "sha256": checksum,
            "sample_rate_hz": rate,
            "channels": channels,
            "sample_count": frames,
            "duration_seconds": duration,
            "requires_offline_crop_for_20s_scope": duration is not None and duration > 20,
            "annotations": annotations,
            "errors": errors,
        })

    audio_ids = {path.stem.removeprefix("vocadito_") for path in paths}
    return {
        "schema_version": "0.1",
        "dataset_root": str(root),
        "scope": "Audio readability, checksums, metadata and annotation-file presence only",
        "not_established": ["annotation accuracy", "suitable preservation anchors", "usage permissions", "repair performance"],
        "summary": {
            "recording_count": len(recordings),
            "metadata_row_count": len(metadata),
            "singer_count": len({row["singer_id"] for row in metadata}),
            "total_duration_seconds": sum(row["duration_seconds"] or 0 for row in recordings),
            "over_20_seconds": sum(row["requires_offline_crop_for_20s_scope"] for row in recordings),
            "recordings_with_errors": sum(bool(row["errors"]) for row in recordings),
            "metadata_without_audio": sorted(set(track_ids) - audio_ids),
        },
        "recordings": recordings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional inventory JSON; use this component's ignored runs folder")
    arguments = parser.parse_args()
    report = audit_vocadito(arguments.dataset_root)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["recordings_with_errors"] or report["summary"]["metadata_without_audio"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
