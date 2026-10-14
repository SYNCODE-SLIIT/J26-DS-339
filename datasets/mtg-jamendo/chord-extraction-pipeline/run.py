"""Full-song, resumable chord extraction. Run with the repository's .venv Python."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import io
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from chords import DATASET, MODEL_VERSION, ROOT, VOCABULARIES, atomic_text, completed, inventory, save  # noqa: E402


MANIFEST_COLUMNS = ("track_id", "mtg_track_id", "artist_id", "album_id", "annotation_path",
                    "source_mp3", "duration_annotation_sec", "duration_audio_sec",
                    "split_0", "split_1", "split_2", "split_3", "split_4", "tags",
                    "status", "chord_segments", "no_chord_fraction", "quality_flags", "error", "updated_utc")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(track_id: str, stage: str, message: str) -> None:
    with (ROOT / "history.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp()}\t{track_id}\t{stage}\t{message.replace(chr(10), ' ')}\n")


def audio_duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                            capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def index_paths(vocabulary: str) -> tuple[Path, Path]:
    stem = "" if vocabulary == "submission" else f"-{vocabulary}"
    return ROOT / f"manifest{stem}.csv", ROOT / f"catalog{stem}.json"


def prior_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["track_id"]: row for row in csv.DictReader(stream)}


def publish(tracks: list[dict[str, str]], results: dict[str, dict],
            failures: dict[str, str], previous: dict[str, dict[str, str]], vocabulary: str) -> None:
    manifest, catalog = index_paths(vocabulary)
    rows = []
    listed = []
    now = timestamp()
    for track in tracks:
        track_id = track["track_id"]
        result = results.get(track_id)
        old = previous.get(track_id, {})
        if result is not None:
            status = "complete"
            error = ""
            duration = result["duration_sec"]
            count = len(result["events"])
            fraction = result["no_chord_fraction"]
            flags = ";".join(result["quality_flags"])
            listed.append({"track_id": track_id, "mtg_track_id": track["mtg_track_id"],
                           "artist_id": track["artist_id"], "split_0": track["split_0"],
                           "duration_sec": duration, "source_mp3": f"../{track['source_mp3']}",
                           "chords_json": f"outputs/{vocabulary}/{track_id}.json",
                           "chords_csv": f"outputs/{vocabulary}/{track_id}.csv",
                           "chord_segments": count, "no_chord_fraction": fraction,
                           "quality_flags": result["quality_flags"]})
        elif track_id in failures:
            status = "failed"
            error = failures[track_id]
            duration = old.get("duration_audio_sec", "")
            count = fraction = flags = ""
        else:
            status = "pending"
            error = duration = count = fraction = flags = ""
        rows.append({**track, "duration_audio_sec": duration, "status": status,
                     "chord_segments": count, "no_chord_fraction": fraction,
                     "quality_flags": flags, "error": error,
                     "updated_utc": now if status != "pending" else ""})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(manifest, buffer.getvalue())
    atomic_text(catalog, json.dumps({"schema_version": 1, "vocabulary": vocabulary,
                                    "updated_utc": now, "completed_count": len(listed),
                                    "tracks": listed}, ensure_ascii=False, indent=2) + "\n")


def verify_model() -> None:
    import lv_chordia
    from lv_chordia.chord_recognition import MODEL_NAMES
    from lv_chordia.mir.common import CACHE_DATA_PATH

    version = importlib.metadata.version("lv-chordia")
    if version != MODEL_VERSION or lv_chordia.__version__ != MODEL_VERSION:
        raise RuntimeError(f"Expected lv-chordia {MODEL_VERSION}; found {version}")
    missing = [name for name in MODEL_NAMES if not (Path(CACHE_DATA_PATH) / f"{name}.sdict").is_file()]
    if missing:
        raise RuntimeError(f"LV-Chordia model weights missing in {CACHE_DATA_PATH}: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Process at most this many pending songs")
    parser.add_argument("--track-id", help="Process one numeric MTG-Jamendo track ID")
    parser.add_argument("--retry-failed", action="store_true", help="Retry earlier failures")
    parser.add_argument("--vocabulary", choices=VOCABULARIES, default="submission",
                        help="submission is the recommended default; full is more detailed but less reliable")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        verify_model()
    except (ImportError, RuntimeError) as error:
        parser.exit(2, f"Model setup error: {error}\nSee README.md for installation.\n")
    from lv_chordia import chord_recognition

    tracks = inventory()
    if args.track_id is not None:
        wanted = str(int(args.track_id.removeprefix("track_")))
        if wanted not in {track["track_id"] for track in tracks}:
            parser.error(f"Track {args.track_id} has no local MP3")
    manifest, _ = index_paths(args.vocabulary)
    previous = prior_rows(manifest)
    results = {}
    failures = {track_id: row.get("error", "") for track_id, row in previous.items()
                if row.get("status") == "failed"}
    for track in tracks:
        existing = completed(track, args.vocabulary)
        if existing is not None:
            results[track["track_id"]] = existing
            failures.pop(track["track_id"], None)
    publish(tracks, results, failures, previous, args.vocabulary)
    pending = [track for track in tracks if track["track_id"] not in results
               and (args.track_id is None or track["track_id"] == wanted)
               and (args.retry_failed or track["track_id"] not in failures)]
    if args.limit is not None:
        pending = pending[:args.limit]
    print(f"{len(results)} complete; {len(pending)} selected; {len(failures)} prior failures; "
          f"vocabulary={args.vocabulary}", flush=True)
    for position, track in enumerate(pending, 1):
        track_id = track["track_id"]
        print(f"[{position}/{len(pending)}] {track_id}", flush=True)
        log(track_id, "start", f"vocabulary={args.vocabulary}")
        try:
            mp3 = DATASET / track["source_mp3"]
            duration_sec = audio_duration(mp3)
            raw = chord_recognition(str(mp3), chord_dict_name=args.vocabulary)
            result = save(track, args.vocabulary, duration_sec, raw)
            results[track_id] = result
            failures.pop(track_id, None)
            log(track_id, "complete", f"vocabulary={args.vocabulary} segments={len(result['events'])} "
                f"flags={','.join(result['quality_flags']) or 'none'}")
            print(f"  saved {len(result['events'])} chord segments"
                  f"; flags={result['quality_flags'] or 'none'}", flush=True)
        except KeyboardInterrupt:
            log(track_id, "interrupted", f"vocabulary={args.vocabulary}")
            publish(tracks, results, failures, previous, args.vocabulary)
            print("Interrupted; completed songs remain saved.", flush=True)
            return 130
        except Exception as error:
            failures[track_id] = f"{type(error).__name__}: {error}"[:500]
            log(track_id, "failed", f"vocabulary={args.vocabulary} {failures[track_id]}")
            traceback.print_exc()
        publish(tracks, results, failures, previous, args.vocabulary)
    print(f"Done: {len(results)} complete; {len(failures)} failed. "
          "Rerun the same command to continue.", flush=True)
    return 0 if not any(track["track_id"] in failures for track in pending) else 1


if __name__ == "__main__":
    raise SystemExit(main())
