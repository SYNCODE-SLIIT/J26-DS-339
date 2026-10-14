"""Extract melody, chords, melody-derived key, and relative chord plans per song."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from beats import complete as beats_complete  # noqa: E402
from bpdr.export import Export as BPDRExport  # noqa: E402
from chords import completed as chord_complete, verify_model  # noqa: E402
from combine import combine  # noqa: E402
from comparison import completed as comparison_complete, save as save_comparison  # noqa: E402
from common import (ASSETS, CHORDS, DATASET, MELODIES, ROOT, STAGES, WORK, atomic_text, python_in,
                    inventory, melody_complete)  # noqa: E402
from key import completed as key_complete, save as save_key  # noqa: E402
from melody import verify_melody_assets  # noqa: E402
from roman import completed as roman_complete, save as save_roman  # noqa: E402


MANIFEST = ROOT / "manifest.csv"
CATALOG = ROOT / "catalog.json"
HISTORY = ROOT / "history.log"
MANIFEST_COLUMNS = ("track_id", "mtg_track_id", "artist_id", "album_id", "annotation_path",
                    "source_mp3", "duration_annotation_sec", "duration_audio_sec",
                    "split_0", "split_1", "split_2", "split_3", "split_4", "tags",
                    "status", "game_notes", "sheetsage_notes", "combined_notes",
                    "beat_count", "median_beat_period_sec", "median_beat_rate_bpm",
                    "submission_chords", "full_chords", "submission_flags", "full_flags",
                    "key_tonic", "key_mode", "key_confidence", "chord_key_tonic",
                    "chord_key_mode", "key_agreement", "key_review_flag", "key_review_reasons",
                    "roman_submission_mapped",
                    "roman_full_mapped", "roman_submission_unmapped", "roman_full_unmapped",
                    "bpdr_status", "bpdr_screen", "bpdr_crop_start_sec",
                    "error_stage", "error", "updated_utc")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(track_id: str, stage: str, message: str) -> None:
    with HISTORY.open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp()}\t{track_id}\t{stage}\t{message.replace(chr(10), ' ')}\n")
        stream.flush()


def audio_duration(path: Path) -> float:
    value = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                     "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                                     str(path)], text=True).strip()
    return float(value)


def prior_rows() -> dict[str, dict[str, str]]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        return {row["track_id"]: row for row in csv.DictReader(stream)}


def compact_result(result: dict) -> dict:
    """Keep catalog fields and counts, not every note/beat/chord in RAM."""
    if result.get("schema_version") == "bpdr-track-v1":
        return {"status": result["status"], "screen": result.get("screen", {}),
                "crop": result.get("crop", {}), "training_eligible": False}
    return {key: len(value) if key in ("notes", "beats", "events") else value
            for key, value in result.items()
            if key not in ("selection", "downbeat_lab_times_sec")}


def item_count(result: dict, key: str) -> int:
    value = result[key]
    return value if isinstance(value, int) else len(value)


def scan_existing(tracks: list[dict[str, str]], bpdr: BPDRExport) -> dict[str, dict[str, dict]]:
    results: dict[str, dict[str, dict]] = {}
    for track in tracks:
        stages = {}
        for method in MELODIES:
            result = melody_complete(method, track)
            if result is not None:
                stages[method] = compact_result(result)
        result = beats_complete(track)
        if result is not None:
            stages["beats"] = compact_result(result)
        if "combined" in stages:
            result = key_complete(track)
            if result is not None:
                stages["key"] = compact_result(result)
        for vocabulary in CHORDS:
            result = chord_complete(track, vocabulary)
            if result is not None:
                stages[vocabulary] = compact_result(result)
        if "key" in stages and "submission" in stages:
            result = comparison_complete(track)
            if result is not None:
                stages["key_comparison"] = compact_result(result)
        if "key" in stages:
            for vocabulary in CHORDS:
                if vocabulary in stages:
                    result = roman_complete(track, vocabulary)
                    if result is not None:
                        stages[f"roman_{vocabulary}"] = compact_result(result)
        if bpdr.includes(track["track_id"]):
            result = bpdr.complete(track)
            if result is not None:
                stages["bpdr"] = compact_result(result)
        if stages:
            results[track["track_id"]] = stages
    return results


def song_complete(track: dict[str, str], stages: dict[str, dict], bpdr: BPDRExport) -> bool:
    return (all(stage in stages for stage in STAGES)
            and (not bpdr.includes(track["track_id"]) or "bpdr" in stages))


def publish(tracks: list[dict[str, str]], results: dict[str, dict[str, dict]],
            errors: dict[str, tuple[str, str]], durations: dict[str, float],
            bpdr: BPDRExport) -> None:
    rows = []
    listed = []
    now = timestamp()
    for track in tracks:
        track_id = track["track_id"]
        stages = results.get(track_id, {})
        error_stage, error = errors.get(track_id, ("", ""))
        status = ("complete" if song_complete(track, stages, bpdr) else "failed" if error
                  else "partial" if stages else "pending")
        counts = {method: item_count(stages[method], "notes") if method in stages else ""
                  for method in MELODIES}
        chord_counts = {vocab: item_count(stages[vocab], "events") if vocab in stages else ""
                        for vocab in CHORDS}
        flags = {vocab: ";".join(stages[vocab]["quality_flags"]) if vocab in stages else ""
                 for vocab in CHORDS}
        key = stages.get("key", {})
        beats = stages.get("beats", {})
        comparison = stages.get("key_comparison", {})
        bpdr_result = stages.get("bpdr", {})
        melodic_key = key.get("melody_key", {}).get("selected") or {}
        chord_key = comparison.get("chord_key", {}).get("selected") or {}
        roman_counts = {vocab: stages.get(f"roman_{vocab}", {}).get("mapping_counts", {})
                        for vocab in CHORDS}
        duration = durations.get(track_id, next((float(result["duration_sec"])
                                               for result in stages.values()), None))
        rows.append({**track, "duration_audio_sec": duration if duration is not None else "",
                     "status": status, "game_notes": counts["game_vocal"],
                     "sheetsage_notes": counts["sheetsage2_instrumental"],
                     "combined_notes": counts["combined"],
                     "beat_count": item_count(beats, "beats") if beats else "",
                     "median_beat_period_sec": beats.get("median_beat_period_sec", ""),
                     "median_beat_rate_bpm": beats.get("median_beat_rate_bpm", ""),
                     "submission_chords": chord_counts["submission"],
                     "full_chords": chord_counts["full"],
                     "submission_flags": flags["submission"], "full_flags": flags["full"],
                     "key_tonic": melodic_key.get("tonic", ""),
                     "key_mode": melodic_key.get("mode", ""),
                     "key_confidence": melodic_key.get("confidence", ""),
                     "chord_key_tonic": chord_key.get("tonic", ""),
                     "chord_key_mode": chord_key.get("mode", ""),
                     "key_agreement": comparison.get("agreement", ""),
                     "key_review_flag": comparison.get("review_flag", ""),
                     "key_review_reasons": ";".join(comparison.get("review_reasons", [])),
                     "roman_submission_mapped": roman_counts["submission"].get("mapped", ""),
                     "roman_full_mapped": roman_counts["full"].get("mapped", ""),
                     "roman_submission_unmapped": sum(value for status, value in roman_counts["submission"].items()
                                                       if status not in ("mapped", "no_chord")) if roman_counts["submission"] else "",
                     "roman_full_unmapped": sum(value for status, value in roman_counts["full"].items()
                                                 if status not in ("mapped", "no_chord")) if roman_counts["full"] else "",
                     "bpdr_status": bpdr_result.get("status", "") if bpdr.includes(track_id) else "ineligible",
                     "bpdr_screen": bpdr_result.get("screen", {}).get("status", ""),
                     "bpdr_crop_start_sec": bpdr_result.get("crop", {}).get("offset_seconds", ""),
                     "error_stage": error_stage, "error": error,
                     "updated_utc": now if status != "pending" else ""})
        if stages:
            folder = f"outputs/{track['mtg_track_id']}"
            listed.append({
                "track_id": track_id, "mtg_track_id": track["mtg_track_id"],
                "artist_id": track["artist_id"], "split_0": track["split_0"],
                "duration_sec": duration, "audio": f"../{track['source_mp3']}",
                "status": status, "error_stage": error_stage,
                "melody": {method: {
                    "midi": f"{folder}/melody/{method}.mid",
                    "json": f"{folder}/melody/{method}.json",
                    "csv": f"{folder}/melody/{method}.csv",
                    "note_count": counts[method],
                } for method in MELODIES if method in stages},
                "beats": {"json": f"{folder}/beats/beats.json",
                          "csv": f"{folder}/beats/beats.csv",
                          "beat_count": item_count(beats, "beats")} if beats else None,
                "chords": {vocab: {
                    "json": f"{folder}/chords/{vocab}.json",
                    "csv": f"{folder}/chords/{vocab}.csv",
                    "segment_count": chord_counts[vocab],
                    "quality_flags": stages[vocab]["quality_flags"],
                } for vocab in CHORDS if vocab in stages},
                "key": {"json": f"{folder}/key.json", "melody": melodic_key}
                       if "key" in stages else None,
                "key_comparison": {"json": f"{folder}/key_comparison.json",
                                   "chord_diagnostic": chord_key,
                                   "agreement": comparison.get("agreement"),
                                   "review_reasons": comparison.get("review_reasons", [])}
                                  if "key_comparison" in stages else None,
                "roman": {vocab: {
                    "json": f"{folder}/roman/{vocab}.json",
                    "csv": f"{folder}/roman/{vocab}.csv",
                    "mapping_counts": roman_counts[vocab],
                } for vocab in CHORDS if f"roman_{vocab}" in stages},
                "bpdr": {"record": f"{folder}/bpdr/record.json",
                         "vocal": f"{folder}/bpdr/vocal.wav" if bpdr_result.get("status") == "exported" else None,
                         "status": bpdr_result.get("status"),
                         "screen": bpdr_result.get("screen", {}).get("status")}
                        if bpdr_result else None,
            })
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(MANIFEST, buffer.getvalue())
    atomic_text(CATALOG, json.dumps({"schema_version": 1, "updated_utc": now,
                                    "inventory_count": len(tracks),
                                    "completed_count": sum(row["status"] == "complete" for row in rows),
                                    "tracks": listed}, ensure_ascii=False, indent=2) + "\n")


class SheetSageWorker:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.log_file = WORK / "sheetsage.log"

    def _read_message(self, expected: str | None = None) -> list[str]:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            line = line.rstrip()
            parts = line.split("\t")
            if parts[0] in ("READY", "DONE", "ERROR") and (expected is None or
                    parts[0] == "READY" or len(parts) > 1 and parts[1] == expected):
                return parts
            with self.log_file.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        raise RuntimeError(f"SheetSage2 worker exited {self.process.poll()}; see {self.log_file}")

    def start(self) -> None:
        if self.process is not None:
            return
        env = os.environ.copy()
        env.update({"PYTORCH_ENABLE_MPS_FALLBACK": "1", "HF_HUB_OFFLINE": "1"})
        self.process = subprocess.Popen(
            [str(python_in(ASSETS / ".venv-sheetsage2")),
             str(ROOT / "code" / "sheetsage_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env)
        message = self._read_message()
        if message[0] != "READY":
            raise RuntimeError(f"SheetSage2 worker did not initialize: {message}")
        print(f"SheetSage2 device: {message[1]}", flush=True)

    def _transcribe_once(self, track: dict[str, str], duration: float) -> dict:
        self.start()
        assert self.process is not None and self.process.stdin is not None
        row = {**track, "duration_sec": duration}
        self.process.stdin.write(json.dumps(row) + "\n")
        self.process.stdin.flush()
        message = self._read_message(track["track_id"])
        if message[0] == "ERROR":
            raise RuntimeError(" ".join(message[2:]))
        if message[0] != "DONE":
            raise RuntimeError(f"Unexpected SheetSage2 response: {message}")
        result = melody_complete("sheetsage2_instrumental", track)
        if result is None:
            raise RuntimeError("SheetSage2 reported DONE without validated output")
        return result

    def transcribe(self, track: dict[str, str], duration: float) -> dict:
        for attempt_number in (1, 2):
            try:
                return self._transcribe_once(track, duration)
            except (BrokenPipeError, RuntimeError) as exc:
                if attempt_number == 2 or not sheet_worker_failure(exc):
                    raise
                log(track["track_id"], "sheetsage2_instrumental",
                    f"worker_restart after {type(exc).__name__}: {exc}")
            finally:
                # Each song gets a new process so its MPS allocations are released.
                self.close()
        raise AssertionError("unreachable")

    def extract_beats(self, track: dict[str, str], duration: float) -> dict:
        self.transcribe(track, duration)
        result = beats_complete(track)
        if result is None:
            raise RuntimeError("SheetSage2 reported DONE without validated beat output")
        return result

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None


def sheet_worker_failure(exc: BaseException | str) -> bool:
    message = str(exc).lower()
    return (isinstance(exc, BrokenPipeError)
            or "out of memory" in message
            or "broken pipe" in message
            or "worker exited" in message)


def model_resource_failure(exc: BaseException | str) -> bool:
    message = str(exc).lower()
    return ("out of memory" in message
            or "no metal device available" in message
            or "worker exited -9" in message)


def extract_game(track: dict[str, str], duration: float, *, keep_stem: bool = False) -> dict:
    result = subprocess.run(
        [sys.executable, str(ROOT / "code" / "game_worker.py")],
        input=json.dumps({"track": track, "duration_sec": duration,
                          "keep_stem": keep_stem}),
        capture_output=True, text=True)
    if result.returncode:
        lines = (result.stderr or result.stdout).strip().splitlines()
        detail = lines[-1][-400:] if lines else "no error details"
        raise RuntimeError(f"GAME worker exited {result.returncode}: {detail}")
    saved = melody_complete("game_vocal", track)
    if saved is None:
        raise RuntimeError("GAME worker exited without validated output")
    return saved


def obtain_game_stem(track: dict[str, str], duration: float) -> Path:
    """Regenerate a historical song's stem only when an opt-in export needs it."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "code" / "game_worker.py")],
        input=json.dumps({"track": track, "duration_sec": duration,
                          "mode": "stem_only"}), capture_output=True, text=True)
    if result.returncode:
        lines = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(f"Vocal stem worker exited {result.returncode}: "
                           f"{lines[-1][-400:] if lines else 'no error details'}")
    stem = WORK / "stems" / f"{track['track_id']}.wav"
    if not stem.is_file() or not stem.stat().st_size:
        raise RuntimeError("Vocal stem worker produced no WAV")
    return stem


def extract_chords(track: dict[str, str], vocabulary: str, duration: float) -> dict:
    result = subprocess.run(
        [sys.executable, str(ROOT / "code" / "chord_worker.py")],
        input=json.dumps({"track": track, "vocabulary": vocabulary,
                          "duration_sec": duration}),
        capture_output=True, text=True)
    if result.returncode:
        lines = (result.stderr or result.stdout).strip().splitlines()
        detail = lines[-1][-400:] if lines else "no error details"
        raise RuntimeError(f"LV-Chordia worker exited {result.returncode}: {detail}")
    saved = chord_complete(track, vocabulary)
    if saved is None:
        raise RuntimeError("LV-Chordia worker exited without validated output")
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Process this many next incomplete songs")
    parser.add_argument("--track-id", help="Process one numeric or track_0000000 ID")
    parser.add_argument("--retry-failed", action="store_true", help="Retry songs with recorded errors")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        verify_melody_assets()
        verify_model()
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"Model setup error: {exc}\nSee README.md.\n")
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "run.lock").open("w") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                lock.write("0")
                lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            parser.exit(2, "Another extraction run is active.\n")
        tracks = inventory()
        try:
            bpdr = BPDRExport(tracks)
        except (ValueError, FileNotFoundError) as exc:
            parser.exit(2, f"BPDR setup error: {exc}\n")
        if not python_in(ROOT / ".venv-bpdr").is_file():
            parser.exit(2, "BPDR environment missing; install bpdr-requirements.txt (see README.md).\n")
        if args.track_id is not None:
            try:
                wanted = str(int(args.track_id.removeprefix("track_")))
            except ValueError:
                parser.error("--track-id must be numeric or track_ followed by digits")
            if wanted not in {track["track_id"] for track in tracks}:
                parser.error(f"Track {args.track_id} has no local MP3")
        previous = prior_rows()
        results = scan_existing(tracks, bpdr)
        errors = {track_id: (row.get("error_stage", ""), row["error"])
                  for track_id, row in previous.items() if row.get("error")
                  and row.get("error_stage") not in results.get(track_id, {})}
        durations = {track_id: float(row["duration_audio_sec"])
                     for track_id, row in previous.items() if row.get("duration_audio_sec")}
        publish(tracks, results, errors, durations, bpdr)
        selected = [track for track in tracks
                    if (args.track_id is None or track["track_id"] == wanted)
                    and not song_complete(track, results.get(track["track_id"], {}), bpdr)
                    and (args.retry_failed or args.track_id is not None
                         or track["track_id"] not in errors)]
        # Backfill songs with saved stages first, finishing core-complete songs
        # before older partial ones; only then begin untouched MP3s.
        selected.sort(key=lambda track: (
            not bool(results.get(track["track_id"])),
            not all(stage in results.get(track["track_id"], {}) for stage in STAGES),
            int(track["track_id"])))
        if args.limit is not None:
            selected = selected[:args.limit]
        print(f"Selected {len(selected)} full songs; {len(tracks)} local MP3s. "
              f"Complete: {sum(song_complete(t, results.get(t['track_id'], {}), bpdr) for t in tracks)}.", flush=True)
        log("*", "run_start", f"selected={len(selected)} retry_failed={args.retry_failed}")
        sheet = SheetSageWorker()
        try:
            for number, track in enumerate(selected, 1):
                track_id = track["track_id"]
                stages = results.setdefault(track_id, {})
                duration = durations.get(track_id)
                if duration is None:
                    try:
                        duration = audio_duration(DATASET / track["source_mp3"])
                        durations[track_id] = duration
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        errors[track_id] = ("audio", f"{type(exc).__name__}: {exc}"[:500])
                        log(track_id, "audio", f"failed {errors[track_id][1]}")
                        publish(tracks, results, errors, durations, bpdr)
                        print(f"[{number}/{len(selected)}] {track['mtg_track_id']}: {errors[track_id][1]}", flush=True)
                        continue
                print(f"[{number}/{len(selected)}] {track['mtg_track_id']}", flush=True)
                required = (*STAGES, *(("bpdr",) if bpdr.includes(track_id) else ()))
                log(track_id, "song_start", f"remaining={','.join(stage for stage in required if stage not in stages)}")

                def attempt(stage: str, action) -> None:
                    if stage in stages:
                        return
                    print(f"  {stage}…", flush=True)
                    log(track_id, stage, "start")
                    try:
                        result = action()
                        stages[stage] = compact_result(result)
                        if errors.get(track_id, ("", ""))[0] == stage:
                            errors.pop(track_id, None)
                        count = (len(result["events"]) if stage in CHORDS or stage.startswith("roman_")
                                 else len(result["beats"]) if stage == "beats"
                                 else 1 if stage in ("key", "key_comparison", "bpdr") else len(result["notes"]))
                        log(track_id, stage, f"complete count={count}")
                        print(f"  {stage}: {count}", flush=True)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        errors[track_id] = (stage, f"{type(exc).__name__}: {exc}"[:500])
                        log(track_id, stage, f"failed {errors[track_id][1]}")
                        print(f"  {stage}: {errors[track_id][1]}", flush=True)
                    publish(tracks, results, errors, durations, bpdr)

                bpdr_needed = bpdr.includes(track_id) and "bpdr" not in stages
                if "game_vocal" not in stages:
                    attempt("game_vocal", lambda: extract_game(track, duration,
                                                               keep_stem=bpdr_needed))
                if ("game_vocal" not in stages
                        and errors.get(track_id, ("", ""))[0] == "game_vocal"
                        and model_resource_failure(errors[track_id][1])):
                    log(track_id, "song_end", "partial_or_failed; batch stopped after GAME worker failure")
                    log("*", "halted", f"GAME worker unavailable at {track_id}; saved stages remain available")
                    print("GAME worker ran out of memory or has no Metal device. "
                          "Stopping this batch to prevent more incomplete songs.", flush=True)
                    return 2
                if bpdr_needed and "game_vocal" in stages:
                    def export_bpdr() -> dict:
                        stem = obtain_game_stem(track, duration)
                        bpdr_result = bpdr.export(track, stem)
                        stem.unlink(missing_ok=True)
                        stem.with_suffix(".receipt.json").unlink(missing_ok=True)
                        return bpdr_result
                    attempt("bpdr", export_bpdr)
                    if "bpdr" not in stages:
                        log("*", "halted", f"BPDR failed at {track_id}; saved stages remain available")
                        print("BPDR failed; stopping this batch. Rerun with --retry-failed after resolving the error.", flush=True)
                        return 2
                attempt("sheetsage2_instrumental", lambda: sheet.transcribe(track, duration))
                if ("sheetsage2_instrumental" not in stages
                        and errors.get(track_id, ("", ""))[0] == "sheetsage2_instrumental"
                        and sheet_worker_failure(errors[track_id][1])):
                    log(track_id, "song_end", "partial_or_failed; batch stopped after worker failure")
                    log("*", "halted", f"SheetSage2 worker failed twice at {track_id}; saved stages remain available")
                    print("SheetSage2 failed twice after a fresh worker start. "
                          "Stopping this batch to prevent more incomplete songs.", flush=True)
                    return 2
                if "sheetsage2_instrumental" in stages and "beats" not in stages:
                    beat_result = beats_complete(track)
                    if beat_result is not None:
                        stages["beats"] = compact_result(beat_result)
                        log(track_id, "beats", f"complete count={len(beat_result['beats'])} (with SheetSage2 melody)")
                        publish(tracks, results, errors, durations, bpdr)
                    else:
                        attempt("beats", lambda: sheet.extract_beats(track, duration))
                if "game_vocal" in stages and "sheetsage2_instrumental" in stages:
                    def combined_result() -> dict:
                        combine(track, duration)
                        result = melody_complete("combined", track)
                        if result is None:
                            raise RuntimeError("Combined melody files failed validation")
                        return result
                    attempt("combined", combined_result)
                if "combined" in stages:
                    attempt("key", lambda: save_key(track))
                for vocabulary in CHORDS:
                    attempt(vocabulary, lambda vocab=vocabulary: extract_chords(
                        track, vocab, duration))
                    if (vocabulary not in stages
                            and errors.get(track_id, ("", ""))[0] == vocabulary
                            and model_resource_failure(errors[track_id][1])):
                        log(track_id, "song_end", "partial_or_failed; batch stopped after chord worker failure")
                        log("*", "halted", f"LV-Chordia worker failed at {track_id}; saved stages remain available")
                        print("LV-Chordia worker ran out of memory. "
                              "Stopping this batch to prevent more incomplete songs.", flush=True)
                        return 2
                if "key" in stages and "submission" in stages:
                    attempt("key_comparison", lambda: save_comparison(track))
                if "key" in stages:
                    for vocabulary in CHORDS:
                        if vocabulary in stages:
                            attempt(f"roman_{vocabulary}",
                                    lambda vocab=vocabulary: save_roman(track, vocab))
                completed = song_complete(track, stages, bpdr)
                log(track_id, "song_end", "complete" if completed else "partial_or_failed")
                print(f"  {'complete' if completed else 'incomplete'}; saved {len(stages)}/{len(required)} stages", flush=True)
            log("*", "run_end", f"selected={len(selected)} complete={sum(song_complete(t, results.get(t['track_id'], {}), bpdr) for t in selected)}")
        except KeyboardInterrupt:
            log("*", "interrupted", "Ctrl-C; rerun the same command")
            print("Stopped. Saved stages remain available for resume.", flush=True)
            return 130
        finally:
            sheet.close()
            publish(tracks, results, errors, durations, bpdr)
        return 0 if not any(track["track_id"] in errors for track in selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
