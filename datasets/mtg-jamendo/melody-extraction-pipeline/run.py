"""Resume full-song GAME vocal + SheetSage2 instrumental extraction.

From the repository root: .venv/bin/python datasets/mtg-jamendo/melody-extraction-pipeline/run.py --limit 10
Omit --limit to process all remaining songs.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from combine import combine  # noqa: E402
from common import ASSETS, DATASET, ROOT, WORK, atomic_text, complete, inventory, paths, read_notes, write_result  # noqa: E402


MANIFEST = ROOT / "manifest.csv"
CATALOG = ROOT / "catalog.json"
HISTORY = ROOT / "history.log"
GAME_ROOT = ASSETS / "tools" / "GAME"
GAME_MODEL = ASSETS / "models" / "GAME-1.0-medium" / "model.pt"
SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MANIFEST_COLUMNS = ("track_id", "mtg_track_id", "artist_id", "album_id", "annotation_path",
                    "source_mp3", "duration_annotation_sec", "duration_audio_sec",
                    "split_0", "split_1", "split_2", "split_3", "split_4", "tags",
                    "status", "game_notes", "sheetsage_notes", "combined_notes", "error", "updated_utc")
NOTE_COUNTS: dict[str, dict[str, int]] = {}


def log(track_id: str, stage: str, message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with HISTORY.open("a") as stream:
        stream.write(f"{stamp}\t{track_id}\t{stage}\t{message}\n")
        stream.flush()


def duration(path: Path) -> float:
    value = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
                                     "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                                     str(path)], text=True).strip()
    return float(value)


def prior_fields() -> tuple[dict[str, str], dict[str, str]]:
    if not MANIFEST.exists():
        return {}, {}
    with MANIFEST.open(newline="") as stream:
        rows = {row["track_id"]: row for row in csv.DictReader(stream)}
    for track_id, row in rows.items():
        NOTE_COUNTS[track_id] = {method: int(row[column]) for method, column in (
            ("game_vocal", "game_notes"), ("sheetsage2_instrumental", "sheetsage_notes"),
            ("combined", "combined_notes")) if row.get(column, "") != ""}
    return ({track_id: row["error"] for track_id, row in rows.items() if row.get("error")},
            {track_id: row["duration_audio_sec"] for track_id, row in rows.items()
             if row.get("duration_audio_sec")})


def publish_index(tracks: list[dict[str, str]], errors: dict[str, str],
                  durations: dict[str, str]) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    completed = []
    listed = []
    rows = []
    for track in tracks:
        track_id = track["track_id"]
        states = {method: complete(method, track_id) for method in
                  ("game_vocal", "sheetsage2_instrumental", "combined")}
        if states["combined"] and all(states.values()):
            status = "complete"
            errors.pop(track_id, None)
        elif track_id in errors:
            status = "failed"
        elif any(states.values()):
            status = "partial"
        else:
            status = "pending"
        known = NOTE_COUNTS.setdefault(track_id, {})
        counts = {}
        for method, exists in states.items():
            if exists and method not in known:
                known[method] = len(read_notes(method, track_id))
            counts[method] = known[method] if exists else ""
        rows.append({**track, "duration_audio_sec": durations.get(track_id, ""),
                     "status": status, "game_notes": counts["game_vocal"],
                     "sheetsage_notes": counts["sheetsage2_instrumental"],
                     "combined_notes": counts["combined"],
                     "error": errors.get(track_id, ""), "updated_utc": now if status != "pending" else ""})
        if any(states.values()):
            item = {
                "track_id": track_id, "mtg_track_id": track["mtg_track_id"],
                "artist_id": track["artist_id"], "split_0": track["split_0"],
                "duration_sec": float(durations.get(track_id) or track["duration_annotation_sec"] or 0),
                "audio": f"../{track['source_mp3']}",
                "notes": {method: f"outputs/{method}/{track_id}.json" for method in states if states[method]},
                "midi": {method: f"outputs/{method}/{track_id}.mid" for method in states if states[method]},
                "csv": {method: f"outputs/{method}/{track_id}.csv" for method in states if states[method]},
                "counts": counts, "available": states, "status": status,
            }
            listed.append(item)
            if status == "complete":
                completed.append(item)
    temporary = MANIFEST.with_suffix(".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, MANIFEST)
    atomic_text(CATALOG, json.dumps({"schema_version": 1, "updated_utc": now,
                                     "completed_count": len(completed), "tracks": listed},
                                    separators=(",", ":")) + "\n")


def parse_game_pitch(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        import pretty_midi
        match = re.fullmatch(r"([A-G][#b]?\d+)([+-]\d+)?", value)
        if match is None:
            raise ValueError(f"Unexpected GAME pitch {value!r}")
        return float(pretty_midi.note_name_to_number(match.group(1))) + float(match.group(2) or 0) / 100


class GameExtractor:
    def __init__(self) -> None:
        self.separator = None
        self.stems = WORK / "stems"
        self.mixes = WORK / "mixes"
        self.raw = WORK / "game_raw"
        self.stems.mkdir(parents=True, exist_ok=True)
        self.mixes.mkdir(parents=True, exist_ok=True)
        self.raw.mkdir(parents=True, exist_ok=True)

    def stem(self, track: dict[str, str], seconds: float) -> Path:
        import soundfile as sf

        track_id = track["track_id"]
        target = self.stems / f"{track_id}.wav"
        if target.exists():
            try:
                if abs(sf.info(target).duration - seconds) < 2.0:
                    return target
            except RuntimeError:
                pass
            target.unlink()
        stereo_mix = self.mixes / f"{track_id}.wav"
        if not stereo_mix.exists() or not stereo_mix.stat().st_size:
            temporary_mix = self.mixes / f".{track_id}.tmp.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                            "-i", str(DATASET / track["source_mp3"]),
                            "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
                            str(temporary_mix)], check=True)
            os.replace(temporary_mix, stereo_mix)
        if self.separator is None:
            from mlx_audio_separator import Separator
            self.separator = Separator(output_dir=str(self.stems),
                                       model_file_dir=str(ASSETS / "models" / "mlx-audio-separator"),
                                       output_format="WAV", output_single_stem="Vocals")
            self.separator.load_model(SEPARATOR_MODEL)
        temporary_stem = self.stems / f"{track_id}.vocal_tmp.wav"
        temporary_stem.unlink(missing_ok=True)
        found = self.separator.separate(
            str(stereo_mix), custom_output_names={"Vocals": f"{track_id}.vocal_tmp"})
        available = [Path(path) for path in found]
        available = [path if path.is_absolute() else self.stems / path for path in available]
        available = [path for path in available if path.exists()]
        if len(available) == 1 and abs(sf.info(available[0]).duration - seconds) < 2.0:
            available[0].replace(target)
        if not target.exists() or not target.stat().st_size:
            raise RuntimeError(f"Separator produced no vocal WAV: {found}")
        stereo_mix.unlink(missing_ok=True)
        return target

    def extract(self, track: dict[str, str], seconds: float) -> int:
        track_id = track["track_id"]
        if complete("game_vocal", track_id):
            return len(read_notes("game_vocal", track_id))
        raw_csv = self.raw / f"{track_id}.csv"
        raw_mid = self.raw / f"{track_id}.mid"
        # A previous inference may have stopped mid-write; only final triplets are reusable.
        raw_csv.unlink(missing_ok=True)
        raw_mid.unlink(missing_ok=True)
        stem = self.stem(track, seconds)
        command = [str(ASSETS / ".venv-game" / "bin" / "python"),
                   str(GAME_ROOT / "infer.py"), "extract", str(stem),
                   "-m", str(GAME_MODEL), "--output-formats", "mid,csv",
                   "--pitch-format", "number", "--output-dir", str(self.raw)]
        result = subprocess.run(command, cwd=GAME_ROOT, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"GAME exited {result.returncode}: {(result.stderr or result.stdout)[-1500:]}")
        if raw_csv.exists() != raw_mid.exists():
            raise RuntimeError("GAME created only one of its MIDI/CSV files")
        notes = []
        if raw_csv.exists():
            with raw_csv.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    original = round(parse_game_pitch(row["pitch"]), 5)
                    shifted = round(original + 12.0, 5)
                    rounded = round(shifted)
                    if not 0 <= rounded <= 127:
                        raise ValueError(f"Octave-shifted pitch out of MIDI range: {shifted}")
                    notes.append({"onset_sec": float(row["onset"]),
                                  "offset_sec": float(row["offset"]),
                                  "midi_pitch": shifted, "midi_pitch_rounded": rounded,
                                  "model_midi_pitch": original, "velocity": 100,
                                  "source": "game_vocal"})
        write_result("game_vocal", track_id, notes, metadata={
            "duration_sec": seconds, "pitch_shift_semitones": 12,
            "model_source": "GAME-1.0-medium on separated vocal stem",
        })
        (self.stems / f"{track_id}.wav").unlink(missing_ok=True)
        raw_csv.unlink(missing_ok=True)
        raw_mid.unlink(missing_ok=True)
        return len(notes)


def run_sheet_queue(queue: list[dict], tracks: list[dict[str, str]],
                    errors: dict[str, str], durations: dict[str, str]) -> None:
    queue_path = WORK / "sheet_queue.json"
    atomic_text(queue_path, json.dumps(queue))
    env = os.environ.copy()
    env.update({"PYTORCH_ENABLE_MPS_FALLBACK": "1", "HF_HUB_OFFLINE": "1"})
    command = [str(ASSETS / ".venv-sheetsage2" / "bin" / "python"),
               str(ROOT / "code" / "sheetsage_worker.py"), str(queue_path)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=env)
    assert process.stdout is not None
    try:
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith("DONE\t"):
                track_id = line.split("\t")[1]
                try:
                    counts = combine(track_id, float(durations[track_id]))
                    errors.pop(track_id, None)
                    log(track_id, "complete", f"{counts}")
                    publish_index(tracks, errors, durations)
                    print(f"{track_id}: complete ({counts['combined']} combined notes)", flush=True)
                except Exception as exc:
                    errors[track_id] = f"combine: {type(exc).__name__}: {exc}"
                    log(track_id, "failed", errors[track_id])
                    publish_index(tracks, errors, durations)
                    print(f"{track_id}: {errors[track_id]}", flush=True)
            elif line.startswith("ERROR\t"):
                _, track_id, message = line.split("\t", 2)
                errors[track_id] = f"sheetsage2: {message}"
                log(track_id, "failed", errors[track_id])
                publish_index(tracks, errors, durations)
                print(f"{track_id}: {errors[track_id]}", flush=True)
            elif line.startswith("DEVICE\t"):
                print(f"SheetSage2 device: {line.split(chr(9), 1)[1]}", flush=True)
            else:
                with (WORK / "sheetsage.log").open("a") as stream:
                    stream.write(line + "\n")
        if process.wait():
            raise RuntimeError(f"SheetSage2 worker exited {process.returncode}; see {WORK / 'sheetsage.log'}")
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Number of next incomplete songs to process, then stop")
    parser.add_argument("--retry-failed", action="store_true", help="Include songs previously marked failed")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another extraction run is active") from exc
        tracks = inventory()
        errors, durations = prior_fields()
        publish_index(tracks, errors, durations)
        selected = [track for track in tracks if not all(complete(method, track["track_id"])
                    for method in ("game_vocal", "sheetsage2_instrumental", "combined"))
                    and (args.retry_failed or track["track_id"] not in errors)]
        if args.limit is not None:
            selected = selected[:args.limit]
        print(f"Selected {len(selected)} full songs; {len(tracks)} local MP3s in inventory.", flush=True)
        log("*", "run_start", f"selected={len(selected)} inventory={len(tracks)} retry_failed={args.retry_failed}")
        if not selected:
            log("*", "run_end", "nothing to do")
            return
        extractor = GameExtractor()
        ready = []
        for index, track in enumerate(selected, 1):
            track_id = track["track_id"]
            print(f"[{index}/{len(selected)}] {track_id}: GAME vocal", flush=True)
            try:
                seconds = float(durations.get(track_id) or duration(DATASET / track["source_mp3"]))
                durations[track_id] = str(round(seconds, 3))
                resumed_game = complete("game_vocal", track_id)
                count = extractor.extract(track, seconds)
                log(track_id, "game_vocal", f"{'resumed' if resumed_game else 'ok'} notes={count} octave_shift=12")
                publish_index(tracks, errors, durations)
                if complete("sheetsage2_instrumental", track_id):
                    counts = combine(track_id, seconds)
                    errors.pop(track_id, None)
                    log(track_id, "complete", f"{counts}")
                    publish_index(tracks, errors, durations)
                else:
                    ready.append({"track_id": track_id, "source_mp3": track["source_mp3"],
                                  "duration_sec": seconds})
            except KeyboardInterrupt:
                log(track_id, "interrupted", "Ctrl-C; existing outputs remain for resume")
                raise
            except Exception as exc:
                errors[track_id] = f"game_vocal/combine: {type(exc).__name__}: {exc}"
                log(track_id, "failed", errors[track_id])
                publish_index(tracks, errors, durations)
                print(f"{track_id}: {errors[track_id]}", flush=True)
        if ready:
            print(f"SheetSage2 instrumental: {len(ready)} full songs", flush=True)
            run_sheet_queue(ready, tracks, errors, durations)
        finished = sum(complete("combined", track["track_id"]) for track in selected)
        log("*", "run_end", f"complete={finished}/{len(selected)}")
        print(f"Finished this selection: {finished}/{len(selected)} songs. Run the same command to continue.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("*", "interrupted", "Ctrl-C; rerun the same command to continue")
        print("Stopped. Completed files are saved; rerun the same command to continue.", flush=True)
        sys.exit(130)
