"""Pick fresh songs, run every extractor, and replace round-2/compare.html.

Run with the repository's .venv Python. The current comparison stays in place
while the next one is computed in a sibling staging directory.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
ROUND = PROJECT / "round-2"
STAGE = PROJECT / ".round-2-next"
BACKUP = PROJECT / ".round-2-previous"
PIPELINE = PROJECT / "code" / "pipeline.py"
PYTHON = REPO / ".venv" / "bin" / "python"
HISTORY = ROUND / "seen_tracks.csv"
STATUS_STAGES = ("prepare", "separate", "game_vocal", "game_mix", "sheetsage2", "yourmt3", "combined")


def ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as stream:
        return {row["track_id"] for row in csv.DictReader(stream)}


def save_ids(path: Path, track_ids: set[str]) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["track_id"])
        writer.writerows([track_id] for track_id in sorted(track_ids))
    os.replace(temp, path)


def run(name: str, command: list[str], env: dict[str, str]) -> None:
    print(f"\n== {name} ==", flush=True)
    log = STAGE / f"{name}.log"
    with log.open("w") as output:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        tail: list[str] = []
        for line in process.stdout:
            output.write(line)
            output.flush()
            tail.append(line.rstrip())
            tail = tail[-25:]
            if re.match(r"^\[\d+/\d+\]", line) or line.startswith("Selected ") or line.startswith("Exported "):
                print(line.rstrip(), flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"{name} failed (exit {code}); see {log}\n" + "\n".join(tail))


def check_status(track_ids: set[str], stage: str) -> None:
    status = json.loads((STAGE / "run_status.json").read_text())
    failed = {track_id: status.get(track_id, {}).get(stage) for track_id in track_ids
              if status.get(track_id, {}).get(stage, {}).get("status") not in ("ok", "empty_prediction")}
    if failed:
        raise RuntimeError(f"{stage} failed for {failed}")


def validate(track_ids: set[str]) -> None:
    import pretty_midi

    rows = list(csv.DictReader((STAGE / "sample.csv").open(newline="")))
    assert {row["track_id"] for row in rows} == track_ids
    for stage in STATUS_STAGES:
        check_status(track_ids, stage)
    html = (STAGE / "compare.html").read_text()
    match = re.search(r"const songs = (\[.*?\]);", html)
    assert match is not None
    songs = json.loads(match.group(1))
    assert [song["id"] for song in songs] == [row["track_id"] for row in rows]
    duration_by_id = {row["track_id"]: float(row["clip_duration_sec"]) for row in rows}
    for song in songs:
        assert (STAGE / song["original"]).exists()
        assert (STAGE / song["mp3"]).exists()
        assert abs(sf.info(STAGE / song["original"]).duration - duration_by_id[song["id"]]) < 0.1
        labels = [choice["label"] for choice in song["choices"]]
        assert labels[0].startswith("Combined")
        assert any("YourMT3+" in label for label in labels)
        assert any("SheetSage2" in label and "instrumental" in label for label in labels)
        for choice in song["choices"]:
            for field in ("audio", "midi", "json"):
                assert (STAGE / choice[field]).exists(), (song["id"], field)
        combined = STAGE / "results" / "combined" / f"{song['id']}.json"
        notes = json.loads(combined.read_text())["notes"]
        assert all(a["offset_sec"] <= b["onset_sec"] + 1e-6 for a, b in zip(notes, notes[1:]))
        midi = pretty_midi.PrettyMIDI(str(combined.with_suffix(".mid")))
        assert sum(len(instrument.notes) for instrument in midi.instruments) == len(notes)
    print(f"Validated {len(songs)} songs and all linked prediction files.", flush=True)


def publish(seen: set[str], new_ids: set[str]) -> None:
    if BACKUP.exists():
        raise RuntimeError(f"Previous publish backup exists; inspect it before rerunning: {BACKUP}")
    BACKUP.mkdir()
    names = ["sample.csv", "clips", "stems", "results", "run_status.json", "review.csv",
             "summary.csv", "report.html", "run-id.txt", "compare.html", "run-version.js"]
    names += sorted({path.name for path in ROUND.glob("*.log")} | {path.name for path in STAGE.glob("*.log")})
    moved_old: list[str] = []
    moved_new: list[str] = []
    try:
        for name in names:
            old = ROUND / name
            if old.exists():
                old.rename(BACKUP / name)
                moved_old.append(name)
        for name in names:
            new = STAGE / name
            if new.exists():
                new.rename(ROUND / name)
                moved_new.append(name)
        save_ids(HISTORY, seen | new_ids)
    except Exception:
        for name in reversed(moved_new):
            (ROUND / name).rename(STAGE / name)
        for name in reversed(moved_old):
            (BACKUP / name).rename(ROUND / name)
        raise
    shutil.rmtree(BACKUP)
    shutil.rmtree(STAGE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--seed", type=int, help="Optional repeatable random seed")
    args = parser.parse_args()
    if args.count < 1 or args.seconds <= 0:
        parser.error("count and seconds must be positive")
    if BACKUP.exists():
        raise RuntimeError(f"Previous publish backup exists; inspect it first: {BACKUP}")
    ROUND.mkdir(exist_ok=True)
    with (ROUND / ".next-comparison.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another comparison run is already active") from exc
        seen = set().union(*(ids(path) for path in (
            PROJECT / "sample.csv", ROUND / "rejected_sample.csv", ROUND / "sample.csv", HISTORY)))
        if STAGE.exists():
            shutil.rmtree(STAGE)
        STAGE.mkdir()
        save_ids(STAGE / "excluded.csv", seen)
        seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
        run_id = f"{int(time.time())}-{seed}"
        (STAGE / "run-id.txt").write_text(run_id + "\n")
        (STAGE / "run-version.js").write_text(f"window.MELODY_RUN_VERSION = {json.dumps(run_id)};\n")
        env = os.environ.copy()
        env["MELODY_RUN_DIR"] = str(STAGE)
        run("sample", [str(PYTHON), str(PIPELINE), "sample", "--count", str(args.count),
                       "--seconds", str(args.seconds), "--seed", str(seed), "--unique-artists",
                       "--exclude-sample", str(STAGE / "excluded.csv")], env)
        new_ids = ids(STAGE / "sample.csv")
        assert len(new_ids) == args.count and not new_ids & seen
        for stage in ("prepare", "separate", "game_vocal", "game_mix"):
            run(stage, [str(PYTHON), str(PIPELINE), stage], env)
            check_status(new_ids, stage)
        sheet_env = env | {"PYTORCH_ENABLE_MPS_FALLBACK": "1", "HF_HUB_OFFLINE": "1"}
        run("sheetsage2", [str(PROJECT / ".venv-sheetsage2" / "bin" / "python"),
                          str(PROJECT / "code" / "sheetsage2_worker.py")], sheet_env)
        check_status(new_ids, "sheetsage2")
        run("combined", [str(PYTHON), str(PROJECT / "code" / "combine_melody.py")], env)
        check_status(new_ids, "combined")
        run("yourmt3", [str(PROJECT / ".venv-yourmt3" / "bin" / "python"),
                        str(PROJECT / "code" / "yourmt3_worker.py")], env)
        check_status(new_ids, "yourmt3")
        for stage in ("yourmt3_parts", "review_template", "summary", "report", "compare"):
            run(stage, [str(PYTHON), str(PIPELINE), stage], env)
        validate(new_ids)
        publish(seen, new_ids)
        print(f"\nUpdated {ROUND / 'compare.html'} with {len(new_ids)} new songs (seed {seed}).", flush=True)


if __name__ == "__main__":
    main()
