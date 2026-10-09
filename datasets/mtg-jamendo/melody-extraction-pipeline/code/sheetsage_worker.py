"""Transcribe full MP3s with one loaded SheetSage2 model."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
from pathlib import Path

from common import ASSETS, DATASET, complete, write_result


HF_HOME = ASSETS / "models" / "hf-home"
os.environ.setdefault("HF_HOME", str(HF_HOME))
MODEL_REVISION = "32c7c7473e7b59e1ed36bd8a8fefbd11ae8d468a"
BASE_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
BASE = HF_HOME / "hub" / "models--m-a-p--MERT-v2-FullSong" / "snapshots" / BASE_REVISION


def median_beat(path: Path) -> float:
    if not path.exists():
        return 0.5
    times = []
    for line in path.read_text().splitlines():
        try:
            times.append(float(line.split()[0]))
        except (IndexError, ValueError):
            continue
    intervals = [right - left for left, right in zip(times, times[1:])
                 if 0.2 <= right - left <= 2.0]
    return float(statistics.median(intervals)) if len(intervals) >= 3 else 0.5


def transcribe(model, row: dict) -> int:
    import pretty_midi

    track_id = row["track_id"]
    with tempfile.TemporaryDirectory(prefix="sheetsage2-") as folder:
        directory = Path(folder)
        model.transcribe(str(DATASET / row["source_mp3"]), output_dir=directory, dtype="fp32")
        midi_path = directory / "melody_instrumental.mid"
        if not midi_path.exists():
            raise FileNotFoundError(midi_path)
        notes = []
        for instrument in pretty_midi.PrettyMIDI(str(midi_path)).instruments:
            for note in instrument.notes:
                notes.append({"onset_sec": round(float(note.start), 5),
                              "offset_sec": round(float(note.end), 5),
                              "midi_pitch": int(note.pitch),
                              "midi_pitch_rounded": int(note.pitch),
                              "velocity": int(note.velocity),
                              "source": "sheetsage2_instrumental"})
        write_result("sheetsage2_instrumental", track_id, notes, metadata={
            "duration_sec": row["duration_sec"],
            "median_beat_sec": median_beat(directory / "beat.lab"),
            "model_revision": MODEL_REVISION,
            "base_model_revision": BASE_REVISION,
        })
        return len(notes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path, help="JSON list of track IDs, MP3 paths, and durations")
    args = parser.parse_args()
    import torch
    from transformers import AutoModel

    if not BASE.joinpath("model.safetensors").exists():
        raise FileNotFoundError("Pinned MERT-v2-FullSong weights are missing")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = AutoModel.from_pretrained(
        "m-a-p/SheetSage2", revision=MODEL_REVISION, trust_remote_code=True,
        local_files_only=True, base_model_path=str(BASE), torch_dtype=torch.float32,
    ).eval().to(device)
    print(f"DEVICE\t{device}", flush=True)
    for row in json.loads(args.queue.read_text()):
        track_id = row["track_id"]
        if complete("sheetsage2_instrumental", track_id):
            print(f"DONE\t{track_id}\tresumed", flush=True)
            continue
        start = time.monotonic()
        try:
            count = transcribe(model, row)
            print(f"DONE\t{track_id}\t{count}\t{time.monotonic() - start:.1f}s", flush=True)
        except Exception as exc:
            print(f"ERROR\t{track_id}\t{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
