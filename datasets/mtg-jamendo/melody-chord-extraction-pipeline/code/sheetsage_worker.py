"""Keep one SheetSage2 model loaded while the parent sends full songs on stdin."""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from beats import complete as beats_complete, save as save_beats
from common import ASSETS, DATASET, melody_complete, write_notes


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


def transcribe(model, row: dict) -> tuple[int, int]:
    import pretty_midi

    track_id = row["track_id"]
    melody_done = melody_complete("sheetsage2_instrumental", row)
    beats_done = beats_complete(row)
    with tempfile.TemporaryDirectory(prefix="sheetsage2-") as folder:
        directory = Path(folder)
        model.transcribe(str(DATASET / row["source_mp3"]), output_dir=directory, dtype="fp32")
        if melody_done is None:
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
            melody_done = write_notes("sheetsage2_instrumental", row, notes,
                                      duration_sec=float(row["duration_sec"]), metadata={
                                          "median_beat_sec": median_beat(directory / "beat.lab"),
                                          "model_revision": MODEL_REVISION,
                                          "base_model_revision": BASE_REVISION,
                                      })
        if beats_done is None:
            beats_done = save_beats(row, float(row["duration_sec"]),
                                    directory / "beat.lab", directory / "downbeat.lab")
        return len(melody_done["notes"]), len(beats_done["beats"])


def main() -> None:
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
    print(f"READY\t{device}", flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue
        row = json.loads(line)
        track_id = row["track_id"]
        if melody_complete("sheetsage2_instrumental", row) and beats_complete(row):
            print(f"DONE\t{track_id}\tresumed", flush=True)
            continue
        start = time.monotonic()
        try:
            melody_count, beat_count = transcribe(model, row)
            print(f"DONE\t{track_id}\tmelody={melody_count}\tbeats={beat_count}\t{time.monotonic() - start:.1f}s", flush=True)
        except Exception as exc:
            print(f"ERROR\t{track_id}\t{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
