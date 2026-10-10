"""Run the official YourMT3+ Space checkpoint on the selected mixed-audio clips."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import torch
import torchaudio

from pipeline import CLIPS, RESULTS, ROOT, STATUS, TOOLS, normalize_midi, rows


SPACE = TOOLS / "YourMT3-Space"
sys.path.insert(0, str(SPACE))
sys.path.insert(0, str(SPACE / "amt" / "src"))
os.chdir(SPACE)

from model_helper import load_model_checkpoint  # noqa: E402
from utils.audio import slice_padded_array  # noqa: E402
from utils.event2note import merge_zipped_note_events_and_ties_to_notes  # noqa: E402
from utils.note2event import mix_notes  # noqa: E402
from utils.utils import write_model_output_as_midi  # noqa: E402


CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
MODEL_ARGS = [
    CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
    "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe", "-wf", "4",
    "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1",
    "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "32", "-w", "1",
]


def transcribe(model, track_id: str) -> None:
    result_dir = RESULTS / "yourmt3"
    result_dir.mkdir(parents=True, exist_ok=True)
    target = result_dir / f"{track_id}.mid"
    if target.exists() and target.with_suffix(".json").exists():
        return
    audio, sr = torchaudio.load(str(CLIPS / f"{track_id}.wav"))
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg["sample_rate"])
    segment_size = model.audio_cfg["input_frames"]
    segments = slice_padded_array(audio, segment_size, segment_size)
    segments = torch.from_numpy(segments.astype("float32")).unsqueeze(1)
    with torch.inference_mode():
        predicted, _ = model.inference_file(bsz=4, audio_segments=segments)
    start_secs = [segment_size * i / model.audio_cfg["sample_rate"] for i in range(len(segments))]
    decoded = []
    for channel in range(model.task_manager.num_decoding_channels):
        token_arrays = [array[:, channel, :] for array in predicted]
        zipped_events, _, _ = model.task_manager.detokenize_list_batches(
            token_arrays, start_secs, return_events=True
        )
        notes, _ = merge_zipped_note_events_and_ties_to_notes(zipped_events)
        decoded.append(notes)
    notes = mix_notes(decoded)
    with tempfile.TemporaryDirectory(prefix="yourmt3-") as temp:
        write_model_output_as_midi(notes, temp, track_id, model.midi_output_inverse_vocab)
        shutil.move(f"{temp}/model_output/{track_id}.mid", target)
    normalize_midi(target, target.with_suffix(".json"), "yourmt3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model_checkpoint(args=MODEL_ARGS, device="cpu")
    model.to(device)
    model.eval()
    print(f"YourMT3+ device: {device}", flush=True)
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    selected = rows()[:args.limit]
    for index, row in enumerate(selected, 1):
        track_id = row["track_id"]
        began = time.monotonic()
        try:
            transcribe(model, track_id)
            count = len(json.loads((RESULTS / "yourmt3" / f"{track_id}.json").read_text())["notes"])
            outcome = {"status": "ok" if count else "empty_prediction", "notes": count,
                       "seconds": round(time.monotonic() - began, 2)}
        except Exception as exc:
            outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        status.setdefault(track_id, {})["yourmt3"] = outcome
        STATUS.write_text(json.dumps(status, indent=2) + "\n")
        print(f"[{index}/{len(selected)}] yourmt3 {track_id}: {outcome}", flush=True)


if __name__ == "__main__":
    main()
