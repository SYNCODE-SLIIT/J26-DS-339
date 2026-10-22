"""Run the pinned SheetSage2 model on the same excerpts as the pilot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from pipeline import CLIPS, MODELS, RESULTS, STATUS, normalize_midi, preview, rows


HF_HOME = MODELS / "hf-home"
os.environ.setdefault("HF_HOME", str(HF_HOME))
MODEL_REVISION = "32c7c7473e7b59e1ed36bd8a8fefbd11ae8d468a"
BASE_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"
MODEL = HF_HOME / "hub" / "models--m-a-p--SheetSage2" / "snapshots" / MODEL_REVISION
BASE = HF_HOME / "hub" / "models--m-a-p--MERT-v2-FullSong" / "snapshots" / BASE_REVISION


def transcribe(model, row: dict[str, str]) -> dict[str, int]:
    track_id = row["track_id"]
    result_dir = RESULTS / "sheetsage2"
    result_dir.mkdir(parents=True, exist_ok=True)
    targets = {part: result_dir / f"{track_id}.{part}.mid" for part in ("vocal", "instrumental")}
    if all(path.exists() and path.with_suffix(".json").exists() for path in targets.values()):
        return {part: len(json.loads(path.with_suffix(".json").read_text())["notes"])
                for part, path in targets.items()}

    with tempfile.TemporaryDirectory(prefix="sheetsage2-") as folder:
        output = Path(folder)
        model.transcribe(str(CLIPS / f"{track_id}.wav"), output_dir=output, dtype="fp32")
        for part, target in targets.items():
            source = output / f"melody_{part}.mid"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, target)
            normalize_midi(target, target.with_suffix(".json"), f"sheetsage2_{part}")
            preview(target, target.with_suffix(".preview.wav"), float(row["clip_duration_sec"]))
        for source_name, suffix in (("events.json", "events.json"), ("beat.lab", "beat.lab"),
                                    ("downbeat.lab", "downbeat.lab"), ("chord.lab", "chord.lab"),
                                    ("score.abc", "score.abc")):
            source = output / source_name
            if source.exists():
                shutil.copy2(source, result_dir / f"{track_id}.{suffix}")

    return {part: len(json.loads(path.with_suffix(".json").read_text())["notes"])
            for part, path in targets.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel

    if not MODEL.joinpath("model.safetensors").exists() or not BASE.joinpath("model.safetensors").exists():
        raise FileNotFoundError("Download the pinned SheetSage2 and MERT-v2-FullSong snapshots first")
    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = AutoModel.from_pretrained(
        "m-a-p/SheetSage2", revision=MODEL_REVISION,
        trust_remote_code=True, local_files_only=True,
        base_model_path=str(BASE), torch_dtype=torch.float32,
    ).eval().to(device)
    print(f"SheetSage2 device: {device}", flush=True)

    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    selected = rows()[:args.limit]
    for index, row in enumerate(selected, 1):
        track_id = row["track_id"]
        began = time.monotonic()
        try:
            counts = transcribe(model, row)
            outcome = {"status": "ok", "notes": counts,
                       "seconds": round(time.monotonic() - began, 2)}
        except Exception as exc:
            outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        status.setdefault(track_id, {})["sheetsage2"] = outcome
        STATUS.write_text(json.dumps(status, indent=2) + "\n")
        print(f"[{index}/{len(selected)}] sheetsage2 {track_id}: {outcome}", flush=True)


if __name__ == "__main__":
    main()
