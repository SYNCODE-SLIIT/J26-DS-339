"""The established vocal separation + GAME extraction, with a +12 semitone vocal shift."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from common import ASSETS, DATASET, WORK, atomic_text, melody_complete, python_in, read_notes, write_notes


GAME_ROOT = ASSETS / "tools" / "GAME"
GAME_MODEL = ASSETS / "models" / "GAME-1.0-medium" / "model.pt"
SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def verify_melody_assets() -> None:
    required = (GAME_MODEL, GAME_ROOT / "infer.py",
                ASSETS / "models" / "mlx-audio-separator" / SEPARATOR_MODEL,
                python_in(ASSETS / ".venv-game"),
                python_in(ASSETS / ".venv-sheetsage2"),
                ASSETS / "models" / "hf-home")
    absent = [str(path) for path in required if not path.exists()]
    if absent:
        raise FileNotFoundError(f"Melody model assets missing: {absent}")


def parse_game_pitch(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        import pretty_midi

        match = re.fullmatch(r"([A-G][#b]?\d+)([+-]\d+)?", value)
        if match is None:
            raise ValueError(f"Unexpected GAME pitch {value!r}") from None
        return float(pretty_midi.note_name_to_number(match.group(1))) + float(match.group(2) or 0) / 100


class GameExtractor:
    def __init__(self) -> None:
        self.separator = None
        self.stems = WORK / "stems"
        self.mixes = WORK / "mixes"
        self.raw = WORK / "game_raw"
        for folder in (self.stems, self.mixes, self.raw):
            folder.mkdir(parents=True, exist_ok=True)

    def stem(self, track: dict[str, str], seconds: float, *, bpdr_verified: bool = False) -> Path:
        import soundfile as sf

        track_id = track["track_id"]
        target = self.stems / f"{track_id}.wav"
        receipt_path = self.stems / f"{track_id}.receipt.json"
        model_path = ASSETS / "models" / "mlx-audio-separator" / SEPARATOR_MODEL

        def digest(path: Path) -> str:
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()

        def expected_receipt() -> dict:
            return {"track_id": track_id, "source_mp3": track["source_mp3"],
                    "source_sha256": digest(DATASET / track["source_mp3"]),
                    "separator_sha256": digest(model_path),
                    "backend": "audio-separator-pytorch" if os.name == "nt" else "mlx-audio-separator"}

        source_mtime = (DATASET / track["source_mp3"]).stat().st_mtime_ns
        if target.exists():
            try:
                if (target.stat().st_mtime_ns >= source_mtime and
                        abs(sf.info(target).duration - seconds) < 2.0):
                    if not bpdr_verified:
                        return target
                    if receipt_path.is_file():
                        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                        if all(receipt.get(k) == v for k, v in expected_receipt().items()) \
                                and receipt.get("stem_sha256") == digest(target):
                            return target
            except (RuntimeError, ValueError, OSError):
                pass
            target.unlink()
            receipt_path.unlink(missing_ok=True)
        stereo_mix = self.mixes / f"{track_id}.wav"
        if (not stereo_mix.exists() or not stereo_mix.stat().st_size or
                stereo_mix.stat().st_mtime_ns < source_mtime):
            temporary_mix = self.mixes / f".{track_id}.tmp.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                            "-i", str(DATASET / track["source_mp3"]),
                            "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
                            str(temporary_mix)], check=True)
            temporary_mix.replace(stereo_mix)
        if os.name == "nt":
            worker = python_in(ASSETS / ".venv-separator-windows")
            if not worker.is_file():
                raise FileNotFoundError("Windows separator environment missing")
            result = subprocess.run([str(worker), str(Path(__file__).with_name("windows_separator_worker.py"))],
                                    input=__import__("json").dumps({
                                        "source": str(stereo_mix), "destination": str(target),
                                        "model": str(ASSETS / "models" / "mlx-audio-separator" / SEPARATOR_MODEL)}),
                                    capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(f"Windows vocal separator exited {result.returncode}: "
                                   f"{(result.stderr or result.stdout)[-900:]}")
            if abs(sf.info(target).duration - seconds) >= 2.0:
                target.unlink(missing_ok=True)
                raise RuntimeError("Windows separator stem duration mismatch")
            stereo_mix.unlink(missing_ok=True)
            if bpdr_verified:
                atomic_text(receipt_path, json.dumps({**expected_receipt(),
                            "stem_sha256": digest(target)}, indent=2) + "\n")
            return target
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
        if bpdr_verified:
            atomic_text(receipt_path, json.dumps({**expected_receipt(),
                        "stem_sha256": digest(target)}, indent=2) + "\n")
        return target

    def extract(self, track: dict[str, str], seconds: float, *, keep_stem: bool = False) -> dict:
        existing = melody_complete("game_vocal", track)
        if existing is not None:
            return existing
        track_id = track["track_id"]
        raw_csv = self.raw / f"{track_id}.csv"
        raw_mid = self.raw / f"{track_id}.mid"
        raw_csv.unlink(missing_ok=True)
        raw_mid.unlink(missing_ok=True)
        stem = self.stem(track, seconds, bpdr_verified=keep_stem)
        command = [str(python_in(ASSETS / ".venv-game")),
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
        data = write_notes("game_vocal", track, notes, duration_sec=seconds, metadata={
            "pitch_shift_semitones": 12,
            "model_source": "GAME-1.0-medium on separated vocal stem",
        })
        if not keep_stem:
            stem.unlink(missing_ok=True)
        raw_csv.unlink(missing_ok=True)
        raw_mid.unlink(missing_ok=True)
        return data
