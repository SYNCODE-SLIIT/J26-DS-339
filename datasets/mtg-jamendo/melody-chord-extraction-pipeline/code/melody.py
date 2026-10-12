"""The established vocal separation + GAME extraction, with a +12 semitone vocal shift."""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

from common import ASSETS, DATASET, WORK, melody_complete, read_notes, write_notes


GAME_ROOT = ASSETS / "tools" / "GAME"
GAME_MODEL = ASSETS / "models" / "GAME-1.0-medium" / "model.pt"
SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def verify_melody_assets() -> None:
    required = (GAME_MODEL, GAME_ROOT / "infer.py",
                ASSETS / "models" / "mlx-audio-separator" / SEPARATOR_MODEL,
                ASSETS / ".venv-game" / "bin" / "python",
                ASSETS / ".venv-sheetsage2" / "bin" / "python",
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

    def stem(self, track: dict[str, str], seconds: float) -> Path:
        import soundfile as sf

        track_id = track["track_id"]
        target = self.stems / f"{track_id}.wav"
        source_mtime = (DATASET / track["source_mp3"]).stat().st_mtime_ns
        if target.exists():
            try:
                if (target.stat().st_mtime_ns >= source_mtime and
                        abs(sf.info(target).duration - seconds) < 2.0):
                    return target
            except RuntimeError:
                pass
            target.unlink()
        stereo_mix = self.mixes / f"{track_id}.wav"
        if (not stereo_mix.exists() or not stereo_mix.stat().st_size or
                stereo_mix.stat().st_mtime_ns < source_mtime):
            temporary_mix = self.mixes / f".{track_id}.tmp.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y",
                            "-i", str(DATASET / track["source_mp3"]),
                            "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
                            str(temporary_mix)], check=True)
            temporary_mix.replace(stereo_mix)
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

    def extract(self, track: dict[str, str], seconds: float) -> dict:
        existing = melody_complete("game_vocal", track)
        if existing is not None:
            return existing
        track_id = track["track_id"]
        raw_csv = self.raw / f"{track_id}.csv"
        raw_mid = self.raw / f"{track_id}.mid"
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
        data = write_notes("game_vocal", track, notes, duration_sec=seconds, metadata={
            "pitch_shift_semitones": 12,
            "model_source": "GAME-1.0-medium on separated vocal stem",
        })
        stem.unlink(missing_ok=True)
        raw_csv.unlink(missing_ok=True)
        raw_mid.unlink(missing_ok=True)
        return data
