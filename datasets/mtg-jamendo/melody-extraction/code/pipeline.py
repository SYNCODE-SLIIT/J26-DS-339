"""Small, reproducible audio-to-note comparison on MTG-Jamendo songs.

All generated files stay under melody-extraction.  Each transcriber owns one
flat results directory, keyed by the original Jamendo track id.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf


PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("MELODY_RUN_DIR", PROJECT)).resolve()
DATASET = PROJECT.parent
INPUT = DATASET / "vocal-tagged-audio"
SAMPLE = ROOT / "sample.csv"
CLIPS = ROOT / "clips"
STEMS = ROOT / "stems"
RESULTS = ROOT / "results"
MODELS = PROJECT / "models"
TOOLS = PROJECT / "tools"
STATUS = ROOT / "run_status.json"
GAME_PYTHON = PROJECT / ".venv-game" / "bin" / "python"
GAME_ROOT = TOOLS / "GAME"
GAME_MODEL = MODELS / "GAME-1.0-medium" / "model.pt"
SEPARATOR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def call(argv: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def duration(path: Path) -> float:
    output = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)], text=True
    )
    return float(output.strip())


def sample_songs(count: int, seconds: float, seed: int, overwrite: bool,
                 exclude_samples: list[Path] | None = None,
                 unique_artists: bool = False) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if SAMPLE.exists() and not overwrite:
        raise FileExistsError(f"{SAMPLE} already exists; pass --overwrite-sample to replace it")
    paths = sorted(INPUT.rglob("*.mp3"))
    previous: set[str] = set()
    for excluded in exclude_samples or []:
        with excluded.open(newline="") as stream:
            previous.update(row["track_id"] for row in csv.DictReader(stream))
    if previous:
        paths = [path for path in paths if path.name.split(".")[0] not in previous]
    if len(paths) < count:
        raise ValueError(f"Only {len(paths)} MP3s found, requested {count}")
    rng = random.Random(seed)
    if unique_artists:
        artists = {}
        with (DATASET / "annotations" / "autotagging.tsv").open() as stream:
            next(stream)
            for line in stream:
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) >= 4:
                    artists[Path(fields[3]).stem] = fields[1]
        groups: dict[str, list[Path]] = {}
        for path in paths:
            track_id = path.name.split(".")[0]
            groups.setdefault(artists.get(track_id, f"unknown:{track_id}"), []).append(path)
        if len(groups) < count:
            raise ValueError(f"Only {len(groups)} artists found, requested {count}")
        paths = [rng.choice(groups[artist]) for artist in rng.sample(sorted(groups), count)]
        rng.shuffle(paths)
    else:
        paths = rng.sample(paths, count)
    rows = []
    for path in paths:
        source_duration = duration(path)
        max_start = max(0.0, source_duration - seconds)
        start = rng.uniform(max_start * 0.15, max_start * 0.65)
        rows.append({
            "track_id": path.name.split(".")[0],
            "source_path": str(path.relative_to(DATASET)),
            "source_duration_sec": round(source_duration, 3),
            "clip_start_sec": round(start, 3),
            "clip_duration_sec": min(seconds, round(source_duration - start, 3)),
            "seed": seed,
        })
    with SAMPLE.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected {len(rows)} songs from {len(paths)} MP3s: {SAMPLE}")


def rows() -> list[dict[str, str]]:
    with SAMPLE.open(newline="") as stream:
        return list(csv.DictReader(stream))


def clip_path(track_id: str) -> Path:
    return CLIPS / f"{track_id}.wav"


def stem_path(track_id: str) -> Path:
    return STEMS / f"{track_id}.wav"


def prepare(row: dict[str, str]) -> None:
    CLIPS.mkdir(exist_ok=True)
    target = clip_path(row["track_id"])
    if target.exists():
        return
    source = DATASET / row["source_path"]
    call(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", row["clip_start_sec"],
          "-i", str(source), "-t", row["clip_duration_sec"], "-ar", "44100",
          "-ac", "2", "-c:a", "pcm_s16le", str(target)])


class VocalSeparator:
    def __init__(self) -> None:
        from mlx_audio_separator import Separator
        STEMS.mkdir(exist_ok=True)
        self.separator = Separator(
            output_dir=str(STEMS),
            model_file_dir=str(MODELS / "mlx-audio-separator"),
            output_format="WAV",
            output_single_stem="Vocals",
        )
        self.separator.load_model(SEPARATOR_MODEL)

    def run(self, row: dict[str, str]) -> None:
        track_id = row["track_id"]
        target = stem_path(track_id)
        if target.exists():
            return
        paths = self.separator.separate(
            str(clip_path(track_id)), custom_output_names={"Vocals": track_id}
        )
        found = [Path(p) for p in paths]
        found = [p if p.is_absolute() else STEMS / p for p in found]
        found = [p for p in found if p.exists()]
        if not target.exists() and len(found) == 1:
            found[0].replace(target)
        if not target.exists():
            raise RuntimeError(f"No vocal stem for {track_id}: {paths}")


def game(row: dict[str, str], *, vocal: bool) -> None:
    track_id = row["track_id"]
    method = "game_vocal" if vocal else "game_mix"
    result_dir = RESULTS / method
    result_dir.mkdir(parents=True, exist_ok=True)
    target = result_dir / f"{track_id}.mid"
    json_target = result_dir / f"{track_id}.json"
    if target.exists() and json_target.exists():
        return
    source = stem_path(track_id) if vocal else clip_path(track_id)
    if not source.exists():
        raise FileNotFoundError(source)
    if not GAME_MODEL.exists():
        raise FileNotFoundError(GAME_MODEL)
    if not target.exists():
        call([str(GAME_PYTHON), str(GAME_ROOT / "infer.py"), "extract", str(source),
              "-m", str(GAME_MODEL), "--output-formats", "mid,csv",
              "--pitch-format", "number", "--output-dir", str(result_dir)], cwd=GAME_ROOT)
    if not target.exists():
        raise RuntimeError(f"GAME did not create {target}")
    normalize_game(target.with_suffix(".csv"), json_target, method)


def normalize_game(csv_path: Path, json_path: Path, method: str) -> None:
    notes = []
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            pitch_text = row["pitch"]
            if pitch_text.replace(".", "", 1).isdigit():
                pitch = float(pitch_text)
            else:
                # GAME also supports note names with cents, e.g. G#2+6.
                import re
                match = re.fullmatch(r"([A-G][#b]?\d+)([+-]\d+)?", pitch_text)
                if not match:
                    raise ValueError(f"Unexpected GAME pitch: {pitch_text!r}")
                pitch = float(pretty_midi.note_name_to_number(match.group(1)))
                pitch += float(match.group(2) or 0) / 100.0
            notes.append({
                "onset_sec": float(row["onset"]),
                "offset_sec": float(row["offset"]),
                "midi_pitch": pitch,
                "part": "lead_vocal" if method == "game_vocal" else "lead_candidate",
            })
    json_path.write_text(json.dumps({
        "track_id": csv_path.stem, "method": method, "time_origin": "clip_start",
        "pitch_unit": "MIDI semitones; decimals preserve cents", "notes": notes,
    }, indent=2) + "\n")


def normalize_midi(midi_path: Path, json_path: Path, method: str) -> None:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append({
                "onset_sec": round(float(note.start), 5),
                "offset_sec": round(float(note.end), 5),
                "midi_pitch": int(note.pitch),
                "velocity": int(note.velocity),
                "part": instrument.name or ("drums" if instrument.is_drum else "unknown"),
                "program": int(instrument.program),
                "is_drum": bool(instrument.is_drum),
            })
    notes.sort(key=lambda n: (n["onset_sec"], n["midi_pitch"]))
    json_path.write_text(json.dumps({
        "track_id": midi_path.stem.split(".")[0], "method": method, "time_origin": "clip_start",
        "notes": notes,
    }, indent=2) + "\n")


def export_yourmt3_parts() -> None:
    part_dir = RESULTS / "yourmt3" / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows():
        track_id = row["track_id"]
        source = RESULTS / "yourmt3" / f"{track_id}.mid"
        if not source.exists():
            continue
        midi = pretty_midi.PrettyMIDI(str(source))
        for index, instrument in enumerate(midi.instruments):
            if instrument.is_drum or not instrument.notes:
                continue
            target = part_dir / f"{track_id}.p{instrument.program:03d}.{index}.mid"
            if not target.exists():
                part_midi = pretty_midi.PrettyMIDI()
                part_midi.instruments.append(instrument)
                part_midi.write(str(target))
            normalize_midi(target, target.with_suffix(".json"), "yourmt3_part")
            count += 1
    print(f"Exported {count} non-drum instrument parts to {part_dir}")


def preview(midi_path: Path, audio_path: Path, seconds: float) -> None:
    if audio_path.exists():
        return
    sr = 22050
    samples = np.zeros(int(math.ceil(seconds * sr)), dtype=np.float32)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            start = max(0, int(note.start * sr))
            end = min(len(samples), int(note.end * sr))
            if end <= start:
                continue
            t = np.arange(end - start, dtype=np.float32) / sr
            freq = pretty_midi.note_number_to_hz(note.pitch)
            envelope = np.minimum(1, t / 0.015) * np.minimum(1, (t[-1] - t) / 0.04)
            samples[start:end] += (0.08 * np.sin(2 * np.pi * freq * t) * envelope)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0
    if peak > 0.95:
        samples *= 0.95 / peak
    sf.write(audio_path, samples, sr)


def make_report() -> None:
    blocks = []
    for row in rows():
        track_id = row["track_id"]
        parts = [f"<h2>{html.escape(track_id)}</h2>",
                 f"<p>Original: {html.escape(row['source_path'])}; excerpt starts at {row['clip_start_sec']} s</p>"]
        for label, path in [("Mix", clip_path(track_id)), ("Vocal stem", stem_path(track_id))]:
            if path.exists():
                parts.append(f"<label>{label}<audio controls preload='none' src='{path.relative_to(ROOT)}'></audio></label>")
        for method in ("combined", "game_vocal", "game_mix", "yourmt3"):
            midi = RESULTS / method / f"{track_id}.mid"
            if not midi.exists():
                continue
            sound = midi.with_suffix(".preview.wav")
            preview(midi, sound, float(row["clip_duration_sec"]))
            data = json.loads(midi.with_suffix(".json").read_text())
            display = "GAME vocal + SheetSage2 instrumental" if method == "combined" else method
            parts.append(f"<label>{display} ({len(data['notes'])} notes)<audio controls preload='none' src='{sound.relative_to(ROOT)}'></audio>"
                         f" <a href='{midi.relative_to(ROOT)}'>MIDI</a> <a href='{midi.with_suffix('.json').relative_to(ROOT)}'>JSON</a></label>")
        for part in ("instrumental", "vocal"):
            midi = RESULTS / "sheetsage2" / f"{track_id}.{part}.mid"
            if not midi.exists():
                continue
            sound = midi.with_suffix(".preview.wav")
            preview(midi, sound, float(row["clip_duration_sec"]))
            notes = json.loads(midi.with_suffix(".json").read_text())["notes"]
            parts.append(f"<label>SheetSage2 {part} ({len(notes)} notes)"
                         f"<audio controls preload='none' src='{sound.relative_to(ROOT)}'></audio>"
                         f" <a href='{midi.relative_to(ROOT)}'>MIDI</a>"
                         f" <a href='{midi.with_suffix('.json').relative_to(ROOT)}'>JSON</a></label>")
        part_files = list((RESULTS / "yourmt3" / "parts").glob(f"{track_id}.p*.mid"))
        def priority(path: Path) -> tuple[int, int]:
            instrument = pretty_midi.PrettyMIDI(str(path)).instruments[0]
            name = instrument.name.lower()
            preferred = any(word in name for word in ("piano", "singing", "lead"))
            return (0 if preferred else 1, -len(instrument.notes))
        for part_midi in sorted(part_files, key=priority)[:5]:
            instrument = pretty_midi.PrettyMIDI(str(part_midi)).instruments[0]
            sound = part_midi.with_suffix(".preview.wav")
            preview(part_midi, sound, float(row["clip_duration_sec"]))
            parts.append(f"<label>↳ {html.escape(instrument.name)} ({len(instrument.notes)} notes)"
                         f"<audio controls preload='none' src='{sound.relative_to(ROOT)}'></audio>"
                         f" <a href='{part_midi.relative_to(ROOT)}'>MIDI</a>"
                         f" <a href='{part_midi.with_suffix('.json').relative_to(ROOT)}'>JSON</a></label>")
        blocks.append("<section>" + "\n".join(parts) + "</section>")
    (ROOT / "report.html").write_text("<!doctype html><meta charset='utf-8'><title>MTG-Jamendo extraction test</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#f8f7f3}"
        "section{background:white;padding:1rem;margin:1rem 0;border:1px solid #ddd;border-radius:10px}"
        "label{display:block;margin:.5rem 0}audio{vertical-align:middle;margin-left:1rem;max-width:60%}</style>"
        f"<h1>MTG-Jamendo extraction test</h1><p>{len(rows())} seeded random songs; all audio and notes are excerpt-relative."
        " These outputs are unreviewed predictions, not ground truth.</p>" + "\n".join(blocks))
    print(ROOT / "report.html")


def make_compare() -> None:
    """Write a small, self-contained A/B player for the frozen sample."""
    songs = []
    for row in rows():
        track_id = row["track_id"]
        choices = []
        for method, label in (("combined", "Combined · GAME vocal + SheetSage2 instrumental"),
                              ("game_vocal", "GAME · vocal stem"),
                              ("game_mix", "GAME · original mix"),
                              ("yourmt3", "YourMT3+ · all parts")):
            midi = RESULTS / method / f"{track_id}.mid"
            if not midi.exists():
                continue
            sound = midi.with_suffix(".preview.wav")
            preview(midi, sound, float(row["clip_duration_sec"]))
            notes = json.loads(midi.with_suffix(".json").read_text())["notes"]
            detail = ""
            if method == "combined":
                vocal_count = sum(note["source"] == "game_vocal" for note in notes)
                detail = f"{vocal_count} GAME vocal + {len(notes) - vocal_count} SheetSage2 instrumental"
            choices.append({"label": label, "notes": len(notes),
                            "detail": detail,
                            "audio": str(sound.relative_to(ROOT)),
                            "midi": str(midi.relative_to(ROOT)),
                            "json": str(midi.with_suffix(".json").relative_to(ROOT))})
        for part in ("instrumental", "vocal"):
            midi = RESULTS / "sheetsage2" / f"{track_id}.{part}.mid"
            if not midi.exists():
                continue
            sound = midi.with_suffix(".preview.wav")
            preview(midi, sound, float(row["clip_duration_sec"]))
            notes = json.loads(midi.with_suffix(".json").read_text())["notes"]
            choices.append({"label": f"SheetSage2 · {part} melody", "notes": len(notes),
                            "audio": str(sound.relative_to(ROOT)),
                            "midi": str(midi.relative_to(ROOT)),
                            "json": str(midi.with_suffix(".json").relative_to(ROOT))})
        for midi in sorted((RESULTS / "yourmt3" / "parts").glob(f"{track_id}.p*.mid")):
            sound = midi.with_suffix(".preview.wav")
            if not sound.exists():
                preview(midi, sound, float(row["clip_duration_sec"]))
            notes = json.loads(midi.with_suffix(".json").read_text())["notes"]
            name = notes[0]["part"] if notes else midi.stem
            choices.append({"label": f"YourMT3+ · {name}", "notes": len(notes),
                            "audio": str(sound.relative_to(ROOT)),
                            "midi": str(midi.relative_to(ROOT)),
                            "json": str(midi.with_suffix(".json").relative_to(ROOT))})
        songs.append({"id": track_id, "start": float(row["clip_start_sec"]),
                      "original": str(clip_path(track_id).relative_to(ROOT)),
                      "mp3": os.path.relpath(DATASET / row["source_path"], ROOT), "choices": choices})

    data = json.dumps(songs, separators=(",", ":")).replace("<", "\\u003c")
    version_file = ROOT / "run-id.txt"
    version = version_file.read_text().strip() if version_file.exists() else ""
    page = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Melody extraction · listen and compare</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;background:#f5f4f0;color:#20231f;max-width:780px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.65rem;margin-bottom:.25rem}p{color:#555d52}main{background:white;border:1px solid #d9ddd3;border-radius:14px;padding:1.25rem;box-shadow:0 2px 12px #0001}
.row{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}.field{flex:1;min-width:230px}label{display:block;font-weight:650;margin-bottom:.3rem}
select,button{font:inherit;border:1px solid #aeb7a8;border-radius:7px;padding:.55rem;background:white;color:inherit}select{width:100%}
.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:1rem;margin:1.2rem 0}.player{background:#f8faf6;border:1px solid #dfe6db;border-radius:10px;padding:1rem}
.player strong{display:block;margin-bottom:.5rem}audio{width:100%}.buttons{display:flex;gap:.5rem;flex-wrap:wrap}button{cursor:pointer}button:hover{background:#eaf0e6}
.meta{font-size:.9rem;color:#555d52}a{color:#245f44;margin-right:.7rem}#count{font-weight:650}
</style>
<h1>Listen and compare</h1>
<p>Original excerpt beside a simple audio rendering of its predicted MIDI notes. These predictions have not been checked by hand. This page refreshes when the next run finishes.</p>
<main>
<div class="row"><div class="field"><label for="song">Song</label><select id="song"></select></div>
<div class="field"><label for="choice">MIDI prediction</label><select id="choice"></select></div></div>
<div class="meta" id="info"></div>
<div class="players"><div class="player"><strong>Original excerpt</strong><audio id="original" controls preload="none"></audio></div>
<div class="player"><strong>MIDI preview</strong><audio id="midiAudio" controls preload="none"></audio></div></div>
<div class="buttons"><button id="playOriginal">Play original</button><button id="playMidi">Play MIDI</button><button id="playBoth">Play together</button><button id="pauseBoth">Pause both</button></div>
<p class="meta"><span id="count"></span> · <a id="midiLink">Download MIDI</a><a id="jsonLink">View JSON</a><a id="mp3Link">Full original MP3</a></p>
<p class="meta">The MIDI preview uses a plain sine tone so pitches and timing are easy to compare. Playback may drift slightly when both players run together.</p>
</main>
<script>
const songs = __DATA__;
const pageVersion = __VERSION__;
const $ = id => document.getElementById(id);
const song = $('song'), choice = $('choice'), original = $('original'), midi = $('midiAudio');
for (const s of songs) song.add(new Option(s.id, s.id));
function stop(){original.pause();midi.pause()}
function loadChoice(){
  stop(); const c=songs[song.selectedIndex].choices[choice.selectedIndex];
  midi.src=c.audio; midi.load(); $('count').textContent=`${c.notes} predicted notes${c.detail ? ` · ${c.detail}` : ''}`;
  $('midiLink').href=c.midi; $('jsonLink').href=c.json;
}
function loadSong(){
  stop(); const s=songs[song.selectedIndex]; original.src=s.original; original.load();
  $('info').textContent=`Track ${s.id} · excerpt starts at ${s.start.toFixed(3)} seconds in the original song`;
  $('mp3Link').href=s.mp3; choice.replaceChildren();
  for (const c of s.choices) choice.add(new Option(`${c.label} (${c.notes} notes)`,c.midi));
  loadChoice();
}
song.onchange=loadSong; choice.onchange=loadChoice;
$('playOriginal').onclick=()=>{midi.pause();original.play()};
$('playMidi').onclick=()=>{original.pause();midi.play()};
$('playBoth').onclick=()=>{original.currentTime=0;midi.currentTime=0;original.play();midi.play()};
$('pauseBoth').onclick=stop;
loadSong();
if (pageVersion) setInterval(() => {
  const probe = document.createElement('script');
  probe.src = `run-version.js?t=${Date.now()}`;
  probe.onload = () => { probe.remove(); if (window.MELODY_RUN_VERSION !== pageVersion) location.reload(); };
  probe.onerror = () => probe.remove();
  document.head.append(probe);
}, 10000);
</script></html>'''
    target = ROOT / "compare.html"
    target.write_text(page.replace("__DATA__", data).replace("__VERSION__", json.dumps(version)))
    print(target)


def make_summary() -> None:
    review_path = ROOT / "review.csv"
    reviews = {}
    if review_path.exists():
        with review_path.open(newline="") as stream:
            reviews = {row["track_id"]: row for row in csv.DictReader(stream)}
    annotations = {}
    annotation_path = DATASET / "annotations" / "autotagging.tsv"
    with annotation_path.open() as stream:
        next(stream)
        for line in stream:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) >= 6:
                annotations[Path(fields[3]).stem] = {
                    "artist_id": fields[1], "tags": "; ".join(fields[5:])
                }
    output_rows = []
    for row in rows():
        track_id = row["track_id"]
        stem = stem_path(track_id)
        clip = clip_path(track_id)
        mix_audio, _ = sf.read(clip, dtype="float32") if clip.exists() else (None, None)
        mix_rms = float(np.sqrt(np.mean(mix_audio * mix_audio))) if mix_audio is not None else None
        audio, _ = sf.read(stem, dtype="float32") if stem.exists() else (None, None)
        stem_rms = float(np.sqrt(np.mean(audio * audio))) if audio is not None else None
        counts = {}
        for method in ("combined", "game_vocal", "game_mix", "yourmt3"):
            result = RESULTS / method / f"{track_id}.json"
            counts[method] = len(json.loads(result.read_text())["notes"]) if result.exists() else None
        for part in ("instrumental", "vocal"):
            result = RESULTS / "sheetsage2" / f"{track_id}.{part}.json"
            counts[f"sheetsage2_{part}"] = len(json.loads(result.read_text())["notes"]) if result.exists() else None
        combined = RESULTS / "combined" / f"{track_id}.json"
        combined_notes = json.loads(combined.read_text())["notes"] if combined.exists() else None
        instruments = ""
        multi = RESULTS / "yourmt3" / f"{track_id}.json"
        if multi.exists():
            notes = json.loads(multi.read_text())["notes"]
            instrument_counts = {}
            for note in notes:
                label = note["part"] or ("Drums" if note["is_drum"] else pretty_midi.program_to_instrument_name(note["program"]))
                instrument_counts[label] = instrument_counts.get(label, 0) + 1
            instruments = "; ".join(f"{name}:{count}" for name, count in sorted(instrument_counts.items()))
        output_rows.append({
            **row,
            **annotations.get(track_id, {"artist_id": "", "tags": ""}),
            "clip_wav": str(clip_path(track_id).relative_to(ROOT)) if clip_path(track_id).exists() else "",
            "vocal_wav": str(stem.relative_to(ROOT)) if stem.exists() else "",
            "game_vocal_json": f"results/game_vocal/{track_id}.json" if counts["game_vocal"] is not None else "",
            "combined_json": f"results/combined/{track_id}.json" if counts["combined"] is not None else "",
            "game_mix_json": f"results/game_mix/{track_id}.json" if counts["game_mix"] is not None else "",
            "yourmt3_json": f"results/yourmt3/{track_id}.json" if counts["yourmt3"] is not None else "",
            "sheetsage2_instrumental_json": f"results/sheetsage2/{track_id}.instrumental.json" if counts["sheetsage2_instrumental"] is not None else "",
            "sheetsage2_vocal_json": f"results/sheetsage2/{track_id}.vocal.json" if counts["sheetsage2_vocal"] is not None else "",
            "yourmt3_part_jsons": "; ".join(str(path.relative_to(ROOT)) for path in sorted(
                (RESULTS / "yourmt3" / "parts").glob(f"{track_id}.p*.json")
            )),
            "mix_rms": round(mix_rms, 6) if mix_rms is not None else "",
            "stem_rms": round(stem_rms, 6) if stem_rms is not None else "",
            "game_vocal_notes": counts["game_vocal"],
            "combined_notes": counts["combined"],
            "combined_vocal_notes": sum(note["source"] == "game_vocal" for note in combined_notes) if combined_notes is not None else None,
            "combined_instrumental_notes": sum(note["source"] == "sheetsage2_instrumental" for note in combined_notes) if combined_notes is not None else None,
            "game_mix_notes": counts["game_mix"],
            "yourmt3_notes": counts["yourmt3"],
            "sheetsage2_instrumental_notes": counts["sheetsage2_instrumental"],
            "sheetsage2_vocal_notes": counts["sheetsage2_vocal"],
            "yourmt3_instruments": instruments,
            "review_status": "reviewed" if any(
                value for key, value in reviews.get(track_id, {}).items() if key != "track_id"
            ) else "unreviewed",
            "review_hint": "quiet_vocal_stem" if stem_rms is not None and stem_rms < 0.001 else "",
        })
    target = ROOT / "summary.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(target)


def make_review_template() -> None:
    target = ROOT / "review.csv"
    if target.exists():
        print(f"Preserved existing {target}")
        return
    columns = ["track_id", "vocal_present", "best_vocal_method", "pitch_rating_1_to_5",
               "note_coverage_rating_1_to_5", "instrument_quality_rating_1_to_5", "notes"]
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({"track_id": row["track_id"]} for row in rows())
    print(target)


def run_stage(stage: str) -> None:
    status = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    selected = rows()
    if stage in {"game_vocal", "game_mix"}:
        method = stage
        source_dir = STEMS if stage == "game_vocal" else CLIPS
        result_dir = RESULTS / method
        result_dir.mkdir(parents=True, exist_ok=True)
        if any(not (result_dir / f"{row['track_id']}.mid").exists() for row in selected):
            call([str(GAME_PYTHON), str(GAME_ROOT / "infer.py"), "extract",
                  str(source_dir), "-m", str(GAME_MODEL), "--glob", "*.wav",
                  "--batch-size", "2", "--output-formats", "mid,csv",
                  "--pitch-format", "number", "--output-dir", str(result_dir)], cwd=GAME_ROOT)
        for index, row in enumerate(selected, 1):
            track_id = row["track_id"]
            try:
                midi = result_dir / f"{track_id}.mid"
                if not midi.exists():
                    # GAME omits output files when it detects no notes.
                    blank = pretty_midi.PrettyMIDI()
                    blank.instruments.append(pretty_midi.Instrument(0, name="lead_vocal"))
                    blank.write(str(midi))
                    midi.with_suffix(".csv").write_text("onset,offset,pitch\n")
                normalize_game(midi.with_suffix(".csv"), midi.with_suffix(".json"), method)
                count = len(json.loads(midi.with_suffix(".json").read_text())["notes"])
                result = {"status": "ok" if count else "empty_prediction", "notes": count}
            except Exception as exc:
                result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            status.setdefault(track_id, {})[stage] = result
            STATUS.write_text(json.dumps(status, indent=2) + "\n")
            print(f"[{index}/{len(selected)}] {stage} {track_id}: {result}", flush=True)
        return
    separator = VocalSeparator() if stage == "separate" else None
    for index, row in enumerate(selected, 1):
        track_id = row["track_id"]
        began = time.monotonic()
        try:
            if stage == "prepare":
                prepare(row)
            elif stage == "separate":
                assert separator is not None
                separator.run(row)
            else:
                raise ValueError(stage)
            result = {"status": "ok", "seconds": round(time.monotonic() - began, 2)}
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        status.setdefault(track_id, {})[stage] = result
        STATUS.write_text(json.dumps(status, indent=2) + "\n")
        print(f"[{index}/{len(selected)}] {stage} {track_id}: {result}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sample", "prepare", "separate", "game_vocal", "game_mix", "yourmt3_parts", "report", "compare", "summary", "review_template"])
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--overwrite-sample", action="store_true")
    parser.add_argument("--exclude-sample", type=Path, action="append")
    parser.add_argument("--unique-artists", action="store_true")
    args = parser.parse_args()
    if args.stage == "sample":
        sample_songs(args.count, args.seconds, args.seed, args.overwrite_sample,
                     args.exclude_sample, args.unique_artists)
    elif args.stage == "report":
        make_report()
    elif args.stage == "compare":
        make_compare()
    elif args.stage == "summary":
        make_summary()
    elif args.stage == "yourmt3_parts":
        export_yourmt3_parts()
    elif args.stage == "review_template":
        make_review_template()
    else:
        run_stage(args.stage)


if __name__ == "__main__":
    main()
