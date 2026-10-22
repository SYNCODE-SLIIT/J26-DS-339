"""Build a read-only full-corpus manifest and independent MIDI timing audit."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pretty_midi
import soundfile as sf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from melody_understanding.annotations import read_midi_notes, overlap_count


# Auditing Corpus Pairing, Split Membership and MIDI Timing
def main():
    """Audit file pairing, official split coverage and two MIDI readers."""
    # Read dataset and output paths from the command line.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "datasets/humtrans/downloads")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    # Index official split assignments and pair files by their exact filename stems.
    split_path = args.data / "train_valid_test_keys.json"
    splits = json.loads(split_path.read_text())
    assignments = [(key, name) for name, keys in splits.items() for key in keys]
    membership = dict(assignments)
    wavs = {p.stem: p for p in (args.data / "wav_data_sync_with_midi").glob("*.wav")}
    midis = {p.stem: p for p in (args.data / "midi_data").glob("*.mid")}
    rows, counts = [], Counter()
    # Inspect every available stem, retaining missing pairs and read failures as flags.
    for i, key in enumerate(sorted(wavs.keys() | midis.keys())):
        row = {"key": key, "split": membership.get(key), "singer": key.split("_")[0],
               "composition": key.split("_")[1], "flags": []}
        try:
            info = sf.info(wavs[key])
            notes, anomalies = read_midi_notes(midis[key])
            # Cross-check decoded note times and pitches with a second MIDI reader.
            other = sorted((n.start, n.end, n.pitch) for inst in pretty_midi.PrettyMIDI(str(midis[key])).instruments for n in inst.notes)
            agree = len(notes) == len(other) and np.allclose(notes, other, atol=1e-7, rtol=0)
            # Check note tails, overlaps and source metadata for structural problems.
            tail = max((n[1] for n in notes), default=0) - info.duration
            overlaps = overlap_count(notes)
            row.update(duration_s=info.duration, sample_rate=info.samplerate,
                channels=info.channels, note_count=len(notes), tail_overrun_s=max(0, tail),
                overlap_count=overlaps, reader_agreement=bool(agree),
                midi_sha256=hashlib.sha256(midis[key].read_bytes()).hexdigest(),
                wav_bytes=wavs[key].stat().st_size, **anomalies)
            for name, condition in [("empty_midi", not notes), ("reader_disagreement", not agree),
                ("tail_overrun", tail > .05), ("overlapping_notes", overlaps > 0),
                ("missing_split", key not in membership), *[(k, v > 0) for k, v in anomalies.items()]]:
                if condition:
                    row["flags"].append(name)
        except Exception as exc:
            row["flags"].append("read_error")
            row["error"] = f"{type(exc).__name__}: {exc}"
        counts.update(row["flags"])
        rows.append(row)
        if (i + 1) % 3000 == 0:
            print(f"Audited {i + 1} pairs", flush=True)
    # Keep detailed recording identifiers local; commit only aggregate evidence.
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    # Summarize coverage and anomalies, with hashes and versions for reproducibility.
    report = {"recorded_date": "2026-10-21", "protocol": "mido-vs-pretty-midi-v1",
        "scope": "all split metadata; no TEST predictions or metric evaluation",
        "wav_count": len(wavs), "midi_count": len(midis),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "duplicate_split_assignments": len(assignments) - len(membership),
        "split_keys_without_pair": len(set(membership) - (wavs.keys() & midis.keys())),
        "paired_keys_without_split": len((wavs.keys() & midis.keys()) - membership.keys()),
        "note_count": sum(r.get("note_count", 0) for r in rows),
        "flag_counts": dict(counts), "unflagged_pairs": sum(not r["flags"] for r in rows),
        "flags_by_split": {name: dict(Counter(flag for r in rows if r['split'] == name for flag in r['flags'])) for name in splits},
        "max_tail_overrun_s": max(r.get("tail_overrun_s", 0) for r in rows),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "versions": {p: version(p) for p in ["mido", "pretty_midi", "numpy", "soundfile"]},
        "implementation_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), ROOT / "src/melody_understanding/annotations.py"]},
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "limitation": "Agreement of readers verifies decoding, not perceptual alignment; PrettyMIDI also uses Mido for low-level parsing."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
