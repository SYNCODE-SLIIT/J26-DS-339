"""R-5. Blinded paired-listening packet generator (proposal section 3.5 / appendix 1). No human ratings are created.

For each of N Vocadito training-role windows the packet holds a pair (original, processed) in randomised X/Y order,
plus identity controls (X and Y are the same file), seeded repeat trials, and a sealed owner key.
'Processed' = what Studio delivers today for that window (cleaned + loudness-normalised prepared audio); pitch repair
is applied only where the current pipeline accepts it, so the packet tests the delivered output, not a hypothetical.
All files are gain-matched (RMS), neutral-named, and trial order is shuffled with seed 339.
Output: results/studio/listening_packet/ (trial wavs, rating_sheet.csv/html, owner_answer_key_DO_NOT_SHARE.json).
"""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / "source"), str(WS)]

from bpdr_input.studio import clean, evidence as E
from tools.benchmark_studio_vocadito import load_vocadito_windows

OUT = WS / "results" / "studio" / "listening_packet"
SEED = 339
N_PAIRS = 12
N_IDENTITY = 10
N_REPEATS = 4
TARGET_RMS = 0.08
FIELDS = ["tune_preservation_1to5", "expression_vibrato_preservation_1to5", "processing_artifacts_1to5", "prefer_X_or_Y"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def match(x: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
    return np.clip(x * (TARGET_RMS / r), -0.99, 0.99).astype(np.float32)


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    wins = load_vocadito_windows()
    rng.shuffle(wins)
    wins = wins[: N_PAIRS + N_IDENTITY]
    pair_wins, ident_wins = wins[:N_PAIRS], wins[N_PAIRS:N_PAIRS + N_IDENTITY]

    def load(w):
        x, _ = E.load_mono(w["wav_path"], sr=E.SR, max_s=None)
        s0 = int(round(w["offset_s"] * E.SR))
        win = x[s0:s0 + int(round(w["duration_s"] * E.SR))]
        xn = E.normalize(win)
        cl = clean.spectral_gate(win)
        t_o, t_c = E.pesto_track(xn), E.pesto_track(cl)
        use, _ = clean.choose(t_o, t_c)
        return match(win), match(cl if use else xn)

    trials = []  # (kind, window, orig, processed)
    for w in pair_wins:
        o, p = load(w)
        trials.append(("pair", w["id"], o, p))
    for w in ident_wins:
        o, _ = load(w)
        trials.append(("identity", w["id"], o, o.copy()))
    repeats = [("repeat", t[1], t[2], t[3]) for t in rng.sample([t for t in trials if t[0] == "pair"], N_REPEATS)]
    trials += repeats
    rng.shuffle(trials)

    key, rows = [], []
    for i, (kind, wid, o, p) in enumerate(trials, 1):
        x_is_original = rng.random() < 0.5 if kind != "identity" else True
        xa, ya = (o, p) if x_is_original else (p, o)
        stem = f"trial_{i:03d}"
        for tag, arr in (("X", xa), ("Y", ya), ("guide", o)):
            sf.write(OUT / f"{stem}_{tag}.wav", arr, E.SR, subtype="PCM_24")
        rows.append({"trial": stem, **{f: "" for f in FIELDS}})
        key.append({
            "trial": stem, "kind": kind, "source_window": wid, "X_is_original": bool(x_is_original),
            "sha256_X": sha((OUT / f"{stem}_X.wav").read_bytes()), "sha256_Y": sha((OUT / f"{stem}_Y.wav").read_bytes()),
            "identity_files_bit_identical": bool(kind == "identity"),
        })

    with open(OUT / "rating_sheet.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["trial"] + FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    html = ["<!doctype html><meta charset='utf-8'><title>Listening rating sheet</title><style>body{font-family:sans-serif;max-width:900px;margin:2em auto}td,th{border:1px solid #888;padding:6px}</style>",
            "<h1>Listening rating sheet</h1><p>Listen to the guide, then X and Y. Rate 1 (poor) to 5 (excellent). No answers are pre-filled.</p><table><tr><th>Trial</th>"
            + "".join(f"<th>{f}</th>" for f in FIELDS) + "</tr>"]
    html += [f"<tr><td>{r['trial']}</td>" + "<td>&nbsp;</td>" * len(FIELDS) + "</tr>" for r in rows]
    html.append("</table>")
    (OUT / "rating_sheet.html").write_text("\n".join(html), encoding="utf-8")
    (OUT / "owner_answer_key_DO_NOT_SHARE.json").write_text(json.dumps({"seed": SEED, "trials": key}, indent=2), encoding="utf-8")
    print(f"wrote {len(trials)} trials ({N_PAIRS} pairs, {N_IDENTITY} identity, {N_REPEATS} repeats) to {OUT}")


if __name__ == "__main__":
    main()
