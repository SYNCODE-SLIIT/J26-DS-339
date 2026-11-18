"""Studio sanity runs: (1) false rejections on every opened sung file, (2) F1 deltas on the annotated demo examples.
Run: python tools/studio_validate.py   -> results/studio/evidence/validation.json"""
import json, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
import librosa, numpy as np, soundfile as sf
from bpdr_input.studio import evidence as E, examples, pipeline, scoring

IDS = [1, 3, 6, 8, 10, 13, 14, 18, 20, 25, 26, 27, 28, 29, 31, 32, 35, 36, 37, 38, 39]
OUT = ROOT / "results/studio/evidence"; OUT.mkdir(parents=True, exist_ok=True)
tmp = Path(tempfile.mkdtemp())
res = {"status_counts": {}, "rejections": [], "examples": []}
noise = {k: librosa.load(ROOT / "datasets/environmental_noise" / v, sr=E.SR)[0] for k, v in examples.NOISES.items()}

def mix(x, name, snr):
    n = np.resize(noise[name], len(x)); g = np.sqrt((x ** 2).mean() / ((n ** 2).mean() * 10 ** (snr / 10)))
    y = x + g * n; return (y / np.abs(y).max() * .9).astype(np.float32)

cases = []
for i in IDS:
    x, _ = librosa.load(ROOT / "datasets/vocadito/Audio" / f"vocadito_{i}.wav", sr=E.SR, duration=8.0)
    for cond, y in (("clean", x), ("rain0", mix(x, "rain", 0)), ("vacuum0", mix(x, "vacuum cleaner", 0)), ("ticks10", mix(x, "clock tick", 10))):
        cases.append((f"vocadito_{i}:{cond}", y.astype(np.float32)))
for hp in sorted((ROOT / "datasets/humtrans/subset/train").glob("*/original.wav")):      # 12 real hums, training subset
    x, _ = librosa.load(hp, sr=E.SR, duration=8.0); cases.append((f"humtrans:{hp.parent.name}", x.astype(np.float32)))
for k, (name, y) in enumerate(cases):
    p = tmp / f"{k}.wav"; sf.write(p, y, E.SR, subtype="PCM_24")
    r = pipeline.prepare(p, tmp / f"o{k}", with_pyin=False)
    res["status_counts"][r["status"]] = res["status_counts"].get(r["status"], 0) + 1
    if not r["status"].startswith("ACCEPT"):
        res["rejections"].append({"case": name, "status": r["status"], "reason": r["reasons"][:1],
                                  "voiced": r.get("quality", {}).get("voiced_ratio"), "plateau": r.get("quality", {}).get("speech", {}).get("plateau_fraction")})
for name in ("speech",):
    pass
for ex in examples.build():
    if not ex["reference"]: continue
    r = pipeline.prepare(ex["path"], tmp / ("ex_" + ex["id"]))
    n = r["notes"]
    res["examples"].append({"example": ex["id"], "status": r["status"],
        "f1_today": scoring.note_f1(ex["reference"], n["before_current_extractor_pyin"], 8),
        "f1_shared_on_prepared": scoring.note_f1(ex["reference"], n["shared_extractor_on_prepared"], 8),
        "f1_proposed_pesto": scoring.note_f1(ex["reference"], n["after_prepared_pesto"], 8)})
(OUT / "validation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
