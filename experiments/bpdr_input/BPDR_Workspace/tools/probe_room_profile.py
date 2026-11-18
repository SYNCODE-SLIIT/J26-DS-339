"""Plan build-plan F2a comparison. DECLARED BEFORE RUN: on the 21 training-role Vocadito sources (first 8 s) at 10 dB and
0 dB noise, does using a voice-free room-check clip (2 s of the SAME noise at the SAME gain, a different stretch) instead
of the quietest-frames guess change (a) onset+pitch F1 of PESTO notes after cleaning, (b) the number of files where the
cleaned candidate is chosen? Report the singer-level 95% interval of the F1 difference (room minus quiet-frames), either way.
Development evidence only; the room clip here is synthetic (taken from the same noise file)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source")); sys.path.insert(0, str(ROOT / "tools"))
import librosa, numpy as np
import studio_evidence as SE
from studio_frame_control import scores
from bpdr_input.studio import clean, evidence as E

noise, _ = librosa.load(SE.NOISE, sr=E.SR)
rows = []
for i in SE.TRAIN_ROLE_IDS:
    x0 = SE.load8(i)
    for cond, snr in (("10dB", 10), ("0dB", 0)):
        n = np.resize(noise, len(x0)); g = np.sqrt((x0 ** 2).mean() / ((n ** 2).mean() * 10 ** (snr / 10)))
        x = (x0 + g * n).astype(np.float32); peak = np.abs(x).max() / 0.9; x = (x / peak).astype(np.float32)
        room = (g * np.roll(n, 3 * E.SR)[: 2 * E.SR] / peak).astype(np.float32)
        dur = len(x) / E.SR; trk_o = E.pesto_track(E.normalize(x)); r = dict(id=i, singer=SE.SINGER_OF[i], cond=cond)
        r["f1_original"] = float(scores(i, E.notes_from_track(trk_o, dur))[0])
        for name, kw in (("quiet", {}), ("room", {"noise": room})):
            c = clean.spectral_gate(x, **kw); trk_c = E.pesto_track(c)
            use, _ = clean.choose(trk_o, trk_c)
            r[name + "_chosen"] = bool(use); r[name + "_f1_cleaned"] = float(scores(i, E.notes_from_track(trk_c, dur))[0])
            r[name + "_f1_studio"] = r[name + "_f1_cleaned"] if use else r["f1_original"]
        rows.append(r); print(r, flush=True)
sing = np.array([r["singer"] for r in rows]); us = sorted(set(sing)); rng = np.random.default_rng(339); res = {}
for cond in ("10dB", "0dB", "all"):
    sel = [r for r in rows if cond == "all" or r["cond"] == cond]
    for key in ("f1_cleaned", "f1_studio"):
        d = np.array([r["room_" + key] - r["quiet_" + key] for r in sel]); sg = np.array([r["singer"] for r in sel]); u = sorted(set(sg)); bs = []
        for _ in range(2000):
            idx = np.concatenate([np.where(sg == s)[0] for s in rng.choice(u, len(u))]); bs.append(d[idx].mean())
        res[f"{cond}|{key}"] = dict(diff=float(d.mean()), ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])])
    res[f"{cond}|chosen"] = dict(room=sum(r["room_chosen"] for r in sel), quiet=sum(r["quiet_chosen"] for r in sel), n=len(sel))
print(json.dumps(res, indent=1))
json.dump(dict(summary=res, rows=rows), open(ROOT / "results/studio/evidence/room_profile_comparison.json", "w"), indent=1)
