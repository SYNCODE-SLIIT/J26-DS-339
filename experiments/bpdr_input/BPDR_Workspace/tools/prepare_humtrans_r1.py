"""Analysis arrays + MIDI note labels for the collected HumTrans R1 clips (resumable). No model fit.
Crop: first 10 s, highest-energy channel, 22.05 kHz, same shared analysis (model_recovery.measured) as the other training windows."""
import io, json, sys, zipfile
from math import gcd
from pathlib import Path
import numpy as np, soundfile as sf, mido, torch
from scipy.signal import resample_poly
WS = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(WS / 'source'), str(WS / 'tools')]
import prepare_short_phrase_training as P
OUT = WS / 'datasets/humtrans_r1'; ZIP = WS / 'datasets/humtrans/all_midi.zip'
def notes_from_midi(blob):
    m = mido.MidiFile(file=io.BytesIO(blob)); t = 0.; on = {}; out = []
    for msg in m:
        t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0: on[msg.note] = t
        elif msg.type in ('note_off', 'note_on') and msg.note in on:
            s = on.pop(msg.note); out.append([s, 440 * 2 ** ((msg.note - 69) / 12), t - s, msg.note])
    return sorted(out)
def main():
    torch.set_num_threads(2); (OUT / 'analysis').mkdir(exist_ok=True); zf = zipfile.ZipFile(ZIP); man = json.load(open(OUT / 'manifest.json')); done = 0
    for i, s in enumerate(man):
        if 'sha256' not in s: continue
        f = OUT / 'analysis' / f"{s['key']}.npz"; nf = OUT / 'analysis' / f"{s['key']}.notes.json"
        if f.exists() and nf.exists(): done += 1; continue
        try:
            y, sr = sf.read(OUT / 'audio' / f"{s['key']}.wav", dtype='float32', always_2d=True)
            y = y[:10 * sr, int(np.argmax((y ** 2).mean(0)))]; g = gcd(sr, 22050)
            if sr != 22050: y = resample_poly(y, 22050 // g, sr // g).astype(np.float32)
            if len(y) < 3 * 22050 or not np.isfinite(y).all(): s['skip'] = 'short_or_invalid'; continue
            a = P.recovery.measured(y); P.save_analysis(f, a)
            nf.write_text(json.dumps(notes_from_midi(zf.read(f"midi_data/{s['key']}.mid"))))
        except Exception as e:
            s['skip'] = repr(e)[:120]
        if i % 20 == 0: print(i, len(man), flush=True)
    json.dump(man, open(OUT / 'manifest_analysis.json', 'w'), indent=1); print('finished', flush=True)
if __name__ == '__main__': main()
