"""Controlled proof of concept of the proposal's BPDR method on synthetic hummed phrases.

Purpose: show, end to end in the implemented code, the mechanism the proposal claims (sections 1.3, 3.2, 3.6):
four aligned views, a local in-note detuning, false split/merge/jitter boundary changes, and three matched
transformers (augmentation-only 71, frame-only 70, BPDR 71) trained with the anchor, smoothness, boundary
invariance (L_B) and counter-detuning (L_D) terms. Every result here is on CONSTRUCTED voices, not on real singing,
and is not evidence of real-voice accuracy.

Data path (same component code as the real data):
  synthetic accepted reference (A audio) -> interventions.render_local (local shift, 80-ms ramps) -> C audio
  analysis.extract (PESTO, log-mel, energy, onset) -> fine_pitch.fine_columns (YIN pitch/pitch change)
  -> timing-only boundary track (shared Gaussian encoding) from nominal edges (A, C) and corrupted edges (B, D)
Training: squared-error anchor (fine_pitch_mse_check.squared_objective), phrase-relative pitch input, matched init,
optimizer, budget and schedule for all three conditions; BPDR adds L_B and L_D after the 100-update warm-up.
Evaluation on held-out synthetic voices (different timbre, range, tuning offset and vibrato draws).

Commands: build (CPU venv: audio + features), train (XPU venv).
"""
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np

COMPONENT = Path(__file__).resolve().parents[2]
ROOT = COMPONENT/'runs/controlled_study'
SR = 22050
SHIFTS = (-30., -20., -10., 10., 20., 30.)
CORRUPTIONS = ('split', 'merge', 'jitter')


@dataclass(frozen=True)
class Design:
    version: str = 'controlled-bpdr-0.1'
    train_voices: int = 16
    heldout_voices: int = 6
    train_phrases_per_voice: int = 8
    heldout_phrases_per_voice: int = 6
    train_variants_per_phrase: int = 3
    heldout_variants_per_phrase: int = 2
    phrase_seconds: float = 6.
    seed: int = 339
    updates: int = 1500
    warmup: int = 100
    batch: int = 8
    evaluate_every: int = 250


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def voice(rng):
    return dict(centre=float(rng.uniform(-1500, -400)), offset=float(rng.uniform(-35, 35)), tilt=float(rng.uniform(.45, .8)),
                vibrato_depth=float(rng.uniform(0, 18)), vibrato_rate=float(rng.uniform(4.8, 6.5)), intonation_sd=float(rng.uniform(1.5, 4.)),
                noise=float(rng.uniform(.002, .008)), formant=float(rng.uniform(600, 1400)))


def phrase(v, rng, seconds):
    """Note list [(onset, offset, cents)], one long note guaranteed, optional short rests."""
    notes, t, pitch = [], .25, v['centre']+v['offset']
    long_index = int(rng.integers(1, 4))
    while t < seconds-.6:
        k = len(notes)
        dur = float(rng.uniform(1.25, 1.6)) if k == long_index else float(rng.uniform(.4, 1.))
        dur = min(dur, seconds-.3-t)
        if dur < .3:
            break
        cents = pitch+float(rng.normal(0, v['intonation_sd']))
        notes.append((t, t+dur, cents))
        t += dur+(float(rng.uniform(.1, .18)) if rng.random() < .3 else 0.)
        pitch += 100*int(rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))
        pitch = float(np.clip(pitch, v['centre']+v['offset']-700, v['centre']+v['offset']+700))
    return notes


def synthesize(v, notes, seconds, rng):
    n = int(seconds*SR); t = np.arange(n)/SR
    f0 = np.full(n, np.nan); amp = np.zeros(n)
    for i, (a, b, c) in enumerate(notes):
        m = (t >= a) & (t < b)
        cents = np.full(m.sum(), c)
        if i+1 < len(notes) and abs(notes[i+1][0]-b) < 1e-9:   # 50-ms glide into a connected next note
            g = t[m] > b-.05
            cents[g] = c+(notes[i+1][2]-c)*((t[m][g]-(b-.05))/.05)
        if b-a >= .6:
            cents = cents+v['vibrato_depth']*np.sin(2*np.pi*v['vibrato_rate']*(t[m]-a))*np.clip((t[m]-a-.2)/.3, 0, 1)
        f0[m] = 440*2**(cents/1200)
        env = np.minimum(1, np.minimum((t[m]-a)/.03, (b-t[m])/.04))
        amp[m] = np.clip(env, 0, 1)
    phase = 2*np.pi*np.cumsum(np.nan_to_num(f0))/SR
    y = np.zeros(n)
    for h in range(1, 9):
        fh = np.nan_to_num(f0)*h
        gain = v['tilt']**(h-1)*(1+.8*np.exp(-((fh-v['formant'])/400)**2))
        y += gain*np.sin(h*phase)
    y = y*amp
    y = .3*y/max(np.abs(y).max(), 1e-9)+v['noise']*rng.standard_normal(n)
    return np.clip(y, -.95, .95).astype(np.float32)


def features(audio, edges):
    from .analysis import extract
    from .fine_pitch import fine_columns
    from .rule_correction import fine_pitch
    from .shared_pair_audit import game_boundary_track
    f, _ = extract(audio, SR, COMPONENT/'checkpoints')
    times = f['times_seconds']
    raw = f['boundary_features_71'].copy()
    raw[:, 70] = game_boundary_track(times, edges)
    fixed, _ = fine_columns(raw, fine_pitch(audio, SR, times))
    return times, fixed


def make_record(v, notes, audio, cents, corruption, seed, folder):
    from .interventions import render_local, corrupt_boundaries
    from .shared_pair_audit import game_boundary_track
    a, b, _ = max(notes, key=lambda n: n[1]-n[0])
    support = (a+.15, b-.15)
    shifted, _ = render_local(audio, SR, cents, support)
    edges = np.asarray(sorted({x for n in notes for x in n[:2]}))
    times, A = features(audio, edges)
    _, C = features(shifted, edges)
    corrupted = corrupt_boundaries(edges, times, corruption, seed=seed, split_time=(support[0]+support[1])/2 if corruption == 'split' else None)
    B, D = A.copy(), C.copy()
    B[:, 70] = D[:, 70] = game_boundary_track(times, corrupted)
    voiced = (A[:, 66] > .5) & (C[:, 66] > .5)
    ramp = ((times >= support[0]) & (times < support[0]+.1)) | ((times > support[1]-.1) & (times <= support[1]))
    plateau = (times >= support[0]+.1) & (times <= support[1]-.1)
    mask = voiced & ~ramp
    target = np.where(mask & plateau, -cents, 0.).astype(np.float32)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'views.npz', A=A, B=B, C=C, D=D, times=times, mask=mask, correction_target_cents=target)
    import soundfile as sf
    sf.write(folder/'A.wav', audio, SR, subtype='PCM_24'); sf.write(folder/'C.wav', shifted, SR, subtype='PCM_24')
    return dict(path=str(folder/'views.npz'), cents=float(cents), corruption=corruption, support_seconds=list(support),
                nominal_edges=edges.tolist(), corrupted_edges=corrupted.tolist(), plateau_frames=int((mask & plateau).sum()),
                mask_frames=int(mask.sum()))


def build():
    d = Design()
    out = ROOT/uuid.uuid4().hex; out.mkdir(parents=True)
    rng = np.random.default_rng(d.seed)
    catalog = []
    for split, n_voices, n_phr, n_var in (('train', d.train_voices, d.train_phrases_per_voice, d.train_variants_per_phrase),
                                           ('heldout', d.heldout_voices, d.heldout_phrases_per_voice, d.heldout_variants_per_phrase)):
        for vi in range(n_voices):
            v = voice(rng); vid = f'{split}_voice_{vi:02d}'
            for pi in range(n_phr):
                notes = phrase(v, rng, d.phrase_seconds)
                audio = synthesize(v, notes, d.phrase_seconds, rng)
                for k in range(n_var):
                    cents = float(rng.choice(SHIFTS)); corruption = CORRUPTIONS[(pi+k) % 3]
                    rec = make_record(v, notes, audio, cents, corruption, int(rng.integers(1e9)), out/split/vid/f'phrase_{pi:02d}'/f'variant_{k}')
                    catalog.append(dict(rec, split=split, voice_id=vid, phrase_id=f'{vid}_p{pi:02d}', voice=v, notes=notes))
            print('built', split, vid, len(catalog), flush=True)
    (out/'catalog.json').write_text(json.dumps(dict(design=asdict(d), records=catalog, created_utc=datetime.now(timezone.utc).isoformat(),
                                                    source_sha256=sha(__file__)), indent=1), encoding='utf-8')
    (ROOT/'current.json').write_text(json.dumps(dict(path=str(out)), indent=2), encoding='utf-8')
    print('CONTROLLED_DATA:', out)
    return out


def load_pair(r):
    with np.load(r['path']) as z:
        inputs = {v: z[v].astype(np.float32) for v in 'ABCD'}
        mask = z['mask'].astype(bool); det = -z['correction_target_cents'].astype(np.float32)
    return dict(pair_id=f"{r['phrase_id']}|{r['cents']:+g}|{r['corruption']}", phrase_id=r['phrase_id'], singer_id=r['voice_id'],
                cents=r['cents'], corruption=r['corruption'], inputs=inputs, mask=mask, detuning=det, data_status='controlled_synthetic')


def evaluate(model, pairs, normalizer, width, device):
    import torch
    from .examples import collate
    from .expanded_comparison import to_device
    model.eval(); rows = []
    with torch.inference_mode():
        for p in pairs:
            b = to_device(collate([p], normalizer, width), device)
            r = {v: model(x, b['padding'])[0].cpu().numpy() for v, x in b['inputs'].items()}
            m = p['mask']; P = m & (p['detuning'] != 0); O = m & (p['detuning'] == 0); t = -p['detuning']
            rows.append(dict(cents=p['cents'], corruption=p['corruption'],
                             correction_mae=float(np.mean([np.abs(r[v][P]-t[P]).mean() for v in 'CD'])), zero_mae=float(np.abs(t[P]).mean()),
                             response_gain=float((r['C'][P]-r['A'][P]).mean()/t[P].mean()),
                             boundary_sensitivity=float((np.abs(r['A'][m]-r['B'][m]).mean()+np.abs(r['C'][m]-r['D'][m]).mean())/2),
                             unshifted_alteration=float(np.abs(r['A'][m]).mean()), outside_alteration=float(np.abs(r['C'][O]).mean())))
    model.train()
    return rows


def summary(rows):
    out = dict(records=len(rows), correction_mae=float(np.mean([r['correction_mae'] for r in rows])), zero_mae=float(np.mean([r['zero_mae'] for r in rows])),
               median_response_gain=float(np.median([r['response_gain'] for r in rows])),
               boundary_sensitivity=float(np.mean([r['boundary_sensitivity'] for r in rows])),
               unshifted_alteration=float(np.mean([r['unshifted_alteration'] for r in rows])),
               outside_alteration=float(np.mean([r['outside_alteration'] for r in rows])))
    out['correction_gain'] = 1-out['correction_mae']/out['zero_mae']
    return out


def train(seed=None):
    """seed=None uses the design seed (339) and becomes the current run; other seeds are extra replicates."""
    import torch
    from .examples import Normalizer, collate
    from .expanded_comparison import to_device
    from .fine_pitch_mse_check import squared_objective
    from .fine_pitch_relative_check import relative_pair
    from .losses import LossWeights
    from .matched_comparison import matched_model
    data = Path(json.loads((ROOT/'current.json').read_text())['path'])
    cat = json.loads((data/'catalog.json').read_text()); d = Design(**cat['design'])
    replicate = seed is not None and seed != d.seed
    seed = d.seed if seed is None else int(seed)
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cpu'
    out = data/'training'/uuid.uuid4().hex; out.mkdir(parents=True)
    usable = [r for r in cat['records'] if r['plateau_frames'] >= 10]   # every record needs audited correction frames
    train_pairs = [relative_pair(load_pair(r)) for r in usable if r['split'] == 'train']
    held_pairs = [relative_pair(load_pair(r)) for r in usable if r['split'] == 'heldout']
    normalizer = Normalizer.fit(train_pairs)
    np.savez(out/'normalizer.npz', mean=normalizer.mean, scale=normalizer.scale)
    rng = np.random.default_rng(seed)
    schedule = np.stack([rng.choice(len(train_pairs), size=d.batch, replace=False) for _ in range(d.updates)])
    np.save(out/'schedule.npy', schedule)
    protocol = dict(fit=dict(seed=seed), model=dict(hidden=128, layers=4, heads=4, feed_forward=512, dropout=.1))
    (out/'declaration.json').write_text(json.dumps(dict(declared_utc=datetime.now(timezone.utc).isoformat(), data=str(data), design=asdict(d), seed=seed, device=device,
        conditions=dict(augmentation_only=71, frame_only=70, bpdr=71), objective='squared L_A + 0.1 L_S; BPDR adds L_B + L_D after warm-up',
        input='fine pitch, phrase-relative; training-only standardizer', source_sha256=sha(__file__),
        label='Controlled synthetic proof of concept; not real-voice accuracy'), indent=2), encoding='utf-8')
    torch.set_num_threads(2); lw = LossWeights()
    results = {}
    for condition, width in (('augmentation_only', 71), ('frame_only', 70), ('bpdr', 71)):
        model = matched_model(protocol, width).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
        curve, started = [], time.monotonic()
        model.train()
        for step, idx in enumerate(schedule, 1):
            b = to_device(collate([train_pairs[i] for i in idx], normalizer, width), device)
            pred = {v: model(x, b['padding']) for v, x in b['inputs'].items()}
            loss, parts = squared_objective(pred, b, condition, lw, step > d.warmup)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            if step % d.evaluate_every == 0 or step == d.updates:
                s = summary(evaluate(model, held_pairs, normalizer, width, device))
                curve.append(dict(update=step, loss=float(loss.detach()), heldout=s))
                print(condition, step, {k: round(v, 4) for k, v in s.items()}, 'elapsed', round(time.monotonic()-started), flush=True)
        rows = evaluate(model, held_pairs, normalizer, width, device)
        torch.save(dict(state={k: v.cpu() for k, v in model.state_dict().items()}, width=width, condition=condition, protocol=protocol,
                        normalizer=dict(mean=normalizer.mean.tolist(), scale=normalizer.scale.tolist()), input_transform='phrase_relative_fine_pitch',
                        data_status='controlled_synthetic'), out/f'{condition}.pt')
        by = lambda key: {str(k): summary([r for r in rows if r[key] == k]) for k in sorted({r[key] for r in rows})}
        results[condition] = dict(heldout=summary(rows), by_corruption=by('corruption'), by_shift=by('cents'), curve=curve)
    (out/'evidence.json').write_text(json.dumps(dict(results=results, heldout_records=len(held_pairs), train_records=len(train_pairs)), indent=2), encoding='utf-8')
    if replicate:
        with (ROOT/'training_replicates.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(seed=seed, path=str(out)))+'\n')
    else:
        (ROOT/'training_current.json').write_text(json.dumps(dict(path=str(out)), indent=2), encoding='utf-8')
    print('CONTROLLED_TRAINING:', out)
    for c, r in results.items():
        print(c, json.dumps({k: round(v, 4) for k, v in r['heldout'].items()}))
    return out


if __name__ == '__main__':
    {'build': build, 'train': train}[sys.argv[1]](*sys.argv[2:])
