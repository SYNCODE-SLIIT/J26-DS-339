"""FastTNet v5 cumulative ablation ladder. Declared in results/studio/r1/declaration_transformer_v5.json.
Resumable (per-config OOF cache, per-fit result files). Vocadito and reserved HumTrans splits are never read for decisions.
Run: ..\\.venv\\Scripts\\python.exe tools\\fit_r1_transformer_v5.py   (add --smoke for a 3-step quick check)
"""
import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / 'tools'), str(WS / 'source')]
import fit_r1_boundary as B  # noqa: E402
from bpdr_input.studio import boundary  # noqa: E402

torch.set_num_threads(2)
OUT = WS / 'results/studio/r1'
V5 = OUT / 'v5'
CACHE = V5 / 'cache'
DECL = OUT / 'declaration_transformer_v5.json'
N = 501
HOP = 0.02
STEPS, LR, BATCH, POS_WEIGHT = 250, 3e-3, 24, 10.0
OOF_SEED, FINAL_SEEDS = 339, (339, 340, 341)
THRS = tuple(round(0.20 + 0.05 * i, 2) for i in range(15))
SHIFTS_MS = tuple(range(-30, 31, 5))
ADOPT_MARGIN = 0.003
NOISE_VIEW_STRENGTH = 0.8
LADDER = [
    ('a_base', {}), ('b_feats', {'feats': 1}), ('c_samepitch', {'samepitch': 1}),
    ('d_bridge', {'bridge': 1}), ('e_noise', {'noise': 1}), ('f_soft', {'soft': 1}),
]


# ---------------------------------------------------------------- augmentation (numpy, on (N,5) feature arrays)
def seg_bounds(r):
    on = np.sort(np.round(np.asarray(r['onsets']) / HOP).astype(int))
    on = on[(on >= 0) & (on < N)]
    if len(on) == 0:
        return []
    end_all = int(np.flatnonzero(r['valid'])[-1]) + 1 if r['valid'].any() else N
    ends = np.r_[on[1:], end_all]
    return [(int(a), int(b)) for a, b in zip(on, ends) if b > a]


def _med(X, a, b):
    m = X[a:b, 1] > 0.5
    return float(np.median(X[a:b, 0][m])) if m.sum() >= 3 else None


def same_pitch(X, segs, rng, p=0.20):
    for k in range(1, len(segs)):
        if rng.random() < p:
            (a0, b0), (a, b) = segs[k - 1], segs[k]
            mp, mk = _med(X, a0, b0), _med(X, a, b)
            if mp is not None and mk is not None:
                X[a:b, 0] += (mp - mk) * X[a:b, 1]


def vibrato(X, segs, rng, p=0.5):
    for a, b in segs:
        if b - a >= 12 and rng.random() < p:
            t = np.arange(b - a) * HOP
            amp, f, ph = rng.uniform(0.20, 0.55), rng.uniform(4.5, 7.0), rng.uniform(0, 2 * np.pi)
            X[a:b, 0] += amp * np.sin(2 * np.pi * f * t + ph) * np.hanning(b - a) * X[a:b, 1]


def glide(X, segs, rng, p=0.5):
    for a, _ in segs[1:]:
        if rng.random() < p:
            w = int(rng.integers(2, 6)); i0, i1 = max(0, a - w), min(N, a + w + 1)
            ok = X[:, 1]; k = np.ones(2 * w + 1)
            num = np.convolve(X[:, 0] * ok, k, 'same'); den = np.convolve(ok, k, 'same')
            sm = np.where(den > 0, num / np.maximum(den, 1e-9), X[:, 0])
            X[i0:i1, 0] = np.where(ok[i0:i1] > 0.5, sm[i0:i1], 0.0)


def noise(X, rng, s):
    """Mimic measured 0 dB shifts: flux floor up, pitch jitter, short voicing dropouts, tiny confidence drop."""
    X[:, 4] = np.clip(X[:, 4] * (1 - 0.3 * s) + s * rng.uniform(0, 0.25) + s * np.abs(rng.normal(0, 0.08, N)), 0, 1)
    X[:, 0] += rng.normal(0, 0.13 * s, N) * X[:, 1]
    if rng.random() < s * 0.5:
        for _ in range(int(s * 0.015 * N)):
            i = int(rng.integers(0, N)); L = int(rng.integers(1, 4))
            X[i:i + L, 1] = 0.0; X[i:i + L, 0] = 0.0
    X[:, 2] = np.clip(X[:, 2] - s * rng.uniform(0, 0.03), 0, 1)
    X[:, 3] += rng.normal(0, 0.05 * s, N)


def augment(r, X, flags, rng):
    X = X.copy(); segs = seg_bounds(r)
    if flags.get('samepitch'): same_pitch(X, segs, rng)
    if flags.get('bridge'): vibrato(X, segs, rng); glide(X, segs, rng)
    if flags.get('noise'): noise(X, rng, float(rng.random()))
    X[:, 0] *= X[:, 1]
    return X.astype(np.float32)


def noise_view(r, X):
    rng = np.random.default_rng(zlib.crc32(r['id'].encode()) & 0x7fffffff)
    X = X.copy(); noise(X, rng, NOISE_VIEW_STRENGTH); X[:, 0] *= X[:, 1]
    return X.astype(np.float32)


# ---------------------------------------------------------------- data / training / scoring
def base_X(r):
    return B.feats(r, r['edges'], False).astype(np.float32)


def soft_target(r):
    fr = np.arange(N); y = np.zeros(N)
    for o in r['onsets']:
        y = np.maximum(y, np.exp(-0.5 * ((fr - o / HOP) / 0.8) ** 2))
    return y.astype(np.float32)


def make_net(flags):
    if flags.get('feats'):
        return boundary.FastTNetV5(nin=5, eval_logit_bias=0.0)
    return boundary.FastTNet(nin=5, eval_logit_bias=0.0)


def train(flags, train_recs, seed, steps=STEPS):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = make_net(flags); opt = torch.optim.AdamW(net.parameters(), LR)
    bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(POS_WEIGHT))
    tgt = {r['id']: (soft_target(r) if flags.get('soft') else r['y'].astype(np.float32)) for r in train_recs}
    bx = {r['id']: base_X(r) for r in train_recs}
    net.train()
    for _ in range(steps):
        sub = [train_recs[i] for i in rng.choice(len(train_recs), BATCH, replace=False)]
        XA, XD, Y, M = [], [], [], []
        for r in sub:
            X = augment(r, bx[r['id']], flags, rng)
            okf = np.flatnonzero(r['ok']); s = np.zeros(N, bool)
            if len(okf):
                a = int(rng.choice(okf)); s[a:a + 14] = True; s &= r['ok']
            Xd = X.copy(); Xd[:, 0] += rng.choice([-.3, -.2, .2, .3]) * s
            XA.append(X); XD.append(Xd); Y.append(tgt[r['id']]); M.append(r['valid'])
        f = lambda a: torch.tensor(np.stack(a), dtype=torch.float32)
        y, m = f(Y), torch.tensor(np.stack(M)).float()
        lA, lD = net(f(XA)), net(f(XD))
        loss = 2 * (bce(lA, y) * m).sum() / m.sum() + (bce(lD, y) * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def predict(net, Xs):
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 64):
            out.append(torch.sigmoid(net(torch.from_numpy(np.stack(Xs[i:i + 64])).float())).numpy())
    return np.concatenate(out) if out else np.zeros((0, N))


def pick(p, r, thr, subframe=False, shift_ms=0.0):
    pm = np.where(r['valid'], p, 0.0)
    pk, _ = find_peaks(pm, height=thr, distance=5)
    return np.array(boundary.refine_peak_times(pm, pk, HOP, subframe=subframe, shift_s=shift_ms / 1000.0))


def person_macro(recs, probs_by_id, view, thr, subframe=False, shift_ms=0.0):
    per = {}
    for r in recs:
        p = probs_by_id[r['id']][view]
        per.setdefault(r['group'], []).append(B.f1(r['onsets'], pick(p, r, thr, subframe, shift_ms)))
    return float(np.mean([np.mean(v) for v in per.values()]))


def score(recs, probs, thr, subframe=False, shift_ms=0.0):
    return 0.5 * (person_macro(recs, probs, 0, thr, subframe, shift_ms) + person_macro(recs, probs, 1, thr, subframe, shift_ms))


def best_thr(recs, probs):
    sc = [(score(recs, probs, t), t) for t in THRS]
    s, t = max(sc, key=lambda z: (z[0], -abs(z[1] - 0.5)))
    return float(t), float(s)


def oof(name, flags, train_recs, steps):
    f = CACHE / f'oof_{name}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    persons = sorted({r['group'] for r in train_recs}); probs = {}
    for p in persons:
        t0 = time.time(); trn = [r for r in train_recs if r['group'] != p]; held = [r for r in train_recs if r['group'] == p]
        net = train(flags, trn, OOF_SEED, steps)
        views = [(base_X(r), noise_view(r, base_X(r))) for r in held]
        pc = predict(net, [v[0] for v in views]); pn = predict(net, [v[1] for v in views])
        for r, a, b in zip(held, pc, pn): probs[r['id']] = np.stack([a, b]).astype(np.float32)
        print(f'  [{name}] fold {p}: {len(held)} clips ({time.time() - t0:.0f}s)', flush=True)
        del net; gc.collect()
    pickle.dump(probs, open(f, 'wb'))
    return probs


def dev_eval(net, dev, thr, subframe=False, shift_ms=0.0):
    pc = predict(net, [base_X(r) for r in dev]); pn = predict(net, [noise_view(r, base_X(r)) for r in dev])
    f = lambda P: [B.f1(r['onsets'], pick(p, r, thr, subframe, shift_ms)) for r, p in zip(dev, P)]
    a, b = f(pc), f(pn)
    return dict(clean=float(np.mean(a)), noisy=float(np.mean(b)), clean_per_clip=a, noisy_per_clip=b)


def main():
    global CACHE
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    steps = 3 if args.smoke else STEPS
    if args.smoke:
        CACHE = WS / 'tmp' / 'v5_smoke_cache'
    V5.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(DECL.read_bytes()).hexdigest()
    recs = B.load(); tr = [r for r in recs if r['role'] == 'train']; dev = [r for r in recs if r['role'] == 'development']
    if args.smoke: tr = tr[::9]
    print(f'train {len(tr)} dev {len(dev)} persons {sorted({r["group"] for r in tr})} declaration {sha[:12]}', flush=True)
    assert not any(r['role'] == 'vocadito' for r in tr + dev)

    # ---- ladder
    cur, best, summary, oofs = {}, None, [], {}
    for name, add in LADDER:
        flags = {**cur, **add}; t0 = time.time()
        probs = oof(name, flags, tr, steps); oofs[name] = probs
        thr, s = best_thr(tr, probs); clean_s = person_macro(tr, probs, 0, thr); noisy_s = person_macro(tr, probs, 1, thr)
        adopted = best is None or s >= best + ADOPT_MARGIN
        if adopted: cur, best = flags, s
        summary.append(dict(name=name, flags=flags, thr=thr, score=s, oof_clean=clean_s, oof_noisy=noisy_s, adopted=bool(adopted)))
        print(f'{name:12s} flags={flags} thr={thr:.2f} score={s:.4f} clean={clean_s:.4f} noisy={noisy_s:.4f} '
              f'{"ADOPT" if adopted else "reject"} ({time.time() - t0:.0f}s)', flush=True)
    chosen = [x for x in summary if x['adopted']][-1]
    # ---- decoding on chosen config OOF predictions
    cp = oofs[chosen['name']]; base_s = score(tr, cp, chosen['thr']); dec = dict(subframe=False, shift_ms=0.0, score=base_s)
    grid = []
    for sub in (False, True):
        for sh in SHIFTS_MS:
            sc = score(tr, cp, chosen['thr'], sub, sh); grid.append(dict(subframe=sub, shift_ms=sh, score=sc))
            if sc > dec['score']: dec = dict(subframe=sub, shift_ms=float(sh), score=sc)
    if dec['score'] < base_s + ADOPT_MARGIN: dec = dict(subframe=False, shift_ms=0.0, score=base_s)
    print(f'decoding: plain {base_s:.4f} -> chosen {dec}', flush=True)

    # ---- final fits (3 seeds) for every evaluated config + v1 reference on dev_M02
    final = []
    v1 = boundary.load_boundary_model(boundary.V1_CHECKPOINT)
    ref = dev_eval(v1, dev, boundary.DEFAULT_THRESHOLD)
    print(f'studio_v1 reference dev_M02 onset F1: clean {ref["clean"]:.4f} noisy {ref["noisy"]:.4f}', flush=True)
    for item in summary:
        for seed in FINAL_SEEDS:
            f = V5 / f'{item["name"]}_s{seed}.json'
            if f.exists() and not args.smoke:
                final.append(json.load(open(f))); continue
            t0 = time.time(); net = train(item['flags'], tr, seed, steps)
            use_dec = item['name'] == chosen['name']
            d = dev_eval(net, dev, item['thr'], dec['subframe'] if use_dec else False, dec['shift_ms'] if use_dec else 0.0)
            row = dict(name=item['name'], seed=seed, flags=item['flags'], thr=item['thr'], decoded=bool(use_dec), **d)
            if not args.smoke:
                torch.save(net.state_dict(), V5 / f'{item["name"]}_s{seed}.pt'); json.dump(row, open(f, 'w'), default=float)
            final.append(row)
            print(f'  final {item["name"]} seed {seed}: dev clean {d["clean"]:.4f} noisy {d["noisy"]:.4f} ({time.time() - t0:.0f}s)', flush=True)
            del net; gc.collect()

    # ---- promote candidate checkpoint (seed 339 of chosen config) as boundary_transformer_v5.pt (not the Studio default)
    if not args.smoke:
        net = make_net(chosen['flags']); net.load_state_dict(torch.load(V5 / f'{chosen["name"]}_s339.pt', weights_only=True))
        torch.save(net.state_dict(), OUT / 'boundary_transformer_v5.pt')
        meta = dict(model_class='FastTNetV5' if chosen['flags'].get('feats') else 'FastTNet', arch='transformer_v5', model='FastTNet v5 ' + chosen['name'],
                    nin=5, d=32, stride=4, nhead=4, nlayers=2, eval_logit_offset=0.0, calibrated_threshold=chosen['thr'],
                    subframe_peaks=bool(dec['subframe']), onset_shift_s=dec['shift_ms'] / 1000.0, flags=chosen['flags'],
                    declaration_sha256=sha, seed=339)
        json.dump(meta, open(OUT / 'boundary_transformer_v5_meta.json', 'w'), indent=2, default=float)
    json.dump(dict(declaration_sha256=sha, smoke=bool(args.smoke), ladder=summary, chosen=chosen, decoding=dec, decoding_grid=grid,
                   studio_v1_dev_M02=ref, final=final),
              open(V5 / ('ladder_results_smoke.json' if args.smoke else 'ladder_results.json'), 'w'), indent=1, default=float)
    print('done', flush=True)


if __name__ == '__main__':
    main()
