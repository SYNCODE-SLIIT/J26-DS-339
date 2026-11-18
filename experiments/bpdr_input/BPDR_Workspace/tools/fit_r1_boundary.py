"""R1 boundary model, three matched arms. Declared in results/studio/r1/declaration_boundary.json. Resumable partial saves.
Run: ..\\.venv\\Scripts\\python.exe tools\\fit_r1_boundary.py"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, torch
from scipy.signal import find_peaks
WS = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(WS / 'tools'), str(WS / 'source')]
import fit_r1_estimator as F
from bpdr_input.repair.interventions import corrupt_boundaries
from bpdr_input.preparation.pairs.shared_pair_audit import game_boundary_track
A = WS / 'datasets/humtrans_r1/analysis'; V = WS / 'datasets/vocadito/Annotations/Notes'
DATA = WS / 'results/transformer_usability/2026-11-04_v64_reliable_melodic_pairs'; OUT = WS / 'results/studio/r1'
STEPS, BATCH, SEEDS, N = 300, 24, (339, 340), 501
KINDS = ('clean', 'split', 'merge', 'jitter'); THRS = (.3, .4, .5, .6, .7, .8)


def f1(ref, est, tol=.05):
    ref, est = np.sort(ref), np.sort(est)
    if len(ref) == 0 or len(est) == 0: return 0.0
    used, hit = set(), 0
    for r in ref:
        j = [k for k in range(len(est)) if k not in used and abs(est[k] - r) <= tol]
        if j: used.add(min(j, key=lambda k: abs(est[k] - r))); hit += 1
    if hit == 0: return 0.0
    p, rc = hit / len(est), hit / len(ref); return 2 * p * rc / (p + rc)


def mk(z, group, role, rid, onsets, valid_lo, valid_hi):
    raw = z['raw_views'][0]; c = z['fine'].astype(float); ok = z['reliable'].astype(bool) & np.isfinite(c)
    if ok.sum() < 30: return None
    t = np.arange(len(raw)) * .02; pad = lambda a, v=0.: np.r_[a, np.full(N - len(a), v)] if len(a) < N else a[:N]
    pk, _ = find_peaks(raw[:, 70], height=.5, distance=5); y = np.zeros(len(raw))
    for o in onsets:
        i = int(round(o / .02))
        if 0 <= i < len(raw):
            y[i] = 1.; y[max(i - 1, 0)] = max(y[max(i - 1, 0)], .5); y[min(i + 1, len(raw) - 1)] = max(y[min(i + 1, len(raw) - 1)], .5)
    return dict(group=group, role=role, id=rid, t=np.arange(N) * .02, ok=pad(ok, False).astype(bool), pitch=pad(np.where(ok, c - np.median(c[ok]), 0.) / 100.),
                conf=pad(raw[:, 65]), energy=pad(raw[:, 68], -100.), onset=pad(raw[:, 69]), edges=t[pk], onsets=np.asarray(onsets), y=pad(y),
                valid=pad((t >= valid_lo) & (t <= valid_hi), False).astype(bool), target=np.zeros(N))


def load():
    recs = []
    for s in json.load(open(WS / 'datasets/humtrans_r1/manifest_analysis.json')):
        f = A / f"{s['key']}.ref.npz"
        if 'sha256' not in s or not f.exists(): continue
        rf = np.load(f)
        if not bool(rf['kept']): continue
        t = np.arange(501) * .02; nid = rf['nid']; ons = np.array([t[np.flatnonzero(nid == k)[0]] for k in np.unique(nid[nid >= 0])])
        if len(ons) < 3: continue
        r = mk(np.load(A / f"{s['key']}.npz"), s['person'], s['role'], s['key'], ons, ons.min() - .1, ons.max() + .5)
        if r: recs.append(r)
    for r0 in json.load(open(DATA / 'train_originals.json')):
        if 'excluded' in r0 or not r0['source_id'].startswith('vocadito'): continue
        f = V / f"{r0['parent_source_id']}_notesA1.csv"
        if not f.exists(): continue
        a = np.loadtxt(f, delimiter=',', ndmin=2); off = r0['crop_offset_seconds']; ons = a[:, 0] - off; ons = ons[(ons >= 0) & (ons < 10)]
        if len(ons) < 3: continue
        r = mk(np.load(r0['analysis_A']), r0['group'], 'vocadito', r0['source_id'], ons, 0., 10.)
        if r: recs.append(r)
    return recs


def feats(r, edges, use_b, pitch=None):
    cols = [r['pitch'] if pitch is None else pitch, r['ok'].astype(float), r['conf'], F.zs(r['energy']), r['onset']]
    if use_b: cols.append(game_boundary_track(r['t'], edges))
    return np.stack(cols, -1).astype(np.float32)


def corrupted(edges, t, kind, seed):
    try: return corrupt_boundaries(edges, t, kind, seed=seed)
    except ValueError: return edges


def batch(sub, rng, use_b, aug):
    X, XB, Xd, Y, M = [], [], [], [], []
    for r in sub:
        e2 = corrupted(r['edges'], r['t'], ('split', 'merge', 'jitter')[rng.integers(3)], int(rng.integers(1 << 30))) if (aug and rng.random() < .5) else r['edges']
        okf = np.flatnonzero(r['ok']); s = np.zeros(N, bool)
        if len(okf): a = int(rng.choice(okf)); s[a:a + 14] = True; s &= r['ok']
        p2 = r['pitch'] + rng.choice([-.3, -.2, .2, .3]) * s
        X.append(feats(r, r['edges'], use_b)); XB.append(feats(r, e2, use_b)); Xd.append(feats(r, e2, use_b, p2)); Y.append(r['y']); M.append(r['valid'])
    f = lambda a: torch.tensor(np.stack(a), dtype=torch.float32)
    return f(X), f(XB), f(Xd), f(Y), torch.tensor(np.stack(M)).float()


def fit(arm, train, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); use_b = arm != 'frame_only'; net = F.Net(5 + use_b); opt = torch.optim.Adam(net.parameters(), 3e-3)
    bce = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(10.))
    for step in range(STEPS):
        sub = [train[i] for i in rng.choice(len(train), BATCH, replace=False)]
        X, XB, Xd, Y, M = batch(sub, rng, use_b, aug=use_b); lA, lB, lD = net(X), net(XB), net(Xd)
        loss = sum((bce(l, Y) * M).sum() / M.sum() for l in (lA, lB, lD))
        if arm == 'bpdr':
            pA, pB, pD = torch.sigmoid(lA), torch.sigmoid(lB), torch.sigmoid(lD)
            loss = loss + ((pA - pB).abs() * M).sum() / M.sum() * 10 + ((pB - pD).abs() * M).sum() / M.sum() * 10
        opt.zero_grad(); loss.backward(); opt.step()
    return net, use_b


def probs(net, r, use_b, edges):
    with torch.no_grad(): return torch.sigmoid(net(torch.tensor(feats(r, edges, use_b))[None])[0]).numpy()


def picks(p, r, thr):
    pk, _ = find_peaks(np.where(r['valid'], p, 0), height=thr, distance=5); return r['t'][pk]


def extractor_f1(r, edges):
    idx = np.clip(np.round(edges / .02).astype(int), 0, N - 1)
    return f1(r['onsets'], edges[r['valid'][idx]])


def pick_threshold(net, use_b, train):
    best = (-1, .5)
    for thr in THRS:
        v = np.mean([f1(r['onsets'], picks(probs(net, r, use_b, r['edges']), r, thr)) for r in train[:80]])
        if v > best[0]: best = (v, thr)
    return best[1], best[0]


def evaluate(net, use_b, thr, test):
    rows = []
    for r in test:
        row = dict(person=r['group'], id=r['id'])
        for k in KINDS:
            e = r['edges'] if k == 'clean' else corrupted(r['edges'], r['t'], k, 339)
            row['f1_' + k] = f1(r['onsets'], picks(probs(net, r, use_b, e), r, thr)); row['ext_' + k] = extractor_f1(r, e)
            row['f1100_' + k] = f1(r['onsets'], picks(probs(net, r, use_b, e), r, thr), .10)
        rows.append(row)
    return rows


def main():
    recs = load(); tr = [r for r in recs if r['role'] == 'train']; dev = [r for r in recs if r['role'] == 'development']; voc = [r for r in recs if r['role'] == 'vocadito']
    persons = sorted({r['group'] for r in tr}); print('train', len(tr), 'dev', len(dev), 'vocadito', len(voc), flush=True)
    sha = hashlib.sha256((OUT / 'declaration_boundary.json').read_bytes()).hexdigest(); res = []
    pf = OUT / 'boundary_fit_partial.json'
    if pf.exists():
        prev = json.load(open(pf)); assert prev['declaration_sha256'] == sha, 'declaration changed since the partial run'; res = prev['rows']
    # a (fold, seed, arm) is complete only if all of its test splits were written
    done = {(r['fold'], r['seed'], r['arm']) for r in res}; print('resuming; completed fits:', len(done), flush=True)
    jobs = [(p, [r for r in tr if r['group'] != p], {'heldout': [r for r in tr if r['group'] == p]}) for p in persons] + [('ALL', tr, {'dev_M02': dev, 'vocadito': voc})]
    for tag, train, tests in jobs:
        for seed in SEEDS:
            for arm in F.ARMS:
                if (tag, seed, arm) in done: continue
                net, use_b = fit(arm, train, seed); thr, trf1 = pick_threshold(net, use_b, train)
                ext_tr = np.mean([extractor_f1(r, r['edges']) for r in train[:80]]); healthy = bool(trf1 >= .9 * ext_tr)
                for name, test in tests.items():
                    for row in evaluate(net, use_b, thr, test): res.append(dict(fold=tag, seed=seed, arm=arm, split=name, thr=thr, healthy=healthy, **row))
                print(tag, seed, arm, 'thr', thr, 'train F1', round(trf1, 3), 'extractor', round(ext_tr, 3), 'healthy', healthy, flush=True)
                json.dump(dict(declaration_sha256=sha, partial=True, rows=res), open(OUT / 'boundary_fit_partial.json', 'w'), default=float)
    json.dump(dict(declaration_sha256=sha, rows=res), open(OUT / 'boundary_fit_results.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
