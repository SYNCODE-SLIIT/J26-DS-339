"""R1 estimator on aligned HumTrans clips. Declared in results/studio/r1/declaration_humtrans.json. Resumable partial saves."""
import json, sys, hashlib
from pathlib import Path
import numpy as np, torch
from scipy.signal import find_peaks
WS = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(WS / 'tools'), str(WS / 'source')]
import fit_r1_estimator as F
from bpdr_input.repair.interventions import corrupt_boundaries
A = WS / 'datasets/humtrans_r1/analysis'; OUT = WS / 'results/studio/r1'; STEPS, BATCH, SEEDS = 300, 24, (339, 340)


def spanmed(c, ok, ids):
    o = np.full(len(c), np.nan)
    for s in np.unique(ids):
        m = (ids == s) & ok
        if m.sum() >= 3: o[ids == s] = np.median(c[m])
    return o


def rolling(c, ok, w=10):
    o = np.full(len(c), np.nan)
    for i in range(len(c)):
        m = ok[max(0, i - w // 2): i + w // 2 + 1]; v = c[max(0, i - w // 2): i + w // 2 + 1][m]
        if m.sum() >= 3: o[i] = np.median(v)
    return o


def load():
    man = json.load(open(WS / 'datasets/humtrans_r1/manifest_analysis.json')); recs = []; N = 501
    pad = lambda a, v=0.: np.r_[a, np.full(N - len(a), v)] if len(a) < N else a[:N]
    for s in man:
        f = A / f"{s['key']}.ref.npz"
        if 'sha256' not in s or not f.exists(): continue
        rf = np.load(f)
        if not bool(rf['kept']): continue
        z = np.load(A / f"{s['key']}.npz"); raw = z['raw_views'][0]; c = z['fine'].astype(float); ref = rf['ref']
        ok = z['reliable'].astype(bool) & np.isfinite(c) & ~np.isnan(ref)
        if ok.sum() < 30: continue
        t = np.arange(len(raw)) * .02; med = np.median(c[ok]); off = np.median((c - ref)[ok]); pk, _ = find_peaks(raw[:, 70], height=.5, distance=5)
        runs = np.cumsum(np.r_[0, (np.abs(np.diff(np.nan_to_num(ref))) > 1).astype(int)])
        cc = np.where(ok, c - off, 0.)
        base = dict(frame=np.abs(cc - np.nan_to_num(ref)), ext=np.abs(spanmed(cc, ok, np.digitize(t, t[pk])) - ref),
                    roll10=np.abs(rolling(cc, ok) - ref), true=np.abs(spanmed(cc, ok, runs) - ref))
        m = ok & ~np.isnan(base['ext']) & ~np.isnan(base['roll10']) & ~np.isnan(base['true'])
        if m.sum() < 30: continue
        recs.append(dict(group=s['person'], role=s['role'], id=s['key'], t=np.arange(N) * .02, ok=pad(ok, False).astype(bool),
                         pitch=pad(np.where(ok, c - med, 0.) / 100.), target=pad(np.where(ok, ref + off - med, 0.) / 100.), conf=pad(raw[:, 65]),
                         energy=pad(raw[:, 68], -100.), onset=pad(raw[:, 69]), edges=t[pk], base={k: float(v[m].mean()) for k, v in base.items()}))
    return recs


def fit(arm, train, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); use_b = arm != 'frame_only'
    net = F.Net(5 + use_b); opt = torch.optim.Adam(net.parameters(), 3e-3)
    for step in range(STEPS):
        sub = [train[i] for i in rng.choice(len(train), BATCH, replace=False)]
        X, XB, Xd, Y, M, S = F.batch(sub, rng, use_b, aug=use_b); m = M.float()
        L1 = lambda a, b: ((a - b).abs() * m).sum() / m.sum()
        yA, yB, yD = net(X), net(XB), net(Xd); loss = L1(yA, Y) + L1(yB, Y) + L1(yD, Y + S)
        if arm == 'bpdr': loss = loss + L1(yA, yB) + L1(yD - yA, S)
        opt.zero_grad(); loss.backward(); opt.step()
    return net, use_b


def score(net, use_b, rec):
    y = F.predict(net, rec, use_b); row = dict(mae=F.mae(y, rec), **{'base_' + k: v for k, v in rec['base'].items()})
    for k in ('split', 'merge', 'jitter'):
        if not use_b: row['chg_' + k] = 0.; continue
        try: e2 = corrupt_boundaries(rec['edges'], rec['t'], k, seed=339)
        except ValueError: continue
        row['chg_' + k] = float(np.abs(F.predict(net, rec, use_b, e2) - y)[rec['ok']].mean() * 100)
    return row


def main():
    recs = load(); tr = [r for r in recs if r['role'] == 'train']; dev = [r for r in recs if r['role'] == 'development']
    persons = sorted({r['group'] for r in tr}); print('train clips', len(tr), 'dev clips', len(dev), 'persons', persons, flush=True)
    sha = hashlib.sha256((OUT / 'declaration_humtrans.json').read_bytes()).hexdigest(); res = []
    jobs = [(p, [r for r in tr if r['group'] != p], [r for r in tr if r['group'] == p]) for p in persons] + [('DEV_M02', tr, dev)]
    for tag, train, test in jobs:
        for seed in SEEDS:
            for arm in F.ARMS:
                net, use_b = fit(arm, train, seed)
                ref = np.mean([F.mae(r['pitch'], r) for r in train[:80]])
                ok = bool(np.mean([F.mae(F.predict(net, r, use_b), r) for r in train[:80]]) < .9 * ref)
                for r in test: res.append(dict(fold=tag, seed=seed, arm=arm, person=r['group'], id=r['id'], healthy=ok, **score(net, use_b, r)))
                print(tag, seed, arm, 'healthy', ok, flush=True)
                json.dump(dict(declaration_sha256=sha, partial=True, rows=res), open(OUT / 'humtrans_fit_partial.json', 'w'), default=float)
    json.dump(dict(declaration_sha256=sha, rows=res), open(OUT / 'humtrans_fit_results.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
