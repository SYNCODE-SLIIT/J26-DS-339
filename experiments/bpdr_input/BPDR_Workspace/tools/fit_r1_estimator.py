"""R1 note-centre estimator, three matched arms, leave-group-out. See results/studio/r1/declaration.json (declared first).
Run from the owner's terminal, one job at a time:  ..\.venv\Scripts\python.exe tools\fit_r1_estimator.py"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.signal import find_peaks
WS = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(WS / 'source')]
from bpdr_input.repair.interventions import corrupt_boundaries
from bpdr_input.preparation.pairs.shared_pair_audit import game_boundary_track
DATA = WS / 'results/transformer_usability/2026-11-04_v64_reliable_melodic_pairs'
V = WS / 'datasets/vocadito/Annotations/Notes'; OUT = WS / 'results/studio/r1'
DECL = OUT / 'declaration.json'; SEEDS = (339, 340, 341); ARMS = ('frame_only', 'augmentation_only', 'bpdr'); EPOCHS = 150
torch.set_num_threads(2)

def load():
    recs = []
    for r in json.load(open(DATA / 'train_originals.json')):
        if 'excluded' in r or not r['source_id'].startswith('vocadito'): continue
        f = V / f"{r['parent_source_id']}_notesA1.csv"
        if not f.exists(): continue
        a = np.loadtxt(f, delimiter=',', ndmin=2); z = np.load(r['analysis_A']); raw = z['raw_views'][0]
        t = np.arange(len(raw)) * .02; T = t + r['crop_offset_seconds']
        ok = z['reliable'].astype(bool) & np.isfinite(z['fine']); c = z['fine'].astype(float)
        ref = np.full(len(t), np.nan)
        for on, hz, du in a[:, :3]:
            ref[(T >= on) & (T < on + du)] = 1200 * np.log2(hz / 440)
        ok = ok & ~np.isnan(ref)
        if ok.sum() < 10: continue
        med = np.median(c[ok]); off = np.median((c - ref)[ok])
        pk, _ = find_peaks(raw[:, 70], height=.5, distance=5)
        nid = np.digitize(T, a[:, 0])  # not used by the model
        N = 501; pad = lambda a, v=0.: np.r_[a, np.full(N - len(a), v)] if len(a) < N else a[:N]
        recs.append(dict(group=r['group'], id=r['source_id'], t=np.arange(N) * .02, ok=pad(ok, False).astype(bool), pitch=pad(np.where(ok, c - med, 0.) / 100.),
                         target=pad(np.where(ok, ref + off - med, 0.) / 100.), conf=pad(raw[:, 65]), energy=pad(raw[:, 68], -100.), onset=pad(raw[:, 69]), edges=t[pk]))
    return recs
def zs(x): return (x - x.mean()) / (x.std() + 1e-6)

class Net(nn.Module):
    """Dilated residual temporal convolutions (receptive field about +-1.2 s); same size in all arms."""
    def __init__(self, nin, h=32):
        super().__init__(); self.i = nn.Conv1d(nin, h, 1)
        self.c = nn.ModuleList([nn.Conv1d(h, h, 5, padding=2 * d, dilation=d) for d in (1, 2, 4, 8, 1, 2, 4, 8)]); self.o = nn.Conv1d(h, 1, 1)
    def forward(self, x):
        z = self.i(x.transpose(1, 2))
        for c in self.c: z = z + torch.nn.functional.gelu(c(z))
        return self.o(z)[:, 0]

def feats(rec, pitch, edges, use_b):
    cols = [pitch, rec['ok'].astype(float), rec['conf'], zs(rec['energy']), rec['onset']]
    if use_b: cols.append(game_boundary_track(rec['t'], edges))
    return np.stack(cols, -1).astype(np.float32)
def corrupt(edges, t, rng):
    k = ('split', 'merge', 'jitter')[rng.integers(3)]
    try: return corrupt_boundaries(edges, t, k, seed=int(rng.integers(1 << 30)))
    except ValueError: return edges

def batch(recs, rng, use_b, aug):
    X, XB, Y, M, S, Xd = [], [], [], [], [], []
    for r in recs:
        e2 = corrupt(r['edges'], r['t'], rng) if aug else r['edges']
        X.append(feats(r, r['pitch'], r['edges'], use_b)); XB.append(feats(r, r['pitch'], e2, use_b))
        okf = np.flatnonzero(r['ok']); s = np.zeros(len(r['t']), bool)
        if len(okf):
            a = int(rng.choice(okf)); s[a:a + 14] = True; s &= r['ok']
        e = rng.choice([-.3, -.2, .2, .3]); p2 = r['pitch'] + e * s
        Xd.append(feats(r, p2, e2 if aug else r['edges'], use_b)); S.append(e * s)
        Y.append(r['target']); M.append(r['ok'])
    f = lambda a: torch.tensor(np.stack(a), dtype=torch.float32)
    return f(X), f(XB), f(Xd), f(Y), torch.tensor(np.stack(M)), f(S)

def fit(arm, train, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    use_b = arm != 'frame_only'; net = Net(5 + use_b); opt = torch.optim.Adam(net.parameters(), 3e-3)
    for ep in range(EPOCHS):
        X, XB, Xd, Y, M, S = batch(train, rng, use_b, aug=use_b)
        m = M.float(); L1 = lambda a, b: ((a - b).abs() * m).sum() / m.sum()
        yA, yB, yD = net(X), net(XB), net(Xd)
        loss = L1(yA, Y) + L1(yB, Y) + L1(yD, Y + S)           # all arms: supervised on originals, altered boundaries, detuned copies
        if arm == 'bpdr':
            loss = loss + L1(yA, yB) + L1(yD - yA, S)           # L_B and L_D
        opt.zero_grad(); loss.backward(); opt.step()
    return net, use_b

def predict(net, rec, use_b, edges=None):
    with torch.no_grad(): return net(torch.tensor(feats(rec, rec['pitch'], rec['edges'] if edges is None else edges, use_b))[None])[0].numpy()
def mae(y, rec):
    m = rec['ok']; d = (y - rec['target'])[m]; d = d - np.median(d); return float(np.abs(d).mean() * 100)

def main():
    recs = load(); groups = sorted({r['group'] for r in recs}); folds = [groups[i::7] for i in range(7)]
    print('records', len(recs), 'groups', len(groups), flush=True)
    decl_sha = hashlib.sha256(DECL.read_bytes()).hexdigest(); res = []
    for fi, fg in enumerate(folds):
        train = [r for r in recs if r['group'] not in fg]; test = [r for r in recs if r['group'] in fg]
        for seed in SEEDS:
            for arm in ARMS:
                net, use_b = fit(arm, train, seed); healthy_ref = np.mean([mae(r['pitch'], r) for r in train])
                healthy = float(np.mean([mae(predict(net, r, use_b), r) for r in train])) < .9 * healthy_ref
                healthy = bool(healthy)
                for r in test:
                    y = predict(net, r, use_b); row = dict(fold=fi, seed=seed, arm=arm, group=r['group'], id=r['id'], mae=mae(y, r), healthy=healthy)
                    if use_b:
                        for k in ('split', 'merge', 'jitter'):
                            try: e2 = corrupt_boundaries(r['edges'], r['t'], k, seed=339)
                            except ValueError: continue
                            row['chg_' + k] = float(np.abs(predict(net, r, use_b, e2) - y)[r['ok']].mean() * 100)
                    else: row.update(chg_split=0., chg_merge=0., chg_jitter=0.)
                    res.append(row)
                print(fi, seed, arm, 'healthy', healthy, flush=True)
                json.dump(dict(declaration_sha256=decl_sha, partial=True, rows=res), open(OUT / 'fit_results_partial.json', 'w'))
    json.dump(dict(declaration_sha256=decl_sha, rows=res), open(OUT / 'fit_results.json', 'w'), indent=1, default=float)
if __name__ == '__main__': main()
