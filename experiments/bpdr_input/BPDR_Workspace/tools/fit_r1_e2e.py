"""End-to-end note F1, Transformer vs TCN, and lean-feature BPDR test. Declared in results/studio/r1/declaration_e2e.json.
Resumable: partial results are written after every fit. Run: ..\\.venv\\Scripts\\python.exe tools\\fit_r1_e2e.py"""
import json, sys, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as tnf
from scipy.signal import find_peaks
WS = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(WS / 'tools'), str(WS / 'source')]
import fit_r1_boundary as B
F = B.F; OUT = WS / 'results/studio/r1'; V = B.V; DATA = B.DATA; N = 501
SEEDS_BY = {'tcn': (339, 340), 'transformer': (339,)}; ARMS = ('frame_only', 'augmentation_only', 'bpdr'); KINDS = ('clean', 'split', 'merge', 'jitter')
CONFIGS = [('tcn', 'full'), ('tcn', 'lean'), ('transformer', 'full'), ('transformer', 'lean')]
SETTINGS = {'tcn': dict(steps=300, lr=3e-3), 'transformer': dict(steps=250, lr=3e-3)}


class FastTransformerBlock(nn.Module):
    def __init__(self, d=32, h=4):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        B_, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        f = lambda t: t.view(B_, T, self.h, C // self.h).transpose(1, 2)
        a = tnf.scaled_dot_product_attention(f(q), f(k), f(v)).transpose(1, 2).contiguous().view(B_, T, C)
        x = x + self.proj(a)
        return x + self.ff(self.n2(x))


class FastTNet(nn.Module):
    """Stride-4 dilated residual highway + multi-scale pitch step deltas + depthwise conv positional encoding + fast pre-norm attention."""
    def __init__(self, nin, d=32, nhead=4, nlayers=2):
        super().__init__()
        self.in_proj = nn.Conv1d(nin + 5, d, 3, padding=1)
        self.fine_convs = nn.ModuleList([nn.Conv1d(d, d, 5, padding=2 * k, dilation=k) for k in (1, 2, 4, 8)])
        self.down = nn.Conv1d(d, d, 4, stride=4, padding=1)
        self.pos_conv = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.enc = nn.Sequential(*[FastTransformerBlock(d, nhead) for _ in range(nlayers)])
        self.up = nn.ConvTranspose1d(d, d, 4, stride=4, padding=0)
        self.gate = nn.Parameter(torch.tensor(0.2))
        self.out = nn.Conv1d(d, 1, 1)

    def _deltas(self, x):
        p, ok = x[:, :, 0], x[:, :, 1]
        dp1 = torch.zeros_like(p)
        dp1[:, 1:] = (p[:, 1:] - p[:, :-1]).abs() * ok[:, 1:] * ok[:, :-1]
        dok = torch.zeros_like(ok)
        dok[:, 1:] = (ok[:, 1:] - ok[:, :-1]).abs()
        steps = [dp1.clamp(0, 2.0), dok]
        po = (p * ok)[:, None]
        ok1 = ok[:, None]
        for w in (3, 6, 10):
            pad_p = tnf.pad(po, (w, w))
            pad_ok = tnf.pad(ok1, (w, w))
            cs_p = pad_p.cumsum(-1)
            cs_ok = pad_ok.cumsum(-1)
            T = p.shape[1]
            lp = (cs_p[:, 0, w:w+T] - cs_p[:, 0, :T]) / (cs_ok[:, 0, w:w+T] - cs_ok[:, 0, :T]).clamp_min(1.0)
            rp = (cs_p[:, 0, w+1:w+1+T] - cs_p[:, 0, 1:1+T]) / (cs_ok[:, 0, w+1:w+1+T] - cs_ok[:, 0, 1:1+T]).clamp_min(1.0)
            valid = (pad_ok[:, 0, w-1:w-1+T] > 0.5) & (pad_ok[:, 0, w+1:w+1+T] > 0.5)
            steps.append(((rp - lp).abs() * valid.float()).clamp(0, 2.0))
        return torch.stack(steps, dim=-1)

    def forward(self, x):
        dx = self._deltas(x)
        z = self.in_proj(torch.cat([x, dx], dim=-1).transpose(1, 2))
        for c in self.fine_convs:
            z = z + tnf.gelu(c(z))
        zc = tnf.gelu(self.down(z))
        zc = zc + self.pos_conv(zc)
        zc = self.enc(zc.transpose(1, 2)).transpose(1, 2)
        zu = self.up(zc)
        zu = tnf.pad(zu, (0, max(0, x.shape[1] - zu.shape[-1])), mode='replicate')[..., :x.shape[1]]
        z = z + self.gate * tnf.gelu(zu)
        out = self.out(z)[:, 0]
        return out if self.training else out + 0.9


TNet = FastTNet


_orig_feats = B.feats
def lean_feats(r, edges, use_b, pitch=None):
    cols = [r['pitch'] if pitch is None else pitch, r['ok'].astype(float), r['conf']]
    if use_b: cols.append(B.game_boundary_track(r['t'], edges))
    return np.stack(cols, -1).astype(np.float32)


def fit(arch, feat, arm, train, seed):
    B.feats = lean_feats if feat == 'lean' else _orig_feats
    torch.manual_seed(seed); rng = np.random.default_rng(seed); use_b = arm != 'frame_only'
    nin = (3 if feat == 'lean' else 5) + use_b; net = F.Net(nin) if arch == 'tcn' else FastTNet(nin)
    s = SETTINGS[arch]; opt = (torch.optim.Adam if arch == 'tcn' else torch.optim.AdamW)(net.parameters(), s['lr'])
    bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(10.)); net.train()
    for step in range(s['steps']):
        sub = [train[i] for i in rng.choice(len(train), B.BATCH, replace=False)]
        X, XB, Xd, Y, M = B.batch(sub, rng, use_b, aug=use_b); lA, lB, lD = net(X), net(XB), net(Xd)
        loss = sum((bce(l, Y) * M).sum() / M.sum() for l in (lA, lB, lD))
        if arm == 'bpdr':
            pA, pB, pD = torch.sigmoid(lA), torch.sigmoid(lB), torch.sigmoid(lD)
            loss = loss + ((pA - pB).abs() * M).sum() / M.sum() * 10 + ((pB - pD).abs() * M).sum() / M.sum() * 10
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net, use_b


def make_notes(edges, fine, rel):
    t = np.arange(len(fine)) * .02; seg = np.digitize(t, np.sort(edges)); out = []
    for s in np.unique(seg):
        idx = np.flatnonzero((seg == s) & rel)
        if len(idx) >= 4: out.append((float(t[idx[0]]), int(round(69 + np.median(fine[idx]) / 100))))
    return out


def note_f1(est, ref):
    used, hit = set(), 0
    for ro, rm in ref:
        c = [k for k, (eo, em) in enumerate(est) if k not in used and abs(eo - ro) <= .05 and abs((em - rm) * 100) <= 50]
        if c: used.add(min(c, key=lambda k: abs(est[k][0] - ro))); hit += 1
    if hit == 0: return 0.0
    p, rc = hit / len(est), hit / len(ref); return 2 * p * rc / (p + rc)


def voc_meta():
    meta = {}
    for r in json.load(open(DATA / 'train_originals.json')):
        if 'excluded' in r or not r['source_id'].startswith('vocadito'): continue
        refs = []
        for a in ('A1', 'A2'):
            f = V / f"{r['parent_source_id']}_notes{a}.csv"
            if not f.exists(): continue
            m = np.loadtxt(f, delimiter=',', ndmin=2); off = r['crop_offset_seconds']; on = m[:, 0] - off; k = (on >= 0) & (on < 10)
            refs.append([(float(o), 69 + 12 * np.log2(h / 440)) for o, h in zip(on[k], m[k, 1])])
        z = np.load(r['analysis_A']); c = z['fine'].astype(float); rel = z['reliable'].astype(bool) & np.isfinite(c)
        meta[r['source_id']] = dict(fine=np.where(rel, c, 0.), rel=rel, refs=refs)
    return meta


def evaluate(net, use_b, thr, test, meta):
    rows = []
    for r in test:
        row = dict(person=r['group'], id=r['id']); m = meta.get(r['id'])
        for k in KINDS:
            e = r['edges'] if k == 'clean' else B.corrupted(r['edges'], r['t'], k, 339)
            pk = B.picks(B.probs(net, r, use_b, e), r, thr)
            row['f1_' + k] = B.f1(r['onsets'], pk); row['ext_' + k] = B.extractor_f1(r, e)
            if m is not None and m['refs']:
                row['note_model_' + k] = float(np.mean([note_f1(make_notes(pk, m['fine'], m['rel']), ref) for ref in m['refs'] if ref]))
                row['note_ext_' + k] = float(np.mean([note_f1(make_notes(e, m['fine'], m['rel']), ref) for ref in m['refs'] if ref]))
        rows.append(row)
    return rows


def main():
    recs = B.load(); tr = [r for r in recs if r['role'] == 'train']; dev = [r for r in recs if r['role'] == 'development']; voc = [r for r in recs if r['role'] == 'vocadito']
    meta = voc_meta(); print('train', len(tr), 'dev', len(dev), 'vocadito', len(voc), 'with notes', sum(r['id'] in meta for r in voc), flush=True)
    sha = hashlib.sha256((OUT / 'declaration_e2e.json').read_bytes()).hexdigest(); res = []; pf = OUT / 'e2e_fit_partial.json'
    if pf.exists():
        prev = json.load(open(pf))
        if prev.get('declaration_sha256') == sha:
            res = prev.get('rows', [])
        else:
            # Declaration amended: preserve completed TCN rows, refit Transformer
            res = [r for r in prev.get('rows', []) if r.get('arch') == 'tcn']
            print(f"Declaration amended ({sha[:8]}); preserving {len(res)} completed TCN rows", flush=True)
            json.dump(dict(declaration_sha256=sha, partial=True, rows=res), open(pf, 'w'), default=float)
    done = {(r['arch'], r['feat'], r['arm'], r['seed']) for r in res}; print('resuming; completed fits:', len(done), flush=True)
    best_trans_trf1 = -1.0
    for arch, feat in CONFIGS:
        for seed in SEEDS_BY[arch]:
            for arm in ARMS:
                if (arch, feat, arm, seed) in done: continue
                net, use_b = fit(arch, feat, arm, tr, seed); thr, trf1 = B.pick_threshold(net, use_b, tr)
                ext_tr = np.mean([B.extractor_f1(r, r['edges']) for r in tr[:80]]); healthy = bool(trf1 >= .9 * ext_tr)
                if arch == 'transformer':
                    if trf1 > best_trans_trf1:
                        best_trans_trf1 = trf1
                        torch.save(net.state_dict(), OUT / 'boundary_transformer.pt')
                        print(f"Saved best transformer checkpoint to {OUT / 'boundary_transformer.pt'} (feat={feat}, arm={arm}, trF1={trf1:.4f})", flush=True)
                    if feat == 'full' and arm == 'bpdr':
                        torch.save(net.state_dict(), OUT / 'boundary_transformer_bpdr.pt')
                for name, test in (('vocadito', voc), ('dev_M02', dev)):
                    for row in evaluate(net, use_b, thr, test, meta): res.append(dict(arch=arch, feat=feat, arm=arm, seed=seed, split=name, thr=thr, healthy=healthy, **row))
                print(arch, feat, arm, seed, 'thr', thr, 'train F1', round(trf1, 3), 'extractor', round(ext_tr, 3), 'healthy', healthy, flush=True)
                json.dump(dict(declaration_sha256=sha, partial=True, rows=res), open(pf, 'w'), default=float)
    json.dump(dict(declaration_sha256=sha, rows=res), open(OUT / 'e2e_fit_results.json', 'w'), indent=1, default=float)
    print(f"Done. Produced {OUT / 'e2e_fit_results.json'} with {len(res)} rows.", flush=True)


if __name__ == '__main__':
    main()
