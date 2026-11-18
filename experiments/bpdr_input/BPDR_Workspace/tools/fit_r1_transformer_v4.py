"""FastTNet v4 & v1_rep boundary model experiment (Item 1a).
Declared in results/studio/r1/declaration_transformer_v4.json.
Resumable: partial results are written after every fit.
Run: ..\\.venv\\Scripts\\python.exe tools\\fit_r1_transformer_v4.py
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / 'tools'), str(WS / 'source')]
import fit_r1_boundary as B
import fit_r1_e2e as E

OUT = WS / 'results/studio/r1'
DECL_PATH = OUT / 'declaration_transformer_v4.json'
PARTIAL_PATH = OUT / 'transformer_v4_partial.json'
RESULTS_PATH = OUT / 'transformer_v4_results.json'
CKPT_PATH = OUT / 'boundary_transformer_v4.pt'
CKPT_BPDR_PATH = OUT / 'boundary_transformer_v4_bpdr.pt'
META_PATH = OUT / 'boundary_transformer_v4_meta.json'

STEPS = 250
LR = 3e-3
POS_WEIGHT = 10.0

THRS_V1 = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
THRS_V4 = tuple(round(0.20 + 0.05 * i, 2) for i in range(16))  # 0.20 .. 0.95

JOBS = [
    ('transformer_v1_rep', 'full', 'frame_only', 340, 0.9, THRS_V1),
    ('transformer_v1_rep', 'full', 'bpdr', 340, 0.9, THRS_V1),
    ('transformer_v4', 'full', 'frame_only', 339, 0.0, THRS_V4),
    ('transformer_v4', 'full', 'bpdr', 339, 0.0, THRS_V4),
    ('transformer_v4', 'full', 'frame_only', 340, 0.0, THRS_V4),
    ('transformer_v4', 'full', 'bpdr', 340, 0.0, THRS_V4),
]

_ORIG_FEATS = B.feats


class FastTNetV4(E.FastTNet):
    """Exact stride-4 FastTNet with configurable eval_logit_offset (0.9 for v1_rep, 0.0 for v4)."""

    def __init__(self, nin, d=32, nhead=4, nlayers=2, eval_logit_offset=0.0):
        super().__init__(nin=nin, d=d, nhead=nhead, nlayers=nlayers)
        self.nin = nin
        self.d = d
        self.nhead = nhead
        self.nlayers = nlayers
        self.eval_logit_offset = float(eval_logit_offset)

    def forward(self, x):
        dx = self._deltas(x)
        z = self.in_proj(torch.cat([x, dx], dim=-1).transpose(1, 2))
        for c in self.fine_convs:
            z = z + tnf.gelu(c(z))
        zc = tnf.gelu(self.down(z))
        zc = zc + self.pos_conv(zc)
        zc = self.enc(zc.transpose(1, 2)).transpose(1, 2)
        zu = self.up(zc)
        zu = tnf.pad(zu, (0, max(0, x.shape[1] - zu.shape[-1])), mode='replicate')[..., : x.shape[1]]
        z = z + self.gate * tnf.gelu(zu)
        out = self.out(z)[:, 0]
        if self.training or self.eval_logit_offset == 0.0:
            return out
        return out + self.eval_logit_offset


def pick_threshold_grid(net, use_b, clips, thrs):
    """Pick threshold in thrs maximizing mean onset F1 at 50 ms on tr[:80]."""
    net.eval()
    probs_list = [B.probs(net, r, use_b, r['edges']) for r in clips]
    best_f1, best_thr = -1.0, 0.5
    for thr in thrs:
        scores = [B.f1(r['onsets'], B.picks(p, r, thr)) for r, p in zip(clips, probs_list)]
        v = float(np.mean(scores))
        if v > best_f1:
            best_f1, best_thr = v, float(thr)
    return best_thr, best_f1


def fit_job(arm, train, seed, eval_logit_offset):
    B.feats = _ORIG_FEATS
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    use_b = arm != 'frame_only'
    nin = 5 + int(use_b)
    net = FastTNetV4(nin=nin, eval_logit_offset=eval_logit_offset)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(POS_WEIGHT))

    net.train()
    for _ in range(STEPS):
        sub = [train[i] for i in rng.choice(len(train), B.BATCH, replace=False)]
        X, XB, Xd, Y, M = B.batch(sub, rng, use_b, aug=use_b)
        lA, lB, lD = net(X), net(XB), net(Xd)
        loss = sum((bce(l, Y) * M).sum() / M.sum() for l in (lA, lB, lD))
        if arm == 'bpdr':
            pA, pB, pD = torch.sigmoid(lA), torch.sigmoid(lB), torch.sigmoid(lD)
            loss = (
                loss
                + ((pA - pB).abs() * M).sum() / M.sum() * 10.0
                + ((pB - pD).abs() * M).sum() / M.sum() * 10.0
            )
        opt.zero_grad()
        loss.backward()
        opt.step()

    net.eval()
    return net, use_b, nin


def build_meta_json(fit_meta):
    v4_ff = [m for m in fit_meta if m['arch'] == 'transformer_v4' and m['feat'] == 'full' and m['arm'] == 'frame_only']
    v4_bp = [m for m in fit_meta if m['arch'] == 'transformer_v4' and m['feat'] == 'full' and m['arm'] == 'bpdr']
    best_ff = max(v4_ff, key=lambda m: m['train_f1']) if v4_ff else None
    best_bp = max(v4_bp, key=lambda m: m['train_f1']) if v4_bp else None
    return {
        'model_class': 'FastTNet',
        'arch': 'transformer_v4',
        'nin': best_ff['nin'] if best_ff else 5,
        'd': 32,
        'stride': 4,
        'nhead': 4,
        'nlayers': 2,
        'eval_logit_offset': 0.0,
        'eval_logit_bias': 0.0,
        'calibrated_threshold': best_ff['thr'] if best_ff else 0.5,
        'checkpoint_path': 'results/studio/r1/boundary_transformer_v4.pt',
        'selected_fit': best_ff,
        'bpdr_nin': best_bp['nin'] if best_bp else 6,
        'bpdr_calibrated_threshold': best_bp['thr'] if best_bp else 0.5,
        'bpdr_checkpoint_path': 'results/studio/r1/boundary_transformer_v4_bpdr.pt',
        'bpdr_selected_fit': best_bp,
        'all_fit_thresholds': fit_meta,
    }


def main():
    t_start = time.time()
    recs = B.load()
    tr = [r for r in recs if r['role'] == 'train']
    dev = [r for r in recs if r['role'] == 'development']
    voc = [r for r in recs if r['role'] == 'vocadito']
    meta = E.voc_meta()
    print(
        f"Loaded train={len(tr)}, dev_M02={len(dev)}, vocadito={len(voc)} "
        f"(with notes={sum(r['id'] in meta for r in voc)})",
        flush=True,
    )

    sha = hashlib.sha256(DECL_PATH.read_bytes()).hexdigest()
    res = []
    fit_meta = []
    if PARTIAL_PATH.exists():
        prev = json.load(open(PARTIAL_PATH))
        if prev.get('declaration_sha256') == sha:
            res = prev.get('rows', [])
            fit_meta = prev.get('fit_meta', [])
    done = {(r['arch'], r['feat'], r['arm'], r['seed']) for r in res}
    print(f"Resuming from partial; completed fits: {len(done)}/{len(JOBS)}", flush=True)

    best_ff_trf1 = max(
        (m['train_f1'] for m in fit_meta if m['arch'] == 'transformer_v4' and m['feat'] == 'full' and m['arm'] == 'frame_only'),
        default=-1.0,
    )
    best_bp_trf1 = max(
        (m['train_f1'] for m in fit_meta if m['arch'] == 'transformer_v4' and m['feat'] == 'full' and m['arm'] == 'bpdr'),
        default=-1.0,
    )

    ext_tr = float(np.mean([B.extractor_f1(r, r['edges']) for r in tr[:80]]))
    for arch, feat, arm, seed, eval_off, thrs in JOBS:
        if (arch, feat, arm, seed) in done:
            continue
        t0 = time.time()
        net, use_b, nin = fit_job(arm, tr, seed, eval_off)
        thr, trf1 = pick_threshold_grid(net, use_b, tr[:80], thrs)
        healthy = bool(trf1 >= 0.9 * ext_tr)
        params = int(sum(p.numel() for p in net.parameters()))

        if arch == 'transformer_v4' and arm == 'frame_only' and trf1 > best_ff_trf1:
            best_ff_trf1 = trf1
            torch.save(net.state_dict(), CKPT_PATH)
            print(f"Saved best standalone v4 checkpoint -> {CKPT_PATH} (seed={seed}, trF1={trf1:.4f}, thr={thr})", flush=True)
        if arch == 'transformer_v4' and arm == 'bpdr' and trf1 > best_bp_trf1:
            best_bp_trf1 = trf1
            torch.save(net.state_dict(), CKPT_BPDR_PATH)
            print(f"Saved best bpdr v4 checkpoint -> {CKPT_BPDR_PATH} (seed={seed}, trF1={trf1:.4f}, thr={thr})", flush=True)

        for name, test in (('vocadito', voc), ('dev_M02', dev)):
            for row in E.evaluate(net, use_b, thr, test, meta):
                res.append(
                    dict(
                        arch=arch,
                        feat=feat,
                        arm=arm,
                        seed=seed,
                        split=name,
                        thr=thr,
                        healthy=healthy,
                        **row,
                    )
                )

        fit_info = dict(
            arch=arch,
            feat=feat,
            arm=arm,
            seed=seed,
            nin=nin,
            params=params,
            steps=STEPS,
            eval_logit_offset=eval_off,
            thr=thr,
            train_f1=round(trf1, 6),
            extractor_train_f1=round(ext_tr, 6),
            healthy=healthy,
            elapsed_s=round(time.time() - t0, 1),
        )
        fit_meta.append(fit_info)
        print(
            f"{arch:18} {feat:4} {arm:11} seed={seed} off={eval_off:.1f} thr={thr:.2f} "
            f"trF1={trf1:.3f} ext={ext_tr:.3f} healthy={healthy} ({fit_info['elapsed_s']}s)",
            flush=True,
        )
        json.dump(
            dict(declaration_sha256=sha, partial=True, fit_meta=fit_meta, rows=res),
            open(PARTIAL_PATH, 'w'),
            default=float,
        )
        del net
        gc.collect()

    meta_json = build_meta_json(fit_meta)
    json.dump(meta_json, open(META_PATH, 'w'), indent=2, default=float)
    json.dump(
        dict(
            declaration_sha256=sha,
            model_class='FastTNet',
            eval_logit_offset=0.0,
            fit_meta=fit_meta,
            rows=res,
        ),
        open(RESULTS_PATH, 'w'),
        indent=1,
        default=float,
    )
    print(f"Done in {time.time() - t_start:.1f}s. Wrote {RESULTS_PATH} ({len(res)} rows) and {META_PATH}.", flush=True)


if __name__ == '__main__':
    main()
