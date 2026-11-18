"""FastTNet v3 boundary model experiment. Declared in results/studio/r1/declaration_transformer_v3.json.
Keeps v1's compact stride-4 FastTNet architecture while removing the +0.9 eval logit offset
and adding calibrated pos_weight (5.0), AdamW warmup + cosine LR decay, weight decay, and grad clipping.
Resumable: partial results are written after every fit.
Run: ..\\.venv\\Scripts\\python.exe tools\\fit_r1_transformer_v3.py
"""
import gc
import hashlib
import json
import math
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
DECL_PATH = OUT / 'declaration_transformer_v3.json'
PARTIAL_PATH = OUT / 'transformer_v3_partial.json'
RESULTS_PATH = OUT / 'transformer_v3_results.json'
CKPT_PATH = OUT / 'boundary_transformer_v3.pt'
CKPT_BPDR_PATH = OUT / 'boundary_transformer_v3_bpdr.pt'
META_PATH = OUT / 'boundary_transformer_v3_meta.json'

FEATS = ('full', 'lean')
ARMS = ('frame_only', 'augmentation_only', 'bpdr')
SEEDS = (339, 340)

STEPS = 300
WARMUP_STEPS = 20
LR = 3e-3
MIN_LR = 3e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
POS_WEIGHT = 5.0
THRS = tuple(round(0.20 + 0.05 * i, 2) for i in range(15))  # 0.20 .. 0.90

_ORIG_FEATS = B.feats


class FastTNetV3(E.FastTNet):
    """Exact stride-4 FastTNet architecture with calibrated output (eval_logit_offset = 0.0)."""

    def __init__(self, nin, d=32, nhead=4, nlayers=2):
        super().__init__(nin=nin, d=d, nhead=nhead, nlayers=nlayers)
        self.nin = nin
        self.d = d
        self.nhead = nhead
        self.nlayers = nlayers

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
        return self.out(z)[:, 0]


def set_feats(feat):
    B.feats = E.lean_feats if feat == 'lean' else _ORIG_FEATS


def pick_threshold_v3(net, use_b, clips):
    """Pick threshold in THRS maximizing mean onset F1 at 50 ms on tr[:80]."""
    net.eval()
    probs_list = [B.probs(net, r, use_b, r['edges']) for r in clips]
    best_f1, best_thr = -1.0, 0.5
    for thr in THRS:
        scores = [B.f1(r['onsets'], B.picks(p, r, thr)) for r, p in zip(clips, probs_list)]
        v = float(np.mean(scores))
        if v > best_f1:
            best_f1, best_thr = v, float(thr)
    return best_thr, best_f1


def lr_multiplier(step):
    """20-step linear warmup from 0.1*lr to lr, then cosine decay from 3e-3 to 3e-4."""
    if step < WARMUP_STEPS:
        return 0.1 + 0.9 * (step + 1) / float(WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / float(max(1, STEPS - WARMUP_STEPS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = MIN_LR / LR
    return min_ratio + (1.0 - min_ratio) * cosine


def fit_v3(feat, arm, train, seed):
    set_feats(feat)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    use_b = arm != 'frame_only'
    nin = (3 if feat == 'lean' else 5) + int(use_b)
    net = FastTNetV3(nin=nin)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_multiplier)
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
        torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()

    net.eval()
    return net, use_b, nin


def build_meta_json(fit_meta):
    full_frame = [m for m in fit_meta if m['feat'] == 'full' and m['arm'] == 'frame_only']
    full_bpdr = [m for m in fit_meta if m['feat'] == 'full' and m['arm'] == 'bpdr']
    best_ff = max(full_frame, key=lambda m: m['train_f1']) if full_frame else None
    best_bp = max(full_bpdr, key=lambda m: m['train_f1']) if full_bpdr else None
    return {
        'model_class': 'FastTNet',
        'arch': 'transformer_v3',
        'nin': best_ff['nin'] if best_ff else 5,
        'd': 32,
        'stride': 4,
        'nhead': 4,
        'nlayers': 2,
        'eval_logit_offset': 0.0,
        'eval_logit_bias': 0.0,
        'calibrated_threshold': best_ff['thr'] if best_ff else 0.5,
        'checkpoint_path': 'results/studio/r1/boundary_transformer_v3.pt',
        'selected_fit': best_ff,
        'bpdr_nin': best_bp['nin'] if best_bp else 6,
        'bpdr_calibrated_threshold': best_bp['thr'] if best_bp else 0.5,
        'bpdr_checkpoint_path': 'results/studio/r1/boundary_transformer_v3_bpdr.pt',
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
    done = {(r['feat'], r['arm'], r['seed']) for r in res}
    print(f"Resuming from partial; completed fits: {len(done)}/12", flush=True)

    best_ff_trf1 = max(
        (m['train_f1'] for m in fit_meta if m['feat'] == 'full' and m['arm'] == 'frame_only'),
        default=-1.0,
    )
    best_bp_trf1 = max(
        (m['train_f1'] for m in fit_meta if m['feat'] == 'full' and m['arm'] == 'bpdr'),
        default=-1.0,
    )

    ext_tr = float(np.mean([B.extractor_f1(r, r['edges']) for r in tr[:80]]))
    for feat in FEATS:
        for seed in SEEDS:
            for arm in ARMS:
                if (feat, arm, seed) in done:
                    continue
                t0 = time.time()
                net, use_b, nin = fit_v3(feat, arm, tr, seed)
                thr, trf1 = pick_threshold_v3(net, use_b, tr[:80])
                healthy = bool(trf1 >= 0.9 * ext_tr)
                params = int(sum(p.numel() for p in net.parameters()))

                if feat == 'full' and arm == 'frame_only' and trf1 > best_ff_trf1:
                    best_ff_trf1 = trf1
                    torch.save(net.state_dict(), CKPT_PATH)
                    print(f"Saved best standalone v3 checkpoint -> {CKPT_PATH} (seed={seed}, trF1={trf1:.4f}, thr={thr})", flush=True)
                if feat == 'full' and arm == 'bpdr' and trf1 > best_bp_trf1:
                    best_bp_trf1 = trf1
                    torch.save(net.state_dict(), CKPT_BPDR_PATH)
                    print(f"Saved best bpdr v3 checkpoint -> {CKPT_BPDR_PATH} (seed={seed}, trF1={trf1:.4f}, thr={thr})", flush=True)

                for name, test in (('vocadito', voc), ('dev_M02', dev)):
                    for row in E.evaluate(net, use_b, thr, test, meta):
                        res.append(
                            dict(
                                arch='transformer_v3',
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
                    arch='transformer_v3',
                    feat=feat,
                    arm=arm,
                    seed=seed,
                    nin=nin,
                    params=params,
                    steps=STEPS,
                    thr=thr,
                    train_f1=round(trf1, 6),
                    extractor_train_f1=round(ext_tr, 6),
                    healthy=healthy,
                    elapsed_s=round(time.time() - t0, 1),
                )
                fit_meta.append(fit_info)
                print(
                    f"transformer_v3 {feat:4} {arm:17} seed={seed} thr={thr:.2f} "
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
