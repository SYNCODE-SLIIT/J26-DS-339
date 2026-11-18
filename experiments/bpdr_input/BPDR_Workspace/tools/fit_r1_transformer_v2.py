"""FastTNet v2 boundary model experiment. Declared in results/studio/r1/declaration_transformer_v2.json.
Resumable: partial results are written after every fit.
Run: ..\\.venv\\Scripts\\python.exe tools\\fit_r1_transformer_v2.py
"""
import copy
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
DECL_PATH = OUT / 'declaration_transformer_v2.json'
PARTIAL_PATH = OUT / 'transformer_v2_partial.json'
RESULTS_PATH = OUT / 'transformer_v2_results.json'
CKPT_PATH = OUT / 'boundary_transformer_v2.pt'
CKPT_BPDR_PATH = OUT / 'boundary_transformer_v2_bpdr.pt'
META_PATH = OUT / 'boundary_transformer_v2_meta.json'

FEATS = ('full', 'lean')
ARMS = ('frame_only', 'augmentation_only', 'bpdr')
SEEDS = (339, 340)
KINDS = ('clean', 'split', 'merge', 'jitter')

STEPS = 350
WARMUP_STEPS = 30
LR = 3e-3
MIN_LR = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
POS_WEIGHT = 5.0
EVAL_STEPS = (200, 250, 300, 350)
THRS = tuple(round(0.20 + 0.05 * i, 2) for i in range(15))  # 0.20 .. 0.90

_ORIG_FEATS = B.feats


class FastTransformerBlockV2(nn.Module):
    """Pre-norm multi-head self-attention block with dropout and GELU FFN."""

    def __init__(self, d=40, h=4, drop=0.1):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(2 * d, d),
        )

    def forward(self, x):
        B_, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        f = lambda t: t.view(B_, T, self.h, C // self.h).transpose(1, 2)
        a = tnf.scaled_dot_product_attention(f(q), f(k), f(v))
        a = a.transpose(1, 2).contiguous().view(B_, T, C)
        x = x + self.drop(self.proj(a))
        return x + self.drop(self.ff(self.n2(x)))


class FastTNetV2(nn.Module):
    """Calibrated FastTNet v2:
    - 7-channel exact non-overlapping multi-scale pitch & confidence deltas
    - GroupNorm + 6 residual dilated 20 ms convs (dilations 1,2,4,8,1,2)
    - Stride-2 downsampling + depthwise conv pos encoding + 2 pre-norm Transformer blocks
    - Linear-interpolation upsampling + gated fusion + conv refinement head (no logit offset)
    """

    def __init__(self, nin=5, d=40, stride=2, nhead=4, nlayers=2, drop=0.1):
        super().__init__()
        self.nin = nin
        self.d = d
        self.stride = stride
        self.nhead = nhead
        self.nlayers = nlayers
        self.drop_rate = drop

        self.in_proj = nn.Conv1d(nin + 7, d, 3, padding=1)
        self.in_norm = nn.GroupNorm(4, d)
        self.fine_convs = nn.ModuleList(
            [nn.Conv1d(d, d, 5, padding=2 * k, dilation=k) for k in (1, 2, 4, 8, 1, 2)]
        )
        self.fine_drop = nn.Dropout(drop)
        self.down = nn.Conv1d(d, d, stride + 2, stride=stride, padding=1)
        self.pos_conv = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.enc = nn.Sequential(*[FastTransformerBlockV2(d, nhead, drop) for _ in range(nlayers)])
        self.up_proj = nn.Conv1d(d, d, 3, padding=1)
        self.gate = nn.Parameter(torch.tensor(0.2))
        self.head = nn.Sequential(
            nn.Conv1d(d, d, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d, 1, 1),
        )

    def _deltas(self, x):
        p, ok, conf = x[:, :, 0], x[:, :, 1], x[:, :, 2]
        dp1 = torch.zeros_like(p)
        dp1[:, 1:] = (p[:, 1:] - p[:, :-1]).abs() * ok[:, 1:] * ok[:, :-1]
        dok = torch.zeros_like(ok)
        dok[:, 1:] = (ok[:, 1:] - ok[:, :-1]).abs()
        dc1 = torch.zeros_like(conf)
        dc1[:, 1:] = (conf[:, 1:] - conf[:, :-1]).abs()
        steps = [dp1.clamp(0.0, 2.0), dok, dc1.clamp(0.0, 1.0)]

        po = (p * ok)[:, None]
        ok1 = ok[:, None]
        c1 = conf[:, None]
        T = p.shape[1]
        for w in (3, 6, 12):
            cs_p = tnf.pad(tnf.pad(po, (w, w)).cumsum(-1), (1, 0))
            cs_ok = tnf.pad(tnf.pad(ok1, (w, w)).cumsum(-1), (1, 0))
            cnt_l = cs_ok[:, 0, w : w + T] - cs_ok[:, 0, :T]
            cnt_r = cs_ok[:, 0, 2 * w : 2 * w + T] - cs_ok[:, 0, w : w + T]
            lp = (cs_p[:, 0, w : w + T] - cs_p[:, 0, :T]) / cnt_l.clamp_min(1.0)
            rp = (cs_p[:, 0, 2 * w : 2 * w + T] - cs_p[:, 0, w : w + T]) / cnt_r.clamp_min(1.0)
            valid = (cnt_l >= 1.0) & (cnt_r >= 1.0)
            steps.append(((rp - lp).abs() * valid.float()).clamp(0.0, 2.0))
            if w == 3:
                cs_c = tnf.pad(tnf.pad(c1, (w, w)).cumsum(-1), (1, 0))
                cl = (cs_c[:, 0, w : w + T] - cs_c[:, 0, :T]) / float(w)
                cr = (cs_c[:, 0, 2 * w : 2 * w + T] - cs_c[:, 0, w : w + T]) / float(w)
                steps.append((0.5 * (cl + cr) - conf).clamp(0.0, 1.0))
        return torch.stack(steps, dim=-1)

    def forward(self, x):
        dx = self._deltas(x)
        z = tnf.gelu(self.in_norm(self.in_proj(torch.cat([x, dx], dim=-1).transpose(1, 2))))
        for c in self.fine_convs:
            z = z + self.fine_drop(tnf.gelu(c(z)))
        zc = tnf.gelu(self.down(z))
        zc = zc + self.pos_conv(zc)
        zc = self.enc(zc.transpose(1, 2)).transpose(1, 2)
        zu = self.up_proj(tnf.interpolate(zc, size=x.shape[1], mode='linear', align_corners=False))
        z = z + self.gate * tnf.gelu(zu)
        return self.head(z)[:, 0]


def stratified_train_split(tr):
    """Split 525 HumTrans train clips into 473 train + 52 internal val (every 10th per person)."""
    by_group = {}
    for r in tr:
        by_group.setdefault(r['group'], []).append(r)
    tr_sub, val_int, tr_health = [], [], []
    for g in sorted(by_group):
        clips = sorted(by_group[g], key=lambda r: r['id'])
        g_tr = []
        for idx, r in enumerate(clips):
            if idx % 10 == 0:
                val_int.append(r)
            else:
                tr_sub.append(r)
                g_tr.append(r)
        tr_health.extend(g_tr[:20])
    return tr_sub, val_int, tr_health


def set_feats(feat):
    B.feats = E.lean_feats if feat == 'lean' else _ORIG_FEATS


def pick_threshold_v2(net, use_b, clips):
    """Pick threshold in THRS maximizing mean onset F1 at 50 ms on given clips."""
    net.eval()
    # Precompute probabilities once per clip to keep grid search fast
    probs_list = [B.probs(net, r, use_b, r['edges']) for r in clips]
    best_f1, best_thr = -1.0, 0.5
    for thr in THRS:
        scores = [B.f1(r['onsets'], B.picks(p, r, thr)) for r, p in zip(clips, probs_list)]
        v = float(np.mean(scores))
        if v > best_f1:
            best_f1, best_thr = v, float(thr)
    return best_thr, best_f1


def lr_multiplier(step):
    """Linear warmup over WARMUP_STEPS, then cosine decay to MIN_LR / LR."""
    if step < WARMUP_STEPS:
        return 0.1 + 0.9 * (step + 1) / float(WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / float(max(1, STEPS - WARMUP_STEPS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = MIN_LR / LR
    return min_ratio + (1.0 - min_ratio) * cosine


def fit_v2(feat, arm, tr_sub, val_int, seed):
    set_feats(feat)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    use_b = arm != 'frame_only'
    nin = (3 if feat == 'lean' else 5) + int(use_b)
    net = FastTNetV2(nin=nin)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_multiplier)
    bce = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(POS_WEIGHT))

    best_val_f1 = -1.0
    best_thr = 0.5
    best_step = STEPS
    best_state = None

    net.train()
    for step in range(1, STEPS + 1):
        sub = [tr_sub[i] for i in rng.choice(len(tr_sub), B.BATCH, replace=False)]
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

        if step in EVAL_STEPS:
            thr_s, val_f1_s = pick_threshold_v2(net, use_b, val_int)
            net.train()
            if val_f1_s > best_val_f1:
                best_val_f1 = val_f1_s
                best_thr = thr_s
                best_step = step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, use_b, nin, best_thr, best_val_f1, best_step


def build_meta_json(fit_meta):
    """Build boundary_transformer_v2_meta.json from completed fit metadata."""
    full_frame = [m for m in fit_meta if m['feat'] == 'full' and m['arm'] == 'frame_only']
    full_bpdr = [m for m in fit_meta if m['feat'] == 'full' and m['arm'] == 'bpdr']
    best_ff = max(full_frame, key=lambda m: m['val_f1']) if full_frame else None
    best_bp = max(full_bpdr, key=lambda m: m['val_f1']) if full_bpdr else None
    return {
        'model_class': 'FastTNetV2',
        'nin': best_ff['nin'] if best_ff else 5,
        'd': 40,
        'stride': 2,
        'nhead': 4,
        'nlayers': 2,
        'drop': 0.1,
        'eval_logit_offset': 0.0,
        'calibrated_threshold': best_ff['thr'] if best_ff else 0.5,
        'checkpoint_path': 'results/studio/r1/boundary_transformer_v2.pt',
        'selected_fit': best_ff,
        'bpdr_nin': best_bp['nin'] if best_bp else 6,
        'bpdr_calibrated_threshold': best_bp['thr'] if best_bp else 0.5,
        'bpdr_checkpoint_path': 'results/studio/r1/boundary_transformer_v2_bpdr.pt',
        'bpdr_selected_fit': best_bp,
        'all_fit_thresholds': fit_meta,
    }


def main():
    t_start = time.time()
    recs = B.load()
    tr = [r for r in recs if r['role'] == 'train']
    dev = [r for r in recs if r['role'] == 'development']
    voc = [r for r in recs if r['role'] == 'vocadito']
    tr_sub, val_int, tr_health = stratified_train_split(tr)
    meta = E.voc_meta()
    print(
        f"Loaded train={len(tr)} (fit={len(tr_sub)}, int_val={len(val_int)}, health={len(tr_health)}), "
        f"dev_M02={len(dev)}, vocadito={len(voc)} (with notes={sum(r['id'] in meta for r in voc)})",
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

    best_ff_val = max(
        (m['val_f1'] for m in fit_meta if m['feat'] == 'full' and m['arm'] == 'frame_only'),
        default=-1.0,
    )
    best_bp_val = max(
        (m['val_f1'] for m in fit_meta if m['feat'] == 'full' and m['arm'] == 'bpdr'),
        default=-1.0,
    )

    ext_tr = float(np.mean([B.extractor_f1(r, r['edges']) for r in tr_health]))
    for feat in FEATS:
        for seed in SEEDS:
            for arm in ARMS:
                if (feat, arm, seed) in done:
                    continue
                t0 = time.time()
                net, use_b, nin, thr, val_f1, best_step = fit_v2(feat, arm, tr_sub, val_int, seed)
                set_feats(feat)
                trf1 = float(np.mean([B.f1(r['onsets'], B.picks(B.probs(net, r, use_b, r['edges']), r, thr)) for r in tr_health]))
                healthy = bool(trf1 >= 0.9 * ext_tr)
                params = int(sum(p.numel() for p in net.parameters()))

                if feat == 'full' and arm == 'frame_only' and val_f1 > best_ff_val:
                    best_ff_val = val_f1
                    torch.save(net.state_dict(), CKPT_PATH)
                    print(f"Saved best standalone checkpoint -> {CKPT_PATH} (seed={seed}, val_f1={val_f1:.4f}, thr={thr})", flush=True)
                if feat == 'full' and arm == 'bpdr' and val_f1 > best_bp_val:
                    best_bp_val = val_f1
                    torch.save(net.state_dict(), CKPT_BPDR_PATH)
                    print(f"Saved best bpdr checkpoint -> {CKPT_BPDR_PATH} (seed={seed}, val_f1={val_f1:.4f}, thr={thr})", flush=True)

                for name, test in (('vocadito', voc), ('dev_M02', dev)):
                    for row in E.evaluate(net, use_b, thr, test, meta):
                        res.append(
                            dict(
                                arch='transformer_v2',
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
                    arch='transformer_v2',
                    feat=feat,
                    arm=arm,
                    seed=seed,
                    nin=nin,
                    params=params,
                    best_step=best_step,
                    thr=thr,
                    val_f1=round(val_f1, 6),
                    train_f1=round(trf1, 6),
                    extractor_train_f1=round(ext_tr, 6),
                    healthy=healthy,
                    elapsed_s=round(time.time() - t0, 1),
                )
                fit_meta.append(fit_info)
                print(
                    f"transformer_v2 {feat:4} {arm:17} seed={seed} step={best_step} thr={thr:.2f} "
                    f"valF1={val_f1:.3f} trF1={trf1:.3f} ext={ext_tr:.3f} healthy={healthy} ({fit_info['elapsed_s']}s)",
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
            model_class='FastTNetV2',
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
