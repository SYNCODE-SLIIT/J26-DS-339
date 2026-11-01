"""Matched point-supervised and adjacent-frame-supervised Transformer fits."""
from collections import defaultdict
from pathlib import Path
import hashlib
import time

import numpy as np
import torch
from torch.nn import functional as F

from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.experiments.learning import joint_transformer_training as joint
from bpdr_input.modeling.joint_transformer import JointTransformer, joint_tokens, predict_raw, save, load
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.held_voice_policy import HeldPolicy


def adjacent_loss(output_cents, target_cents, valid):
    """Supervise differences only when BOTH adjacent target labels are known.

    Never force a genuine labelled change to zero. Never supervise across a
    missing/ramp/unreliable label or an artificial clipped-window boundary.
    Five cents is a fixed loss scale, not a new renderer or permission limit.
    """
    mask = valid[:, 1:] & valid[:, :-1]
    delta = (output_cents[:, 1:]-output_cents[:, :-1]) - (target_cents[:, 1:]-target_cents[:, :-1])
    return F.smooth_l1_loss(delta[mask]/5., torch.zeros_like(delta[mask])) if mask.any() else output_cents.sum()*0.


def fit(pairs, anchors, arm, seed, updates, folder, declaration_hash):
    if arm not in ('matched_baseline', 'adjacent_supervision'):
        raise ValueError('Unknown temporal experiment arm')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = JointTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    valid_pairs = [p for p in pairs if 'excluded' not in p]
    grouped, originals = defaultdict(list), defaultdict(list)
    for p in valid_pairs:
        grouped[p['group']].append(p)
    for p in anchors:
        originals[p['group']].append(p)
    groups, original_groups = sorted(grouped), sorted(originals)
    if not groups or not original_groups:
        raise ValueError('No supervised pairs or reliable original anchors')
    cache, labels = {}, {}
    def cached(key, raw, target, mask):
        if key not in cache:
            cache[key] = joint_tokens(raw, np.arange(len(raw)), model.config)
            labels[key] = (np.asarray(target, np.float32), np.asarray(mask, bool))
    for p in valid_pairs:
        for view in ('A', 'C'):
            raw = p[view]['views'][0]
            target = np.zeros(len(raw), np.float32) if view == 'A' else p['target']
            mask = p[view]['reliable'] if view == 'A' else p['mask'] & p[view]['reliable']
            cached(p['path_'+view], raw, target, mask)
    for p in anchors:
        raw = p['A']['views'][0]
        cached(p['path_A'], raw, np.zeros(len(raw)), p['A']['reliable'])
    augmentation = joint.augmented_pairs(anchors, seed)
    for i, p in enumerate(augmentation):
        cached(('aug', i, 'A'), p['clean'], np.zeros(len(p['target'])), p['mask'])
        cached(('aug', i, 'C'), p['shifted'], p['target'], p['mask'])
    folder.mkdir(parents=True, exist_ok=True)
    started, history, sample_hash = time.perf_counter(), [], hashlib.sha256()
    model.train()
    for step in range(updates):
        xx, pads, yy, vv, weights = [], [], [], [], []
        def append(key, index, weight):
            x, pad = cache[key]
            target, mask = labels[key]
            positions = index+np.array([-1, 0, 1])
            present = (positions >= 0) & (positions < len(x))
            indices = np.clip(positions, 0, len(x)-1)
            xx.append(x[indices]); pads.append(pad[indices])
            yy.append(target[indices]); vv.append(mask[indices] & present)
            weights.append(weight)
            sample_hash.update((str(key)+':'+str(index)+'\n').encode())
        for _ in range(12):
            if augmentation and rng.random() < .25:
                i = int(rng.integers(len(augmentation)))
                p = augmentation[i]
                mask = p['plateau'] if rng.random() < .7 else p['mask'] & (p['target'] == 0)
                index = int(rng.choice(np.flatnonzero(mask)))
                ka, kc = ('aug', i, 'A'), ('aug', i, 'C')
            else:
                group = groups[int(rng.integers(len(groups)))]
                p = grouped[group][int(rng.integers(len(grouped[group])))]
                plateau, outside = p['mask'] & (p['target'] != 0), p['mask'] & (p['target'] == 0)
                mask = plateau if rng.random() < .7 or not outside.any() else outside
                index = int(rng.choice(np.flatnonzero(mask)))
                ka, kc = p['path_A'], p['path_C']
            append(ka, index, 4.)
            append(kc, index, 2. if labels[kc][0][index] == 0 else 1.)
            og = original_groups[int(rng.integers(len(original_groups)))]
            op = originals[og][int(rng.integers(len(originals[og])))]
            oi = int(rng.choice(np.flatnonzero(op['A']['reliable'])))
            append(op['path_A'], oi, 4.)
        x, pad = np.stack(xx), np.stack(pads)
        targets, known = torch.from_numpy(np.stack(yy)), torch.from_numpy(np.stack(vv))
        outputs, logits, amplitudes = model(torch.from_numpy(x.reshape(-1, *x.shape[2:])),
                                         torch.from_numpy(pad.reshape(-1, pad.shape[-1])), True)
        outputs, logits, amplitudes = [a.reshape(36, 3) for a in (outputs, logits, amplitudes)]
        pred, target = outputs[:, 1]/30., targets[:, 1]/30.
        weight, positive = torch.tensor(weights), target != 0
        anchor = ((pred-target).square()*weight).mean()
        paired, wanted = pred.reshape(12, 3), target.reshape(12, 3)[:, 1]
        response = (paired[:, 1]-paired[:, 0]-wanted).square().mean()
        activity = F.binary_cross_entropy_with_logits(logits[:, 1], positive.float())
        amplitude = ((amplitudes[positive, 1]/30.-target[positive]).square().mean() if positive.any() else amplitudes.sum()*0.)
        temporal = adjacent_loss(outputs, targets, known)
        loss = anchor+.25*response+.5*activity+.25*amplitude + (.1*temporal if arm == 'adjacent_supervision' else 0.)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite temporal Transformer loss')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == updates-1:
            history.append(dict(update=step+1, loss=float(loss.detach()), temporal=float(temporal.detach())))
            ex.write(folder/'training_progress.json', dict(update=step+1, planned_updates=updates, history=history))
    metadata = dict(arm=arm, seed=seed, updates=updates, training_groups=sorted(set(groups) | set(originals)),
                    training_sources=sorted({p['source']['sha256'] for p in valid_pairs+anchors}),
                    declaration_sha256=declaration_hash, sampling_sha256=sample_hash.hexdigest(),
                    feature_only_training_examples=len(augmentation), original_anchor_recordings=len(anchors),
                    temporal_weight=.1 if arm == 'adjacent_supervision' else 0.,
                    eligible_for_demo=False, research_validation_passed=False)
    save(folder/'checkpoint.pt', model.eval(), metadata)
    restored, payload = load(folder/'checkpoint.pt')
    probe = valid_pairs[0]['A']['views'][0]
    np.testing.assert_allclose(predict_raw(model, probe)[0], predict_raw(restored, probe)[0], atol=1e-6)
    receipt = dict(**metadata, checkpoint_sha256=ex.sha(folder/'checkpoint.pt'),
                   reload_verified=True, fit_seconds=time.perf_counter()-started, history=history)
    ex.write(folder/'fit_receipt.json', receipt)
    return restored, payload, receipt


def cv_worker(task):
    root, fold, arm = task
    root = Path(root)
    d = ex.read(root/'declaration.json')
    excluded = set(d['folds'][fold])
    rows = ex.read(root/'train_pairs.json')
    train = [p for p in rows if ex.group(p) not in excluded]
    test = [p for p in rows if ex.group(p) in excluded]
    folder = root/f'cross_validation/fold_{fold+1}/{arm}'
    if (folder/'validation/accuracy.json').exists():
        return ex.read(folder/'validation/accuracy.json')
    if (folder/'fit_receipt.json').exists():
        assert ex.sha(folder/'checkpoint.pt') == ex.read(folder/'fit_receipt.json')['checkpoint_sha256']
        model, payload = load(folder/'checkpoint.pt')
    else:
        model, payload, _ = fit(held.prepared(train), held.anchor_pool(ex.read(root/'anchor_sources.json'), excluded),
                                arm, d['cv_seed'], d['cv_updates'], folder, ex.sha(root/'declaration.json'))
    assert_independent(payload['training_groups'], payload['training_sources'], test)
    return joint.evaluate(model, payload, held.prepared(test), folder/'validation', HeldPolicy(**d['policy']))
