"""Singer-grouped joint-context experiment; augmentations are features only."""
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.joint_transformer import JointTransformer, joint_tokens, predict_raw, save, load
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.held_voice_policy import HeldPolicy, spans


def augmented_pairs(anchors, seed):
    """Exactly labelled pitch-feature pulses, never passed off as audio pairs."""
    rng = np.random.default_rng(seed+9100)
    pairs = []
    for anchor in anchors:
        original = anchor['A']['views'][0]
        regions = [(a, b) for a, b in spans(original[:, 66] > .5) if b-a >= 100]
        if not regions:
            continue
        a, b = regions[int(rng.integers(len(regions)))]
        width = int(rng.integers(48, min(76, b-a-24)+1))
        left = int(rng.integers(a+12, b-width-11))
        pulse = np.zeros(len(original), np.float32)
        pulse[left:left+width] = 1.
        ramp = .5-.5*np.cos(np.linspace(0, np.pi, 5))
        pulse[left:left+5] = ramp
        pulse[left+width-5:left+width] = ramp[::-1]
        t = np.arange(len(original))*.02
        variation = (rng.uniform(-6., 6.)*(t-t.mean()) +
                     rng.uniform(0., 8.)*np.sin(2*np.pi*rng.uniform(4., 7.)*t+rng.uniform(0., 6.28)))
        clean = original.copy()
        clean[:, 64] += variation.astype(np.float32)
        shifted = clean.copy()
        cents = float(rng.choice([-30., -20., -12., 12., 20., 30.]))
        shifted[:, 64] += cents*pulse
        for raw in (clean, shifted):
            raw[:, 67] = 0.
            raw[1:, 67] = np.where((raw[1:, 66] > .5) & (raw[:-1, 66] > .5), np.diff(raw[:, 64]), 0.)
        mask = clean[:, 66] > .5
        # Ramps have known feature labels but plateau sampling is kept away
        # from them, matching the real-pair supervision geometry.
        plateau = np.zeros(len(clean), bool)
        plateau[left+5:left+width-5] = True
        target = -cents*pulse
        pairs.append(dict(group=anchor['group'], source=anchor['source'],
                          clean=clean, shifted=shifted, mask=mask, plateau=plateau,
                          target=target, role='training_features_only'))
    return pairs


def fit(pairs, anchors, arm, seed, updates, folder, declaration_hash):
    if arm not in ('joint', 'joint_augmented'):
        raise ValueError('Unknown joint experiment arm')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = JointTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    valid = [p for p in pairs if 'excluded' not in p]
    grouped, originals = defaultdict(list), defaultdict(list)
    for p in valid:
        grouped[p['group']].append(p)
    for p in anchors:
        originals[p['group']].append(p)
    groups, original_groups = sorted(grouped), sorted(originals)
    if not groups or not original_groups:
        raise ValueError('No supervised pairs or original anchors')
    cache = {}
    def cached(key, raw):
        if key not in cache:
            cache[key] = joint_tokens(raw, np.arange(len(raw)), model.config)
        return cache[key]
    for p in valid:
        for view in ('A', 'C'):
            cached(p['path_'+view], p[view]['views'][0])
    for p in anchors:
        cached(p['path_A'], p['A']['views'][0])
    augmentation = augmented_pairs(anchors, seed) if arm == 'joint_augmented' else []
    for i, p in enumerate(augmentation):
        cached(('aug', i, 'A'), p['clean'])
        cached(('aug', i, 'C'), p['shifted'])
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = []
    model.train()
    for step in range(updates):
        xx, pads, targets, weights = [], [], [], []
        for _ in range(12):
            if augmentation and rng.random() < .25:
                i = int(rng.integers(len(augmentation)))
                p = augmentation[i]
                choice = p['plateau'] if rng.random() < .7 else p['mask'] & (p['target'] == 0)
                index = int(rng.choice(np.flatnonzero(choice)))
                entries = [(('aug', i, 'A'), 0.), (('aug', i, 'C'), float(p['target'][index])/30.)]
            else:
                group = groups[int(rng.integers(len(groups)))]
                p = grouped[group][int(rng.integers(len(grouped[group])))]
                plateau, outside = p['mask'] & (p['target'] != 0), p['mask'] & (p['target'] == 0)
                index = int(rng.choice(np.flatnonzero(plateau if rng.random() < .7 or not outside.any() else outside)))
                entries = [(p['path_A'], 0.), (p['path_C'], float(p['target'][index])/30.)]
            for key, target in entries:
                x, pad = cache[key]
                xx.append(x[index]); pads.append(pad[index]); targets.append(target)
                weights.append(4. if target == 0 and key == entries[0][0] else 2. if target == 0 else 1.)
            og = original_groups[int(rng.integers(len(original_groups)))]
            op = originals[og][int(rng.integers(len(originals[og])))]
            oi = int(rng.choice(np.flatnonzero(op['A']['reliable'])))
            x, pad = cache[op['path_A']]
            xx.append(x[oi]); pads.append(pad[oi]); targets.append(0.); weights.append(4.)
        target, weight = torch.tensor(targets), torch.tensor(weights)
        out, logit, amp = model(torch.from_numpy(np.stack(xx)), torch.from_numpy(np.stack(pads)), True)
        pred, positive = out/30., target != 0
        anchor = ((pred-target).square()*weight).mean()
        paired, wanted = pred.reshape(12, 3), target.reshape(12, 3)[:, 1]
        response = (paired[:, 1]-paired[:, 0]-wanted).square().mean()
        activity = F.binary_cross_entropy_with_logits(logit, positive.float())
        amplitude = ((amp[positive]/30.-target[positive]).square().mean() if positive.any() else amp.sum()*0.)
        loss = anchor + .25*response + .5*activity + .25*amplitude
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite joint loss')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == updates-1:
            history.append(dict(update=step+1, loss=float(loss.detach())))
            ex.write(folder/'training_progress.json', dict(update=step+1, planned_updates=updates, history=history))
    metadata = dict(arm=arm, seed=seed, updates=updates,
                    training_groups=sorted(set(groups) | set(originals)),
                    training_sources=sorted({p['source']['sha256'] for p in valid+anchors}),
                    declaration_sha256=declaration_hash, eligible_for_demo=False, research_validation_passed=False,
                    feature_only_training_examples=len(augmentation), original_anchor_recordings=len(anchors))
    save(folder/'checkpoint.pt', model.eval(), metadata)
    restored, payload = load(folder/'checkpoint.pt')
    np.testing.assert_allclose(predict_raw(model, valid[0]['A']['views'][0])[0],
                               predict_raw(restored, valid[0]['A']['views'][0])[0], atol=1e-6)
    receipt = dict(**metadata, checkpoint_sha256=ex.sha(folder/'checkpoint.pt'),
                   fit_seconds=time.perf_counter()-started, reload_verified=True, history=history)
    ex.write(folder/'fit_receipt.json', receipt)
    return restored, payload, receipt


def predictions(model, pair, view):
    # Both input channels exclude boundary metadata; invariance by construction.
    return np.stack([predict_raw(model, pair[view]['views'][0])[0]]*3)


def evaluate(model, payload, pairs, folder, policy=None, render=False):
    previous = held.predictions
    held.predictions = predictions
    try:
        result = held.evaluate(model, payload, pairs, folder,
                               policy or HeldPolicy(version='held-voice-input-policy-2', region_split_cents=70., minimum_step_fit_gain=.75), render)
    finally:
        held.predictions = previous
    result.update(model_version='joint-context-transformer-1',
                  boundary_invariance='Input construction, not learned BPDR-objective superiority')
    ex.write(folder/'accuracy.json', result)
    return result


def cv_worker(task):
    root, fold, arm = task
    root = Path(root)
    d = ex.read(root/'declaration.json')
    held_groups = set(d['folds'][fold])
    rows = ex.read(root/'train_pairs.json')
    train = [p for p in rows if ex.group(p) not in held_groups]
    test = [p for p in rows if ex.group(p) in held_groups]
    folder = root/f'cross_validation/fold_{fold+1}/{arm}'
    if (folder/'validation/accuracy.json').exists():
        return ex.read(folder/'validation/accuracy.json')
    if (folder/'fit_receipt.json').exists():
        assert ex.sha(folder/'checkpoint.pt') == ex.read(folder/'fit_receipt.json')['checkpoint_sha256']
        model, payload = load(folder/'checkpoint.pt')
    else:
        model, payload, _ = fit(held.prepared(train), held.anchor_pool(ex.read(root/'anchor_sources.json'), held_groups),
                                arm, d['cv_seed'], d['cv_updates'], folder, ex.sha(root/'declaration.json'))
    assert_independent(payload['training_groups'], payload['training_sources'], test)
    return evaluate(model, payload, held.prepared(test), folder/'validation')
