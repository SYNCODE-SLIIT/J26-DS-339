"""Warm-started multiscale context with protected original behavior."""
from collections import defaultdict
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from torch.nn import functional as F
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning.joint_transformer_training import augmented_pairs
from bpdr_input.experiments.learning.response_transformer_training import weighted_anchor
from bpdr_input.modeling.multiscale_transformer import MultiscaleTransformer, multiscale_tokens, predict_raw, save, load
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.modeling.joint_transformer import load as load_parent, predict_raw as parent_predict
from bpdr_input.modeling.multiscale_transfer import expand_parent
from bpdr_input.repair.held_voice_policy import HeldPolicy


def difficult_original_pool(x, reliable):
    """Top20% local-residual frames from one known-original training recording."""
    reliable = np.asarray(reliable)
    if (x.ndim != 3 or x.shape[2] != 13 or reliable.shape != (len(x),)
            or reliable.dtype != bool or not np.isfinite(x).all()):
        raise ValueError('Invalid multiscale original features or reliable mask')
    usable = np.flatnonzero(reliable)
    if not len(usable):
        raise ValueError('No reliable original anchors')
    score = np.abs(x[usable, x.shape[1]//2, 5])
    return usable[score >= np.quantile(score, .8)]


def predictions(model, pair, view):
    return np.stack([predict_raw(model, pair[view]['views'][0])[0]]*3)


def evaluate(model, payload, pairs, folder, policy, render=False):
    previous = held.predictions
    held.predictions = predictions
    try:
        result = held.evaluate(model, payload, pairs, folder, policy, render)
    finally:
        held.predictions = previous
    result.update(model_version='multiscale-context-transformer-1',
                  feature_contract='broad-local-step-1',
                  boundary_invariance='Input construction; no learned BPDR objective superiority claim')
    ex.write(folder/'accuracy.json', result)
    return result


def fit(pairs, anchors, arm, seed, updates, folder, declaration_hash, parent_checkpoint):
    if arm not in ('warm_multiscale',):
        raise ValueError('Unknown joint experiment arm')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    parent, parent_payload = load_parent(parent_checkpoint)
    parent.requires_grad_(False)
    model = expand_parent(parent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.001)
    valid = [p for p in pairs if 'excluded' not in p]
    grouped, originals = defaultdict(list), defaultdict(list)
    for p in valid:
        grouped[p['group']].append(p)
    for p in anchors:
        originals[p['group']].append(p)
    groups, original_groups = sorted(grouped), sorted(originals)
    if not groups or not original_groups:
        raise ValueError('No supervised pairs or original anchors')
    expected_groups = set(groups) | set(originals)
    expected_sources = {p['source']['sha256'] for p in valid+anchors}
    if (set(parent_payload['training_groups']) != expected_groups
            or set(parent_payload['training_sources']) != expected_sources or parent_payload['seed'] != seed):
        raise ValueError('Parent source/singer/seed lineage differs from this training partition')
    initial_error = 0.
    for view in ('A', 'C'):
        probe = valid[0][view]['views'][0]
        old = parent_predict(parent, probe)[0]; new = predict_raw(model, probe)[0]
        np.testing.assert_allclose(old, new, atol=1e-4, rtol=1e-6)
        initial_error = max(initial_error, float(abs(old-new).max()))
    cache = {}
    def cached(key, raw):
        if key not in cache:
            cache[key] = multiscale_tokens(raw, np.arange(len(raw)), model.config)
        return cache[key]
    for p in valid:
        for view in ('A', 'C'):
            cached(p['path_'+view], p[view]['views'][0])
    for p in anchors:
        cached(p['path_A'], p['A']['views'][0])
    augmentation = augmented_pairs(anchors, seed)
    for i, p in enumerate(augmentation):
        cached(('aug', i, 'A'), p['clean'])
        cached(('aug', i, 'C'), p['shifted'])
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = []
    sample_hash = hashlib.sha256()
    positive_weight = 1.
    hard_rng = np.random.default_rng(seed+17000)
    reference_hash, paired_hash = hashlib.sha256(), hashlib.sha256()
    hard_draws = 0
    hard_pools = {p['path_A']: difficult_original_pool(cache[p['path_A']][0], p['A']['reliable']) for p in anchors}
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
                sample_hash.update((str(key)+':'+str(index)+'\n').encode())
                reference_hash.update((str(key)+':'+str(index)+'\n').encode())
                paired_hash.update((str(key)+':'+str(index)+'\n').encode())
                weights.append(4. if target == 0 and key == entries[0][0] else 2. if target == 0 else 1.)
            og = original_groups[int(rng.integers(len(original_groups)))]
            op = originals[og][int(rng.integers(len(originals[og])))]
            oi = int(rng.choice(np.flatnonzero(op['A']['reliable'])))
            reference_hash.update((op['path_A']+':'+str(oi)+'\n').encode())
            if hard_rng.random() < .5:
                oi = int(hard_rng.choice(hard_pools[op['path_A']]))
                hard_draws += 1
            x, pad = cache[op['path_A']]
            xx.append(x[oi]); pads.append(pad[oi]); targets.append(0.); weights.append(4.)
            sample_hash.update((op['path_A']+':'+str(oi)+'\n').encode())
        target, weight = torch.tensor(targets), torch.tensor(weights)
        inputs, padding = torch.from_numpy(np.stack(xx)), torch.from_numpy(np.stack(pads))
        with torch.no_grad():
            teacher = parent(inputs[:, :, [0,1,2,3,4,10,11,12]], padding)/30.
        out, logit, amp = model(inputs, padding, True)
        pred, positive = out/30., target != 0
        anchor = weighted_anchor(pred, target, weight, positive_weight)
        paired, wanted = pred.reshape(12, 3), target.reshape(12, 3)[:, 1]
        response = (paired[:, 1]-paired[:, 0]-wanted).square().mean()
        activity = F.binary_cross_entropy_with_logits(logit, positive.float())
        amplitude = ((amp[positive]/30.-target[positive]).square().mean() if positive.any() else amp.sum()*0.)
        preservation = (pred[~positive]-teacher[~positive]).square().mean()
        loss = anchor + .25*response + .5*activity + .25*amplitude + .5*preservation
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
                    feature_only_training_examples=len(augmentation), original_anchor_recordings=len(anchors),
                    positive_weight=positive_weight, parent_checkpoint_sha256=ex.sha(parent_checkpoint),
                    parent_updates=parent_payload['updates'], total_updates=parent_payload['updates']+updates,
                    initial_equivalence_max_error_cents=initial_error, input_channels=13, parameters=sum(p.numel() for p in model.parameters()),
                    sampling_sha256=sample_hash.hexdigest(),
                    uniform_reference_sampling_sha256=reference_hash.hexdigest(),
                    paired_sampling_sha256=paired_hash.hexdigest(), hard_original_draws=hard_draws,
                    total_original_draws=12*updates)
    save(folder/'checkpoint.pt', model.eval(), metadata)
    restored, payload = load(folder/'checkpoint.pt')
    np.testing.assert_allclose(predict_raw(model, valid[0]['A']['views'][0])[0],
                               predict_raw(restored, valid[0]['A']['views'][0])[0], atol=1e-6)
    receipt = dict(**metadata, checkpoint_sha256=ex.sha(folder/'checkpoint.pt'),
                   fit_seconds=time.perf_counter()-started, reload_verified=True, history=history)
    ex.write(folder/'fit_receipt.json', receipt)
    return restored, payload, receipt


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
                                arm, d['cv_seed'], d['cv_updates'], folder, ex.sha(root/'declaration.json'), Path(d['cv_parents'][str(fold+1)]))
    assert_independent(payload['training_groups'], payload['training_sources'], test)
    return evaluate(model, payload, held.prepared(test), folder/'validation', HeldPolicy(**d['policy']))
