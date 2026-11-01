"""Frozen-backbone dense residual learning with original/paired supervision."""
from collections import defaultdict
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from torch.nn import functional as F
from bpdr_input.experiments.learning import context_improvement as ex, held_voice_training as held
from bpdr_input.experiments.learning.joint_transformer_training import augmented_pairs
from bpdr_input.modeling.joint_transformer import load as parent_load, predict_raw as parent_predict
from bpdr_input.modeling.dense_residual_transformer import DenseResidualTransformer, embeddings_raw, predict_raw, save, load
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.held_voice_policy import HeldPolicy


def fit(pairs, anchors, seed, updates, folder, declaration_hash, parent_checkpoint):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    parent, parent_payload = parent_load(parent_checkpoint)
    model = DenseResidualTransformer(parent)
    state = {k: v.clone() for k, v in parent.state_dict().items()}
    valid = [p for p in pairs if 'excluded' not in p]
    groups, originals = defaultdict(list), defaultdict(list)
    for p in valid:
        groups[p['group']].append(p)
    for p in anchors:
        originals[p['group']].append(p)
    if not groups or not originals:
        raise ValueError('No training pairs or originals')
    training_groups = sorted(set(groups) | set(originals))
    training_sources = sorted({p['source']['sha256'] for p in valid + anchors})
    assert training_groups == parent_payload['training_groups']
    assert training_sources == parent_payload['training_sources']
    assert parent_payload['seed'] == seed
    folder.mkdir(parents=True, exist_ok=True)
    cache = {}
    max_initial = 0.

    def cached(key, raw):
        nonlocal max_initial
        if key not in cache:
            cache[key] = embeddings_raw(model, raw)
            old = parent_predict(parent, raw)[0]
            new = predict_raw(model, raw)[0]
            np.testing.assert_array_equal(new, old)
            max_initial = max(max_initial, float(abs(new - old).max()))
        return cache[key]

    for p in valid:
        for view in ('A', 'C'):
            cached(p['path_' + view], p[view]['views'][0])
    for p in anchors:
        cached(p['path_A'], p['A']['views'][0])
    augmentation = augmented_pairs(anchors, seed)
    for i, p in enumerate(augmentation):
        cached(('aug', i, 'A'), p['clean'])
        cached(('aug', i, 'C'), p['shifted'])
    ex.write(folder / 'initial_equivalence.json', dict(exact=True, unique_inputs=len(cache), maximum_error=max_initial))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.001)
    names, onames = sorted(groups), sorted(originals)
    history, sampler = [], hashlib.sha256()
    started = time.perf_counter()
    model.train()
    for update in range(updates):
        xx, yy, ww = [], [], []
        for _ in range(12):
            if augmentation and rng.random() < .25:
                i = int(rng.integers(len(augmentation)))
                p = augmentation[i]
                choice = p['plateau'] if rng.random() < .7 else p['mask'] & (p['target'] == 0)
                index = int(rng.choice(np.flatnonzero(choice)))
                entries = [(('aug', i, 'A'), 0.), (('aug', i, 'C'), float(p['target'][index]) / 30.)]
            else:
                group = names[int(rng.integers(len(names)))]
                p = groups[group][int(rng.integers(len(groups[group])))]
                plateau = p['mask'] & (p['target'] != 0)
                outside = p['mask'] & (p['target'] == 0)
                choice = plateau if rng.random() < .7 or not outside.any() else outside
                index = int(rng.choice(np.flatnonzero(choice)))
                entries = [(p['path_A'], 0.), (p['path_C'], float(p['target'][index]) / 30.)]
            for key, target in entries:
                xx.append(cache[key][index]); yy.append(target)
                ww.append(4. if target == 0 and key == entries[0][0] else 2. if target == 0 else 1.)
            og = onames[int(rng.integers(len(onames)))]
            op = originals[og][int(rng.integers(len(originals[og])))]
            oi = int(rng.choice(np.flatnonzero(op['A']['reliable'])))
            xx.append(cache[op['path_A']][oi]); yy.append(0.); ww.append(4.)
            sampler.update(repr((entries, index, op['path_A'], oi)).encode())
        target, weight = torch.tensor(yy), torch.tensor(ww)
        output, logit, amp = model.forward_embeddings(torch.from_numpy(np.stack(xx)), True)
        pred, positive = output / 30., target != 0
        anchor = ((pred - target).square() * weight).mean()
        triples = pred.reshape(12, 3)
        response = (triples[:, 1] - triples[:, 0] - target.reshape(12, 3)[:, 1]).square().mean()
        activity = F.binary_cross_entropy_with_logits(logit, positive.float())
        amplitude = ((amp[positive] / 30. - target[positive]).square().mean()
                     if positive.any() else amp.sum() * 0.)
        loss = anchor + .25 * response + .5 * activity + .25 * amplitude
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite dense-residual loss')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if update % 100 == 0 or update == updates - 1:
            history.append(dict(update=update + 1, loss=float(loss.detach())))
            ex.write(folder / 'training_progress.json', dict(update=update + 1, planned_updates=updates, history=history))
    assert all(torch.equal(v, parent.state_dict()[k]) for k, v in state.items())
    metadata = dict(arm='dense_residual', seed=seed, updates=updates, training_groups=training_groups,
                    training_sources=training_sources, declaration_sha256=declaration_hash,
                    parent_checkpoint_sha256=ex.sha(parent_checkpoint), parent_updates=parent_payload['updates'],
                    frozen_parent_verified=True, head_only_finetuning=True, trainable_parameters=72,
                    initial_equivalence_exact=True, initial_inputs=len(cache),
                    feature_only_training_examples=len(augmentation), original_anchor_recordings=len(anchors),
                    sampling_sha256=sampler.hexdigest(), eligible_for_demo=False, research_validation_passed=False)
    save(folder / 'checkpoint.pt', model.eval(), metadata)
    restored, payload = load(folder / 'checkpoint.pt')
    for view in ('A', 'C'):
        raw = valid[0][view]['views'][0]
        np.testing.assert_array_equal(predict_raw(model, raw)[0], predict_raw(restored, raw)[0])
    ex.write(folder / 'fit_receipt.json', dict(**metadata, checkpoint_sha256=ex.sha(folder / 'checkpoint.pt'),
             reload_verified=True, fit_seconds=time.perf_counter() - started, history=history))
    return restored, payload


def predictions(model, pair, view):
    return np.stack([predict_raw(model, pair[view]['views'][0])[0]] * 3)


def evaluate(model, payload, pairs, folder, policy):
    previous = held.predictions
    held.predictions = predictions
    try:
        result = held.evaluate(model, payload, pairs, folder, policy)
    finally:
        held.predictions = previous
    result.update(model_version='dense-residual-transformer-1', boundary_invariance='Input construction, not learned objective superiority')
    ex.write(folder / 'accuracy.json', result)
    return result


def cv_worker(task):
    root, fold = task
    root = Path(root); d = ex.read(root / 'declaration.json')
    excluded = set(d['folds'][fold]); rows = ex.read(root / 'train_pairs.json')
    train = [r for r in rows if ex.group(r) not in excluded]
    test = [r for r in rows if ex.group(r) in excluded]
    folder = root / f'cross_validation/fold_{fold + 1}/dense_residual'
    if (folder / 'validation/accuracy.json').exists():
        return ex.read(folder / 'validation/accuracy.json')
    if (folder / 'fit_receipt.json').exists():
        assert ex.sha(folder / 'checkpoint.pt') == ex.read(folder / 'fit_receipt.json')['checkpoint_sha256']
        model, payload = load(folder / 'checkpoint.pt')
    else:
        model, payload = fit(held.prepared(train), held.anchor_pool(ex.read(root / 'anchor_sources.json'), excluded),
                             d['cv_seed'], d['cv_updates'], folder, ex.sha(root / 'declaration.json'),
                             Path(d['cv_parents'][str(fold + 1)]))
    assert_independent(payload['training_groups'], payload['training_sources'], test)
    return evaluate(model, payload, held.prepared(test), folder / 'validation', HeldPolicy(**d['policy']))
