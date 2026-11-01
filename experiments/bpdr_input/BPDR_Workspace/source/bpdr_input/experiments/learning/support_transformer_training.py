"""Matched parent continuation with or without complete-support supervision."""
from collections import defaultdict
from pathlib import Path
import hashlib
import time
import math
import numpy as np
import torch
from torch.nn import functional as F
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning import joint_transformer_training as joint
from bpdr_input.experiments.learning.joint_transformer_training import augmented_pairs
from bpdr_input.modeling.joint_transformer import JointTransformer, joint_tokens, predict_raw, save, load
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.held_voice_policy import HeldPolicy
from bpdr_input.experiments.learning.support_supervision import block_starts, block_objective, BLOCK_FRAMES


def fit(pairs, anchors, arm, seed, updates, folder, declaration_hash, parent_checkpoint):
    if arm not in ('continued_point', 'support_blocks'):
        raise ValueError('Unknown joint experiment arm')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, parent = load(parent_checkpoint)
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
    if (set(parent['training_groups']) != set(groups)|set(originals)
            or set(parent['training_sources']) != {p['source']['sha256'] for p in valid+anchors}
            or parent['seed'] != seed):
        raise ValueError('Parent singer/source/seed lineage mismatch')
    reference, _ = load(parent_checkpoint)
    initial_error=0.
    for view in ('A','C'):
        probe=valid[0][view]['views'][0]
        old=predict_raw(reference,probe)[0];new=predict_raw(model,probe)[0]
        np.testing.assert_allclose(old,new,atol=1e-6)
        initial_error=max(initial_error,float(abs(old-new).max()))
    del reference
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
    augmentation = augmented_pairs(anchors, seed)
    for i, p in enumerate(augmentation):
        cached(('aug', i, 'A'), p['clean'])
        cached(('aug', i, 'C'), p['shifted'])
    blocks=defaultdict(list)
    for p in valid:
        known=p['mask'] & p['A']['reliable'] & p['C']['reliable']
        starts=block_starts(p['target'],known)
        if len(starts):blocks[p['group']].append((p,starts))
    if not blocks:raise ValueError('No genuinely supervised complete blocks')
    block_groups=sorted(blocks)
    block_rng=np.random.default_rng(seed+23000)
    block_hash=hashlib.sha256()
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = []
    sample_hash = hashlib.sha256()
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
                weights.append(4. if target == 0 and key == entries[0][0] else 2. if target == 0 else 1.)
            og = original_groups[int(rng.integers(len(original_groups)))]
            op = originals[og][int(rng.integers(len(originals[og])))]
            oi = int(rng.choice(np.flatnonzero(op['A']['reliable'])))
            x, pad = cache[op['path_A']]
            xx.append(x[oi]); pads.append(pad[oi]); targets.append(0.); weights.append(4.)
            sample_hash.update((op['path_A']+':'+str(oi)+'\n').encode())
        target, weight = torch.tensor(targets), torch.tensor(weights)
        out, logit, amp = model(torch.from_numpy(np.stack(xx)), torch.from_numpy(np.stack(pads)), True)
        pred, positive = out/30., target != 0
        anchor = ((pred-target).square()*weight).mean()
        paired, wanted = pred.reshape(12, 3), target.reshape(12, 3)[:, 1]
        response = (paired[:, 1]-paired[:, 0]-wanted).square().mean()
        activity = F.binary_cross_entropy_with_logits(logit, positive.float())
        amplitude = ((amp[positive]/30.-target[positive]).square().mean() if positive.any() else amp.sum()*0.)
        bg=block_groups[int(block_rng.integers(len(block_groups)))]
        bp,starts=blocks[bg][int(block_rng.integers(len(blocks[bg])))]
        left=int(block_rng.choice(starts));indices=np.arange(left,left+BLOCK_FRAMES)
        block_hash.update((bp['path_C']+':'+str(left)+'\n').encode())
        bx=np.concatenate([cache[bp['path_'+v]][0][indices] for v in ('A','C')])
        bpad=np.concatenate([cache[bp['path_'+v]][1][indices] for v in ('A','C')])
        bo,bl,_=model(torch.from_numpy(bx),torch.from_numpy(bpad),True)
        by=torch.from_numpy(np.stack([np.zeros(BLOCK_FRAMES,np.float32),bp['target'][indices].astype(np.float32)]))
        support,margin=block_objective(bo.reshape(2,BLOCK_FRAMES),bl.reshape(2,BLOCK_FRAMES),by)
        loss = anchor + .25*response + .5*activity + .25*amplitude + (.25*support if arm=='support_blocks' else 0.)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite joint loss')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == updates-1:
            history.append(dict(update=step+1, loss=float(loss.detach()), block_loss=float(support.detach()), support_margin=float(margin.detach())))
            ex.write(folder/'training_progress.json', dict(update=step+1, planned_updates=updates, history=history))
    metadata = dict(arm=arm, seed=seed, updates=updates,
                    training_groups=sorted(set(groups) | set(originals)),
                    training_sources=sorted({p['source']['sha256'] for p in valid+anchors}),
                    declaration_sha256=declaration_hash, eligible_for_demo=False, research_validation_passed=False,
                    feature_only_training_examples=len(augmentation), original_anchor_recordings=len(anchors),
                    sampling_sha256=sample_hash.hexdigest(), block_sampling_sha256=block_hash.hexdigest(),
                    block_frames=BLOCK_FRAMES, block_weight=.25 if arm=='support_blocks' else 0.,
                    valid_block_pairs_by_singer={g:len(p) for g,p in blocks.items()},
                    parent_checkpoint_sha256=ex.sha(parent_checkpoint),parent_updates=parent['updates'],
                    total_updates=parent['updates']+updates,initial_equivalence_max_error_cents=initial_error)
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
                                arm, d['cv_seed'], d['cv_updates'], folder, ex.sha(root/'declaration.json'),Path(d['cv_parents'][str(fold+1)]))
    assert_independent(payload['training_groups'], payload['training_sources'], test)
    return joint.evaluate(model, payload, held.prepared(test), folder/'validation', HeldPolicy(**d['policy']))
