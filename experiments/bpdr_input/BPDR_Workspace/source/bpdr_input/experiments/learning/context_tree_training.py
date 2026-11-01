"""Materialize exactly the existing frame sampler for a fresh tree comparator."""
from collections import defaultdict
from pathlib import Path
import hashlib
import time
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from bpdr_input.experiments.learning import held_voice_training as held, context_improvement as ex
from bpdr_input.experiments.learning.joint_transformer_training import augmented_pairs
from bpdr_input.modeling import context_tree as tree
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.held_voice_policy import HeldPolicy


def sampled_queries(pairs, anchors, seed, updates):
    rng = np.random.default_rng(seed)
    valid = [p for p in pairs if 'excluded' not in p]
    groups, originals, raw = defaultdict(list), defaultdict(list), {}
    for p in valid:
        groups[p['group']].append(p)
        for view in ('A', 'C'):
            raw[p['path_' + view]] = p[view]['views'][0]
    for p in anchors:
        originals[p['group']].append(p)
        raw[p['path_A']] = p['A']['views'][0]
    names, onames = sorted(groups), sorted(originals)
    if not names or not onames:
        raise ValueError('No supervised tree inputs')
    augmentation = augmented_pairs(anchors, seed)
    for i, p in enumerate(augmentation):
        raw[('aug', i, 'A')], raw[('aug', i, 'C')] = p['clean'], p['shifted']
    queries, targets, weights = [], [], []
    digest = hashlib.sha256()
    for _ in range(updates):
        for _ in range(12):
            if augmentation and rng.random() < .25:
                i = int(rng.integers(len(augmentation))); p = augmentation[i]
                choice = p['plateau'] if rng.random() < .7 else p['mask'] & (p['target'] == 0)
                index = int(rng.choice(np.flatnonzero(choice)))
                entries = [(('aug', i, 'A'), 0.), (('aug', i, 'C'), float(p['target'][index]) / 30.)]
            else:
                g = names[int(rng.integers(len(names)))]; p = groups[g][int(rng.integers(len(groups[g])))]
                plateau, outside = p['mask'] & (p['target'] != 0), p['mask'] & (p['target'] == 0)
                index = int(rng.choice(np.flatnonzero(plateau if rng.random() < .7 or not outside.any() else outside)))
                entries = [(p['path_A'], 0.), (p['path_C'], float(p['target'][index]) / 30.)]
            for key, target in entries:
                queries.append((key, index)); targets.append(target)
                weights.append(4. if target == 0 and key == entries[0][0] else 2. if target == 0 else 1.)
                digest.update((str(key) + ':' + str(index) + '\n').encode())
            og = onames[int(rng.integers(len(onames)))]; op = originals[og][int(rng.integers(len(originals[og])))]
            oi = int(rng.choice(np.flatnonzero(op['A']['reliable'])))
            queries.append((op['path_A'], oi)); targets.append(0.); weights.append(4.)
            digest.update((op['path_A'] + ':' + str(oi) + '\n').encode())
    return raw, queries, np.asarray(targets, np.float32), np.asarray(weights, np.float32), digest.hexdigest(), len(augmentation)


def fit(pairs, anchors, seed, updates, folder, declaration_hash, reference_receipt):
    started = time.perf_counter()
    raw, queries, target, weights, digest, augmentations = sampled_queries(pairs, anchors, seed, updates)
    assert digest == reference_receipt['sampling_sha256'], 'Tree training samples differ from Transformer'
    sources = sorted({p['source']['sha256'] for p in pairs + anchors if 'excluded' not in p})
    groups = sorted({p['group'] for p in pairs + anchors if 'excluded' not in p})
    assert sources == reference_receipt['training_sources'] and groups == reference_receipt['training_groups']
    by_key = defaultdict(list)
    for i, (key, center) in enumerate(queries):
        by_key[key].append((i, center))
    x = np.empty((len(queries), 459), np.float32)
    folder.mkdir(parents=True, exist_ok=True)
    for number, (key, positions) in enumerate(by_key.items(), 1):
        out_indices, centers = np.asarray(positions).T
        unique, inverse = np.unique(centers, return_inverse=True)
        x[out_indices] = tree.features(raw[key], unique)[inverse]
        if number % 20 == 0 or number == len(by_key):
            ex.write(folder / 'training_progress.json', dict(stage='materializing_context', inputs=number, total_inputs=len(by_key)))
    model = ExtraTreesRegressor(n_estimators=256, max_depth=18, min_samples_leaf=8,
                               max_features=1.0, random_state=seed, n_jobs=2)
    ex.write(folder / 'training_progress.json', dict(stage='fitting_trees', trees=256, sampled_rows=len(x)))
    model.fit(x, target, sample_weight=weights)
    metadata = dict(arm='context_extra_trees', seed=seed, sampler_updates=updates, sampled_rows=len(x),
                    training_sources=sources, training_groups=groups, sampling_sha256=digest,
                    declaration_sha256=declaration_hash, original_anchor_recordings=len(anchors),
                    feature_only_training_examples=augmentations, regression_only=True,
                    objective='Weighted squared error; no neural response/activity/amplitude losses',
                    eligible_for_demo=False, research_validation_passed=False)
    tree.save(folder / 'checkpoint.joblib', model, metadata)
    restored, payload = tree.load(folder / 'checkpoint.joblib')
    np.testing.assert_array_equal(model.predict(x[:128]), restored.predict(x[:128]))
    ex.write(folder / 'fit_receipt.json', dict(**metadata, checkpoint_sha256=ex.sha(folder / 'checkpoint.joblib'),
             reload_verified=True, fit_seconds=time.perf_counter() - started, parameters=model.get_params(),
             realized_nodes=sum(t.tree_.node_count for t in model.estimators_)))
    return restored, payload


def predictions(model, pair, view):
    return np.stack([tree.predict_raw(model, pair[view]['views'][0])[0]] * 3)


def worker(task):
    root, fold = task; root = Path(root); d = ex.read(root / 'declaration.json')
    rows = ex.read(root / 'train_pairs.json'); groups = set(d['folds'][fold])
    train = [r for r in rows if ex.group(r) not in groups]; test = [r for r in rows if ex.group(r) in groups]
    folder = root / f'cross_validation/fold_{fold + 1}/context_extra_trees'
    if (folder / 'validation/accuracy.json').exists():
        return ex.read(folder / 'validation/accuracy.json')
    if (folder / 'fit_receipt.json').exists():
        assert ex.sha(folder / 'checkpoint.joblib') == ex.read(folder / 'fit_receipt.json')['checkpoint_sha256']
        model, payload = tree.load(folder / 'checkpoint.joblib')
    else:
        model, payload = fit(held.prepared(train), held.anchor_pool(ex.read(root / 'anchor_sources.json'), groups),
                             d['cv_seed'], d['sampler_updates'], folder, ex.sha(root / 'declaration.json'),
                             ex.read(d['reference_receipts'][str(fold + 1)]))
    assert_independent(payload['training_groups'], payload['training_sources'], test)
    previous = held.predictions; held.predictions = predictions
    try:
        result = held.evaluate(model, payload, held.prepared(test), folder / 'validation', HeldPolicy(**d['policy']))
    finally:
        held.predictions = previous
    result.update(model_version=tree.VERSION, model_family='context_extra_trees')
    ex.write(folder / 'validation/accuracy.json', result)
    return result
