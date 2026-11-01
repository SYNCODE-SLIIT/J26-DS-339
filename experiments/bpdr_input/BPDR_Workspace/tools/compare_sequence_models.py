"""Matched sequence-family comparison with frozen data, loss, sampler and gates."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / 'source'))
from bpdr_input.experiments.learning import response_transformer_training as training
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling import sequence_alternatives as alternative
from bpdr_input.modeling.correction_accuracy import summarize, assert_independent
from bpdr_input.repair.held_voice_policy import HeldPolicy
from bpdr_input.reporting.transformer_accuracy_report import complete_report
import train_dense_residual_transformer as previous
import run_transformer_curve_study as curve

ROOT = WORKSPACE / 'results/transformer_usability/2026-10-30_v27_sequence_comparison'
BASE = previous.BASE
FAMILIES = ('bigru', 'tcn')


def declare():
    ROOT.mkdir(exist_ok=True)
    old = previous.declare()
    if not (ROOT / 'declaration.json').exists():
        retained = ('folds', 'default_seed', 'policy', 'curve_condition', 'renderer', 'input_hashes',
                    'baseline_cv', 'baseline_cv_sha256', 'baseline_evidence', 'baseline_evidence_sha256')
        d = {k: old[k] for k in retained}
        files = dict(old['source_hashes'])
        for p in (Path(__file__).resolve(), Path(alternative.__file__), Path(training.__file__)):
            files[str(p)] = ex.sha(p)
        references = {str(f): str(BASE / f'cross_validation/fold_{f}/matched_baseline/fit_receipt.json') for f in range(1, 5)}
        d.update(created_at=datetime.now(timezone.utc).isoformat(), families=list(FAMILIES), cv_seed=339, cv_updates=1200,
                 source_hashes=files, reference_receipts=references,
                 reference_receipt_hashes={p: ex.sha(p) for p in references.values()},
                 architecture='Same51x8 joint acoustic tokens,32-wide input projection, sinusoidal positions, center norm and bounded amplitude/activity heads. Encoder replacement only:2-layer bidirectional GRU16 per direction, or4 symmetric residual TCN blocks with2 kernel3 convolutions each and dilations1/2/4/8 (61-token receptive field). No dropout. Record parameter/time differences; not parameter-matched.',
                 padding='GRU packs only present context then scatters outputs back. TCN masks absent positions at input and each block.',
                 training='Execute the existing V12 matched-baseline fit function with only constructor/save/load substituted in the isolated worker. Same data/augmentations/loss/AdamW/budget/seed. Every actual sampling trace must equal its frozen reference fold receipt.',
                 control='Reuse authenticated V12 reference trained with exactly this fit function,1200 updates and matching sample digest; no extra reference tuning. No pretrained external encoder or final-audio access.',
                 selection='All-singer raw gate, all50 original WAV identities, then minimum singer-macro delivered after MAE; reference wins ties. Same200 attempts/108 scoreable/92 exclusions and existing renderer/permission. No family adoption from raw accuracy alone.',
                 wider_comparison='Fresh context-feature ExtraTrees is a separately declared next classical baseline; historical active demo metrics are not directly comparable.',
                 demo_activated=False, research_validation_passed=False)
        ex.write(ROOT / 'declaration.json', d)
        snapshot = ROOT / 'executed_sources'; snapshot.mkdir(exist_ok=True)
        for p in files:
            (snapshot / Path(p).name).write_bytes(Path(p).read_bytes())
        for name in ('train_pairs.json', 'anchor_sources.json'):
            (ROOT / name).write_bytes((BASE / name).read_bytes())
    d = ex.read(ROOT / 'declaration.json')
    for label in ('source_hashes', 'input_hashes', 'reference_receipt_hashes'):
        for path, digest in d[label].items():
            assert ex.sha(path) == digest, path
    for label in ('baseline_cv', 'baseline_evidence'):
        assert ex.sha(d[label]) == d[label + '_sha256']
    return d


def worker(task):
    root, fold, family = task
    root = Path(root); d = ex.read(root / 'declaration.json')
    groups = set(d['folds'][fold]); rows = ex.read(root / 'train_pairs.json')
    train = [r for r in rows if ex.group(r) not in groups]
    test = [r for r in rows if ex.group(r) in groups]
    folder = root / f'cross_validation/fold_{fold + 1}/{family}'
    if (folder / 'validation/accuracy.json').exists():
        return ex.read(folder / 'validation/accuracy.json')
    if (folder / 'fit_receipt.json').exists():
        assert ex.sha(folder / 'checkpoint.pt') == ex.read(folder / 'fit_receipt.json')['checkpoint_sha256']
        model, payload = alternative.load(folder / 'checkpoint.pt')
    else:
        old = training.JointTransformer, training.save, training.load
        training.JointTransformer = lambda: alternative.SequenceAlternative(family)
        training.save, training.load = alternative.save, alternative.load
        try:
            model, payload, receipt = training.fit(training.held.prepared(train),
                 training.held.anchor_pool(ex.read(root / 'anchor_sources.json'), groups), 'matched_baseline',
                 d['cv_seed'], d['cv_updates'], folder, ex.sha(root / 'declaration.json'))
        finally:
            training.JointTransformer, training.save, training.load = old
        receipt.update(model_family=family, parameters=sum(p.numel() for p in model.parameters()))
        ex.write(folder / 'fit_receipt.json', receipt)
    receipt = ex.read(folder / 'fit_receipt.json')
    reference = ex.read(d['reference_receipts'][str(fold + 1)])
    assert receipt['sampling_sha256'] == reference['sampling_sha256'], 'Matched sampler differs'
    assert receipt['training_groups'] == reference['training_groups']
    assert receipt['training_sources'] == reference['training_sources']
    assert_independent(payload['training_groups'], payload['training_sources'], test)
    result = training.joint.evaluate(model, payload, training.held.prepared(test), folder / 'validation', HeldPolicy(**d['policy']))
    result.update(model_version=alternative.VERSION, model_family=family)
    ex.write(folder / 'validation/accuracy.json', result)
    ex.write_report(folder / 'validation', result, 'Matched ' + family + ' acoustic sequence model')
    complete_report(folder / 'validation')
    return result


def delivery(d, family):
    rows = ex.read(ROOT / 'train_pairs.json'); pairs = training.held.prepared(rows)
    cache, models = {}, {}
    for fold, groups in enumerate(d['folds'], 1):
        folder = ROOT / f'cross_validation/fold_{fold}/{family}'
        model, payload = alternative.load(folder / 'checkpoint.pt')
        subset = [r for r in rows if ex.group(r) in groups]
        assert_independent(payload['training_groups'], payload['training_sources'], subset)
        for group in groups:
            models[group] = model, payload
        with np.load(folder / 'validation/predictions.npz') as z:
            for i, row in enumerate(subset):
                if 'excluded' not in row:
                    for view in ('A', 'C'):
                        cache[row['path_' + view]] = z[f'{i}_{view}'].copy()
    folder = ROOT / 'arms' / family; folder.mkdir(parents=True, exist_ok=True)
    if not (folder / 'declaration.json').exists():
        ex.write(folder / 'declaration.json', dict(parent=str(ROOT / 'declaration.json'), parent_sha256=ex.sha(ROOT / 'declaration.json')))
    oldroot, oldpredict = curve.ROOT, curve.predict_raw
    curve.ROOT, curve.predict_raw = folder, alternative.predict_raw
    try:
        return curve.run_condition(rows, pairs, cache, models, ex.read(ROOT / 'anchor_sources.json'), 'median11', 'training', ROOT / 'train_pairs.json')
    finally:
        curve.ROOT, curve.predict_raw = oldroot, oldpredict


def main():
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['declare', 'cv', 'delivery'])
    args = parser.parse_args(); d = declare()
    if args.stage == 'cv':
        collected = {f: [] for f in FAMILIES}
        with ProcessPoolExecutor(max_workers=2) as pool:
            tasks = {pool.submit(worker, (str(ROOT), fold, family)): (fold, family) for family in FAMILIES for fold in range(4)}
            for task in as_completed(tasks):
                fold, family = tasks[task]; result = task.result()
                collected[family].extend(result['rows'])
                print('SEQUENCE_CV_FOLD', family, fold + 1, result['summary']['raw_mae_cents'], flush=True)
        results = {family: summarize(rows) for family, rows in collected.items()}
        results['matched_baseline'] = ex.read(d['baseline_cv'])['results']['matched_baseline']
        ex.write(ROOT / 'cross_validation.json', dict(results=results, new_neural_fits=8, matched_sampling_verified=True))
        print('SEQUENCE_CV_COMPLETE', [(f, r['prediction_gate_passed'], r['summary']['raw_mae_cents']) for f, r in results.items()], flush=True)
    elif args.stage == 'delivery':
        results = ex.read(ROOT / 'cross_validation.json')['results']
        evidence = dict(matched_baseline=ex.read(d['baseline_evidence']))
        for family in FAMILIES:
            evidence[family] = delivery(d, family)
        conditions = ['matched_baseline'] + list(FAMILIES)
        selected = min(conditions, key=lambda f: (not results[f]['prediction_gate_passed'],
                       not evidence[f]['gates']['original_controls_all_identity'],
                       evidence[f]['summary']['delivered_after_mae_cents'], conditions.index(f)))
        ex.write(ROOT / 'locked_selection.json', dict(arm=selected, results=evidence, default_seed=339,
                 demo_activated=False, research_validation_passed=False))
        print('SEQUENCE_SELECTION', selected, flush=True)
    declare()


if __name__ == '__main__':
    main()
