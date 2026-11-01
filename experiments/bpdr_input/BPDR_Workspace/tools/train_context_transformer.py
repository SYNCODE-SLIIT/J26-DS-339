"""Declared, bounded Transformer recovery experiment; never activates a demo.

Uses the existing audited VocalSet cache and source-level partitions. Compares
three matched neural conditions, a tree baseline fitted on the same training
examples, and the saved historical Transformer. Calibration is development
material previously inspected by this project, not an untouched final test.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import model_recovery as recovery
from bpdr_input.modeling.context_transformer import (ContextConfig, ContextTransformer,
    load_checkpoint, predict_features, save_checkpoint, window_at)
from bpdr_input.modeling.local_context_pilot import context_features
from bpdr_input.reporting.bpdr_showcase import load_model


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def attach_context(pairs):
    cache = {}
    for p in pairs:
        if 'excluded' in p:
            continue
        p['context'] = {}
        for v in ('A', 'C'):
            key = p['path_'+v]
            if key not in cache:
                cache[key] = [context_features(raw) for raw in p[v]['views']]
            p['context'][v] = cache[key]
    return pairs


def valid(pairs):
    return [p for p in pairs if 'excluded' not in p]


def fit(pairs, condition, steps, seed, folder):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    config = ContextConfig(features=31 if condition == 'frame_only' else 32, window=81)
    model = ContextTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    groups = sorted({p['group'] for p in valid(pairs)})
    grouped = {g:[p for p in valid(pairs) if p['group']==g] for g in groups}
    history = []
    started = time.perf_counter()
    for step in range(steps):
        arrays, pads, targets = [], [], []
        for _ in range(8):
            group = groups[int(rng.integers(len(groups)))]
            p = grouped[group][int(rng.integers(len(grouped[group])))]
            plateau = p['mask'] & (p['target'] != 0)
            outside = p['mask'] & (p['target'] == 0)
            indices = np.flatnonzero(plateau if rng.random()<.7 or not outside.any() else outside)
            center = int(rng.choice(indices))
            alternative = int(rng.integers(1, 3))
            for view, boundary in (('A', 0), ('A', alternative), ('C', 0), ('C', alternative)):
                x, pad, _ = window_at(p['context'][view][boundary][:,:config.features], center, config.window)
                arrays.append(x)
                pads.append(pad)
            targets.append(float(p['target'][center])/30.)
        x = torch.from_numpy(np.stack(arrays))
        pad = torch.from_numpy(np.stack(pads))
        target = torch.tensor(targets, dtype=torch.float32)
        model.train()
        pred = model(x, pad)[:, config.window//2].reshape(-1, 4)/30.
        a, b, c, d = pred.unbind(dim=1)
        anchor = .25*((a*a).mean()+(b*b).mean()+((c-target)**2).mean()+((d-target)**2).mean())
        boundary = .5*(((a-b)**2).mean()+((c-d)**2).mean())
        response = .5*(((c-a-target)**2).mean()+((d-b-target)**2).mean())
        weight = .25 if condition == 'bpdr' and step >= steps//4 else 0.
        loss = anchor + weight*response + (.1*boundary if weight else 0.)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training objective')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step%50 == 0 or step==steps-1:
            row = dict(step=step+1, loss=float(loss.detach()), anchor=float(anchor.detach()),
                       response=float(response.detach()), boundary=float(boundary.detach()))
            history.append(row)
            print(condition, json.dumps(row), flush=True)
    model.eval()
    folder.mkdir(parents=True, exist_ok=True)
    save_checkpoint(folder/'checkpoint.pt', model, dict(condition=condition, updates=steps,
        seed=seed, eligible_for_demo=False, research_validation_passed=False))
    write(folder/'history.json', history)
    restored, _ = load_checkpoint(folder/'checkpoint.pt')
    probe = valid(pairs)[0]['context']['A'][0][:5,:config.features]
    np.testing.assert_allclose(predict_features(model, probe), predict_features(restored, probe), atol=1e-6)
    return model, dict(model=model.describe(), updates=steps, seed=seed,
        seconds=time.perf_counter()-started, checkpoint_sha256=sha(folder/'checkpoint.pt'),
        reload_verified=True, final_training_loss=history[-1])


def score(pairs, predictor, label):
    rows = []
    cache = {}
    for p in pairs:
        if 'excluded' in p:
            rows.append(dict(group=p['source']['singer_id'], cents=p['cents'], excluded=p['excluded']))
            continue
        predictions = {}
        for view in ('A', 'C'):
            key = p['path_'+view]
            if key not in cache:
                cache[key] = np.asarray(predictor(p, view))
            predictions[view] = cache[key]
        a, c = predictions['A'], predictions['C']
        plateau = p['mask'] & (p['target'] != 0)
        outside = p['mask'] & (p['target'] == 0)
        error = np.abs(c[0,plateau]-p['target'][plateau])
        rows.append(dict(group=p['group'], source_id=p['source_id'], cents=p['cents'],
            baseline_mae_cents=float(np.abs(p['target'][plateau]).mean()),
            raw_mae_cents=float(error.mean()), within_5c_percent=float(100*np.mean(error<=5)),
            within_10c_percent=float(100*np.mean(error<=10)),
            response_gain=float(np.mean((c[0,plateau]-a[0,plateau])/p['target'][plateau])),
            original_abs_cents=float(np.abs(a[0,p['A']['reliable']]).mean()),
            original_over_3c_percent=float(100*np.mean(np.abs(a[0,p['A']['reliable']])>3)),
            outside_abs_cents=float(np.abs(c[0,outside]).mean()) if outside.any() else None,
            boundary_disagreement_cents=float(np.mean(np.abs(c[1:]-c[0])))))
    scored = [r for r in rows if 'excluded' not in r]
    metrics = ('baseline_mae_cents','raw_mae_cents','within_5c_percent','within_10c_percent',
               'response_gain','original_abs_cents','original_over_3c_percent','outside_abs_cents','boundary_disagreement_cents')
    by_group = {}
    for group in sorted({r['group'] for r in rows}):
        rr = [r for r in scored if r['group']==group]
        by_group[group] = dict(attempted=sum(r['group']==group for r in rows), scoreable=len(rr))
        if rr:
            by_group[group].update({k:float(np.mean([r[k] for r in rr if r[k] is not None])) for k in metrics})
    represented = [r for r in by_group.values() if r['scoreable']]
    summary = {k:float(np.mean([r[k] for r in represented])) for k in metrics}
    summary.update(attempted=len(rows), scoreable=len(scored), singers=len(represented),
        raw_error_reduction_percent=float(100*np.mean([1-r['raw_mae_cents']/r['baseline_mae_cents'] for r in represented])))
    print(label, json.dumps(summary), flush=True)
    return dict(summary=summary, by_singer=by_group, rows=rows)


def neural_predictor(model):
    def predict(p, view):
        features = p['context'][view]
        first = predict_features(model, features[0][:,:model.config.features])
        if model.config.features == 31:
            return np.stack([first]*3)
        return np.stack([first]+[predict_features(model, x) for x in features[1:]])
    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates', type=int, default=300)
    args = parser.parse_args()
    if args.updates < 1:
        raise ValueError('Updates must be positive')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run = Path(read(WORKSPACE/'results/model_recovery/current.json')['path'])
    out = WORKSPACE/'results/transformer_recovery'/uuid.uuid4().hex
    out.mkdir(parents=True)
    train_rows = read(run/'train_pairs.json')
    calibration_rows = read(run/'calibration_pairs.json')
    train_groups = sorted({p['group'] for p in train_rows if 'excluded' not in p})
    cal_groups = sorted({p.get('group',p['source']['singer_id']) for p in calibration_rows})
    assert not set(train_groups) & set(cal_groups)
    # Source provenance is known before fitting; targets are never model inputs.
    assert not ({p['source']['sha256'] for p in train_rows} & {p['source']['sha256'] for p in calibration_rows})
    protected_paths = [WORKSPACE/'selected_model/bpdr.pt',
        WORKSPACE/'results/notebook_training/current.json',
        WORKSPACE/'exports/selected_run_inference/selected_model.joblib',
        WORKSPACE/'exports/selected_run_inference/manifest.json']
    protected = {str(p):sha(p) for p in protected_paths}
    inputs = {str(run/f):sha(run/f) for f in ('train_pairs.json','calibration_pairs.json','declaration.json')}
    for p in train_rows+calibration_rows:
        if 'excluded' not in p:
            folder = Path(p['path_A']).parent
            for file in (Path(p['path_A']),Path(p['path_C']),Path(p['target_path']),folder/'original_features.npz',folder/f"features_{p['cents']:+g}.npz"):
                inputs[str(file)] = sha(file)
    declaration = dict(version='context-transformer-recovery-1', created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Development experiment, no demo activation or final-pilot access',
        data_run=str(run), train_groups=train_groups, calibration_groups=cal_groups,
        conditions=['augmentation_only','frame_only','bpdr'], updates=args.updates, seed=339,
        config=ContextTransformer(ContextConfig(window=81)).describe(),
        optimizer=dict(name='AdamW', learning_rate=.0005, weight_decay=.001, clip_norm=1.),
        sampling='Uniform singer, uniform audited pair; 70% plateau / 30% audited outside; 8 aligned pairs per update',
        targets='Only target archive labels on audited mask; original zero; detuned signed correction',
        features='31 existing own-audio context features; boundary added for augmentation_only and bpdr',
        objective='30-cent normalized MSE: .5 original/.35 plateau/.15 outside in expectation; bpdr adds .25 paired response + .1 boundary consistency after 25% warmup',
        inference='Every frame is center of its own 81-frame window; explicit edge padding',
        training_diagnostic='Hold out female4/male4 from training side; fixed 150 updates, no selection from diagnostic results',
        calibration_policy='One fixed final checkpoint per condition, 300 updates by default; no calibration tuning or checkpoint selection',
        interpretation='Feature/architecture redesign; not a rerun of original 71-feature proposal',
        limitations=['Small, previously inspected calibration set','One random seed','Long held-tone controlled shifts only',
                     'Frame estimates only; delivered waveform/controller evaluation still required'],
        sources={str(Path(__file__)):sha(__file__),
                 str(WORKSPACE/'source/bpdr_input/modeling/context_transformer.py'):sha(WORKSPACE/'source/bpdr_input/modeling/context_transformer.py')},
        protected_artifacts=protected,input_hashes=inputs)
    write(out/'declaration.json', declaration)
    print('RUN',out,flush=True)
    train = attach_context(recovery.cached_pairs(train_rows))
    write(out/'data_counts.json', dict(train_attempted=len(train), train_audited=len(valid(train)),
        audited_by_singer=dict(Counter(p['group'] for p in valid(train)))))
    # Learning proof uses only a training pair; the endpoint is fixed in advance.
    sanity, sanity_fit = fit([valid(train)[0]],'bpdr',100,339,out/'sanity')
    sanity_score = score([valid(train)[0]],neural_predictor(sanity),'sanity_train_only')
    write(out/'sanity.json',dict(fit=sanity_fit,score=sanity_score))
    held = {'female4','male4'}
    internal_train = [p for p in train if p.get('group',p['source']['singer_id']) not in held]
    internal_hold = [p for p in train if p.get('group',p['source']['singer_id']) in held]
    diagnostic, diagnostic_fit = fit(internal_train,'bpdr',150,339,out/'training_diagnostic')
    diagnostic_score = score(internal_hold,neural_predictor(diagnostic),'training_singer_holdout')
    write(out/'training_diagnostic.json',dict(fit=diagnostic_fit,score=diagnostic_score))
    models, fits = {}, {}
    for condition in declaration['conditions']:
        models[condition], fits[condition] = fit(train,condition,args.updates,339,out/condition)
    # Tree refit uses exactly the old audited arrays and declared configuration.
    x,y,weights = recovery.training_arrays(train)
    settings = read(run/'declaration.json')['models']['extra_trees']
    estimator = ExtraTreesRegressor(**settings)
    with threadpool_limits(limits=2):
        estimator.fit(x,y,sample_weight=weights)
    write(out/'locked_endpoints.json',dict(fits=fits, tree_settings=settings,
        declaration_sha256=sha(out/'declaration.json'), calibration_access='Features loaded only after all endpoints saved'))
    calibration = attach_context(recovery.cached_pairs(calibration_rows))
    results = {}
    for name,model in models.items():
        results[name] = score(calibration,neural_predictor(model),name)
    results['matched_extra_trees'] = score(calibration,
        lambda p,v:np.stack([estimator.predict(p['context'][v][0][:,:31])]*3),'matched_extra_trees')
    old,normalizer,width = load_model(WORKSPACE/'selected_model/bpdr.pt')
    def historical(p,v):
        with torch.inference_mode():
            return old(torch.from_numpy(np.stack([normalizer(a[:,:width]) for a in p[v]['views']]))).numpy()
    results['historical_transformer'] = score(calibration,historical,'historical_transformer')
    assert all(sha(p)==digest for p,digest in protected.items())
    assert all(sha(p)==digest for p,digest in inputs.items())
    evidence = dict(declaration_sha256=sha(out/'declaration.json'), results=results,
        protected_artifacts_unchanged=True, input_artifacts_unchanged=True,
        demo_activated=False, research_validation_passed=False,
        calibration_used_for_selection=False, final_pilot_accessed=False,
        environment=dict(python=sys.version,torch=torch.__version__,device='cpu',threads=2))
    write(out/'evidence.json',evidence)
    lines = ['# Transformer recovery experiment', '',
        'Development calibration only. One seed; controlled long tones. No demo activation.', '',
        '| Model | MAE (cents) | Within 10 cents | Original change (cents) | Response gain |',
        '|---|---:|---:|---:|---:|']
    for name,result in results.items():
        s = result['summary']
        lines.append(f"| {name} | {s['raw_mae_cents']:.3f} | {s['within_10c_percent']:.1f}% | {s['original_abs_cents']:.3f} | {s['response_gain']:.3f} |")
    lines += ['', 'All conditions use the same calibration cases. Historical Transformer retains different features and training history.',
        'Read evidence.json for every attempted case, exclusions, singers, preservation and boundary sensitivity.',
        'These are raw correction estimates. Permission, rendered-WAV replay and listening remain to be verified.',
        'The calibration set has been inspected in earlier experiments and is not an independent final test.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out.parent/'latest_experiment.json',dict(path=str(out),evidence_sha256=sha(out/'evidence.json')))
    print('COMPLETE',out,flush=True)


if __name__ == '__main__':
    main()
