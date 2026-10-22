"""Frozen synthetic-to-real development evaluation; no final test or musical approvals.

All seeds and all three proposal models are reported. Existing fine-pitch real
audio archives are read, never regenerated. References and intervention masks
are used only to score outputs, not as model inputs.
"""
from bpdr_input import COMPONENT_ROOT
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

COMPONENT = COMPONENT_ROOT
ROOT = COMPONENT / 'runs/presentation_study'
FINE = COMPONENT / 'runs/fine_pitch/66e3619e611241dd80d33a32578b73c6'
CONDITIONS = ('augmentation_only', 'frame_only', 'bpdr')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def frozen_runs():
    root = COMPONENT / 'runs/controlled_study'
    runs = [(339, Path(read(root / 'training_current.json')['path']))]
    runs += [(r['seed'], Path(r['path'])) for r in
             (json.loads(line) for line in (root / 'training_replicates.jsonl').read_text().splitlines() if line.strip())]
    if len({s for s, _ in runs}) != len(runs):
        raise ValueError('Duplicate replicate seed')
    return sorted(runs)


def summaries(rows):
    from bpdr_input.experiments.learning.controlled_study import summary
    result = dict(all=summary(rows))
    for key in ('dataset', 'corruption', 'cents'):
        result['by_' + key] = {str(v): summary([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
    # Variants are correlated. Give each source group equal weight, not each frame.
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['group'])].append(row)
    rng = np.random.default_rng(20261017)
    ci = {}
    for dataset in sorted({k[0] for k in groups}):
        values = np.array([[np.mean([r[k] for r in rr]) for k in
                           ('correction_mae', 'response_gain', 'unshifted_alteration')]
                          for (ds, _), rr in sorted(groups.items()) if ds == dataset])
        boot = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
        ci[dataset] = dict(groups=len(values), independent_unit='singer (Vocadito), artist proxy (Jamendo)',
                           group_equal_mean=values.mean(axis=0).tolist(),
                           bootstrap_95_percentile=np.quantile(boot, [.025, .975], axis=0).tolist(),
                           columns=['correction_mae', 'response_gain', 'unshifted_alteration'])
    result['group_uncertainty'] = ci
    return result


def score_predictions(pair, predictions):
    """Score compensating correction; a constant phrase bias is not response."""
    mask = pair['mask']; target = -pair['detuning']
    plateau = mask & (target != 0); outside = mask & ~plateau
    if not plateau.any() or not outside.any():
        raise ValueError('Empty correction or outside support')
    r = predictions
    return dict(correction_mae=float(np.mean([np.abs(r[v][plateau]-target[plateau]).mean() for v in 'CD'])),
                zero_mae=float(np.abs(target[plateau]).mean()),
                response_gain=float((r['C'][plateau]-r['A'][plateau]).mean()/target[plateau].mean()),
                boundary_sensitivity=float(np.mean([np.abs(r[a][mask]-r[b][mask]).mean() for a, b in [('A', 'B'), ('C', 'D')]])),
                unshifted_alteration=float(np.mean([np.abs(r[v][mask]).mean() for v in 'AB'])),
                outside_alteration=float(np.mean([np.abs(r[v][outside]).mean() for v in 'CD'])))


def predictions(model, normalizer, width, pair, device):
    import torch
    # Every view is processed independently, without the target or scoring mask.
    x = np.stack([normalizer(pair['inputs'][v][:, :width]) for v in 'ABCD'])
    with torch.inference_mode():
        output = model(torch.from_numpy(x).to(device)).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite prediction')
    return dict(zip('ABCD', output))


def evaluate_records(checkpoint, records, out, device):
    from bpdr_input.reporting.bpdr_showcase import load_model
    from bpdr_input.modeling.examples import read_pair
    from bpdr_input.experiments.learning.fine_pitch_relative_check import relative_pair
    model, normalizer, width = load_model(checkpoint)
    model = model.to(device); rows = []; arrays = {}
    started = time.monotonic()
    for i, record in enumerate(records):
        pair = relative_pair(read_pair(record, 'engineering'))
        pred = predictions(model, normalizer, width, pair, device)
        ds = record.get('dataset', 'vocadito' if record['phrase_id'].startswith('vocadito') else 'jamendo')
        rows.append(dict(score_predictions(pair, pred), index=i, dataset=ds, group=record['singer_id'],
                         phrase_id=record['phrase_id'], cents=record['cents'], corruption=record['corruption']))
        for v in 'ABCD':
            arrays[f'{i}_{v}'] = pred[v]
        if (i+1) % 75 == 0:
            print(checkpoint.name, i+1, '/', len(records), 'seconds', round(time.monotonic()-started), flush=True)
    np.savez_compressed(out/'predictions.npz', **arrays)
    write(out/'rows.json', rows)
    result = summaries(rows)
    write(out/'summary.json', result)
    del model
    return result


def transfer():
    import torch
    if not torch.xpu.is_available():
        raise RuntimeError('Declared XPU required; no silent device fallback')
    torch.set_num_threads(2)
    records = read(FINE/'development_catalog.json')['records']
    if len(records) != 304 or any(r['split'] != 'development' for r in records):
        raise ValueError('Frozen development inventory changed')
    runs = frozen_runs()
    out = ROOT / uuid.uuid4().hex
    out.mkdir(parents=True)
    frozen = {str(FINE/'development_catalog.json'): sha(FINE/'development_catalog.json')}
    for _, run in runs:
        for condition in CONDITIONS:
            p = run/(condition+'.pt'); frozen[str(p)] = sha(p)
    # Verify all input views/targets against their recorded provenance.
    for record in records:
        for v in 'ABCD':
            p = Path(record['input_directory'])/(v+'_input.npz')
            if sha(p) != record['input_sha256'][v]:
                raise ValueError(f'Changed input {p}')
            frozen[str(p)] = sha(p)
        p = Path(record['target_path']); frozen[str(p)] = sha(p)
    pipeline = COMPONENT.parents[1]/'datasets/mtg-jamendo/melody-chord-extraction-pipeline'
    shared = {str(p): sha(p) for p in sorted(pipeline.rglob('*.py')) if '.venv' not in p.parts and '__pycache__' not in p.parts}
    declaration = dict(version='presentation-real-transfer-1', declared_utc=datetime.now(timezone.utc).isoformat(),
                       scope='Exploratory development transfer, not a final test or musical validation',
                       optimizer_updates=0, source_sha256=sha(__file__), device='xpu', records=len(records),
                       seeds=[s for s, _ in runs], conditions=CONDITIONS, frozen=frozen, shared_pipeline=shared,
                       gate=dict(per_dataset_median_response_gain_min=.25, per_dataset_original_alteration_max_cents=1.,
                                 per_dataset_correction_gain_min=0.),
                       selection='Report every seed and condition. No development-fitted normalizer or threshold. Final test untouched.')
    write(out/'declaration.json', declaration)
    print('TRANSFER_RUN', out, flush=True)
    results = {}
    for seed, run in runs:
        results[str(seed)] = {}
        for condition in CONDITIONS:
            folder = out/str(seed)/condition; folder.mkdir(parents=True)
            result = evaluate_records(run/(condition+'.pt'), records, folder, 'xpu')
            result['engineering_gate_passes'] = all(
                s['median_response_gain'] >= .25 and s['unshifted_alteration'] <= 1. and s['correction_gain'] > 0
                for s in result['by_dataset'].values())
            results[str(seed)][condition] = result
            print('RESULT', seed, condition, json.dumps(result['by_dataset']), flush=True)
    for p, digest in {**frozen, **shared}.items():
        if sha(p) != digest:
            raise ValueError('Frozen file changed: '+p)
    write(out/'evidence.json', dict(declaration_sha256=sha(out/'declaration.json'), results=results,
                                  frozen_files_unchanged=len(frozen), shared_pipeline_unchanged=len(shared),
                                  optimizer_updates=0, final_test_accessed=False))
    write(ROOT/'transfer_current.json', dict(path=str(out)))
    return out


def adapt():
    """One bounded matched transfer-learning experiment, retaining frozen controls.

    This is engineering training on real recordings with injected errors, not
    human-approved musical ground truth. All conditions receive the same 300
    updates, seed and schedule: four real and four synthetic records per update.
    No development case is used for fitting, normalization or early stopping.
    """
    import torch
    from bpdr_input.reporting.bpdr_showcase import load_model
    from bpdr_input.experiments.learning.controlled_study import load_pair
    from bpdr_input.modeling.examples import read_pair, collate
    from bpdr_input.experiments.learning.expanded_comparison import to_device
    from bpdr_input.experiments.learning.fine_pitch_mse_check import squared_objective
    from bpdr_input.experiments.learning.fine_pitch_relative_check import relative_pair
    from bpdr_input.modeling.losses import LossWeights
    if not torch.xpu.is_available():
        raise RuntimeError('Declared XPU unavailable')
    torch.set_num_threads(2)
    seed = 339; updates = 300
    base = dict(frozen_runs())[seed]
    data = Path(read(COMPONENT/'runs/controlled_study/current.json')['path'])
    real_records = read(FINE/'training_catalog.json')['records']
    development = read(FINE/'development_catalog.json')['records']
    synthetic = [r for r in read(data/'catalog.json')['records'] if r['split'] == 'train' and r['plateau_frames'] >= 10]
    if len(real_records) != 995 or len(synthetic) != 384 or any(r['split'] != 'train' for r in real_records):
        raise ValueError('Training inventory changed')
    ds = lambda r: r.get('dataset', 'vocadito' if r['phrase_id'].startswith('vocadito') else 'jamendo')
    train_groups = {(ds(r), r['singer_id']) for r in real_records}
    dev_groups = {(ds(r), r['singer_id']) for r in development}
    if train_groups & dev_groups:
        raise ValueError('Singer/artist overlap')
    groups = defaultdict(list)
    for i, r in enumerate(real_records):
        groups[(ds(r), r['singer_id'], r['phrase_id'])].append(i)
    by_ds = {d: [g for g in groups if g[0] == d] for d in ('vocadito', 'jamendo')}
    rng = np.random.default_rng(seed)
    schedule = []
    for step in range(updates):
        chosen = []
        for d in ('vocadito', 'jamendo'):
            for _ in range(2):
                g = by_ds[d][int(rng.integers(len(by_ds[d])))]
                chosen.append(int(rng.choice(groups[g])))
        schedule.append(dict(real=chosen, synthetic=rng.choice(len(synthetic), 4, replace=False).tolist()))
    out = ROOT/uuid.uuid4().hex; out.mkdir(parents=True)
    (out/'executed_presentation_study.py').write_bytes(Path(__file__).read_bytes())
    np.save(out/'schedule.npy', np.array([[*s['real'], *s['synthetic']] for s in schedule]))
    frozen = {str(p): sha(p) for p in [FINE/'training_catalog.json', FINE/'development_catalog.json', data/'catalog.json']}
    for c in CONDITIONS:
        p = base/(c+'.pt'); frozen[str(p)] = sha(p)
    declaration = dict(version='presentation-mixed-adaptation-1', declared_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed, source_sha256=sha(__file__), updates_per_condition=updates, effective_batch=8,
        learning_rate=1e-4, optimizer='AdamW, weight_decay .01, gradient clip 1',
        initialization='Frozen synthetic seed 339, same corresponding condition; original checkpoints never overwritten',
        real_records=len(real_records), synthetic_records=len(synthetic),
        sampling='Two Vocadito, two Jamendo (equal phrase groups within dataset), four synthetic train variants per update',
        normalizer='Keep each original synthetic training normalizer frozen; no development normalization',
        objective='squared L_A + .1 L_S, BPDR L_B/L_D after 100-update warm-up; unchanged 71/70 features',
        schedule_sha256=sha(out/'schedule.npy'), frozen=frozen,
        evaluation='Final update only, every condition on all 304 development variants; no tuning/early stopping on development',
        gate='Both datasets: median paired response >= .25, original mean alteration <= 1 cent, correction gain > 0',
        limitation='Engineering real+synthetic adaptation; no musical approvals or final test; only one matched adaptation seed')
    write(out/'declaration.json', declaration)
    print('ADAPT_RUN', out, flush=True)
    real_cache = {}; synth_cache = {}; results = {}
    for c in CONDITIONS:
        torch.manual_seed(seed)
        model, norm, width = load_model(base/(c+'.pt')); model.to('xpu').train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        history = []; start = time.monotonic()
        for step, item in enumerate(schedule, 1):
            optimizer.zero_grad(); total = 0.
            selected = []
            for i in item['real']:
                if i not in real_cache:
                    real_cache[i] = relative_pair(read_pair(real_records[i], 'engineering'))
                selected.append(real_cache[i])
            for i in item['synthetic']:
                if i not in synth_cache:
                    synth_cache[i] = relative_pair(load_pair(synthetic[i]))
                selected.append(synth_cache[i])
            for pair in selected:
                b = to_device(collate([pair], norm, width), 'xpu')
                r = {v: model(x, b['padding']) for v, x in b['inputs'].items()}
                loss, _ = squared_objective(r, b, c, LossWeights(), step > 100)
                (loss/8).backward(); total += float(loss.detach())/8
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            optimizer.step()
            history.append(dict(update=step, loss=total, gradient_norm=grad))
            if step % 25 == 0:
                print(c, step, 'loss', round(total, 3), 'seconds', round(time.monotonic()-start), flush=True)
        saved = torch.load(base/(c+'.pt'), weights_only=False, map_location='cpu')
        saved.update(state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                     data_status='exploratory_real_plus_synthetic', adaptation_declaration_sha256=sha(out/'declaration.json'))
        torch.save(saved, out/(c+'.pt'))
        write(out/(c+'_history.json'), history)
        del model, optimizer
        folder = out/c; folder.mkdir()
        result = evaluate_records(out/(c+'.pt'), development, folder, 'xpu')
        result['engineering_gate_passes'] = all(s['median_response_gain'] >= .25 and s['unshifted_alteration'] <= 1.
                                               and s['correction_gain'] > 0 for s in result['by_dataset'].values())
        results[c] = result
        print('ADAPT_RESULT', c, json.dumps(result['by_dataset']), flush=True)
    for p, digest in frozen.items():
        if sha(p) != digest:
            raise ValueError('Frozen evidence changed')
    write(out/'evidence.json', dict(declaration_sha256=sha(out/'declaration.json'), results=results,
                                  optimizer_updates=updates*len(CONDITIONS), final_test_accessed=False,
                                  note='Failed candidates are retained; do not replace the working synthetic demo automatically'))
    write(ROOT/'adapt_current.json', dict(path=str(out)))
    return out


if __name__ == '__main__':
    {'transfer': transfer, 'adapt': adapt}[sys.argv[1]]()
