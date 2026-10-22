"""Isolated Gate E learning controls; never a training corpus or deployable model."""
from bpdr_input import COMPONENT_ROOT
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
import torch

from bpdr_input.modeling.examples import Normalizer, collate, read_pair
from bpdr_input.modeling.losses import objective, masked_mean_abs, smoothness
from bpdr_input.experiments.learning.matched_comparison import matched_model, check_catalog, metric_row
from bpdr_input.preparation.sources.shared_exports import digest

ARMS = ('real_original', 'oracle_original', 'real_balanced', 'oracle_balanced')


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def regions(batch):
    plateau = batch['mask'] & (batch['detuning'] != 0)
    outside = batch['mask'] & (batch['detuning'] == 0)
    if not plateau.any(dim=1).all() or not outside.any(dim=1).all():
        raise ValueError('Every diagnostic pair needs plateau and outside frames')
    return plateau, outside


def balanced_objective(r, batch):
    """Diagnostic only: equal plateau/outside weight within each per-example term."""
    plateau, outside = regions(batch)
    distance = lambda a, b: .5*(masked_mean_abs(a, b, plateau)[0]+masked_mean_abs(a, b, outside)[0])
    zero = torch.zeros_like(batch['detuning'])
    anchor = sum(distance(r[v], zero if v in 'AB' else -batch['detuning']) for v in 'ABCD')/4
    boundary = .5*(distance(r['A'], r['B'])+distance(r['C'], r['D']))
    response = .5*(distance(r['C']-r['A'], -batch['detuning'])+distance(r['D']-r['B'], -batch['detuning']))
    regularizer = sum(smoothness(r[v], batch['voiced']['A' if v in 'AB' else 'C'])[0] for v in 'ABCD')/4
    parts = dict(anchor=anchor, boundary=boundary, detuning=response, smoothness=regularizer)
    return anchor+boundary+response+.1*regularizer, parts


def loss(r, batch, balanced):
    if balanced:
        return balanced_objective(r, batch)
    total, parts, _ = objective(r, batch, 'bpdr', response=True)
    return total, parts


def make_control(batch, oracle):
    """Private-label positive control, separately marked and never written as corpus inputs."""
    inputs = {v: x.clone() for v, x in batch['inputs'].items()}
    if oracle:
        # Replace normalized boundary column with a deliberately privileged answer cue.
        # This is unavailable during real inference and is NOT a proposed feature.
        for v in 'ABCD':
            inputs[v][:, :, 70] = 0 if v in 'AB' else -batch['detuning']/30
    return dict(batch, inputs=inputs)


def move(batch, device):
    return dict(batch, inputs={v: x.to(device) for v, x in batch['inputs'].items()},
                voiced={v: x.to(device) for v, x in batch['voiced'].items()},
                **{k: batch[k].to(device) for k in ('mask', 'detuning', 'padding')})


def forward(model, batch):
    return {v: model(batch['inputs'][v], batch['padding']) for v in 'ABCD'}


def measure(model, batch, record, raw_pair, balanced):
    model.eval()
    with torch.no_grad():
        r = forward(model, batch)
        total, parts = loss(r, batch, balanced)
    prediction = {v: r[v][0].cpu().numpy().copy() for v in 'ABCD'}
    row = metric_row(record, prediction, raw_pair['inputs'], raw_pair['mask'], raw_pair['detuning'])
    p, _ = regions(batch)
    row['plateau_response_gain'] = float(torch.cat([
        ((r[v]-r[o])/(-batch['detuning']).masked_fill(~p, 1))[p] for v, o in [('C', 'A'), ('D', 'B')]
    ]).mean().cpu())
    return dict(loss=float(total.cpu()), parts={k: float(v.cpu()) for k, v in parts.items()}, metrics=row), prediction


def zero_gradient_check(batch, balanced):
    r = {v: torch.zeros_like(batch['detuning'], requires_grad=True) for v in 'ABCD'}
    total, _ = loss(r, batch, balanced)
    total.backward()
    p, _ = regions(batch)
    # Descent must move detuned predictions toward -injection and originals oppositely for L_D.
    checks = {v: bool(((r[v].grad[p]*(-batch['detuning'][p])) < 0).all()) for v in 'CD'}
    return dict(loss=float(total), detuned_plateau_descent_sign_correct=all(checks.values()),
                finite_gradients=all(bool(torch.isfinite(x.grad).all()) for x in r.values()),
                output_gradient_abs_sum={v: float(x.grad.abs().sum()) for v, x in r.items()})


def declare(root):
    prep = Path(json.loads((root/'runs/comparison_preparation_current.json').read_text())['path'])
    records = json.loads((prep/'summary.json').read_text())['examples']
    check_catalog(records, 'train')
    # Source selection by existing order before any new outcomes; never the easiest observed case.
    record = records[0]
    pair = read_pair(record, 'engineering')
    with np.load(prep/'normalizer.npz') as z:
        batch = collate([pair], Normalizer(z['mean'], z['scale']))
    regions(batch)
    old = root/'runs/matched_comparison/64dc9a78175c4113929e8a890894dcbd'
    declaration = json.loads((old/'declaration.json').read_text())
    frozen = {**declaration['frozen_inputs'], **declaration['pipeline_files']}
    frozen.update({str(p): digest(p) for p in old.rglob('*') if p.is_file()})
    for v in 'ABCD':
        p = Path(record['input_directory'])/(v+'_input.npz')
        frozen[str(p)] = digest(p)
    frozen[record['target_path']] = digest(record['target_path'])
    # Snapshot root modules used by the diagnostic, excluding notebooks/docs that will be updated.
    for p in (root/'src/bpdr_input').rglob('*.py'):
        frozen[str(p)] = digest(p)
    out = root/'runs/learning_control'/uuid.uuid4().hex
    out.mkdir(parents=True)
    shutil.copyfile(__file__, out/'learning_control_at_declaration.py')
    d = dict(version='gate-e-learning-control-1', declared_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Bounded single-training-pair optimization diagnosis; no generalization or model selection',
        data_status='exploratory_not_research', training_eligible=False, production_eligible=False,
        research_ready_examples=0, record=record, preparation_run=str(prep), baseline_run=str(old),
        arms=list(ARMS), updates_per_arm=100, total_updates=400, seed=339,
        model=json.loads((prep/'locked_protocol.json').read_text())['model'],
        learning_rate=.001, optimizer='AdamW', weight_decay=.01, gradient_clip=1.,
        batch_size=1, full_phrase_frames=len(pair['mask']), warmup=0,
        response='Active from update 1, unlike 100-update warmup in original comparison; this is a separate diagnostic',
        selection='Final update only; intermediate measures at 0,25,50,75,100 are descriptive, no best-checkpoint selection',
        oracle='PRIVATE LABEL CONTROL ONLY: replace normalized column 70 by target/30 for C/D and zero for A/B; never a runtime feature',
        balanced='Only loss reduction changes: equal .5 plateau/.5 outside per anchor, boundary and detuning term; smoothness unchanged',
        paired_initialization='Same seed/common tensors; same dropout RNG reset per arm',
        positive_control_rule='At final update: correct signed plateau response >= .1 and plateau error <= .9 times analytic zero error',
        limits='One predeclared training pair/seed; oracle cannot establish real-input learnability; no development/heldout processing or audio rendering',
        frozen_inputs=frozen, pipeline_files=declaration['pipeline_files'])
    for p, h in frozen.items():
        if digest(p) != h:
            raise ValueError('Frozen evidence changed: '+p)
    write(out/'declaration.json', d)
    write(root/'runs/learning_control_current.json', dict(path=str(out)))
    return out


def run(out, device):
    out = Path(out)
    if (out/'evidence.json').exists() or any((out/a).exists() for a in ARMS):
        raise ValueError('Diagnostic already started; preserve this run and declare a new one')
    d = json.loads((out/'declaration.json').read_text())
    for p, h in d['frozen_inputs'].items():
        if digest(p) != h: raise ValueError('Changed input: '+p)
    if device == 'xpu' and not torch.xpu.is_available(): raise ValueError('XPU unavailable')
    torch.set_num_threads(2)
    pair = read_pair(d['record'], 'engineering')
    with np.load(Path(d['preparation_run'])/'normalizer.npz') as z:
        batch = collate([pair], Normalizer(z['mean'], z['scale']))
    p, o = regions(batch)
    zero = {v: np.zeros_like(pair['detuning']) for v in 'ABCD'}
    zero_metrics = metric_row(d['record'], zero, pair['inputs'], pair['mask'], pair['detuning'])
    checks = {str(b): zero_gradient_check(batch, b) for b in (False, True)}
    if not all(c['finite_gradients'] and c['detuned_plateau_descent_sign_correct'] for c in checks.values()):
        raise ValueError('Incorrect/nonfinite output gradient')
    protocol = dict(fit=dict(seed=d['seed']), model=d['model'])
    initial = matched_model(protocol, 71)
    torch.save(initial.state_dict(), out/'initial_state.pt')
    results = {}
    for arm in ARMS:
        start = time.monotonic()
        directory = out/arm
        directory.mkdir()
        model = matched_model(protocol, 71).to(device)
        control = move(make_control(batch, arm.startswith('oracle')), device)
        balanced = arm.endswith('balanced')
        model.eval()
        # Evaluation-mode gradients isolate wiring without stochastic dropout.
        probe = dict(control, inputs={v: x.detach().clone().requires_grad_() for v, x in control['inputs'].items()})
        total, _ = loss(forward(model, probe), probe, balanced)
        total.backward()
        gradient = dict(finite=all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in model.parameters()),
            project_weight_norm=float(model.project.weight.grad.norm().cpu()),
            head_weight_norm=float(model.head.weight.grad.norm().cpu()),
            input_abs_gradient_by_column=torch.stack([x.grad.abs().mean((0, 1)) for x in probe['inputs'].values()]).mean(0).cpu().tolist())
        write(directory/'gradients.json', gradient)
        model.zero_grad(set_to_none=True)
        # Match training RNG, independently of gradient/evaluation probes.
        torch.manual_seed(d['seed'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=d['learning_rate'], weight_decay=d['weight_decay'])
        history = []
        m, predictions = measure(model, control, d['record'], pair, balanced)
        history.append(dict(update=0, **m))
        np.savez_compressed(directory/'initial_predictions.npz', **predictions)
        for step in range(1, d['updates_per_arm']+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total, _ = loss(forward(model, control), control, balanced)
            if not torch.isfinite(total): raise ValueError('Nonfinite loss')
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), d['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            if step % 25 == 0:
                m, predictions = measure(model, control, d['record'], pair, balanced)
                history.append(dict(update=step, preclip_gradient_norm=float(norm.cpu()), **m))
                print(arm, step, 'plateau_error', round(m['metrics']['repair_error_cents'], 5), 'gain', round(m['metrics']['plateau_response_gain'], 5), flush=True)
                write(directory/'history.json', history)
        np.savez_compressed(directory/'final_predictions.npz', **predictions)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # Diagnostic-only state cannot be consumed as an eligible training checkpoint.
        torch.save(dict(state=state, diagnostic_only=True, oracle_input=arm.startswith('oracle'),
                        production_eligible=False, training_eligible=False), directory/'diagnostic_state.pt')
        delta = float(sum((state[k]-initial.state_dict()[k]).double().square().sum() for k in state).sqrt())
        final = history[-1]['metrics']
        results[arm] = dict(final=history[-1], initial=history[0], parameter_delta_l2=delta,
            zero_loss=checks[str(balanced)]['loss'], gradient=gradient, seconds=time.monotonic()-start,
            passes_predeclared_response_rule=bool(final['plateau_response_gain'] >= .1 and
                final['repair_error_cents'] <= .9*zero_metrics['repair_error_cents']))
        del model, optimizer, control, probe
    evidence = dict(status='complete', declaration_sha256=digest(out/'declaration.json'),
        interpreter=sys.executable, torch_version=torch.__version__, device=device,
        device_name=torch.xpu.get_device_name(0) if device=='xpu' else 'CPU',
        optimizer_updates=400, data_status=d['data_status'], research_ready_examples=0,
        production_eligible=False, training_eligible=False, original_target_fraction_zero=1-.5*float(p.sum()/batch['mask'].sum()),
        plateau_frames=int(p.sum()), outside_frames=int(o.sum()), zero_metrics=zero_metrics,
        zero_gradient_checks=checks, arms=results,
        artifacts={str(f.relative_to(out)): digest(f) for f in out.rglob('*') if f.is_file()})
    for f, h in d['frozen_inputs'].items():
        if digest(f) != h: raise ValueError('Frozen evidence changed: '+f)
    write(out/'evidence.json', evidence)
    print('COMPLETE', str(out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['declare', 'run'])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--device', choices=['cpu', 'xpu'], default='xpu')
    args = parser.parse_args()
    if args.command == 'declare': print(declare(COMPONENT_ROOT))
    else: run(args.run, args.device)


if __name__ == '__main__': main()
