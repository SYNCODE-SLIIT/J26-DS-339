"""Plateau diagnostic: do all three arms fit the frozen fold-1 training batch? (plan section 16.2)

Training batch only. Declared before running. Variants: as_declared, gate_open (activity-head bias +2 on every arm).
Run from this folder: PYTHONPATH=../source python probe_control_fit_plateau.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import compare_balanced_short_objectives as study
import diagnose_short_objective_fit as diag
from probe_balanced_anchor_fit import balanced_anchor

previous = study.previous
ROOT = study.ROOT.parent / '2026-11-12_control_fit_plateau'
UPDATES, WARMUP, LOG = 600, 200, 25


def run_one(condition, variant, declaration, train, originals, trace0):
    study.set_kind(train, study.KINDS[0])
    tokens, padding, batch = previous.training_batch(train, originals, trace0, condition)
    model = previous.short_experiment.new_model(declaration)
    if variant == 'gate_open':
        with torch.no_grad():
            model.activity.bias.fill_(2.0)
    torch.manual_seed(339)
    optimizer = torch.optim.AdamW(model.parameters(), **declaration['optimizer'])
    history = []
    for step in range(UPDATES + 1):
        if step % LOG == 0:
            history.append(dict(update=step, **{k: float(v) for k, v in diag.inspect(model, tokens, padding, batch).items()
                                                  if k in ('positive_mae_cents', 'signed_response', 'known_zero_mean_cents', 'amplitude_max_cents')}))
            print('PLATEAU', condition, variant, history[-1], flush=True)
        if step == UPDATES:
            break
        model.train()
        output = model(tokens, padding).reshape(len(trace0), 4, 2) / 30.
        predictions = dict(zip('ABCD', output.unbind(1)))
        loss, parts, _ = previous.objective(predictions, batch, condition, response=step >= WARMUP)
        loss = loss - parts['anchor'] + balanced_anchor(predictions, batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    start = history[0]['positive_mae_cents']
    escape = next((h['update'] for h in history if h['positive_mae_cents'] < 0.8 * start), None)
    return dict(history=history, start_error=start, escape_update=escape, final=history[-1])


def main():
    torch.set_num_threads(2)
    declaration, rows, originals_rows = study.declare()
    data = study.records(rows, originals_rows)
    train = [r for r in data if r['group'] not in declaration['folds'][0]]
    originals = {r['source_id']: r for r in train if r['kind'] == 'original'}
    trace0 = np.load(previous.parent.ROOT / 'fold_1/upper_tail_preservation/physical_trace.npy')[0]
    results = {}
    for variant in ('as_declared', 'gate_open'):
        for condition in ('augmentation_only', 'frame_only', 'bpdr'):
            results[f'{variant}|{condition}'] = run_one(condition, variant, declaration, train, originals, trace0)
    ROOT.mkdir(exist_ok=True)
    (ROOT / 'results.json').write_text(json.dumps(dict(
        scope='fold-1 first frozen training batch, seed 339, 600 updates, training only; plan section 16.2',
        generalization_measured=False, production_eligible=False, results=results), indent=1), encoding='utf-8')
    for k, v in results.items():
        print(k, 'start', round(v['start_error'], 2), 'escape', v['escape_update'], 'final', {a: round(b, 3) for a, b in v['final'].items()})


if __name__ == '__main__':
    main()
