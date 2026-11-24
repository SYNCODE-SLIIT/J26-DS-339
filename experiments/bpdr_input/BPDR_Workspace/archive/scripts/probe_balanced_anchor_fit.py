"""Test one common-anchor weighting correction on the frozen training batch."""

from pathlib import Path

import numpy as np
import torch

import diagnose_short_objective_fit as baseline


run = baseline.run
ROOT = baseline.ROOT.parent / '2026-11-07_balanced_anchor_fit_probe'


def balanced_anchor(predictions, batch):
    target = -batch['detuning']
    labels = torch.stack((torch.zeros_like(target), torch.zeros_like(target), target, target), dim=1)
    mask = batch['mask'][:, None].expand_as(labels)
    output = torch.stack([predictions[view] for view in 'ABCD'], dim=1)
    error = abs(output - labels)
    losses = []
    for selected in (mask & (labels != 0), mask & (labels == 0)):
        counts = selected.sum(dim=(1, 2))
        used = counts > 0
        if not used.any():
            raise ValueError('The balanced diagnostic requires both target classes')
        per_example = (error * selected).sum(dim=(1, 2)) / counts.clamp_min(1)
        losses.append(per_example[used].mean())
    return .5 * (losses[0] + losses[1])


def main():
    torch.set_num_threads(2)
    recipe = run.read(run.ROOT / 'declaration.json')
    for path, digest in recipe['source_hashes'].items():
        assert run.sha(path) == digest
    _, rows, originals = run.parent.declare()
    data = run.load_records(rows, originals)
    train = [r for r in data if r['group'] not in recipe['folds'][0]]
    original_by_id = {r['source_id']: r for r in train if r['kind'] == 'original'}
    trace_path = run.parent.ROOT / 'fold_1/upper_tail_preservation/physical_trace.npy'
    trace = np.load(trace_path)
    expected, _ = run.parent.paired.previous.sample_trace(train, recipe)
    np.testing.assert_array_equal(trace, expected)
    tokens, padding, batch = run.training_batch(train, original_by_id, trace[0], 'augmentation_only')
    protocol = dict(script_sha256=run.sha(Path(__file__)), recipe_sha256=run.sha(run.ROOT / 'declaration.json'),
                    baseline_sha256=run.sha(baseline.ROOT / 'results.json'),
                    gradient_audit_sha256=run.sha(ROOT.parent / '2026-11-07_objective_gradient_audit/results.json'),
                    changed_variable='Equal nonzero/zero anchor-class mass, per-example means within each class',
                    updates=200, seed=339, training_only=True, production_eligible=False)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert run.read(path) == protocol
    else:
        run.write(path, protocol)
    model = run.short_experiment.new_model(recipe)
    optimizer = torch.optim.AdamW(model.parameters(), **recipe['optimizer'])
    history = []
    for step in range(201):
        model.train()
        output = model(tokens, padding).reshape(36, 4, 2) / 30.
        predictions = dict(zip('ABCD', output.unbind(1)))
        original, parts, _ = run.objective(predictions, batch, 'augmentation_only')
        loss = original - parts['anchor'] + balanced_anchor(predictions, batch)
        if step in (0, 50, 100, 200):
            metrics = dict(update=step, optimized_loss=float(loss.detach()),
                           **baseline.inspect(model, tokens, padding, batch))
            history.append(metrics)
            print('BALANCED_ANCHOR_FIXED_BATCH', metrics, flush=True)
        if step == 200:
            break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    run.write(ROOT / 'results.json', dict(declaration_sha256=run.sha(path), history=history,
                                         generalization_measured=False, production_eligible=False))


if __name__ == '__main__':
    main()
