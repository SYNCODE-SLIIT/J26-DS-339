"""Single-variable fixed-batch probe of the unsupervised activity multiplier."""

from pathlib import Path

import numpy as np
import torch

import diagnose_short_objective_fit as baseline


run = baseline.run
ROOT = baseline.ROOT.parent / '2026-11-07_ungated_objective_fit_probe'


def main():
    torch.set_num_threads(2)
    recipe = run.read(run.ROOT / 'declaration.json')
    for path, expected in recipe['source_hashes'].items():
        assert run.sha(path) == expected
    _, rows, originals = run.parent.declare()
    data = run.load_records(rows, originals)
    train = [r for r in data if r['group'] not in recipe['folds'][0]]
    original_by_id = {r['source_id']: r for r in train if r['kind'] == 'original'}
    trace_path = run.parent.ROOT / 'fold_1/upper_tail_preservation/physical_trace.npy'
    trace = np.load(trace_path)
    expected, _ = run.parent.paired.previous.sample_trace(train, recipe)
    np.testing.assert_array_equal(trace, expected)
    tokens, padding, batch = run.training_batch(train, original_by_id, trace[0], 'augmentation_only')
    assert recipe['config']['gated']
    settings = dict(recipe, config=dict(recipe['config'], gated=False))
    protocol = dict(source_sha256=run.sha(Path(__file__)), recipe_sha256=run.sha(run.ROOT / 'declaration.json'),
                    baseline_result_sha256=run.sha(baseline.ROOT / 'results.json'),
                    updates=200, changed_variable='Bypass activity multiplier only', training_only=True)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert run.read(path) == protocol
    else:
        run.write(path, protocol)
    model = run.short_experiment.new_model(settings)
    control = run.short_experiment.new_model(recipe)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, control.state_dict()[name], atol=0, rtol=0)
    del control
    optimizer = torch.optim.AdamW(model.parameters(), **recipe['optimizer'])
    history = []
    target = -batch['detuning'] * 30.
    positive = batch['mask'] & (target != 0)
    labels = torch.stack((torch.zeros_like(target), torch.zeros_like(target), target, target), 1)
    zero = batch['mask'][:, None].expand_as(labels) & (labels == 0)
    for step in range(201):
        model.train()
        requests, _, _ = model(tokens, padding, True)
        requests = requests.reshape(36, 4, 2)
        prediction = requests / 30.
        loss, _, _ = run.objective(dict(zip('ABCD', prediction.unbind(1))), batch, 'augmentation_only')
        if step in (0, 50, 100, 200):
            selected = requests[:, 2][positive]
            intended = target[positive]
            entry = dict(update=step, loss=float(loss.detach()),
                         positive_mae_cents=float((selected - intended).abs().mean().detach()),
                         signed_response=float((selected * intended).mean().detach() / intended.square().mean()),
                         zero_mean_request_cents=float(requests[zero].abs().mean().detach()),
                         maximum_request_cents=float(requests.abs().max().detach()))
            history.append(entry)
            print('UNGATED_OBJECTIVE_FIXED_BATCH', entry, flush=True)
        if step == 200:
            break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    run.write(ROOT / 'results.json', dict(declaration_sha256=run.sha(path), history=history,
                                         initialization_matches=True, generalization_measured=False,
                                         production_eligible=False))


if __name__ == '__main__':
    main()
