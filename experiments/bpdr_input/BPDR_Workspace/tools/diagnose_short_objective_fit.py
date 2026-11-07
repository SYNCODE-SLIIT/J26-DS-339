"""Fixed-batch diagnosis of effective anchors and the bounded prediction head."""

from pathlib import Path

import numpy as np
import torch

import compare_short_objectives as run


ROOT = run.ROOT.parent / '2026-11-07_short_objective_fit_diagnosis'


def inspect(model, tokens, padding, batch):
    model.eval()
    request, logits, amplitude = model(tokens, padding, True)
    prediction = request.reshape(36, 4, 2) / 30.
    views = dict(zip('ABCD', prediction.unbind(1)))
    total, parts, _ = run.objective(views, batch, 'augmentation_only')
    target = -batch['detuning'] * 30
    selected = batch['mask'] & (target != 0)
    altered = request.reshape(36, 4, 2)[:, 2]
    positives = altered[selected]
    truth = target[selected]
    labels = torch.stack((torch.zeros_like(target), torch.zeros_like(target), target, target), dim=1)
    masks = batch['mask'][:, None].expand_as(labels)
    zero = masks & (labels == 0)
    all_requests = request.reshape_as(labels)
    probability = torch.sigmoid(logits)
    gradient = torch.autograd.grad(total, (model.regression.bias, model.activity.bias), retain_graph=True)
    return dict(loss=float(total.detach()), anchor=float(parts['anchor'].detach()),
                smoothness=float(parts['smoothness'].detach()),
                positive_mae_cents=float((positives - truth).abs().mean().detach()),
                signed_response=float((positives * truth).mean().detach() / truth.square().mean()),
                known_zero_mean_cents=float(all_requests[zero].abs().mean().detach()),
                amplitude_max_cents=float(amplitude.abs().max().detach()),
                activity_min=float(probability.min().detach()), activity_max=float(probability.max().detach()),
                regression_bias_gradient=float(gradient[0].item()),
                activity_bias_gradient=float(gradient[1].item()))


def main():
    torch.set_num_threads(2)
    declaration = run.read(run.ROOT / 'declaration.json')
    for path, expected in declaration['source_hashes'].items():
        assert run.sha(path) == expected
    # Existing parent checks bind every training array and preserve source roles.
    _, rows, originals = run.parent.declare()
    data = run.load_records(rows, originals)
    groups = declaration['folds'][0]
    train = [record for record in data if record['group'] not in groups]
    original_by_id = {r['source_id']: r for r in train if r['kind'] == 'original'}
    trace_path = run.parent.ROOT / 'fold_1/upper_tail_preservation/physical_trace.npy'
    trace = np.load(trace_path)
    expected, _ = run.parent.paired.previous.sample_trace(train, declaration)
    np.testing.assert_array_equal(trace, expected)
    tokens, padding, batch = run.training_batch(train, original_by_id, trace[0], 'augmentation_only')
    mask = batch['mask'].numpy()
    nonzero = (batch['detuning'].numpy() != 0) & mask
    positive_mass = float(.5 * np.mean(nonzero.sum(1) / mask.sum(1)))
    checkpoint = run.ROOT / 'fold_1/augmentation_only/checkpoint.pt'
    receipt = run.read(checkpoint.parent / 'fit_receipt.json')
    assert run.sha(checkpoint) == receipt['checkpoint_sha256']
    ROOT.mkdir(exist_ok=True)
    protocol = dict(source_sha256=run.sha(Path(__file__)), recipe_sha256=run.sha(run.ROOT / 'declaration.json'),
                    checkpoint_sha256=run.sha(checkpoint), trace_sha256=run.sha(trace_path),
                    updates=200, seed=339, training_only=True, production_eligible=False)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert run.read(path) == protocol
    else:
        run.write(path, protocol)
    saved = run.short_experiment.new_model(declaration)
    saved.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
    saved_metrics = inspect(saved, tokens, padding, batch)
    model = run.short_experiment.new_model(declaration)
    optimizer = torch.optim.AdamW(model.parameters(), **declaration['optimizer'])
    history = [dict(update=0, **inspect(model, tokens, padding, batch))]
    for step in range(1, 201):
        model.train()
        prediction = model(tokens, padding).reshape(36, 4, 2) / 30.
        loss, _, _ = run.objective(dict(zip('ABCD', prediction.unbind(1))), batch, 'augmentation_only')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step in (50, 100, 200):
            entry = dict(update=step, **inspect(model, tokens, padding, batch))
            history.append(entry)
            print('OBJECTIVE_FIXED_BATCH', entry, flush=True)
    result = dict(declaration_sha256=run.sha(path), effective_nonzero_anchor_mass=positive_mass,
                  effective_zero_anchor_mass=1 - positive_mass, saved_fit_training_batch=saved_metrics,
                  history=history, independent_replay=False, generalization_measured=False)
    run.write(ROOT / 'results.json', result)


if __name__ == '__main__':
    main()
