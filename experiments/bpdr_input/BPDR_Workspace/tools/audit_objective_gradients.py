"""No-fit audit of the actual four-view anchor normalization and gradients."""

import hashlib
from pathlib import Path

import numpy as np
import torch

import diagnose_short_objective_fit as baseline


run = baseline.run
ROOT = baseline.ROOT.parent / '2026-11-07_objective_gradient_audit'


def vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([gradient.reshape(-1) if gradient is not None else torch.zeros_like(parameter).reshape(-1)
                      for gradient, parameter in zip(gradients, parameters)])


def measure(model, tokens, padding, batch):
    model.eval()
    output = model(tokens, padding).reshape(36, 4, 2) / 30
    target = -batch['detuning']
    labels = torch.stack((torch.zeros_like(target), torch.zeros_like(target), target, target), 1)
    mask = batch['mask'][:, None].expand_as(labels)
    positive, zero = mask & (labels != 0), mask & (labels == 0)
    weight = mask / mask.sum(dim=2, keepdim=True).clamp_min(1) / (4 * 36)
    positive_loss = ((output - labels).abs() * weight * positive).sum()
    zero_loss = ((output - labels).abs() * weight * zero).sum()
    total, parts, _ = run.objective(dict(zip('ABCD', output.unbind(1))), batch, 'augmentation_only')
    torch.testing.assert_close(positive_loss + zero_loss, parts['anchor'])
    parameters = tuple(model.parameters())
    gradients = {name: vector(loss, parameters) for name, loss in
                 [('positive', positive_loss), ('zero', zero_loss), ('total', total)]}
    p, z, t = (gradients[name] for name in ('positive', 'zero', 'total'))
    cosine = torch.dot(p, z) / (p.norm() * z.norm()).clamp_min(1e-30)
    return dict(positive_loss=float(positive_loss.detach()), zero_loss=float(zero_loss.detach()),
                anchor=float(parts['anchor'].detach()), objective=float(total.detach()),
                gradient_norms={name: float(gradient.norm()) for name, gradient in gradients.items()},
                positive_zero_gradient_cosine=float(cosine),
                positive_dot_total_gradient=float(torch.dot(p, t)),
                zero_dot_total_gradient=float(torch.dot(z, t)))


def main():
    torch.set_num_threads(2)
    recipe = run.read(run.ROOT / 'declaration.json')
    for path, expected in recipe['source_hashes'].items():
        assert run.sha(path) == expected
    _, rows, originals = run.parent.declare()
    data = run.load_records(rows, originals)
    train = [record for record in data if record['group'] not in recipe['folds'][0]]
    original_by_id = {record['source_id']: record for record in train if record['kind'] == 'original'}
    trace_path = run.parent.ROOT / 'fold_1/upper_tail_preservation/physical_trace.npy'
    trace = np.load(trace_path)
    expected, _ = run.parent.paired.previous.sample_trace(train, recipe)
    np.testing.assert_array_equal(trace, expected)
    tokens, padding, batch = run.training_batch(train, original_by_id, trace[0], 'augmentation_only')
    checkpoint = run.ROOT / 'fold_1/augmentation_only/checkpoint.pt'
    assert run.read(checkpoint.parent / 'fit_receipt.json')['checkpoint_sha256'] == run.sha(checkpoint)
    protocol = dict(script_sha256=run.sha(Path(__file__)), recipe_sha256=run.sha(run.ROOT / 'declaration.json'),
                    checkpoint_sha256=run.sha(checkpoint), trace_sha256=run.sha(trace_path), fits=0)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert run.read(path) == protocol
    else:
        run.write(path, protocol)
    labels = torch.stack((torch.zeros_like(batch['detuning']), torch.zeros_like(batch['detuning']),
                          -batch['detuning'], -batch['detuning']), 1).flatten().numpy()
    masks = batch['mask'][:, None].expand(36, 4, 2).flatten().numpy()
    by_input = {}
    for token, pad, label, eligible in zip(tokens.numpy(), padding.numpy(), labels, masks):
        if eligible:
            key = hashlib.sha256(token.tobytes() + pad.tobytes()).hexdigest()
            by_input.setdefault(key, set()).add(float(label))
    model = run.short_experiment.new_model(recipe)
    initial = measure(model, tokens, padding, batch)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
    fitted = measure(model, tokens, padding, batch)
    result = dict(declaration_sha256=run.sha(path), initial=initial, fitted=fitted,
                  unique_labeled_token_contexts=len(by_input),
                  contexts_with_conflicting_labels=sum(len(labels) > 1 for labels in by_input.values()),
                  examined_training_batch_only=True)
    output = ROOT / 'results.json'
    if output.exists():
        assert run.read(output) == result
    else:
        run.write(output, result)
    print('OBJECTIVE_GRADIENT_AUDIT', result, flush=True)


if __name__ == '__main__':
    main()
