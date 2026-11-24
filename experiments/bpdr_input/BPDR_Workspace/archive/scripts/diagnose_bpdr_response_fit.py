"""Test the declared BPDR response terms after a collapsed fixed-batch warmup."""

from pathlib import Path

import numpy as np
import torch

import diagnose_short_objective_fit as baseline


run = baseline.run
ROOT = baseline.ROOT.parent / '2026-11-07_bpdr_response_fit_diagnosis'


def main():
    torch.set_num_threads(2)
    proof = run.read(baseline.ROOT / 'integrity_receipt.json')
    assert proof['results_sha256'] == run.sha(baseline.ROOT / 'results.json')
    declaration = run.read(run.ROOT / 'declaration.json')
    for path, digest in declaration['source_hashes'].items():
        assert run.sha(path) == digest
    _, rows, originals = run.parent.declare()
    data = run.load_records(rows, originals)
    train = [r for r in data if r['group'] not in declaration['folds'][0]]
    original_by_id = {r['source_id']: r for r in train if r['kind'] == 'original'}
    trace_path = run.parent.ROOT / 'fold_1/upper_tail_preservation/physical_trace.npy'
    trace = np.load(trace_path)
    expected, _ = run.parent.paired.previous.sample_trace(train, declaration)
    np.testing.assert_array_equal(trace, expected)
    tokens, padding, batch = run.training_batch(train, original_by_id, trace[0], 'bpdr')
    protocol = dict(script_sha256=run.sha(Path(__file__)), baseline_replay_sha256=run.sha(baseline.ROOT / 'integrity_receipt.json'),
                    recipe_sha256=run.sha(run.ROOT / 'declaration.json'), trace_sha256=run.sha(trace_path),
                    warmup_updates=200, full_updates=200, seed=339, training_only=True)
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert run.read(path) == protocol
    else:
        run.write(path, protocol)
    model = run.short_experiment.new_model(declaration)
    optimizer = torch.optim.AdamW(model.parameters(), **declaration['optimizer'])
    history = []
    for step in range(1, 401):
        model.train()
        prediction = model(tokens, padding).reshape(36, 4, 2) / 30.
        loss, parts, _ = run.objective(dict(zip('ABCD', prediction.unbind(1))), batch, 'bpdr', response=step > 200)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step in (50, 100, 200, 250, 300, 400):
            metrics = dict(update=step, response_enabled=step > 200,
                           optimized_loss=float(loss.detach()),
                           parts={key: float(value.detach()) for key, value in parts.items()},
                           **baseline.inspect(model, tokens, padding, batch))
            history.append(metrics)
            run.write(ROOT / 'progress.json', dict(history=history))
            print('BPDR_FIXED_BATCH_RESPONSE', metrics, flush=True)
    # The warm-up is identical to the independently replayed augmentation control.
    warmup = next(row for row in history if row['update'] == 200)
    previous = run.read(baseline.ROOT / 'results.json')['history'][-1]
    for key, value in previous.items():
        assert warmup[key] == value
    run.write(ROOT / 'results.json', dict(declaration_sha256=run.sha(path), history=history,
                                         warmup_matches_baseline=True, generalization_measured=False,
                                         production_eligible=False))


if __name__ == '__main__':
    main()
