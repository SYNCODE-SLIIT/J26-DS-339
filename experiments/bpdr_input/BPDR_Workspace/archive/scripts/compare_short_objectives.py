"""Matched short-phrase BPDR objective comparison on frozen source folds."""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

import compare_short_spectral_context as short_experiment
from bpdr_input.modeling.losses import CONDITIONS, objective


short = short_experiment.short
parent = short_experiment.parent
ROOT = parent.ROOT.parent / '2026-11-06_short_objective_comparison'
read, write, sha = parent.read, parent.write, parent.sha


def boundary_tokens(main, alternate, centers, condition):
    """Use the same acoustic tokens and projection size in all three arms."""
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown condition: {condition}')
    if not np.array_equal(main[:, :70], alternate[:, :70]):
        raise ValueError('Boundary views must preserve the acoustic columns')
    tokens, padding = short.tokens_from_features(short.frame_features(main), centers)
    offsets = np.clip(centers[:, None] + short.OFFSETS, 0, len(main) - 1)
    usable = ~padding & (tokens[:, :, 3] > .5)
    views = []
    for raw in (main, alternate):
        extra = np.zeros_like(tokens)
        if condition != 'frame_only':
            extra[:, :, 0] = np.where(usable, raw[offsets, 70], 0.)
        views.append(np.concatenate((tokens, extra), axis=-1))
    return views, padding


def check_objectives():
    # The first frame has a known detuning, the second has known zero.
    targets = torch.tensor([[-.5, 0.]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    batch = dict(mask=mask, detuning=-targets, voiced={'A': mask, 'C': mask})
    values = {'A': [[.10, .20]], 'B': [[.20, .10]],
              'C': [[-.30, .40]], 'D': [[-.40, .30]]}
    predictions = {name: torch.tensor(value, requires_grad=True) for name, value in values.items()}
    full, parts, _ = objective(predictions, batch, 'bpdr')
    augmentation, _, _ = objective(predictions, batch, 'augmentation_only')
    frame_only, _, _ = objective(predictions, batch, 'frame_only')
    torch.testing.assert_close(augmentation, frame_only)
    torch.testing.assert_close(full, augmentation + parts['boundary'] + parts['detuning'])
    for key in ('boundary', 'detuning'):
        gradients = torch.autograd.grad(parts[key], tuple(predictions.values()), retain_graph=True)
        assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    return {key: float(parts[key].detach()) for key in ('boundary', 'detuning')}


def preflight():
    data = parent.DATA
    receipt = parent.read(data / 'integrity_receipt.json')
    assert receipt['train_pairs_sha256'] == parent.sha(data / 'train_pairs.json')
    assert receipt['input_hashes_sha256'] == parent.sha(data / 'input_hashes.json')
    hashes = parent.read(data / 'input_hashes.json')
    row = next(row for row in parent.read(data / 'train_pairs.json') if 'excluded' not in row)
    assert row['split'] == 'train'
    for key in ('analysis_C', 'target_path'):
        assert hashes[row[key]] == parent.sha(row[key])
    with np.load(row['analysis_C']) as archive:
        main_view, shifted_view, _ = archive['raw_views']
    with np.load(row['target_path']) as archive:
        centers = np.flatnonzero(archive['mask'] & (archive['target'] != 0))
    controls, control_padding = boundary_tokens(main_view, shifted_view, centers, 'frame_only')
    augmentation, padding = boundary_tokens(main_view, shifted_view, centers, 'augmentation_only')
    full, full_padding = boundary_tokens(main_view, shifted_view, centers, 'bpdr')
    np.testing.assert_array_equal(controls[0], controls[1])
    np.testing.assert_array_equal(padding, control_padding)
    np.testing.assert_array_equal(padding, full_padding)
    for a, b, control in zip(augmentation, full, controls):
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[:, :, :8], control[:, :, :8])
        assert np.count_nonzero(a[:, :, 8][padding | (a[:, :, 3] <= .5)]) == 0
    assert np.all(np.any(full[0][:, :, 8] != full[1][:, :, 8], axis=1))
    losses = check_objectives()
    print('SHORT_OBJECTIVE_PREFLIGHT_PASS', len(centers), losses, flush=True)


def declare():
    base, rows, originals = parent.declare()
    paths = [Path(__file__), Path(short_experiment.__file__), Path(short.__file__),
             Path(objective.__code__.co_filename), Path(parent.paired.__file__),
             parent.ROOT / 'declaration.json', parent.DATA / 'integrity_receipt.json']
    for fold in range(1, 5):
        folder = parent.ROOT / f'fold_{fold}/upper_tail_preservation'
        paths.extend((folder / 'physical_trace.npy', folder / 'fit_receipt.json'))
    declaration = parent.norm(dict(
        **{key: base[key] for key in ('folds', 'seed', 'config', 'optimizer', 'updates')},
        conditions=CONDITIONS, warmup_updates=200, fits=12, physical_attempts=len(rows),
        original_attempts=len(originals), loss_weights=dict(boundary=1., detuning=1., smoothness=.1),
        loss='Shared anchor and smoothness, plus boundary and counter-detuning for full BPDR after warmup.',
        sampling='Frozen V68 trace plus next center. Mask unavailable next frames.',
        selection='Existing raw gates, lower held counter-detuning error than both controls.',
        source_hashes={str(path): sha(path) for path in paths}, production_eligible=False,
    ))
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == declaration
    else:
        write(path, declaration)
    return declaration, rows, originals


def load_records(rows, originals):
    data = parent.paired.records(rows, originals)
    original_paths = {row['source_id']: row['analysis_A'] for row in originals if 'excluded' not in row}
    for record in data:
        path = original_paths[record['source_id']] if record['kind'] == 'original' else rows[record['pair_index']]['analysis_C']
        with np.load(path) as archive:
            main, alternate, _ = archive['raw_views']
            np.testing.assert_array_equal(main[:, :70], alternate[:, :70])
            record['boundaries'] = np.stack((main[:, 70], alternate[:, 70]))
    return data


def select_tokens(record, centers, condition):
    acoustic, pad = record['short_tokens'][centers], record['short_pad'][centers]
    positions = np.clip(centers[:, None] + short.OFFSETS, 0, len(record['raw']) - 1)
    extra = np.zeros((2, *acoustic.shape), dtype=acoustic.dtype)
    if condition != 'frame_only':
        extra[..., 0] = np.where(~pad & (acoustic[:, :, 3] > .5),
                                 record['boundaries'][:, positions], 0.)
    return np.concatenate((np.broadcast_to(acoustic, extra.shape), extra), axis=-1), pad


def training_batch(train, originals, queries, condition):
    views, pads, targets, masks, voiced_a, voiced_c = [], [], [], [], [], []
    for index, center in queries:
        altered = train[index]
        clean = originals[altered['source_id']]
        assert len(clean['raw']) == len(altered['raw'])
        following = center + 1
        present = np.array([True, following < len(altered['raw'])])
        centers = np.array([center, min(following, len(altered['raw']) - 1)])
        a, a_pad = select_tokens(clean, centers, condition)
        c, c_pad = select_tokens(altered, centers, condition)
        views.append(np.concatenate((a, c), axis=0))
        pads.append(np.stack((a_pad, a_pad, c_pad, c_pad)))
        targets.append(altered['target'][centers])
        masks.append(present & altered['mask'][centers])
        voiced_a.append(present & (clean['raw'][centers, 66] > .5))
        voiced_c.append(present & (altered['raw'][centers, 66] > .5))
    tokens = torch.from_numpy(np.stack(views)).flatten(0, 2)
    padding = torch.from_numpy(np.stack(pads)).flatten(0, 2)
    batch = dict(mask=torch.from_numpy(np.stack(masks)),
                 detuning=-torch.from_numpy(np.stack(targets)) / 30.,
                 voiced={'A': torch.from_numpy(np.stack(voiced_a)), 'C': torch.from_numpy(np.stack(voiced_c))})
    return tokens, padding, batch


def fit(model, train, trace, declaration, fold, condition, smoke=False):
    optimizer = torch.optim.AdamW(model.parameters(), **declaration['optimizer'])
    originals = {record['source_id']: record for record in train if record['kind'] == 'original'}
    model.train()
    for update, queries in enumerate(trace, start=1):
        tokens, padding, batch = training_batch(train, originals, queries, condition)
        output = model(tokens, padding).reshape(len(queries), 4, 2) / 30.
        predictions = dict(zip('ABCD', output.unbind(dim=1)))
        response = smoke or update > declaration['warmup_updates']
        loss, _, _ = objective(predictions, batch, condition, response=response)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        assert not smoke or norm > 0
        optimizer.step()
        if smoke or update % 200 == 0:
            print('SHORT_OBJECTIVE_FIT', fold, condition, update, float(loss.detach()), flush=True)


def predict(model, record, condition):
    result = []
    with torch.inference_mode():
        for start in range(0, len(record['raw']), 128):
            centers = np.arange(start, min(start + 128, len(record['raw'])))
            views, padding = select_tokens(record, centers, condition)
            tokens = torch.from_numpy(views).flatten(0, 1)
            pads = torch.from_numpy(np.tile(padding, (2, 1)))
            result.append(model(tokens, pads).numpy().reshape(2, len(centers)))
    prediction = np.concatenate(result, axis=1)
    prediction[:, record['raw'][:, 66] <= .5] = 0.
    assert np.isfinite(prediction).all()
    return prediction


def score(records, predictions):
    originals = {r['source_id']: r for r in records if r['kind'] == 'original'}
    grouped = {}
    for record in records:
        main, alternate = predictions[record['id']]
        mask, target = record['mask'], record['target']
        if record['kind'] == 'original':
            continue
        clean, clean_alt = predictions[originals[record['source_id']]['id']]
        groups = grouped.setdefault(record['group'], {'positive': [], 'known_zero': []})
        for label, selected in [('positive', mask & (target != 0)), ('known_zero', mask & (target == 0))]:
            if selected.any():
                boundary = .5 * (abs(main - alternate) + abs(clean - clean_alt))
                response = .5 * (abs(main - clean - target) + abs(alternate - clean_alt - target))
                groups[label].append([float(boundary[selected].mean()), float(response[selected].mean())])
    metrics = {}
    for label in ('positive', 'known_zero'):
        per_group = {name: np.mean(group[label], axis=0).tolist() for name, group in grouped.items() if group[label]}
        boundary, response = np.mean(list(per_group.values()), axis=0)
        metrics[label] = dict(boundary_cents=float(boundary), counter_detuning_error_cents=float(response),
                              by_group_boundary_and_response=per_group)
    return dict(raw=parent.paired.previous.summarize(records, {key: value[0] for key, value in predictions.items()}),
                **metrics)


def run(declaration, data, mode):
    summaries = {}
    for condition in CONDITIONS:
        predictions = {}
        for fold, groups in enumerate(declaration['folds'], start=1):
            train = [r for r in data if r['group'] not in groups]
            held = [r for r in data if r['group'] in groups]
            assert not {r['source_id'] for r in train} & {r['source_id'] for r in held}
            parent_fit = parent.ROOT / f'fold_{fold}/upper_tail_preservation'
            trace = np.load(parent_fit / 'physical_trace.npy')
            digest = hashlib.sha256(trace.tobytes()).hexdigest()
            assert digest == read(parent_fit / 'fit_receipt.json')['physical_trace_sha256']
            expected, _ = parent.paired.previous.sample_trace(train, declaration)
            np.testing.assert_array_equal(trace, expected)
            model = short_experiment.new_model(declaration)
            if mode == 'smoke':
                fit(model, train, trace[:1], declaration, fold, condition, smoke=True)
                example = predict(model.eval(), held[0], condition)
                assert example.shape == (2, len(held[0]['raw']))
                break
            folder = ROOT / f'fold_{fold}' / condition
            folder.mkdir(parents=True, exist_ok=True)
            checkpoint, receipt = folder / 'checkpoint.pt', folder / 'fit_receipt.json'
            binding = dict(declaration_sha256=sha(ROOT / 'declaration.json'), trace_sha256=digest,
                           training_ids=[r['id'] for r in train], held_ids=[r['id'] for r in held])
            if receipt.exists() or mode == 'verify':
                saved = read(receipt)
                assert all(saved[key] == value for key, value in binding.items())
                assert sha(checkpoint) == saved['checkpoint_sha256']
                model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
            else:
                started = time.perf_counter()
                fit(model, train, trace, declaration, fold, condition)
                torch.save(model.state_dict(), checkpoint)
                write(receipt, dict(binding, checkpoint_sha256=sha(checkpoint), fit_seconds=time.perf_counter() - started))
            model.eval()
            predictions.update({r['id']: predict(model, r, condition) for r in held})
        if mode == 'smoke':
            continue
        assert set(predictions) == {r['id'] for r in data}
        path = ROOT / f'{condition}_predictions.npz'
        if mode == 'verify':
            with np.load(path) as archived:
                for key, value in predictions.items():
                    np.testing.assert_array_equal(value, archived[key])
        else:
            np.savez_compressed(path, **predictions)
        summaries[condition] = score(data, predictions)
    if mode == 'smoke':
        print('SHORT_OBJECTIVE_SMOKE_PASS', flush=True)
        return
    full = summaries['bpdr']
    qualifies = full['raw']['raw_gate_passed'] and all(
        full['positive']['counter_detuning_error_cents'] < summaries[condition]['positive']['counter_detuning_error_cents']
        for condition in ('augmentation_only', 'frame_only')
    )
    result = parent.norm(dict(arms=summaries, advance_to_delivery=qualifies, production_eligible=False,
                              declaration_sha256=sha(ROOT / 'declaration.json')))
    path = ROOT / 'results.json'
    if mode == 'verify':
        assert read(path) == result
        write(ROOT / 'integrity_receipt.json', dict(results_sha256=sha(path), checkpoint_reloads=12))
    else:
        write(path, result)
    print('SHORT_OBJECTIVE_RESULTS', qualifies, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('preflight', 'smoke', 'fit', 'verify'), default='preflight')
    args = parser.parse_args()
    torch.set_num_threads(2)
    preflight()
    if args.mode != 'preflight':
        declaration, rows, originals = declare()
        run(declaration, load_records(rows, originals), args.mode)


if __name__ == '__main__':
    main()
