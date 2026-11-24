"""Corrected matched objectives with equal anchor mass and three boundary errors."""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

import audit_short_boundary_corruptions as boundaries
import compare_short_objectives as previous
from probe_balanced_anchor_fit import balanced_anchor


ROOT = previous.ROOT.parent / '2026-11-07_balanced_short_objectives'
KINDS = boundaries.KINDS
read, write, sha = previous.read, previous.write, previous.sha


def declare():
    parent, rows, originals = previous.parent.declare()
    replay = ROOT.parent / '2026-11-07_balanced_anchor_fit_probe/integrity_receipt.json'
    assert read(replay)['results_sha256'] == sha(replay.parent / 'results.json')
    paths = [Path(__file__), Path(previous.__file__), Path(boundaries.__file__),
             Path(balanced_anchor.__code__.co_filename), Path(previous.objective.__code__.co_filename),
             Path(previous.short_experiment.__file__), Path(previous.short.__file__),
             Path(boundaries.corrupt_boundaries.__code__.co_filename),
             Path(boundaries.game_boundary_track.__code__.co_filename),
             previous.parent.DATA / 'integrity_receipt.json', replay, boundaries.ROOT / 'results.json']
    for fold in range(1, 5):
        folder = previous.parent.ROOT / f'fold_{fold}/upper_tail_preservation'
        paths.extend((folder / 'physical_trace.npy', folder / 'fit_receipt.json'))
    declaration = previous.parent.norm(dict(
        **{key: parent[key] for key in ('folds', 'seed', 'updates', 'config', 'optimizer')},
        conditions=previous.CONDITIONS, boundary_kinds=KINDS, warmup_updates=200, fits=12,
        source_hashes={str(path): sha(path) for path in paths},
        anchor='Half nonzero/zero mass, per-example class means, then eligible-example mean.',
        loss='Common balanced anchors and original smoothness; original BPDR response terms after warmup.',
        boundary_schedule='split,merge,jitter repeated identically across arms',
        physical_attempts=len(rows), original_attempts=len(originals), construction_excluded=707,
        selection='Existing raw gates for every kind; lower mean positive response error than both controls; lower boundary error than augmentation for each kind.',
        stopping='Twelve fits and exact replay; no seed, weight or checkpoint sweep.',
        scope='Inspected training-source cross-validation. No fresh confirmation or production qualification.',
        production_eligible=False,
    ))
    ROOT.mkdir(exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == declaration
    else:
        write(path, declaration)
    return declaration, rows, originals


def records(rows, originals):
    data = previous.load_records(rows, originals)
    for record in data:
        raw = record['raw']
        record['boundary_bank'] = {}
        for kind, variant in boundaries.variants(raw).items():
            assert 'excluded' not in variant, (record['id'], kind)
            record['boundary_bank'][kind] = np.stack((raw[:, 70], variant['raw'][:, 70]))
    return data


def set_kind(data, kind):
    for record in data:
        record['boundaries'] = record['boundary_bank'][kind]


def fit(model, train, trace, declaration, condition, fold, smoke=False):
    optimizer = torch.optim.AdamW(model.parameters(), **declaration['optimizer'])
    originals = {r['source_id']: r for r in train if r['kind'] == 'original'}
    history = []
    for update, queries in enumerate(trace, 1):
        kind = KINDS[(update - 1) % len(KINDS)]
        set_kind(train, kind)
        tokens, padding, batch = previous.training_batch(train, originals, queries, condition)
        model.train()
        output = model(tokens, padding).reshape(len(queries), 4, 2) / 30.
        predictions = dict(zip('ABCD', output.unbind(1)))
        loss, parts, _ = previous.objective(
            predictions, batch, condition, response=smoke or update > declaration['warmup_updates'])
        anchor = balanced_anchor(predictions, batch)
        loss = loss - parts['anchor'] + anchor
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if smoke or update % 200 == 0:
            history.append(dict(update=update, corruption=kind, loss=float(loss.detach()),
                                anchor=float(anchor.detach())))
            print('BALANCED_OBJECTIVE_FIT', fold, condition, history[-1], flush=True)
    return history


def run(declaration, data, mode):
    summaries = {}
    for condition in declaration['conditions']:
        predictions = {kind: {} for kind in KINDS}
        for fold, held_groups in enumerate(declaration['folds'], 1):
            train = [r for r in data if r['group'] not in held_groups]
            held = [r for r in data if r['group'] in held_groups]
            assert not {r['source_id'] for r in train} & {r['source_id'] for r in held}
            parent = previous.parent.ROOT / f'fold_{fold}/upper_tail_preservation'
            trace = np.load(parent / 'physical_trace.npy')
            digest = hashlib.sha256(trace.tobytes()).hexdigest()
            assert read(parent / 'fit_receipt.json')['physical_trace_sha256'] == digest
            expected, _ = previous.parent.paired.previous.sample_trace(train, declaration)
            np.testing.assert_array_equal(trace, expected)
            model = previous.short_experiment.new_model(declaration)
            if mode == 'smoke':
                fit(model, train, trace[:3], declaration, condition, fold, smoke=True)
                for kind in KINDS:
                    set_kind(held, kind)
                    actual = previous.predict(model.eval(), held[0], condition)
                    assert actual.shape == (2, len(held[0]['raw']))
                    if condition == 'frame_only':
                        np.testing.assert_array_equal(actual[0], actual[1])
                break
            folder = ROOT / f'fold_{fold}' / condition
            folder.mkdir(parents=True, exist_ok=True)
            checkpoint, receipt_path = folder / 'checkpoint.pt', folder / 'fit_receipt.json'
            binding = dict(declaration_sha256=sha(ROOT / 'declaration.json'), trace_sha256=digest,
                           training_ids=[r['id'] for r in train], held_ids=[r['id'] for r in held])
            if receipt_path.exists() or mode == 'verify':
                receipt = read(receipt_path)
                assert all(receipt[key] == value for key, value in binding.items())
                assert receipt['checkpoint_sha256'] == sha(checkpoint)
                model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
            else:
                started = time.perf_counter()
                history = fit(model, train, trace, declaration, condition, fold)
                torch.save(model.state_dict(), checkpoint)
                write(receipt_path, dict(binding, checkpoint_sha256=sha(checkpoint),
                                         history=history, seconds=time.perf_counter() - started))
            model.eval()
            for kind in KINDS:
                set_kind(held, kind)
                inferred = {r['id']: previous.predict(model, r, condition) for r in held}
                if condition == 'frame_only':
                    for value in inferred.values():
                        np.testing.assert_array_equal(value[0], value[1])
                predictions[kind].update(inferred)
        if mode == 'smoke':
            continue
        summaries[condition] = {}
        for kind, values in predictions.items():
            assert set(values) == {r['id'] for r in data}
            path = ROOT / f'{condition}_{kind}_predictions.npz'
            if mode == 'verify':
                with np.load(path) as saved:
                    for key, value in values.items():
                        np.testing.assert_array_equal(value, saved[key])
            else:
                np.savez_compressed(path, **values)
            summaries[condition][kind] = previous.score(data, values)
    if mode == 'smoke':
        print('BALANCED_OBJECTIVE_SMOKE_PASS', flush=True)
        return
    response = {arm: float(np.mean([summaries[arm][kind]['positive']['counter_detuning_error_cents']
                                   for kind in KINDS])) for arm in summaries}
    advance = (
        all(summaries['bpdr'][kind]['raw']['raw_gate_passed'] for kind in KINDS)
        and all(response['bpdr'] < response[arm] for arm in ('augmentation_only', 'frame_only'))
        and all(summaries['bpdr'][kind]['positive']['boundary_cents'] <
                summaries['augmentation_only'][kind]['positive']['boundary_cents'] for kind in KINDS)
    )
    result = previous.parent.norm(dict(arms=summaries, advance_to_delivery=advance,
                                       declaration_sha256=sha(ROOT / 'declaration.json'), production_eligible=False))
    path = ROOT / 'results.json'
    if mode == 'verify':
        assert read(path) == result
        write(ROOT / 'integrity_receipt.json', dict(results_sha256=sha(path), checkpoint_reloads=12,
                                                   prediction_arrays_replayed=4482))
    else:
        write(path, result)
    print('BALANCED_OBJECTIVE_COMPARISON_COMPLETE', advance, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'fit', 'verify'), default='smoke')
    args = parser.parse_args()
    torch.set_num_threads(2)
    previous.preflight()
    declaration, rows, originals = declare()
    run(declaration, records(rows, originals), args.mode)


if __name__ == '__main__':
    main()
