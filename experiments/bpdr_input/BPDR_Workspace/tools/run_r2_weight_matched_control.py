"""Candidate R2 (Plan Section 3.2): Weight-matched control run with gate open (+2.0 activity bias).

Matches BPDR's post-warmup effective positive-anchor gradient weight from
results/transformer_usability/2026-11-13_loss_redundancy/results.json:
  pos_weight = 1.0 + (0.2066812813282013 * 0.9206768274307251) / (0.5 * 0.6499825119972229) = 1.5855
Applied to `losses[0]` (pos_loss) in balanced_anchor for `augmentation_only` and `frame_only`
after warmup (update > warmup_updates=200), matching BPDR's response schedule.

Supports memory-safe one-fit-per-subprocess execution on 15 GB hosts:
  --mode declare
  --mode smoke
  --mode fit --condition <arm> --fold <1..4>
  --mode score --condition <arm>
  --mode verify
  --mode all
"""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import time

import numpy as np
import torch

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / 'tools'), str(WS / 'source')]

import compare_balanced_short_objectives as study
import compare_gate_open_objectives as gate_open

previous = study.previous
ROOT = study.ROOT.parent / '2026-11-17_r2_weight_matched_control'
POS_WEIGHT_MATCHED = 1.5855
FIT_HEALTH_ANCHOR_THRESHOLD = 0.30
FIT_HEALTH_REQUIRED = 11

read, write, sha = study.read, study.write, study.sha
KINDS = study.KINDS


def gate_open_model(declaration):
    model = gate_open._original_new_model(declaration)
    with torch.no_grad():
        model.activity.bias.fill_(2.0)
    return model


def weighted_balanced_anchor(predictions, batch, pos_weight=1.0):
    """Balanced nonzero/zero anchor loss with optional positive-anchor weight multiplier."""
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
    return 0.5 * (pos_weight * losses[0] + losses[1])


def check_duplicate_4ulp(a, b):
    assert a.dtype == b.dtype == np.float32
    np.testing.assert_array_max_ulp(a, b, maxulp=4)


def declare():
    parent, rows, originals = previous.parent.declare()
    replay = ROOT.parent / '2026-11-07_balanced_anchor_fit_probe/integrity_receipt.json'
    assert read(replay)['results_sha256'] == sha(replay.parent / 'results.json')
    redundancy = ROOT.parent / '2026-11-13_loss_redundancy/results.json'
    assert redundancy.exists()
    paths = [
        Path(__file__),
        Path(gate_open.__file__),
        Path(study.__file__),
        Path(previous.__file__),
        Path(study.boundaries.__file__),
        Path(study.balanced_anchor.__code__.co_filename),
        Path(previous.objective.__code__.co_filename),
        Path(previous.short_experiment.__file__),
        Path(previous.short.__file__),
        Path(study.boundaries.corrupt_boundaries.__code__.co_filename),
        Path(study.boundaries.game_boundary_track.__code__.co_filename),
        previous.parent.DATA / 'integrity_receipt.json',
        replay,
        redundancy,
        study.boundaries.ROOT / 'results.json',
    ]
    for fold in range(1, 5):
        folder = previous.parent.ROOT / f'fold_{fold}/upper_tail_preservation'
        paths.extend((folder / 'physical_trace.npy', folder / 'fit_receipt.json'))
    declaration = previous.parent.norm(
        dict(
            **{key: parent[key] for key in ('folds', 'seed', 'updates', 'config', 'optimizer')},
            conditions=previous.CONDITIONS,
            boundary_kinds=KINDS,
            warmup_updates=200,
            fits=12,
            activity_bias_init=2.0,
            pos_weight_matched=POS_WEIGHT_MATCHED,
            pos_weight_derivation='1 + (0.206681 * 0.920677) / (0.5 * 0.649983) = 1.5855 from 2026-11-13_loss_redundancy/results.json (init_gate_open)',
            pos_weight_schedule='1.0 during warmup (updates 1..200), 1.5855 after warmup (updates 201..1200) for augmentation_only and frame_only; 1.0 throughout for bpdr (matching BPDR response schedule).',
            fit_health_rule=f'{FIT_HEALTH_REQUIRED}-of-12 fits with final logged unweighted balanced anchor < {FIT_HEALTH_ANCHOR_THRESHOLD:.2f} (zero-output baseline ~0.375-0.47).',
            duplicate_check='4 float32 ULPs on frame_only duplicate-view output.',
            source_hashes={str(path): sha(path) for path in paths},
            anchor='Half nonzero/zero mass, per-example class means, then eligible-example mean; controls apply pos_weight=1.5855 to pos_loss after warmup.',
            loss='Common balanced anchors (weight-matched after warmup for controls) and original smoothness; original BPDR response terms after warmup.',
            boundary_schedule='split,merge,jitter repeated identically across arms',
            physical_attempts=len(rows),
            original_attempts=len(originals),
            construction_excluded=707,
            selection='11-of-12 fit health (<0.30 unweighted anchor); existing raw gates for every kind; lower mean positive response error than both controls; lower boundary error than augmentation for each kind.',
            stopping='Twelve fits and exact replay; Candidate R2 honest negative test (no further gate/schedule variants).',
            scope='Inspected training-source cross-validation. No fresh confirmation or production qualification.',
            production_eligible=False,
        )
    )
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == declaration, 'Declaration changed after creation'
    else:
        write(path, declaration)
    return declaration, rows, originals


def load_fold_trace(fold, train, declaration):
    parent = previous.parent.ROOT / f'fold_{fold}/upper_tail_preservation'
    trace = np.load(parent / 'physical_trace.npy')
    digest = hashlib.sha256(trace.tobytes()).hexdigest()
    assert read(parent / 'fit_receipt.json')['physical_trace_sha256'] == digest
    expected, _ = previous.parent.paired.previous.sample_trace(train, declaration)
    np.testing.assert_array_equal(trace, expected)
    return trace, digest


def fit_single(model, train, trace, declaration, condition, fold, smoke=False):
    optimizer = torch.optim.AdamW(model.parameters(), **declaration['optimizer'])
    originals = {r['source_id']: r for r in train if r['kind'] == 'original'}
    history = []
    for update, queries in enumerate(trace, 1):
        kind = KINDS[(update - 1) % len(KINDS)]
        study.set_kind(train, kind)
        tokens, padding, batch = previous.training_batch(train, originals, queries, condition)
        model.train()
        output = model(tokens, padding).reshape(len(queries), 4, 2) / 30.0
        predictions = dict(zip('ABCD', output.unbind(1)))
        post_warmup = smoke or update > declaration['warmup_updates']
        loss, parts, _ = previous.objective(predictions, batch, condition, response=post_warmup)
        pos_w = declaration['pos_weight_matched'] if (condition != 'bpdr' and post_warmup) else 1.0
        weighted_anchor = weighted_balanced_anchor(predictions, batch, pos_weight=pos_w)
        loss = loss - parts['anchor'] + weighted_anchor
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if smoke or update % 200 == 0:
            unweighted_anchor = weighted_balanced_anchor(predictions, batch, pos_weight=1.0)
            history.append(
                dict(
                    update=update,
                    corruption=kind,
                    pos_weight=float(pos_w),
                    loss=float(loss.detach()),
                    anchor=float(unweighted_anchor.detach()),
                    weighted_anchor=float(weighted_anchor.detach()),
                )
            )
            print('R2_FIT', fold, condition, history[-1], flush=True)
    return history


def run_smoke(declaration, data):
    for condition in declaration['conditions']:
        held_groups = declaration['folds'][0]
        train = [r for r in data if r['group'] not in held_groups]
        held = [r for r in data if r['group'] in held_groups]
        trace, _ = load_fold_trace(1, train, declaration)
        model = gate_open_model(declaration)
        fit_single(model, train, trace[:3], declaration, condition, 1, smoke=True)
        for kind in KINDS:
            study.set_kind(held, kind)
            actual = previous.predict(model.eval(), held[0], condition)
            assert actual.shape == (2, len(held[0]['raw']))
            if condition == 'frame_only':
                check_duplicate_4ulp(actual[0], actual[1])
        del model
        gc.collect()
    print('R2_WEIGHT_MATCHED_SMOKE_PASS', flush=True)


def run_fit_one(declaration, data, condition, fold):
    assert condition in declaration['conditions']
    assert 1 <= fold <= len(declaration['folds'])
    held_groups = declaration['folds'][fold - 1]
    train = [r for r in data if r['group'] not in held_groups]
    held = [r for r in data if r['group'] in held_groups]
    assert not {r['source_id'] for r in train} & {r['source_id'] for r in held}
    trace, digest = load_fold_trace(fold, train, declaration)

    folder = ROOT / f'fold_{fold}' / condition
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint, receipt_path = folder / 'checkpoint.pt', folder / 'fit_receipt.json'
    binding = dict(
        declaration_sha256=sha(ROOT / 'declaration.json'),
        trace_sha256=digest,
        training_ids=[r['id'] for r in train],
        held_ids=[r['id'] for r in held],
    )
    if receipt_path.exists():
        receipt = read(receipt_path)
        assert all(receipt[k] == v for k, v in binding.items())
        assert receipt['checkpoint_sha256'] == sha(checkpoint)
        print(f'R2_FIT_SKIP_EXISTING fold={fold} condition={condition} final_anchor={receipt["history"][-1]["anchor"]:.4f}', flush=True)
        return

    model = gate_open_model(declaration)
    started = time.perf_counter()
    history = fit_single(model, train, trace, declaration, condition, fold, smoke=False)
    elapsed = time.perf_counter() - started
    torch.save(model.state_dict(), checkpoint)
    final_anchor = float(history[-1]['anchor'])
    healthy = bool(final_anchor < FIT_HEALTH_ANCHOR_THRESHOLD)
    write(
        receipt_path,
        dict(
            binding,
            checkpoint_sha256=sha(checkpoint),
            history=history,
            final_anchor=final_anchor,
            healthy_anchor_lt_030=healthy,
            seconds=elapsed,
        ),
    )
    print(f'R2_FIT_DONE fold={fold} condition={condition} sec={elapsed:.1f} final_anchor={final_anchor:.4f} healthy={healthy}', flush=True)


def evaluate_condition(declaration, data, condition, verify=False):
    predictions = {kind: {} for kind in KINDS}
    fold_health = {}
    for fold, held_groups in enumerate(declaration['folds'], 1):
        train = [r for r in data if r['group'] not in held_groups]
        held = [r for r in data if r['group'] in held_groups]
        assert not {r['source_id'] for r in train} & {r['source_id'] for r in held}
        _, digest = load_fold_trace(fold, train, declaration)
        folder = ROOT / f'fold_{fold}' / condition
        checkpoint, receipt_path = folder / 'checkpoint.pt', folder / 'fit_receipt.json'
        binding = dict(
            declaration_sha256=sha(ROOT / 'declaration.json'),
            trace_sha256=digest,
            training_ids=[r['id'] for r in train],
            held_ids=[r['id'] for r in held],
        )
        receipt = read(receipt_path)
        assert all(receipt[k] == v for k, v in binding.items())
        assert receipt['checkpoint_sha256'] == sha(checkpoint)
        final_anchor = float(receipt['history'][-1]['anchor'])
        fold_health[f'fold_{fold}'] = dict(
            final_anchor=final_anchor,
            healthy=bool(final_anchor < FIT_HEALTH_ANCHOR_THRESHOLD),
            seconds=float(receipt['seconds']),
        )

        model = gate_open_model(declaration)
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
        model.eval()
        for kind in KINDS:
            study.set_kind(held, kind)
            inferred = {r['id']: previous.predict(model, r, condition) for r in held}
            if condition == 'frame_only':
                for value in inferred.values():
                    check_duplicate_4ulp(value[0], value[1])
            predictions[kind].update(inferred)
        del model
        gc.collect()

    summary = {}
    for kind, values in predictions.items():
        assert set(values) == {r['id'] for r in data}
        path = ROOT / f'{condition}_{kind}_predictions.npz'
        if verify:
            with np.load(path) as saved:
                for key, value in values.items():
                    np.testing.assert_array_equal(value, saved[key])
        else:
            np.savez_compressed(path, **values)
        summary[kind] = previous.score(data, values)
    del predictions
    gc.collect()
    return summary, fold_health


def run_verify(declaration, data, write_results=False):
    summaries = {}
    health_report = {}
    healthy_count = 0
    for condition in declaration['conditions']:
        npz_exists = all((ROOT / f'{condition}_{kind}_predictions.npz').exists() for kind in KINDS)
        summary, fold_health = evaluate_condition(declaration, data, condition, verify=npz_exists)
        summaries[condition] = summary
        health_report[condition] = fold_health
        healthy_count += sum(1 for item in fold_health.values() if item['healthy'])

    fit_health_passed = bool(healthy_count >= FIT_HEALTH_REQUIRED)
    response = {
        arm: float(np.mean([summaries[arm][kind]['positive']['counter_detuning_error_cents'] for kind in KINDS]))
        for arm in summaries
    }
    advance = bool(
        fit_health_passed
        and all(summaries['bpdr'][kind]['raw']['raw_gate_passed'] for kind in KINDS)
        and all(response['bpdr'] < response[arm] for arm in ('augmentation_only', 'frame_only'))
        and all(
            summaries['bpdr'][kind]['positive']['boundary_cents']
            < summaries['augmentation_only'][kind]['positive']['boundary_cents']
            for kind in KINDS
        )
    )
    result = previous.parent.norm(
        dict(
            arms=summaries,
            fit_health=dict(
                healthy_fits=healthy_count,
                total_fits=12,
                required=FIT_HEALTH_REQUIRED,
                threshold_anchor_lt=FIT_HEALTH_ANCHOR_THRESHOLD,
                passed=fit_health_passed,
                by_arm=health_report,
            ),
            advance_to_delivery=advance,
            declaration_sha256=sha(ROOT / 'declaration.json'),
            production_eligible=False,
        )
    )
    path = ROOT / 'results.json'
    if write_results or not path.exists():
        write(path, result)
        write(
            ROOT / 'fit_receipt.json',
            dict(
                declaration_sha256=sha(ROOT / 'declaration.json'),
                healthy_fits=healthy_count,
                total_fits=12,
                fit_health_passed=fit_health_passed,
                by_arm=health_report,
            ),
        )
    else:
        assert read(path) == result
    write(
        ROOT / 'integrity_receipt.json',
        dict(
            results_sha256=sha(path),
            checkpoint_reloads=12,
            prediction_arrays_replayed=4482,
            healthy_fits=healthy_count,
            fit_health_passed=fit_health_passed,
        ),
    )
    print(f'R2_VERIFY_COMPLETE healthy={healthy_count}/12 fit_health_passed={fit_health_passed} advance={advance}', flush=True)


def run_all():
    script = str(Path(__file__).resolve())
    run_cmd = lambda *extra: subprocess.run([sys.executable, '-u', script, *extra], check=True)
    run_cmd('--mode', 'declare')
    run_cmd('--mode', 'smoke')
    for condition in previous.CONDITIONS:
        for fold in (1, 2, 3, 4):
            run_cmd('--mode', 'fit', '--condition', condition, '--fold', str(fold))
        run_cmd('--mode', 'score', '--condition', condition)
    run_cmd('--mode', 'verify')
    run_cmd('--mode', 'verify')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('declare', 'smoke', 'fit', 'score', 'verify', 'all'), default='smoke')
    parser.add_argument('--condition', choices=previous.CONDITIONS, default=None)
    parser.add_argument('--fold', type=int, choices=(1, 2, 3, 4), default=None)
    args = parser.parse_args()

    if args.mode == 'all':
        run_all()
        return

    torch.set_num_threads(2)
    previous.preflight()
    declaration, rows, originals = declare()
    if args.mode == 'declare':
        print('R2_DECLARED', ROOT / 'declaration.json', flush=True)
        return

    data = study.records(rows, originals)
    if args.mode == 'smoke':
        run_smoke(declaration, data)
    elif args.mode == 'fit':
        if args.condition is None or args.fold is None:
            raise ValueError('--fit requires --condition and --fold')
        run_fit_one(declaration, data, args.condition, args.fold)
    elif args.mode == 'score':
        if args.condition is None:
            raise ValueError('--score requires --condition')
        evaluate_condition(declaration, data, args.condition, verify=False)
        print(f'R2_SCORE_SAVED condition={args.condition}', flush=True)
    elif args.mode == 'verify':
        write_flag = not (ROOT / 'results.json').exists()
        run_verify(declaration, data, write_results=write_flag)


if __name__ == '__main__':
    main()
