"""Matched short-phrase spectral context experiment on inspected training sources."""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import compare_tail_preservation as parent


ROOT = parent.ROOT.parent / '2026-11-06_short_spectral_context_v2'
ARMS = ('zero_context', 'spectral_context')
read, write, norm, sha = parent.read, parent.write, parent.norm, parent.sha
short = parent.model_module


class SpectralContrast(short.ShortPhraseTransformer):
    def __init__(self, config):
        super().__init__(config)
        shared = self.project
        self.project = torch.nn.Linear(16, config.hidden)
        with torch.no_grad():
            self.project.weight[:, :8].copy_(shared.weight)
            self.project.bias.copy_(shared.bias)

    def forward(self, tokens, padding=None, details=False):
        encoded = self.encoder(
            self.project(tokens) + self.positions, src_key_padding_mask=padding
        )
        hidden = self.norm(encoded[:, self.center])
        amplitude = 30 * torch.tanh(self.regression(hidden).squeeze(-1))
        logits = self.activity(hidden).squeeze(-1)
        prediction = amplitude * torch.sigmoid(logits) if self.config.gated else amplitude
        return (prediction, logits, amplitude) if details else prediction


def spectral_channels(record):
    raw = record['raw']
    bands = raw[:, :64].reshape(len(raw), 8, 8).mean(axis=2)
    positions = np.arange(len(raw))[:, None] + short.OFFSETS
    values = (bands[np.clip(positions, 0, len(raw) - 1)] - bands[:, None]) / 20
    np.clip(values, -4, 4, out=values)
    values[record['short_pad'] | (record['short_tokens'][:, :, 3] < .5)] = 0
    return values


def new_model(declaration):
    torch.manual_seed(declaration['seed'])
    return SpectralContrast(short.ShortConfig(**declaration['config']))


def declare():
    base, rows, originals = parent.declare()
    integrity = parent.ROOT / 'integrity_receipt.json'
    assert read(integrity)['results_sha256'] == sha(parent.ROOT / 'results.json')
    paths = [Path(__file__), Path(parent.__file__), Path(short.__file__),
             Path(parent.paired.__file__), Path(parent.tail.__file__),
             Path(parent.paired.previous.__file__), integrity,
             parent.ROOT / 'declaration.json', parent.ROOT / 'upper_tail_preservation_all_predictions.npz']
    for fold in range(1, 5):
        folder = parent.ROOT / f'fold_{fold}/upper_tail_preservation'
        paths.extend([folder / 'physical_trace.npy', folder / 'fit_receipt.json'])
    declaration = norm(dict(
        **{key: base[key] for key in ('folds', 'seed', 'config', 'optimizer', 'updates', 'loss')},
        arms=ARMS, source_hashes={str(path.resolve()): sha(path) for path in paths},
        physical_attempts=len(rows), original_attempts=len(originals),
        changed_variable='Eight zero channels versus eight masked, center-relative pooled log-mel channels.',
        selection='Existing all-group raw gates plus lower held positive singer-macro MAE than both control and V68.',
        stopping='Eight fits maximum. Run independent raw replay before any conditional delivery work.',
        nontraining_sources_accessed=False, production_eligible=False,
    ))
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == declaration
    else:
        write(path, declaration)
    return declaration, rows, originals


def preflight(declaration, records):
    assert sum(record['kind'] == 'corrupted' for record in records) == 445
    assert sum(record['kind'] == 'original' for record in records) == 53
    for record in records:
        values = spectral_channels(record)
        assert values.shape == record['short_tokens'].shape
        assert np.isfinite(values).all()
        assert np.count_nonzero(values[record['short_pad']]) == 0
        assert np.count_nonzero(values[record['short_tokens'][:, :, 3] < .5]) == 0
        assert np.count_nonzero(values[:, short.OFFSETS == 0]) == 0
    torch.manual_seed(declaration['seed'])
    baseline = short.ShortPhraseTransformer(short.ShortConfig(**declaration['config'])).eval()
    control = new_model(declaration).eval()
    with torch.no_grad():
        # A nonzero head makes the parity check sensitive to encoder differences.
        baseline.regression.weight.fill_(.05)
        control.regression.weight.copy_(baseline.regression.weight)
        for record in (records[0], records[-1]):
            tokens = torch.from_numpy(record['short_tokens'][::17])
            padding = torch.from_numpy(record['short_pad'][::17])
            widened = torch.cat((tokens, torch.zeros_like(tokens)), dim=-1)
            for actual, expected in zip(control(widened, padding, True), baseline(tokens, padding, True)):
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    print('SPECTRAL_PREFLIGHT_PASS', len(records), flush=True)


def fit(model, train, trace, declaration, fold, arm):
    optimizer = torch.optim.AdamW(model.parameters(), **declaration['optimizer'])
    original_indices = {r['source_id']: i for i, r in enumerate(train) if r['kind'] == 'original'}
    activity = torch.cat((torch.zeros(12), torch.ones(12), torch.zeros(24)))
    model.train()
    for step, batch in enumerate(trace, start=1):
        clean = [(original_indices[train[i]['source_id']], j) for i, j in batch[:12]]
        packed = clean + list(batch)
        tokens = torch.from_numpy(np.stack([train[i]['experiment_tokens'][j] for i, j in packed]))
        padding = torch.from_numpy(np.stack([train[i]['short_pad'][j] for i, j in packed]))
        target = torch.tensor([train[i]['target'][j] for i, j in batch[:12]]) / 30
        prediction, logits, _ = model(tokens, padding, True)
        clean, altered, zero, original = (prediction / 30).split(12)
        anchor = .5 * (clean.square().mean() + (altered - target).square().mean())
        response = .25 * (altered - clean - target).square().mean()
        loss = anchor + response + parent.tail.upper_tail_preservation(zero, original)
        loss = loss + .05 * F.binary_cross_entropy_with_logits(logits, activity)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 200 == 0:
            print('SPECTRAL_FIT', fold, arm, step, float(loss.detach()), flush=True)


def predict(model, record):
    tokens, padding = record['experiment_tokens'], record['short_pad']
    with torch.inference_mode():
        prediction = np.concatenate([
            model(torch.from_numpy(tokens[start:start + 128]),
                  torch.from_numpy(padding[start:start + 128])).numpy()
            for start in range(0, len(tokens), 128)
        ])
    prediction[record['raw'][:, 66] <= .5] = 0.
    assert np.isfinite(prediction).all() and np.max(abs(prediction)) <= 30.00001
    return prediction


def run(declaration, data, verify=False):
    predictions = {arm: {} for arm in (*ARMS, 'frozen_v68')}
    with np.load(parent.ROOT / 'upper_tail_preservation_all_predictions.npz') as archive:
        predictions['frozen_v68'] = {key: archive[key] for key in archive.files}
    for arm in ARMS:
        for record in data:
            tokens = record['short_tokens']
            extra = spectral_channels(record) if arm == 'spectral_context' else np.zeros_like(tokens)
            record['experiment_tokens'] = np.concatenate((tokens, extra), axis=2)
        for fold, groups in enumerate(declaration['folds'], start=1):
            train = [record for record in data if record['group'] not in groups]
            held = [record for record in data if record['group'] in groups]
            assert not {r['source_id'] for r in train} & {r['source_id'] for r in held}
            original = parent.ROOT / f'fold_{fold}/upper_tail_preservation'
            trace = np.load(original / 'physical_trace.npy')
            trace_hash = hashlib.sha256(trace.tobytes()).hexdigest()
            parent_fit = read(original / 'fit_receipt.json')
            assert trace_hash == parent_fit['physical_trace_sha256']
            expected_trace, _ = parent.paired.previous.sample_trace(train, declaration)
            np.testing.assert_array_equal(trace, expected_trace)
            assert trace.shape == (declaration['updates'], 36, 2)
            folder = ROOT / f'fold_{fold}' / arm
            folder.mkdir(parents=True, exist_ok=True)
            checkpoint, receipt = folder / 'checkpoint.pt', folder / 'fit_receipt.json'
            model = new_model(declaration)
            binding = dict(declaration_sha256=sha(ROOT / 'declaration.json'), trace_sha256=trace_hash,
                           train_ids=[r['id'] for r in train], held_ids=[r['id'] for r in held])
            if receipt.exists() or verify:
                saved = read(receipt)
                assert all(saved[key] == value for key, value in binding.items())
                assert saved['checkpoint_sha256'] == sha(checkpoint)
                model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
            else:
                started = time.perf_counter()
                fit(model, train, trace, declaration, fold, arm)
                torch.save(model.state_dict(), checkpoint)
                write(receipt, dict(binding, checkpoint_sha256=sha(checkpoint),
                                    fit_seconds=time.perf_counter() - started))
            model.eval()
            predictions[arm].update({r['id']: predict(model, r) for r in held})
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fit', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    declaration, rows, originals = declare()
    data = parent.paired.records(rows, originals)
    preflight(declaration, data)
    if not (args.fit or args.verify):
        return
    predictions = run(declaration, data, args.verify)
    summarize = parent.paired.previous.summarize
    arms = {}
    for arm, arrays in predictions.items():
        assert set(arrays) == {r['id'] for r in data}
        path = ROOT / f'{arm}_predictions.npz'
        if args.verify:
            with np.load(path) as stored:
                for key, value in arrays.items():
                    np.testing.assert_array_equal(stored[key], value)
        else:
            np.savez_compressed(path, **arrays)
        subsets = dict(all=data, humming=[r for r in data if r['group'].startswith('HumTrans:')],
                       vocadito=[r for r in data if not r['group'].startswith('HumTrans:')])
        arms[arm] = {name: summarize(subset, arrays) for name, subset in subsets.items()}
        positive_errors = {-1: [], 1: []}
        original_peaks = {}
        known_zero_tail = 0
        for record in data:
            target, mask = record['target'], record['mask']
            prediction = arrays[record['id']]
            if record['kind'] == 'original':
                original_peaks[record['id']] = float(np.max(abs(prediction[mask]))) if mask.any() else None
                continue
            known_zero_tail += int(np.count_nonzero(abs(prediction[mask & (target == 0)]) >= 12))
            for sign, errors in positive_errors.items():
                selected = mask & (target * sign > 0)
                if selected.any():
                    errors.append(float(np.mean(abs(prediction[selected] - target[selected]))))
        arms[arm]['diagnostics'] = dict(
            known_zero_frames_at_least_12_cents=known_zero_tail, original_peak_cents=original_peaks,
            mean_case_error_by_target_direction={
                str(sign): dict(cases=len(errors), mae_cents=float(np.mean(errors)) if errors else None)
                for sign, errors in positive_errors.items()
            },
        )
    spectral = arms['spectral_context']['all']
    passed = spectral['raw_gate_passed'] and all(
        spectral['positive_singer_macro_mae_cents'] < arms[arm]['all']['positive_singer_macro_mae_cents']
        for arm in ('zero_context', 'frozen_v68')
    )
    results = norm(dict(arms=arms, advance_to_delivery=passed, production_eligible=False,
                        physical_attempts=len(rows), original_attempts=len(originals),
                        scoreable_pairs=445, original_controls=53,
                        declaration_sha256=sha(ROOT / 'declaration.json')))
    if args.verify:
        assert read(ROOT / 'results.json') == results
        write(ROOT / 'integrity_receipt.json', dict(results_sha256=sha(ROOT / 'results.json'),
                                                   checkpoint_reloads=8, prediction_arrays_replayed=1494))
    else:
        write(ROOT / 'results.json', results)
    print('SPECTRAL_RESULT', {arm: metrics['all']['positive_singer_macro_mae_cents']
                              for arm, metrics in arms.items()}, passed, flush=True)


if __name__ == '__main__':
    main()
