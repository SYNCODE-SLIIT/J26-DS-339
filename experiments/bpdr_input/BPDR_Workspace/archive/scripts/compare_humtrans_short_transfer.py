"""Matched unchanged Transformer/loss: Vocadito-only frozen parents vs added real hums.

Four new grouped training-source fits only. Reuses four verified V46 physical
parents; all actual development/reserved-unseen audio stays closed.
"""
from pathlib import Path
import argparse
import numpy as np
import torch

import train_paired_short_models as paired
import prepare_humtrans_short_training as humming

WS = paired.WS
VOC = paired.DATA
PARENT = paired.ROOT
HUM = humming.ROOT
ROOT = WS / 'results/transformer_usability/2026-11-02_v51_humtrans_transfer'
read, write, norm, sha = paired.read, paired.write, paired.norm, paired.sha


def declare():
    paired.check_lock()
    ROOT.mkdir(parents=True, exist_ok=True)
    voc_receipt = read(VOC / 'integrity_receipt.json')
    hum_receipt = read(HUM / 'integrity_receipt.json')
    parent_receipt = read(PARENT / 'integrity_receipt.json')
    assert voc_receipt['staged_pairs_verified'] == 115 and hum_receipt['planned_attempts'] == 192
    assert parent_receipt['checkpoints_reloaded'] == 8 and parent_receipt['all_summaries_reproduced']
    for data_root in (VOC, HUM):
        for path, expected in read(data_root / 'input_hashes.json').items():
            assert sha(path) == expected
    voc_rows, hum_rows = read(VOC / 'train_pairs.json'), read(HUM / 'train_pairs.json')
    voc_originals, hum_originals = read(VOC / 'train_originals.json'), read(HUM / 'train_originals.json')
    base = read(PARENT / 'declaration.json')
    hum_groups = sorted({r['group'] for r in hum_originals})
    assert hum_groups == ['HumTrans:F01', 'HumTrans:F02', 'HumTrans:F03', 'HumTrans:F04', 'HumTrans:F05', 'HumTrans:M01']
    folds = [list(groups) for groups in base['folds']]
    for index, group in enumerate(hum_groups):
        folds[index % 4].append(group)
    parent_hashes = {}
    for fold in range(1, 5):
        path = PARENT / f'fold_{fold}/paired_physical/checkpoint.pt'
        receipt = read(path.parent / 'fit_receipt.json')
        assert receipt['checkpoint_sha256'] == sha(path)
        parent_hashes[str(path)] = sha(path)
    paths = [Path(__file__), Path(paired.__file__), Path(paired.previous.__file__), Path(paired.model_module.__file__)]
    d = norm(dict(base, version='real-hum-short-transfer-comparison-1',
        folds=folds, arms=['frozen_vocadito_physical', 'added_real_hums'], new_fits=4,
        source_hashes={str(p.resolve()): sha(p) for p in paths}, parent_checkpoints=parent_hashes,
        dataset_integrity_sha256={str(data / 'integrity_receipt.json'): sha(data / 'integrity_receipt.json') for data in (VOC, HUM)},
        parent_integrity_sha256=sha(PARENT / 'integrity_receipt.json'),
        parent_declaration_sha256=sha(PARENT / 'declaration.json'),
        comparison='Same genuine Transformer, seed339,1200 updates, optimizer and paired-preservation loss. Only training pool changes; both use group-balanced sampling. Frozen parents have no HumTrans fitting and the matching original Vocadito holdout. Added-hum arm excludes the held-out hum people too.',
        physical_attempts=len(voc_rows) + len(hum_rows), original_attempts=len(voc_originals) + len(hum_originals),
        nontraining_sources_accessed=False, historical_hum_prior_use=True,
        fresh_confirmation=False, delivered_audio_qualification=False,
        production_eligible=False, research_validation_passed=False))
    declaration_path = ROOT / 'declaration.json'
    if declaration_path.exists():
        assert read(declaration_path) == d
    else:
        write(declaration_path, d)
    return d, voc_rows + hum_rows, voc_originals + hum_originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    d, rows, originals = declare()
    if args.declare_only:
        print('HUM_TRANSFER_DECLARED', d['physical_attempts'], d['folds'], flush=True)
        return
    data = paired.records(rows, originals)
    predictions = {arm: {} for arm in d['arms']}
    for fold, groups in enumerate(d['folds'], 1):
        train = [r for r in data if r['group'] not in groups]
        test = [r for r in data if r['group'] in groups]
        assert not {r['source_id'] for r in train} & {r['source_id'] for r in test}
        folder = ROOT / f'fold_{fold}'
        folder.mkdir(exist_ok=True)
        parent_path = PARENT / f'fold_{fold}/paired_physical/checkpoint.pt'
        parent, meta = paired.model_module.load(parent_path)
        assert not set(meta['training_groups']) & set(groups)
        assert not any(group.startswith('HumTrans:') for group in meta['training_groups'])
        for r in test:
            predictions['frozen_vocadito_physical'][r['id']] = paired.previous.predict(parent, r, 'short_transformer')
        trace, trace_hash = paired.previous.sample_trace(train, d)
        np.save(folder / 'physical_trace.npy', trace)
        receipt_path = folder / 'fit_receipt.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            assert receipt['declaration_sha256'] == sha(ROOT / 'declaration.json')
            assert receipt['physical_trace_sha256'] == trace_hash
            assert receipt['checkpoint_sha256'] == sha(folder / 'checkpoint.pt')
            model, _ = paired.model_module.load(folder / 'checkpoint.pt')
        else:
            # Reuse the exact audited training implementation, redirecting only
            # the new checkpoint's declaration link. No source file is changed.
            old_root = paired.ROOT
            try:
                paired.ROOT = ROOT
                model, info = paired.fit(train, 'paired_physical', trace, d, fold, folder)
            finally:
                paired.ROOT = old_root
            receipt = dict(info, declaration_sha256=sha(ROOT / 'declaration.json'),
                physical_trace_sha256=trace_hash, checkpoint_sha256=sha(folder / 'checkpoint.pt'),
                training_groups=sorted({r['group'] for r in train}), heldout_groups=groups,
                training_sources=sorted({r['source_id'] for r in train}), heldout_sources=sorted({r['source_id'] for r in test}))
            write(receipt_path, receipt)
        for r in test:
            predictions['added_real_hums'][r['id']] = paired.previous.predict(model, r, 'short_transformer')
        print('HUM_TRANSFER_FOLD_COMPLETE', fold, flush=True)
    results = {}
    for arm, pred in predictions.items():
        np.savez_compressed(ROOT / (arm + '_all_predictions.npz'), **pred)
        results[arm] = dict(all=paired.previous.summarize(data, pred),
            humming=paired.previous.summarize([r for r in data if r['group'].startswith('HumTrans:')], pred),
            vocadito=paired.previous.summarize([r for r in data if not r['group'].startswith('HumTrans:')], pred))
    write(ROOT / 'results.json', dict(declaration_sha256=sha(ROOT / 'declaration.json'), arms=results,
        physical_attempts=d['physical_attempts'], scoreable_pairs=sum('excluded' not in r for r in rows),
        original_attempts=d['original_attempts'], original_controls=sum('excluded' not in r for r in originals),
        groups_without_positive_pairs=sorted({r['group'] for r in originals} - {r['group'] for r in rows if 'excluded' not in r}),
        gate_scope='Prediction gate covers represented scoreable groups only; construction-excluded groups remain unassessed, never successful. Delivered and all-attempt practical gates are separate.',
        raw_predictions_only=True, delivered_audio_qualification=False, production_eligible=False))
    print('HUM_TRANSFER_COMPLETE', {a: {k: (s['positive_singer_macro_mae_cents'], s['raw_gate_passed']) for k, s in v.items()} for a, v in results.items()}, flush=True)


if __name__ == '__main__':
    main()
