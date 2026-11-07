"""Measure raw V68 transfer on the replayed longer-hum construction cases."""

import hashlib
from pathlib import Path

import numpy as np
import torch

import compare_tail_preservation as study
import probe_long_hum_physical as physical


def main():
    torch.set_num_threads(2)
    physical.source_study.check_lock()
    read, write, sha = physical.read, physical.write, physical.sha
    protocol = physical.ROOT / 'declaration.json'
    construction = physical.ROOT / 'results.json'
    report = read(construction)
    assert report['declaration_sha256'] == sha(protocol)
    assert read(protocol)['runner_sha256'] == sha(Path(physical.__file__))
    assert report['scoreable'] == 8 and report['negative_control_accepts'] == 0
    source_manifest = physical.source_study.ROOT / 'declaration.json'
    assert read(protocol)['source_manifest_sha256'] == sha(source_manifest)
    sources = {row['key']: row for row in read(source_manifest)['sources']}
    model_plan = study.ROOT / 'declaration.json'
    folds = read(model_plan)['folds']
    checkpoints = {}
    for row in report['rows']:
        if 'excluded' in row:
            continue
        source = sources[row['source_id']]
        assert source['split'] == 'train' and sha(source['path']) == source['sha256']
        group = 'HumTrans:' + source['person_id']
        fold, = [i for i, groups in enumerate(folds, 1) if group in groups]
        folder = study.ROOT / f'fold_{fold}/upper_tail_preservation'
        receipt = read(folder / 'fit_receipt.json')
        checkpoint = folder / 'checkpoint.pt'
        assert group in receipt['heldout_groups'] and group not in receipt['training_groups']
        assert source['key'] not in receipt['training_sources']
        assert sha(checkpoint) == receipt['checkpoint_sha256']
        checkpoints[source['key']] = (checkpoint, fold)
    declaration = dict(
        source_sha256=sha(Path(__file__)), construction_sha256=sha(construction),
        model_plan_sha256=sha(model_plan),
        checkpoint_hashes={key: sha(path) for key, (path, _) in checkpoints.items()},
        attempts=14, model_fits=0, production_eligible=False,
    )
    declaration_path = physical.ROOT / 'transfer_declaration.json'
    if declaration_path.exists():
        assert read(declaration_path) == declaration
    else:
        write(declaration_path, declaration)
    cached, rows = {}, []
    for row in report['rows']:
        key = row['source_id']
        if 'excluded' in row:
            rows.append({name: row[name] for name in ('source_id', 'cents', 'excluded')})
            continue
        checkpoint, fold = checkpoints[key]
        if key not in cached:
            audio = physical.pcm24(physical.working_audio(sources[key]['path']))
            clean = physical.calibration.recovery.measured(audio)
            times = clean['features']['times_seconds']
            model, _ = study.model_module.load(checkpoint)
            clean_request = study.model_module.predict_raw(model, clean['views'][0])[0]
            cached[key] = audio, clean, times, model, clean_request, physical.tracking(audio, times)
        audio, clean, times, model, clean_request, before = cached[key]
        start, end = row['support_seconds']
        cents = row['cents']
        curve = np.where((times >= start) & (times < end), cents, 0.)
        candidate, _ = physical.calibration.render_candidate(
            audio, physical.RATE, times, curve, [(start, end)], physical.calibration.RENDER
        )
        decoded = physical.pcm24(candidate)
        altered = physical.calibration.recovery.measured(decoded)
        after = physical.tracking(decoded, times)
        common = clean['reliable'] & altered['reliable'] & before[0]['voiced_mask'] & after[0]['voiced_mask']
        common &= np.isfinite(before[1]) & np.isfinite(after[1]) & (before[1] > 0) & (after[1] > 0)
        positive = (times >= start + .09) & (times <= end - .09) & common
        positive &= np.isfinite(before[2]) & np.isfinite(after[2]) & (before[2] > 0) & (after[2] > 0)
        zero = ((times < start - .07) | (times > end + .07)) & common
        assert positive.sum() == row['plateau_frames'] and zero.any()
        request = study.model_module.predict_raw(model, altered['views'][0])[0]
        target = -cents
        rows.append(dict(
            source_id=key, fold=fold, cents=cents, support_seconds=[start, end],
            raw_mae_cents=float(np.mean(abs(request[positive] - target))),
            zero_baseline_mae_cents=abs(target),
            correct_direction_frames=int(np.count_nonzero(request[positive] * target > 0)),
            positive_frames=int(positive.sum()),
            original_peak_cents=float(np.max(abs(clean_request[clean['reliable']]))),
            original_frames_at_least_12_cents=int(np.count_nonzero(abs(clean_request[clean['reliable']]) >= 12)),
            known_zero_peak_cents=float(np.max(abs(request[zero]))),
            known_zero_frames_at_least_12_cents=int(np.count_nonzero(abs(request[zero]) >= 12)),
            decoded_pcm_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
        ))
        print('LONG_HUM_TRANSFER_CASE', key, cents, rows[-1]['raw_mae_cents'], flush=True)
    result = dict(declaration_sha256=sha(declaration_path), rows=rows)
    output = physical.ROOT / 'transfer_results.json'
    if output.exists():
        assert read(output) == result
        print('LONG_HUM_TRANSFER_REPLAY_MATCH', flush=True)
    else:
        write(output, result)


if __name__ == '__main__':
    main()
