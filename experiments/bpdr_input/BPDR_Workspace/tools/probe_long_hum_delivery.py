"""Replay the existing short-repair transaction on the longer-hum diagnostic."""

import hashlib
from pathlib import Path

import numpy as np
import torch

import compare_tail_preservation as models
import evaluate_coverage_delivery as delivery
import probe_long_hum_physical as physical


def transaction(audio, features, model, path):
    times = features['features']['times_seconds']
    request, _ = models.model_module.predict_raw(model, features['views'][0])
    proposal = delivery.policy.proposal(audio, times, features['views'], np.stack([request] * 3))
    plan, curve = proposal['plan'], proposal['curve']

    def audit(before, after, rate, intervals):
        assert rate == physical.RATE
        bf, yf, cf = physical.tracking(before, times)
        ba, ya, ca = physical.tracking(after, times)
        return delivery.audit_module.audit_short_curve(
            bf, ba, yf, ya, cf, ca, features['reliable'], intervals, curve
        )

    def render(source, rate, intervals):
        return delivery.render_candidate(source, rate, times, curve, intervals, delivery.RENDER)[0]

    output, status = delivery.transact_fixture(audio, physical.RATE, plan, render, audit)
    if status['outcome'] == 'VERIFIED_FIXTURE_CANDIDATE':
        physical.sf.write(path, output, physical.RATE, subtype='PCM_24')
        try:
            status['encoded_audit'] = delivery.audit_delivered_wav(path, audio, physical.RATE,
                                                                  plan['intervals_seconds'], audit)
            output, _ = physical.sf.read(path, dtype='float32')
        except (ValueError, RuntimeError) as exc:
            status.update(outcome='ROLLED_BACK_ENCODED', rollback=True, encoded_error=str(exc))
            output = audio.copy()
    return output, dict(plan=plan, candidates=proposal['candidates'], transaction=status,
                        unchanged=bool(np.array_equal(output, audio)), output_sha256=delivery.waveform_hash(output))


def main():
    torch.set_num_threads(2)
    physical.source_study.check_lock()
    read, write, sha = physical.read, physical.write, physical.sha
    root = physical.ROOT
    raw_path, raw_plan = root / 'transfer_results.json', root / 'transfer_declaration.json'
    assert read(raw_path)['declaration_sha256'] == sha(raw_plan)
    sources = {row['key']: row for row in read(physical.source_study.ROOT / 'declaration.json')['sources']}
    declaration = dict(source_sha256=sha(Path(__file__)), raw_result_sha256=sha(raw_path),
                       prior_policy_sha256=sha(delivery.ROOT / 'declaration.json'),
                       production_eligible=False)
    path = root / 'delivery_declaration.json'
    if path.exists():
        assert read(path) == declaration
    else:
        write(path, declaration)
    cached, results, originals = {}, [], {}
    for row in read(raw_path)['rows']:
        if 'excluded' in row:
            results.append(row)
            continue
        key, cents = row['source_id'], row['cents']
        if key not in cached:
            source = sources[key]
            assert source['split'] == 'train' and sha(source['path']) == source['sha256']
            checkpoint = models.ROOT / f"fold_{row['fold']}/upper_tail_preservation/checkpoint.pt"
            assert sha(checkpoint) == read(raw_plan)['checkpoint_hashes'][key]
            model, _ = models.model_module.load(checkpoint)
            clean = physical.pcm24(physical.working_audio(source['path']))
            features = physical.calibration.recovery.measured(clean)
            times = features['features']['times_seconds']
            _, originals[key] = transaction(clean, features, model, root / f'{key}_original_candidate.wav')
            cached[key] = model, clean, features, times, physical.tracking(clean, times)
        model, clean, clean_features, times, before = cached[key]
        start, end = row['support_seconds']
        curve = np.where((times >= start) & (times < end), cents, 0.)
        altered, _ = physical.calibration.render_candidate(
            clean, physical.RATE, times, curve, [(start, end)], physical.calibration.RENDER)
        audio = physical.pcm24(altered)
        assert hashlib.sha256(audio.tobytes()).hexdigest() == row['decoded_pcm_sha256']
        features = physical.calibration.recovery.measured(audio)
        after = physical.tracking(audio, times)
        common = clean_features['reliable'] & features['reliable'] & before[0]['voiced_mask'] & after[0]['voiced_mask']
        common &= np.isfinite(before[1]) & np.isfinite(after[1]) & (before[1] > 0) & (after[1] > 0)
        positive = (times >= start + .09) & (times <= end - .09) & common
        positive &= np.isfinite(before[2]) & np.isfinite(after[2]) & (before[2] > 0) & (after[2] > 0)
        zero = ((times < start - .07) | (times > end + .07)) & common
        assert positive.sum() == row['positive_frames'] and zero.any()
        output, record = transaction(audio, features, model, root / f'{key}_{cents:+g}_candidate.wav')
        reference = delivery.fine_pitch(clean, physical.RATE, times)
        original_error = abs(delivery.fine_pitch(audio, physical.RATE, times) - reference)
        delivered_error = abs(delivery.fine_pitch(output, physical.RATE, times) - reference)
        bm, am = float(original_error[positive].mean()), float(delivered_error[positive].mean())
        bz, az = float(original_error[zero].mean()), float(delivered_error[zero].mean())
        harm = az > bz + 1e-6
        results.append(dict(source_id=key, cents=cents, **record, before_mae=bm, after_mae=am,
                            before_zero_mae=bz, after_zero_mae=az, beneficial=am < bm - 1e-6 and not harm,
                            harmful=am > bm + 1e-6 or harm))
        print('LONG_HUM_DELIVERY_CASE', key, cents, record['transaction']['outcome'], flush=True)
    report = physical.calibration.norm(dict(declaration_sha256=sha(path), rows=results, originals=originals))
    output = root / 'delivery_results.json'
    if output.exists():
        assert read(output) == report
        print('LONG_HUM_DELIVERY_REPLAY_MATCH', flush=True)
    else:
        write(output, report)


if __name__ == '__main__':
    main()
