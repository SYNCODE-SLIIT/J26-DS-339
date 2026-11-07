"""Verify first-window reuse and independently replay all later-window cases."""
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

import build_reliable_melodic_pairs as run
from verify_humtrans_short_training import compare_arrays


def main():
    torch.set_num_threads(2)
    d, originals, prior = run.declare()
    root = run.ROOT
    assert d['source_sha256'] == run.sha(run.__file__)
    assert d['audit_source_sha256'] == run.sha(run.calibration.short.__file__)
    for field in ('parent_pair_manifests_sha256','parent_integrities_sha256'):
        for path, expected in d[field].items():
            assert run.sha(path) == expected
    for parent in run.inputs.PARENTS:
        for path, expected in run.read(parent/'input_hashes.json').items():
            assert run.sha(path) == expected
    for path, expected in run.read(root/'input_hashes.json').items():
        assert run.sha(path) == expected
    rows = run.read(root/'train_pairs.json')
    assert len(rows) == 1152 and run.read(root/'train_originals.json') == originals
    by_id = {r['source_id']:r for r in originals}
    assert len({(r['source_id'],r['width_seconds'],r['cents']) for r in rows}) == 1152
    assessable=0;tampers=0
    cache = {}; reused = 0; exclusions = 0; audits = 0; controls = 0; arrays = 0; new_labels = 0
    recreated = root/'verification_recreated'; recreated.mkdir(exist_ok=True)
    for index,row in enumerate(rows):
        original = by_id[row['source_id']]
        assert row['group'] == original['group'] and row['split'] == 'train'
        assert row['width_seconds'] in d['widths_seconds'] and row['cents'] in d['cents']
        if original.get('reuse_historical_case',False):
            old = original['source']['prior_original']
            assert run.sha(old['path_A']) == run.sha(original['path_A'])
            expected = dict(prior[(row['source_id'],row['width_seconds'],row['cents'])],reused_historical_case=True)
            if 'excluded' not in expected:
                expected.update(path_A=original['path_A'],analysis_A=original['analysis_A'])
            assert row == expected
            reused += 1
            continue
        assert not row['reused_historical_case']
        if 'excluded' in original:
            assert original['excluded'] == row['excluded'] == 'SOURCE_SHORTER_THAN_FIVE_SECOND_TARGET'
            exclusions += 1
            continue
        if row['width_seconds'] not in original['widths_with_150ms_context']:
            assert row['excluded'] == 'NO_SINGLE_RELIABLE_REGION_WITH_150MS_CONTEXT'
            exclusions += 1
            continue
        sid = row['source_id']
        if sid not in cache:
            audio,sr = sf.read(original['path_A'],dtype='float32'); assert sr == 22050
            with np.load(original['analysis_A']) as z:
                times = z['times_seconds'].copy(); reliable = z['reliable'].copy()
            bf = run.calibration.coarse_features(audio,sr,run.WS/'checkpoints')
            yb = run.calibration.fine_hz(audio,sr,times)
            cb = run.calibration.short.cross_correlation_pitch(audio,sr,times)
            arrays += compare_arrays(root/sid/'before_measurements.npz',dict(**bf,yin=yb,cc=cb,reliable=reliable))
            cache[sid] = audio,times,reliable,bf,yb,cb
        audio,times,reliable,bf,yb,cb = cache[sid]
        region = original['longest_region_frames']; width = row['width_seconds']
        center = float(np.mean(region)*.02); start,end = center-width/2,center+width/2
        assert row['support_seconds'] == [start,end]
        curve = np.where((times>=start)&(times<end),row['cents'],0.)
        output,render_receipt = run.calibration.render_candidate(audio,22050,times,curve,[(start,end)],run.calibration.RENDER)
        assert run.norm(render_receipt) == row['render_receipt']
        if not np.isfinite(output).all() or abs(output).max()>=1:
            assert row['excluded'] == 'PHYSICAL_RENDER_PEAK_OR_SAMPLE_FAILURE'
            exclusions += 1
            continue
        path = recreated/f'{index:03}.wav'; sf.write(path,output,22050,subtype='PCM_24')
        assert run.sha(path) == run.sha(row['path_C'])
        decoded,sr = sf.read(path,dtype='float32'); assert sr == 22050
        sample_times = np.arange(len(audio))/sr; outside = (sample_times<=start)|(sample_times>=end)
        np.testing.assert_array_equal(decoded[outside],audio[outside])
        af = run.calibration.coarse_features(decoded,sr,run.WS/'checkpoints')
        ya = run.calibration.fine_hz(decoded,sr,times); ca = run.calibration.short.cross_correlation_pitch(decoded,sr,times)
        arrays += compare_arrays(Path(row['path_C']).parent/'measurements.npz',dict(**af,yin=ya,cc=ca,requested=curve))
        audit = run.calibration.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[(start,end)],curve)
        saved = run.read(row['audit_path'])
        assert run.norm(audit) == row['audit'] == saved['audit']; audits += 1
        shift=1. if end+1.<=len(audio)/sr else -1.
        displaced=(start+shift,end+shift)
        assert saved['displaced_support_seconds']==list(displaced)
        location_curve=np.where((times>=displaced[0])&(times<displaced[1]),row['cents'],0.)
        checks=dict(
            unchanged=run.calibration.short.audit_short_curve(bf,bf,yb,yb,cb,cb,reliable,[(start,end)],curve),
            opposite=run.calibration.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[(start,end)],-curve),
            wrong_magnitude=run.calibration.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[(start,end)],curve/3),
            wrong_location=run.calibration.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[displaced],location_curve))
        for name,check in checks.items():
            assert not check['passed'] and run.norm(check)==saved['controls'][name]
            controls+=1
            assessable+=int(check['three_estimator_common_frames']>=3 and check['plateau_coverage_fraction']>=.75 and check['cc']['coverage']>=.75)
        bad=decoded.copy();bad[int(np.flatnonzero(outside)[0])]+=np.float32(2**-15)
        bp=recreated/f'{index:03}_tamper.wav';sf.write(bp,bad,22050,subtype='PCM_24')
        assert run.sha(bp)==run.sha(Path(row['path_C']).parent/'outside_tamper.wav')
        def never(*args):raise AssertionError('Outside sample reached acoustic audit')
        try:run.calibration.audit_delivered_wav(bp,audio,22050,[(start,end)],never)
        except ValueError as error:assert str(error)==saved['tamper']['reason']=='ENCODED_CANDIDATE_OUTSIDE_SUPPORT_CHANGED'
        else:raise AssertionError('Outside tamper accepted')
        tampers+=1
        if not audit['passed']:
            assert row['excluded'] == 'THREE_TRACKER_PHYSICAL_AUDIT_FAILED'
            continue
        c = run.calibration.recovery.measured(decoded)
        common = reliable&c['reliable']&bf['voiced_mask']&af['voiced_mask']&np.isfinite(yb)&np.isfinite(ya)&(yb>0)&(ya>0)
        plateau = (times>=start+.09)&(times<=end-.09)&common&np.isfinite(cb)&np.isfinite(ca)&(cb>0)&(ca>0)
        zero = ((times<start-.07)|(times>end+.07))&common
        left = (times>=region[0]*.02)&(times<start-.07)&c['reliable']
        right = (times>end+.07)&(times<region[1]*.02)&c['reliable']
        if plateau.sum()<3 or min(left.sum(),right.sum())<3:
            assert row['excluded'] == 'INSUFFICIENT_RETAINED_PLATEAU_OR_CONTEXT'
            continue
        assert 'excluded' not in row
        arrays += compare_arrays(row['analysis_C'],dict(raw_views=np.stack(c['views']),reliable=c['reliable'],fine=c['fine'],times_seconds=times))
        arrays += compare_arrays(row['target_path'],dict(target=np.where(plateau,-row['cents'],0.).astype(np.float32),
            mask=plateau|zero,plateau=plateau,known_zero=zero,times=times))
        new_labels += 1
    summary = run.read(root/'summary.json')
    valid = [r for r in rows if 'excluded' not in r]
    assert len(valid) == summary['valid']
    assert reused == summary['cases_reused'] == 16*sum(o.get('reuse_historical_case',False) for o in originals)
    assert new_labels == summary['valid_pairs_new']
    assert sum(r['reused_historical_case'] for r in valid) == summary['valid_pairs_reused']
    run.write(root/'integrity_receipt.json',dict(planned_attempts_retained=len(rows),historical_cases_exactly_matched=reused,
        historical_valid_pairs_retained=summary['valid_pairs_reused'],new_decoded_audits_replayed=audits,negative_controls_replayed=controls,negative_controls_with_adequate_common_coverage=assessable,outside_tampers_rejected=tampers,
        new_supervision_files_recomputed=new_labels,new_arrays_recomputed=arrays,new_construction_exclusions_replayed=exclusions,
        staged_pairs_verified=len(valid),original_attempts=72,valid_originals=53,source_recordings=33,
        declaration_sha256=run.sha(root/'declaration.json'),train_pairs_sha256=run.sha(root/'train_pairs.json'),
        train_originals_sha256=run.sha(root/'train_originals.json'),input_hashes_sha256=run.sha(root/'input_hashes.json'),
        summary_sha256=run.sha(root/'summary.json'),verifier_source_sha256=run.sha(__file__),
        research_training_eligible=False,production_eligible=False))
    print('FULL_WINDOW_SHORT_PAIR_INDEPENDENT_REPLAY_PASS',reused,new_labels,audits,controls,arrays,flush=True)


if __name__=='__main__':
    main()
