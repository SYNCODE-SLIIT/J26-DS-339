"""Independent waveform remeasurement and all-control replay for short-audit calibration."""
from pathlib import Path
from dataclasses import replace
import sys,json
import numpy as np
import soundfile as sf
import torch

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
import calibrate_short_support_audit as run


def main():
    torch.set_num_threads(2);d=run.declare();result=run.read(run.ROOT/'results.json')
    assert result['declaration_sha256']==run.sha(run.ROOT/'declaration.json')
    for p,h in d['source_hashes'].items():assert run.sha(p)==h,p
    for p,h in run.read(run.DATA/'input_hashes.json').items():assert run.sha(p)==h,p
    originals={s['source_id']:s for s in d['originals']};cache={};audits=0;controls=0;tampers=0;context_arrays=0;excluded=0
    baseline=replace(run.short.POLICY,version='retained-ten-frame-calibration-baseline',min_audit_frames=10)
    for row in result['cases']:
        source=originals[row['source_id']];width=row['width_seconds'];cents=row['cents']
        if 'excluded' in source:
            assert row['excluded']=='SOURCE_PREPROCESSING_EXCLUDED';excluded+=1;continue
        region=source['longest_region_frames']
        if region is None or (region[1]-region[0])*.02<width+.30-1e-12:
            assert row['excluded']=='NO_SINGLE_RELIABLE_REGION_WITH_150MS_CONTEXT';excluded+=1;continue
        assert 'excluded' not in row
        if source['source_id'] not in cache:
            audio,sr=sf.read(source['path_A'],dtype='float32');assert sr==22050
            with np.load(source['analysis_A']) as z:times=z['times_seconds'];reliable=z['reliable']
            bf=run.coarse_features(audio,sr,WS/'checkpoints');yb=run.fine_hz(audio,sr,times)
            cb=run.short.cross_correlation_pitch(audio,sr,times)
            expected=dict(**bf,yin=yb,cc=cb,reliable=reliable)
            with np.load(run.ROOT/source['source_id']/'before_measurements.npz') as z:
                assert set(z.files)==set(expected)
                for k,v in expected.items():np.testing.assert_array_equal(z[k],v)
            cache[source['source_id']]=(audio,times,reliable,bf,yb,cb)
        audio,times,reliable,bf,yb,cb=cache[source['source_id']];sr=22050
        center=float(np.mean(region)*.02);support=(center-width/2,center+width/2)
        assert row['support_seconds']==list(support)
        path=Path(row['output']);assert run.sha(path)==row['output_sha256']
        output,rate=sf.read(path,dtype='float32');assert rate==sr and audio.shape==output.shape
        assert np.isfinite(output).all() and abs(output).max()<1
        st=np.arange(len(audio))/sr;outside=(st<=support[0])|(st>=support[1])
        assert np.array_equal(audio[outside],output[outside])
        af=run.coarse_features(output,sr,WS/'checkpoints');ya=run.fine_hz(output,sr,times)
        ca=run.short.cross_correlation_pitch(output,sr,times)
        curve=np.where((times>=support[0])&(times<support[1]),cents,0.)
        expected=dict(**af,yin=ya,cc=ca,requested=curve)
        with np.load(path.parent/'measurements.npz') as z:
            assert set(z.files)==set(expected)
            for k,v in expected.items():np.testing.assert_array_equal(z[k],v)
        ten=run.audit_curve(bf,af,yb,ya,reliable,[support],curve,baseline)
        three=run.audit_curve(bf,af,yb,ya,reliable,[support],curve,run.short.POLICY)
        audit=run.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],curve)
        assert run.norm(ten)==row['ten_frame'] and run.norm(three)==row['three_frame'] and run.norm(audit)==row['three_tracker'];audits+=1
        tests={
            'unchanged':run.short.audit_short_curve(bf,bf,yb,yb,cb,cb,reliable,[support],curve),
            'opposite':run.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],-curve),
            'wrong_magnitude':run.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],curve/3)}
        displaced=(support[0]+1,support[1]+1);dc=np.where((times>=displaced[0])&(times<displaced[1]),cents,0.)
        tests['wrong_location']=run.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[displaced],dc)
        assert run.norm(tests)==row['controls'];assert not any(t['passed'] for t in tests.values());controls+=4
        bad=path.parent/'tampered_outside.wav';assert run.sha(bad)==row['tamper']['sha256']
        def forbidden(*args):raise AssertionError('Tamper reached acoustic callback')
        try:run.audit_delivered_wav(bad,audio,sr,[support],forbidden)
        except ValueError as exc:assert str(exc)=='ENCODED_CANDIDATE_OUTSIDE_SUPPORT_CHANGED'
        else:raise AssertionError('Tamper unexpectedly accepted')
        tampers+=1
        if audit['passed']:
            analysis=run.recovery.measured(output)
            expected=dict(raw_views=np.stack(analysis['views']),reliable=analysis['reliable'],fine=analysis['fine'])
            with np.load(path.parent/'candidate_context_NOT_LABEL.npz') as z:
                for k,v in expected.items():np.testing.assert_array_equal(z[k],v);context_arrays+=1
            a,b=region;left=(times>=a*.02)&(times<support[0]-.07)&analysis['reliable']
            right=(times>support[1]+.07)&(times<b*.02)&analysis['reliable']
            assert row['observed_context']==dict(left_anchor_frames=int(left.sum()),right_anchor_frames=int(right.sum()),both_sides_at_least_three=bool(left.sum()>=3 and right.sum()>=3))
    assert audits+excluded==336 and controls==result['negative_controls'] and tampers==result['outside_sample_tamper_rejections']
    for s in result['summary']:
        rows=[r for r in result['cases'] if r['width_seconds']==s['width_seconds']]
        assert s['attempted']==len(rows)
        for key,field in [('ten_frame_pass','ten_frame'),('three_frame_pass','three_frame'),('three_tracker_pass','three_tracker')]:assert s[key]==sum(r.get(field,{}).get('passed',False) for r in rows)
    run.write(run.ROOT/'integrity_receipt.json',dict(physical_audits_replayed=audits,
        construction_exclusions_reproduced=excluded,negative_pitch_controls_rejected=controls,
        outside_sample_tampers_rejected=tampers,candidate_context_arrays_reinferred=context_arrays,
        results_sha256=run.sha(run.ROOT/'results.json'),verifier_source_sha256=run.sha(__file__),
        finite_calibration_only=True,statistical_independence_claim=False,production_eligible=False))
    print('SHORT_AUDIT_CALIBRATION_REPLAY_PASS',audits,excluded,controls,tampers,context_arrays,flush=True)


if __name__=='__main__':main()
