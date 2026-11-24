"""Independently replay the reliable-melody construction pilot and its controls."""
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

import probe_melodic_construction as run
from bpdr_input.repair.held_voice_policy import spans
from verify_humtrans_short_training import compare_arrays


def main():
    torch.set_num_threads(2);d=run.declare();root=run.ROOT
    assert d['source_sha256']==run.sha(run.__file__)
    for path,expected in d['source_hashes'].items():assert run.sha(path)==expected
    result=run.read(root/'results.json');assert result['declaration_sha256']==run.sha(root/'declaration.json')
    cases={(r['source_id'],r['width_seconds'],r['cents']):r for r in result['cases']}
    assert len(cases)==64
    recreated=root/'verification_recreated';recreated.mkdir(exist_ok=True)
    audits=0;controls=0;tampers=0;arrays=0;geometry_exclusions=0;passed=[];rendered=[];negative=[]
    for source in d['sources']:
        original=source['original'];sid=original['source_id']
        audio,sr=sf.read(original['path_A'],dtype='float32');assert sr==22050
        with np.load(original['analysis_A']) as z:times=z['times_seconds'].copy();reliable=z['reliable'].copy()
        runs=spans(reliable);region=list(map(int,max(runs,key=lambda r:r[1]-r[0])))
        assert region==source['reliable_region']
        bf=run.cal.coarse_features(audio,sr,run.WS/'checkpoints');yb=run.cal.fine_hz(audio,sr,times)
        cb=run.cal.short.cross_correlation_pitch(audio,sr,times)
        arrays+=compare_arrays(root/sid/'before_measurements.npz',dict(**bf,yin=yb,cc=cb,reliable=reliable))
        for width in d['widths_seconds']:
            for cents in d['cents']:
                row=cases[(sid,width,cents)]
                if (region[1]-region[0])*.02<width+.30-1e-12:
                    assert row['excluded']=='NO_CONTINUOUS_RELIABLE_SPAN_WITH_150MS_CONTEXT';geometry_exclusions+=1;continue
                center=float(np.mean(region)*.02);support=(center-width/2,center+width/2)
                requested=np.where((times>=support[0])&(times<support[1]),cents,0.)
                try:output,render_receipt=run.cal.render_candidate(audio,sr,times,requested,[support],run.cal.RENDER)
                except ValueError as error:
                    assert row['excluded']=='RENDER_ERROR' and row['detail']==str(error);continue
                if not np.isfinite(output).all() or abs(output).max()>=1:
                    assert row['excluded']=='RENDER_SAMPLE_FAILURE';continue
                assert 'excluded' not in row and row['support_seconds']==list(support)
                assert row['render_receipt']==run.norm(render_receipt)
                target=recreated/f'{sid}_{width:.2f}_{cents:+g}.wav';sf.write(target,output,sr,subtype='PCM_24')
                assert run.sha(target)==run.sha(row['output'])==row['output_sha256']
                decoded,rate=sf.read(target,dtype='float32');assert rate==sr
                st=np.arange(len(audio))/sr;outside=(st<=support[0])|(st>=support[1])
                np.testing.assert_array_equal(decoded[outside],audio[outside])
                af=run.cal.coarse_features(decoded,sr,run.WS/'checkpoints');ya=run.cal.fine_hz(decoded,sr,times)
                ca=run.cal.short.cross_correlation_pitch(decoded,sr,times)
                arrays+=compare_arrays(Path(row['output']).parent/'measurements.npz',dict(**af,yin=ya,cc=ca,requested=requested))
                audit=run.cal.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],requested)
                assert run.norm(audit)==row['audit'];audits+=1;rendered.append(row)
                shift=1. if support[1]+1.<=len(audio)/sr else -1.
                displaced=(support[0]+shift,support[1]+shift)
                assert row['displaced_support_seconds']==list(displaced)
                location_curve=np.where((times>=displaced[0])&(times<displaced[1]),cents,0.)
                checks=dict(
                    unchanged=run.cal.short.audit_short_curve(bf,bf,yb,yb,cb,cb,reliable,[support],requested),
                    opposite=run.cal.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],-requested),
                    wrong_magnitude=run.cal.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],requested/3),
                    wrong_location=run.cal.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[displaced],location_curve))
                for name,value in checks.items():
                    assert not value['passed'] and run.norm(value)==row['controls'][name]
                    controls+=1;negative.append(value)
                bad=decoded.copy();bad[int(np.flatnonzero(outside)[0])]+=np.float32(2**-15)
                bp=recreated/f'{sid}_{width:.2f}_{cents:+g}_tamper.wav';sf.write(bp,bad,sr,subtype='PCM_24')
                assert run.sha(bp)==run.sha(Path(row['output']).parent/'outside_tamper.wav')
                def never(*args):raise AssertionError('Outside sample reached acoustic audit')
                try:run.cal.audit_delivered_wav(bp,audio,sr,[support],never)
                except ValueError as error:assert str(error)==row['tamper']['reason']=='ENCODED_CANDIDATE_OUTSIDE_SUPPORT_CHANGED'
                else:raise AssertionError('Outside tamper accepted')
                tampers+=1
                if audit['passed']:
                    c=run.cal.recovery.measured(decoded)
                    arrays+=compare_arrays(Path(row['output']).parent/'input_context.npz',dict(raw_views=np.stack(c['views']),reliable=c['reliable'],fine=c['fine'],times_seconds=times))
                    left=(times>=region[0]*.02)&(times<support[0]-.07)&c['reliable']
                    right=(times>support[1]+.07)&(times<region[1]*.02)&c['reliable']
                    assert row['retained_context']==dict(left=int(left.sum()),right=int(right.sum()),both_sides_at_least_three=bool(min(left.sum(),right.sum())>=3))
                    passed.append(row)
        print('MELODIC_CONSTRUCTION_REPLAY_SOURCE',sid,flush=True)
    summary=result['summary']
    assert summary['physical_audit_passes']==len(passed) and summary['rendered']==len(rendered)
    assert summary['negative_controls']==controls and summary['negative_false_accepts']==0
    assert summary['outside_sample_tamper_rejections']==tampers
    assessable=sum(c['three_estimator_common_frames']>=3 and c['plateau_coverage_fraction']>=.75 and c['cc']['coverage']>=.75 for c in negative)
    assert assessable==summary['negative_controls_with_adequate_common_coverage']
    run.write(root/'integrity_receipt.json',dict(planned_attempts_retained=64,source_regions_recomputed=4,
        geometry_exclusions_replayed=geometry_exclusions,decoded_audits_replayed=audits,negative_controls_replayed=controls,
        negative_controls_with_adequate_common_coverage=assessable,outside_tampers_rejected=tampers,
        input_context_and_measurement_arrays_recomputed=arrays,physical_audit_passes=len(passed),
        results_sha256=run.sha(root/'results.json'),verifier_source_sha256=run.sha(__file__),
        reference_construction_only=True,new_model_fits=0,production_eligible=False,research_validation_passed=False))
    print('MELODIC_CONSTRUCTION_INDEPENDENT_REPLAY_PASS',audits,controls,len(passed),arrays,flush=True)


if __name__=='__main__':main()
