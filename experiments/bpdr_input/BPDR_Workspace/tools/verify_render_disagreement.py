"""Independent complete waveform/measurement replay of the disagreement diagnostic."""
from pathlib import Path
import numpy as np
import torch
import diagnose_render_disagreement as study
from bpdr_input.audio.analysis import extract
from bpdr_input.repair.framewise_renderer import render_candidate
from bpdr_input.repair.framewise_audit import audit_curve
from bpdr_input.repair.rule_correction import fine_hz
from bpdr_input.repair.repair_controller import waveform_hash


def main():
    torch.set_num_threads(2);d=study.declare();ex=study.study.ex;root=study.ROOT
    results=ex.read(root/'results.json');assert results['declaration_sha256']==ex.sha(root/'declaration.json')
    pairs=study.study.training.held.prepared(ex.read(d['manifest']));verified=0;frame_count=0
    for arm,folder in d['arms'].items():
        folder=Path(folder);accuracy=ex.read(folder/'accuracy.json');expected=[]
        with np.load(folder/'predictions.npz') as predictions:
            for i,row in enumerate(accuracy['rows']):
                if 'excluded' in row:continue
                trace=row['trace']['C'];plan=trace['permission']
                if not plan['permitted'] or plan['identity']:continue
                expected.append(i);p=pairs[i];out=root/arm/f'case_{i:03d}'
                r=ex.read(out/'receipt.json');assert ex.sha(out/'tracks.npz')==r['tracks_sha256']
                requested=predictions[f'{i}_requested_C'];before=p['C']['features'];times=before['times_seconds']
                candidate,_=render_candidate(p['audio_C'],22050,times,requested,plan['intervals_seconds'])
                assert waveform_hash(candidate)==r['candidate_sha256']
                after,_=extract(candidate,22050,ex.COMPONENT_ROOT/'checkpoints')
                fb=fine_hz(p['audio_C'],22050,times);fa=fine_hz(candidate,22050,times)
                audit=audit_curve(before,after,fb,fa,p['C']['reliable'],plan['intervals_seconds'],requested)
                assert audit==r['audit']==trace['transaction']['post_render_audit']
                with np.load(out/'tracks.npz') as z:
                    np.testing.assert_array_equal(z['candidate_audio'],candidate)
                    np.testing.assert_array_equal(z['requested'],requested)
                    np.testing.assert_array_equal(z['times'],times)
                    np.testing.assert_array_equal(z['reliable'],p['C']['reliable'])
                    for label,v in [('before',before),('after',after)]:
                        for key in ('f0_hz','confidence','voiced_mask','native_pitch_times_seconds','native_pitch_hz','native_pitch_confidence'):
                            np.testing.assert_array_equal(z[label+'_'+key],v[key])
                    np.testing.assert_array_equal(z['fine_before'],fb);np.testing.assert_array_equal(z['fine_after'],fa)
                    # Rebuild audit-frame mask and signs directly, without diagnostic helper.
                    plateau=np.zeros(len(times),bool);limits=audit['base_limits']
                    for lo,hi in plan['intervals_seconds']:
                        plateau|=(times>=lo+limits['ramp_seconds']+limits['analysis_guard_seconds'])&(times<=hi-limits['ramp_seconds']-limits['analysis_guard_seconds'])
                    cb,ca=before['f0_hz'],after['f0_hz']
                    cvalid=before['voiced_mask']&after['voiced_mask']&np.isfinite(cb)&np.isfinite(ca)&(cb>0)&(ca>0)
                    fvalid=p['C']['reliable']&np.isfinite(fb)&np.isfinite(fa)&(fb>0)&(fa>0)
                    valid=cvalid&fvalid
                    checked=plateau&p['C']['reliable']&valid&(abs(requested)>=10)
                    coarse=np.full(len(times),np.nan);coarse[cvalid]=1200*np.log2(ca[cvalid]/cb[cvalid])
                    fine=np.full(len(times),np.nan);fine[fvalid]=1200*np.log2(fa[fvalid]/fb[fvalid])
                    np.testing.assert_array_equal(z['coarse_shift'],coarse);np.testing.assert_array_equal(z['fine_shift'],fine)
                    opposite=checked&(coarse*requested<0)
                    np.testing.assert_array_equal(checked,z['checked']);np.testing.assert_array_equal(opposite,z['opposing'])
                    assert list(np.flatnonzero(opposite))==[x['frame'] for x in r['opposing_frames']]
                    for item in r['opposing_frames']:
                        j=item['frame'];assert item['time_seconds']==times[j] and item['coarse_shift_cents']==coarse[j]
                    inside=np.zeros(len(candidate),bool);st=np.arange(len(candidate))/22050
                    for lo,hi in plan['intervals_seconds']:inside|=(st>=lo)&(st<hi)
                    assert np.array_equal(candidate[~inside],p['audio_C'][~inside])
                verified+=1;frame_count+=int(opposite.sum())
                print('DISAGREEMENT_REPLAY',arm,i,flush=True)
        assert expected==[r['case_index'] for r in results['arms'][arm]['cases']]
        assert len(expected)==results['arms'][arm]['rendered_candidates']==30
    protected=ex.read(root.parent/'2026-10-28_v7_guard/declaration.json')['protected_files']
    for p,h in protected.items():assert ex.sha(p)==h
    assert not (root.parent/'active_model.json').exists();study.declare()
    receipt=dict(candidate_waveforms_and_audits_replayed=verified,opposing_frames_replayed=frame_count,all60_candidates_included=verified==60,attempts_per_arm=200,scoreable_per_arm=108,protected_files_unchanged=len(protected),results_sha256=ex.sha(root/'results.json'),verifier_sha256=ex.sha(Path(__file__)),selection_eligible=False,activation_absent=True,research_validation_passed=False)
    ex.write(root/'integrity_receipt.json',receipt);print('RENDER_DISAGREEMENT_VERIFIED',receipt,flush=True)

if __name__=='__main__':main()
