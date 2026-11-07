"""Independent reload/reinference and proposal replay for the V56 diagnostic."""
import numpy as np
import soundfile as sf
import torch

import diagnose_short_preservation_transfer as run


def main():
    torch.set_num_threads(2)
    study=run.study; d,_,originals=study.declare();root=run.ROOT
    result=study.read(root/'results.json')
    assert result['source_sha256']==study.sha(run.__file__)
    assert result['policy_source_sha256']==study.sha(run.policy.__file__)
    assert result['model_integrity_sha256']==study.sha(study.ROOT/'integrity_receipt.json')
    assert result['original_predictions_sha256']==study.sha(root/'original_predictions.npz')
    controls={r['source_id']:r for r in originals if 'excluded' not in r}
    by_key={(r['fold'],r['source_id']):r for r in result['rows']}
    assert len(by_key)==112
    with np.load(root/'original_predictions.npz') as z:predictions={k:z[k].copy() for k in z.files}
    replayed=[];arrays=0
    for fold,groups in enumerate(d['folds'],1):
        checkpoint=study.ROOT/f'fold_{fold}/checkpoint.pt'
        assert result['checkpoint_hashes'][str(checkpoint)]==study.sha(checkpoint)
        model,meta=study.paired.model_module.load(checkpoint)
        for sid,source in controls.items():
            with np.load(source['analysis_A']) as z:
                views=z['raw_views'].copy();times=z['times_seconds'].copy();mask=z['reliable'].copy()
            features=study.paired.model_module.frame_features(views[0])
            tokens,padding=study.paired.model_module.tokens_from_features(features,np.arange(len(features)))
            outputs=[]
            with torch.inference_mode():
                for start in range(0,len(tokens),128):
                    outputs.extend(model(torch.from_numpy(tokens[start:start+128]),torch.from_numpy(padding[start:start+128])).numpy())
            p=np.asarray(outputs,dtype=np.float32);p[views[0][:,66]<=.5]=0
            np.testing.assert_array_equal(p,predictions[f'fold_{fold}_'+sid]);arrays+=1
            saved=by_key[(fold,sid)]
            assert saved['role']==('held_out' if source['group'] in groups else 'fitted_source')
            assert (sid in meta['training_sources'])==(saved['role']=='fitted_source')
            assert saved['mean_abs_prediction_cents']==(float(np.mean(abs(p[mask]))) if mask.any() else None)
            assert saved['max_abs_prediction_cents']==(float(np.max(abs(p[mask]))) if mask.any() else None)
            assert saved['frames_at_or_above_activation']==int((mask&(abs(p)>=12)).sum())
            audio,sr=sf.read(source['path_A'],dtype='float32');assert sr==22050
            proposal=run.policy.proposal(audio,times,views,np.stack([p]*3))
            assert study.norm(proposal['plan'])==saved['plan']
            assert saved['permitted_nonidentity_proposal']==bool(proposal['plan']['permitted'] and not proposal['plan']['identity'])
            assert saved['admitted_short_candidates']==sum(c['reason'] is None for c in proposal['candidates'])
            replayed.append(saved)
        print('PRESERVATION_DIAGNOSTIC_REPLAY_FOLD',fold,flush=True)
    for role in ('fitted_source','held_out'):
        rows=[r for r in replayed if r['role']==role];s=result['summaries'][role]
        assert s['model_source_combinations']==len(rows)
        assert s['sources_with_activation_tail']==sum(r['frames_at_or_above_activation']>0 for r in rows)
        assert s['permitted_nonidentity_proposals']==sum(r['permitted_nonidentity_proposal'] for r in rows)
        assert s['max_source_prediction_cents']==max(r['max_abs_prediction_cents'] for r in rows if r['max_abs_prediction_cents'] is not None)
    delivery=study.read(root.parent/'2026-11-02_v52_humtrans_delivery/results.json')
    changed={r['source_id'] for r in delivery['arms']['added_real_hums']['records'] if r['kind']=='original' and not r['waveform_unchanged']}
    proposed={r['source_id'] for r in replayed if r['role']=='held_out' and r['permitted_nonidentity_proposal']}
    assert changed==proposed and len(changed)==4
    study.write(root/'integrity_receipt.json',dict(checkpoints_reloaded=4,original_prediction_arrays_reinferred=arrays,
        proposal_plans_replayed=len(replayed),held_original_proposals_match_delivered_changes=True,
        actual_changed_original_count=4,fitted_proposals_are_not_delivered_harm_measurements=True,
        results_sha256=study.sha(root/'results.json'),verifier_source_sha256=study.sha(__file__),
        new_model_fits=0,nontraining_audio_accessed=False,production_eligible=False,research_validation_passed=False))
    print('PRESERVATION_DIAGNOSTIC_INDEPENDENT_REPLAY_PASS',arrays,len(changed),flush=True)


if __name__=='__main__':main()
