"""Independent V67 token/inference/permission and summary replay."""
import numpy as np
import soundfile as sf
import torch
import diagnose_preservation_sampling as run


def main():
    torch.set_num_threads(2);study=run.study;d,_,originals=study.declare();root=run.ROOT
    result=study.read(root/'results.json');delivery=study.read(run.DELIVERY/'results.json')
    assert result['source_sha256']==study.sha(run.__file__)
    assert result['model_integrity_sha256']==study.sha(study.ROOT/'integrity_receipt.json')
    assert result['delivery_integrity_sha256']==study.sha(run.DELIVERY/'integrity_receipt.json')
    assert result['policy_source_sha256']==study.sha(run.policy.__file__)
    assert result['original_predictions_sha256']==study.sha(root/'original_predictions.npz')
    controls={r['source_id']:r for r in originals if 'excluded' not in r}
    saved={(r['arm'],r['fold'],r['source_id']):r for r in result['rows']};assert len(saved)==424
    with np.load(root/'original_predictions.npz') as z:predictions={k:z[k].copy() for k in z.files}
    arrays=0;heldchecks=0
    for arm in study.ARMS:
        actual={r['source_id']:r for r in delivery['arms'][arm]['records'] if r['kind']=='original'}
        for fold,groups in enumerate(d['folds'],1):
            checkpoint=study.ROOT/f'fold_{fold}'/arm/'checkpoint.pt';assert result['checkpoint_hashes'][str(checkpoint)]==study.sha(checkpoint)
            model,meta=study.paired.model_module.load(checkpoint)
            for sid,source in controls.items():
                row=saved[(arm,fold,sid)]
                with np.load(source['analysis_A']) as z:views=z['raw_views'].copy();times=z['times_seconds'].copy();mask=z['reliable'].copy()
                features=study.paired.model_module.frame_features(views[0]);tokens,padding=study.paired.model_module.tokens_from_features(features,np.arange(len(features)));outputs=[]
                with torch.inference_mode():
                    for start in range(0,len(tokens),128):outputs.extend(model(torch.from_numpy(tokens[start:start+128]),torch.from_numpy(padding[start:start+128])).numpy())
                p=np.asarray(outputs,np.float32);p[views[0][:,66]<=.5]=0
                np.testing.assert_array_equal(p,predictions[f'{arm}_fold{fold}_'+sid]);arrays+=1
                held=source['group'] in groups;assert row['role']==('held_out' if held else 'fitted_source')
                assert (sid in meta['training_sources'])==(not held)
                assert row['mean_abs_prediction_cents']==(float(np.mean(abs(p[mask]))) if mask.any() else None) and row['max_abs_prediction_cents']==(float(np.max(abs(p[mask]))) if mask.any() else None)
                assert row['has_reliable_pitch_evidence']==bool(mask.any())
                assert row['frames_at_or_above_activation']==int(np.sum(mask&(abs(p)>=12)))
                audio,sr=sf.read(source['path_A'],dtype='float32');assert sr==22050
                proposal=run.policy.proposal(audio,times,views,np.stack([p]*3));assert study.norm(proposal['plan'])==row['plan']
                assert row['permitted_nonidentity_proposal']==bool(proposal['plan']['permitted'] and not proposal['plan']['identity'])
                if held:
                    assert row['plan']==actual[sid]['plan'];assert row['actual_original_changed']==(not actual[sid]['waveform_unchanged']);heldchecks+=1
                else:assert row['actual_original_changed'] is None
            print('PRESERVATION_SAMPLING_DIAGNOSIS_REPLAY',arm,fold,flush=True)
        for role,count in [('fitted_source',159),('held_out',53)]:
            rows=[r for r in saved.values() if r['arm']==arm and r['role']==role];summary=result['summaries'][arm][role]
            assert len(rows)==summary['model_source_combinations']==count
            assert summary['model_source_combinations_with_reliable_pitch']==sum(r['has_reliable_pitch_evidence'] for r in rows)
            assert summary['sources_with_activation_tail']==sum(r['frames_at_or_above_activation']>0 for r in rows)
            assert summary['permitted_nonidentity_proposals']==sum(r['permitted_nonidentity_proposal'] for r in rows)
            assert summary['max_prediction_cents']==max((r['max_abs_prediction_cents'] for r in rows if r['max_abs_prediction_cents'] is not None),default=None)
            assert summary['actual_changed_originals']==(sum(r['actual_original_changed'] for r in rows) if role=='held_out' else None)
    study.write(root/'integrity_receipt.json',dict(checkpoints_reloaded=8,original_prediction_arrays_reinferred=arrays,
        proposal_plans_replayed=arrays,held_original_plans_and_delivery_events_matched=heldchecks,
        fitted_proposals_are_not_delivered_harm_measurements=True,results_sha256=study.sha(root/'results.json'),verifier_source_sha256=study.sha(__file__),
        new_model_fits=0,production_eligible=False,research_validation_passed=False))
    print('PRESERVATION_SAMPLING_DIAGNOSIS_REPLAY_PASS',arrays,heldchecks,flush=True)


if __name__=='__main__':main()
