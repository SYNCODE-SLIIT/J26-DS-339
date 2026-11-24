"""Compare original-preservation tails on fitted and held sources after V65/V66.

No fit, threshold sweep or model promotion. Checks held proposals against verified
actual-WAV output records; fitted proposals are diagnostic, not measured harms.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import compare_preservation_sampling as study
from bpdr_input.repair import short_phrase_policy as policy

ROOT=study.WS/'results/transformer_usability/2026-11-04_v67_preservation_diagnosis'
DELIVERY=ROOT.parent/'2026-11-04_v66_preservation_delivery'


def main():
    torch.set_num_threads(2);d,_,originals=study.declare()
    proof=study.read(study.ROOT/'integrity_receipt.json');delivered=study.read(DELIVERY/'results.json');delivery_proof=study.read(DELIVERY/'integrity_receipt.json')
    assert proof['checkpoints_reloaded']==8 and proof['results_sha256']==study.sha(study.ROOT/'results.json')
    assert delivery_proof['results_sha256']==study.sha(DELIVERY/'results.json')
    controls=[r for r in originals if 'excluded' not in r];assert len(controls)==53
    ROOT.mkdir(exist_ok=True);rows=[];predictions={};hashes={};metadata={}
    for arm in study.ARMS:
        with np.load(study.ROOT/(arm+'_all_predictions.npz')) as z:heldout={k:z[k].copy() for k in z.files}
        output={r['id']:r for r in delivered['arms'][arm]['records'] if r['kind']=='original'}
        for fold,held in enumerate(d['folds'],1):
            path=study.ROOT/f'fold_{fold}'/arm/'checkpoint.pt';model,meta=study.paired.model_module.load(path)
            hashes[str(path)]=study.sha(path);assert study.read(path.parent/'fit_receipt.json')['checkpoint_sha256']==hashes[str(path)]
            for source in controls:
                audio,sr=sf.read(source['path_A'],dtype='float32');assert sr==22050
                with np.load(source['analysis_A']) as z:views=z['raw_views'].copy();times=z['times_seconds'].copy();mask=z['reliable'].copy()
                p,_=study.paired.model_module.predict_raw(model,views[0]);is_held=source['group'] in held
                assert (source['source_id'] in meta['training_sources'])==(not is_held)
                assert (source['group'] in meta['training_groups'])==(not is_held)
                proposal=policy.proposal(audio,times,views,np.stack([p]*3));plan=proposal['plan'];actual=None
                if is_held:
                    np.testing.assert_array_equal(p,heldout['A_'+source['source_id']])
                    saved=output['A_'+source['source_id']];assert study.norm(plan)==saved['plan']
                    actual=not saved['waveform_unchanged']
                key=f'{arm}_fold{fold}_'+source['source_id'];predictions[key]=p
                rows.append(dict(arm=arm,fold=fold,source_id=source['source_id'],group=source['group'],role='held_out' if is_held else 'fitted_source',
                    has_reliable_pitch_evidence=bool(mask.any()),reliable_frames=int(mask.sum()),mean_abs_prediction_cents=float(np.mean(abs(p[mask]))) if mask.any() else None,max_abs_prediction_cents=float(np.max(abs(p[mask]))) if mask.any() else None,
                    frames_at_or_above_activation=int(np.sum(mask&(abs(p)>=12))),permitted_nonidentity_proposal=bool(plan['permitted'] and not plan['identity']),
                    actual_original_changed=actual,plan=plan))
            print('PRESERVATION_SAMPLING_ORIGINAL_DIAGNOSIS',arm,fold,flush=True)
    summaries={}
    for arm in study.ARMS:
        summaries[arm]={}
        for role,count in [('fitted_source',159),('held_out',53)]:
            subset=[r for r in rows if r['arm']==arm and r['role']==role];assert len(subset)==count
            summaries[arm][role]=dict(model_source_combinations=len(subset),model_source_combinations_with_reliable_pitch=sum(r['has_reliable_pitch_evidence'] for r in subset),sources_with_activation_tail=sum(r['frames_at_or_above_activation']>0 for r in subset),
                permitted_nonidentity_proposals=sum(r['permitted_nonidentity_proposal'] for r in subset),
                max_prediction_cents=max((r['max_abs_prediction_cents'] for r in subset if r['max_abs_prediction_cents'] is not None),default=None),
                actual_changed_originals=sum(r['actual_original_changed'] for r in subset) if role=='held_out' else None,
                proposal_sources=[dict(fold=r['fold'],source_id=r['source_id'],group=r['group']) for r in subset if r['permitted_nonidentity_proposal']])
    np.savez_compressed(ROOT/'original_predictions.npz',**predictions)
    study.write(ROOT/'results.json',dict(source_sha256=study.sha(__file__),policy_source_sha256=study.sha(policy.__file__),
        model_integrity_sha256=study.sha(study.ROOT/'integrity_receipt.json'),delivery_integrity_sha256=study.sha(DELIVERY/'integrity_receipt.json'),
        checkpoint_hashes=hashes,original_predictions_sha256=study.sha(ROOT/'original_predictions.npz'),rows=rows,summaries=summaries,
        held_proposals_exactly_match_verified_delivery=True,
        scope='Fitted_source denotes a training-assigned record/model combination, not proof that every window was sampled; reliable-pitch evidence is separately counted. Repeated combinations are not independent people; fitted proposals are not rendered harm measurements. Held delivered original events are separately verified in V66.',
        new_model_fits=0,nontraining_audio_accessed=False,production_eligible=False,research_validation_passed=False))
    print('PRESERVATION_SAMPLING_ORIGINAL_DIAGNOSIS_COMPLETE',summaries,flush=True)


if __name__=='__main__':main()
