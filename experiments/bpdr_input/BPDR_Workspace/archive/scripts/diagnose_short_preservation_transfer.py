"""Locate preservation errors on fitted versus held-out original controls.

Diagnostic inference only: no refitting, operating-point selection or activation.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

import compare_humtrans_short_transfer as study
from bpdr_input.repair import short_phrase_policy as policy

ROOT = study.WS/'results/transformer_usability/2026-11-04_v56_preservation_diagnosis'


def main():
    torch.set_num_threads(2)
    d,_,originals = study.declare()
    receipt = study.read(study.ROOT/'integrity_receipt.json')
    assert receipt['checkpoints_reloaded']==8 and receipt['prediction_arrays_reinferred']==408
    controls = [r for r in originals if 'excluded' not in r]
    assert len(controls)==28
    ROOT.mkdir(parents=True,exist_ok=True)
    rows=[]; predictions={}; checkpoint_hashes={}
    with np.load(study.ROOT/'added_real_hums_all_predictions.npz') as z:
        heldout={key:z[key].copy() for key in z.files}
    for fold,groups in enumerate(d['folds'],1):
        checkpoint=study.ROOT/f'fold_{fold}/checkpoint.pt'
        model,meta=study.paired.model_module.load(checkpoint)
        checkpoint_hashes[str(checkpoint)]=study.sha(checkpoint)
        for source in controls:
            audio,sr=sf.read(source['path_A'],dtype='float32'); assert sr==22050
            with np.load(source['analysis_A']) as z:
                views=z['raw_views'].copy(); times=z['times_seconds'].copy(); mask=z['reliable'].copy()
            pred,_=study.paired.model_module.predict_raw(model,views[0])
            is_held=source['group'] in groups
            assert (source['source_id'] in meta['training_sources']) == (not is_held)
            if is_held:
                np.testing.assert_array_equal(pred,heldout['A_'+source['source_id']])
            proposal=policy.proposal(audio,times,views,np.stack([pred]*3))
            plan=proposal['plan']
            key=f'fold_{fold}_'+source['source_id']; predictions[key]=pred
            rows.append(dict(fold=fold,source_id=source['source_id'],group=source['group'],
                role='held_out' if is_held else 'fitted_source',
                mean_abs_prediction_cents=float(np.mean(abs(pred[mask]))) if mask.any() else None,
                max_abs_prediction_cents=float(np.max(abs(pred[mask]))) if mask.any() else None,
                reliable_frames=int(mask.sum()),frames_at_or_above_activation=int((mask&(abs(pred)>=12)).sum()),
                admitted_short_candidates=sum(c['reason'] is None for c in proposal['candidates']),
                permitted_nonidentity_proposal=bool(plan['permitted'] and not plan['identity']),
                plan=plan,original_zero_control=True))
        print('PRESERVATION_TRANSFER_DIAGNOSTIC_FOLD',fold,flush=True)
    summaries={}
    for role in ('fitted_source','held_out'):
        subset=[r for r in rows if r['role']==role]
        summaries[role]=dict(model_source_combinations=len(subset),
            sources_with_activation_tail=sum(r['frames_at_or_above_activation']>0 for r in subset),
            permitted_nonidentity_proposals=sum(r['permitted_nonidentity_proposal'] for r in subset),
            proposal_sources=[dict(fold=r['fold'],source_id=r['source_id'],group=r['group']) for r in subset if r['permitted_nonidentity_proposal']],
            max_source_prediction_cents=max(r['max_abs_prediction_cents'] for r in subset if r['max_abs_prediction_cents'] is not None))
    assert summaries['fitted_source']['model_source_combinations']==84 and summaries['held_out']['model_source_combinations']==28
    np.savez_compressed(ROOT/'original_predictions.npz',**predictions)
    study.write(ROOT/'results.json',dict(source_sha256=study.sha(__file__),policy_source_sha256=study.sha(policy.__file__),
        model_integrity_sha256=study.sha(study.ROOT/'integrity_receipt.json'),checkpoint_hashes=checkpoint_hashes,
        original_predictions_sha256=study.sha(ROOT/'original_predictions.npz'),rows=rows,summaries=summaries,
        scope='Fitted-source combinations repeat originals across models; not independent people or fresh validation. A proposal is not a delivered edit.',
        new_model_fits=0,nontraining_audio_accessed=False,production_eligible=False,research_validation_passed=False))
    print('PRESERVATION_TRANSFER_DIAGNOSTIC_COMPLETE',summaries,flush=True)


if __name__=='__main__':
    main()
