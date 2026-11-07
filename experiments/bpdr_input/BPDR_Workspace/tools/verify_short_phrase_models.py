"""Fresh checkpoint inference, source separation and matched-sampler verification."""
from pathlib import Path
import json,sys
import numpy as np
import torch,joblib

WS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WS/'source'));sys.path.insert(0,str(WS/'tools'))
import compare_short_phrase_models as study
from probe_short_phrase_renderer import sha


def main():
    torch.set_num_threads(2);d,rows,originals=study.declare();data=study.records(rows,originals)
    root=study.ROOT;result=study.read(root/'results.json');predictions={};checkpoints=0;arrays=0;samplers=0
    assert result['declaration_sha256']==sha(root/'declaration.json')
    for arm in ['frozen_joint',*study.ARMS]:
        with np.load(root/(arm+'_all_predictions.npz')) as z:predictions[arm]={k:z[k].copy() for k in z.files}
        assert set(predictions[arm])=={r['id'] for r in data}
    frozen,payload=study.old.load(d['frozen_model'])
    for r in data:
        p=study.predict(frozen,r,'frozen_joint');np.testing.assert_array_equal(p,predictions['frozen_joint'][r['id']]);arrays+=1
    for index,groups in enumerate(d['folds'],1):
        train=[r for r in data if r['group'] not in groups];test=[r for r in data if r['group'] in groups]
        trace,digest=study.sample_trace(train,d)
        np.testing.assert_array_equal(trace,np.load(root/f'fold_{index}/sampler_trace.npy'));samplers+=1
        train_sources={r['source_id'] for r in train};test_sources={r['source_id'] for r in test}
        assert train_sources.isdisjoint(test_sources)
        for arm in study.ARMS:
            folder=root/f'fold_{index}'/arm;receipt=study.read(folder/'fit_receipt.json')
            assert receipt['sampler_sha256']==digest
            assert set(receipt['training_groups'])=={r['group'] for r in train}
            assert receipt['heldout_groups']==groups
            assert set(receipt['training_sources'])==train_sources and set(receipt['heldout_sources'])==test_sources
            assert receipt['declaration_sha256']==sha(root/'declaration.json')
            path=folder/receipt['checkpoint'];assert sha(path)==receipt['checkpoint_sha256']
            if arm=='short_extra_trees':model=joblib.load(path)
            else:
                model,payload=(study.old if arm=='joint_retrained' else study.short).load(path)
                assert set(payload['training_sources'])==train_sources and set(payload['training_groups'])=={r['group'] for r in train}
                assert payload['updates']==800 and payload['seed']==339
            checkpoints+=1
            with np.load(folder/'heldout_predictions.npz') as z:
                assert set(z.files)=={r['id'] for r in test}
                for r in test:
                    p=study.predict(model,r,arm)
                    np.testing.assert_array_equal(p,z[r['id']]);np.testing.assert_array_equal(p,predictions[arm][r['id']]);arrays+=1
    for arm,pred in predictions.items():assert study.norm(study.summarize(data,pred))==result['arms'][arm]
    receipt=dict(checkpoints_reloaded=checkpoints,frozen_checkpoint_reloaded=True,
        prediction_arrays_reinferred=arrays,matched_sampler_traces_recreated=samplers,
        singer_and_source_overlap_found=False,all_arm_summaries_reproduced=True,
        verifier_source_sha256=sha(__file__),results_sha256=sha(root/'results.json'),
        source_declaration_sha256=sha(root/'declaration.json'),raw_predictions_only=True,
        delivered_audio_qualification=False,research_validation_passed=False,production_eligible=False)
    study.write(root/'integrity_receipt.json',receipt)
    print('SHORT_MODEL_INDEPENDENT_REPLAY_PASS',checkpoints,arrays,samplers,flush=True)


if __name__=='__main__':main()
