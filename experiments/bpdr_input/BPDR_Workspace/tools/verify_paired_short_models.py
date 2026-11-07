"""Reload every paired-short checkpoint and replay physical/virtual sample lineage."""
from pathlib import Path
import sys
import numpy as np
import torch

WS=Path(__file__).resolve().parents[1];sys.path.insert(0,str(WS/'tools'))
import train_paired_short_models as run


def main():
    torch.set_num_threads(2);d,rows,originals=run.declare();data=run.records(rows,originals)
    result=run.read(run.ROOT/'results.json');assert result['declaration_sha256']==run.sha(run.ROOT/'declaration.json')
    predictions={};checkpoints=0;arrays=0;virtual_count=0
    for arm in run.ARMS:
        with np.load(run.ROOT/(arm+'_all_predictions.npz')) as z:predictions[arm]={k:z[k].copy() for k in z.files}
        assert set(predictions[arm])=={r['id'] for r in data}
    for fold,groups in enumerate(d['folds'],1):
        train=[r for r in data if r['group'] not in groups];test=[r for r in data if r['group'] in groups]
        trace,digest=run.previous.sample_trace(train,d)
        np.testing.assert_array_equal(trace,np.load(run.ROOT/f'fold_{fold}/physical_trace.npy'))
        assert {r['source_id'] for r in train}.isdisjoint({r['source_id'] for r in test})
        for arm in run.ARMS:
            folder=run.ROOT/f'fold_{fold}'/arm;receipt=run.read(folder/'fit_receipt.json')
            assert receipt['declaration_sha256']==run.sha(run.ROOT/'declaration.json')
            assert receipt['physical_trace_sha256']==digest and receipt['checkpoint_sha256']==run.sha(folder/'checkpoint.pt')
            assert receipt['training_groups']==sorted({r['group'] for r in train}) and receipt['heldout_groups']==groups
            assert receipt['training_sources']==sorted({r['source_id'] for r in train})
            assert receipt['heldout_sources']==sorted({r['source_id'] for r in test})
            model,payload=run.model_module.load(folder/'checkpoint.pt');checkpoints+=1
            assert payload['training_sources']==receipt['training_sources'] and payload['training_groups']==receipt['training_groups']
            assert payload['updates']==1200 and payload['seed']==339 and payload['arm']==arm
            with np.load(folder/'heldout_predictions.npz') as z:
                assert set(z.files)=={r['id'] for r in test}
                for record in test:
                    p=run.previous.predict(model,record,'short_transformer')
                    np.testing.assert_array_equal(p,z[record['id']]);np.testing.assert_array_equal(p,predictions[arm][record['id']]);arrays+=1
            saved=run.read(folder/'virtual_trace.json');assert run.sha(folder/'virtual_trace.json')==receipt['virtual_trace_sha256']
            if arm=='paired_physical':assert saved==[]
            else:
                assert len(saved)==4800==receipt['virtual_examples'];rng=np.random.default_rng(d['seed']+7919+fold)
                pools=run.virtual_pools(train,d)
                for index,expected in enumerate(saved):
                    *_,actual=run.draw_virtual(train,pools,rng,d)
                    assert run.norm(dict(update=index//12,**actual))==expected
                    assert actual['group'] in receipt['training_groups'] and actual['group'] not in groups
                    virtual_count+=1
            print('PAIRED_MODEL_REPLAY_FIT',fold,arm,flush=True)
    for arm,pred in predictions.items():assert run.norm(run.previous.summarize(data,pred))==result['arms'][arm]
    run.write(run.ROOT/'integrity_receipt.json',dict(checkpoints_reloaded=checkpoints,prediction_arrays_reinferred=arrays,
        physical_sampler_traces_recreated=4,virtual_examples_recreated=virtual_count,
        source_singer_overlap_found=False,all_summaries_reproduced=True,
        results_sha256=run.sha(run.ROOT/'results.json'),verifier_source_sha256=run.sha(__file__),
        raw_predictions_only=True,production_eligible=False,research_validation_passed=False))
    print('PAIRED_MODEL_INDEPENDENT_REPLAY_PASS',checkpoints,arrays,virtual_count,flush=True)


if __name__=='__main__':main()
