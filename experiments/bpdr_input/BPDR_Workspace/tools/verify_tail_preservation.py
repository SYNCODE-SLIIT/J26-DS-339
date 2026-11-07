"""Reload four frozen parents/four tail-loss models and exact held traces."""
import numpy as np
import torch
import compare_tail_preservation as run


def main():
    torch.set_num_threads(2);d,rows,originals=run.declare();result=run.read(run.ROOT/'results.json')
    assert result['declaration_sha256']==run.sha(run.ROOT/'declaration.json')
    for p,h in d['source_hashes'].items():assert run.sha(p)==h
    data=run.paired.records(rows,originals);predictions={}
    for arm in run.ARMS:
        with np.load(run.ROOT/(arm+'_all_predictions.npz')) as z:predictions[arm]={k:z[k].copy() for k in z.files}
        assert set(predictions[arm])=={r['id'] for r in data}
    arrays=0;models=0;traces=0
    for fold,groups in enumerate(d['folds'],1):
        train=[r for r in data if r['group'] not in groups];test=[r for r in data if r['group'] in groups]
        trace,digest=run.paired.previous.sample_trace(train,d)
        np.testing.assert_array_equal(trace,np.load(run.previous.ROOT/f'fold_{fold}/uniform/physical_trace.npy'))
        np.testing.assert_array_equal(trace,np.load(run.ROOT/f'fold_{fold}/upper_tail_preservation/physical_trace.npy'));traces+=1
        for arm in run.ARMS:
            path=(run.previous.ROOT/f'fold_{fold}/uniform/checkpoint.pt' if arm=='frozen_uniform' else run.ROOT/f'fold_{fold}/upper_tail_preservation/checkpoint.pt')
            receipt=run.read(path.parent/'fit_receipt.json');assert receipt['checkpoint_sha256']==run.sha(path) and receipt['physical_trace_sha256']==digest
            model,meta=run.model_module.load(path);assert meta['training_groups']==sorted({r['group'] for r in train}) and meta['training_sources']==sorted({r['source_id'] for r in train})
            assert not set(meta['training_groups'])&set(groups);assert receipt['parameters']==21666
            assert meta['updates']==1200 and meta['seed']==339
            if arm=='upper_tail_preservation':assert meta['declaration_sha256']==run.sha(run.ROOT/'declaration.json') and meta['arm']==arm
            else:assert d['parent_checkpoints'][str(path)]==run.sha(path)
            for record in test:
                prediction=run.paired.previous.predict(model,record,'short_transformer');np.testing.assert_array_equal(prediction,predictions[arm][record['id']]);arrays+=1
            models+=1
        print('TAIL_PRESERVATION_REPLAY_FOLD',fold,flush=True)
    for arm,pred in predictions.items():
        for name,subset in [('all',data),('humming',[r for r in data if r['group'].startswith('HumTrans:')]),('vocadito',[r for r in data if not r['group'].startswith('HumTrans:')])]:assert run.paired.previous.summarize(subset,pred)==result['arms'][arm][name]
    with np.load(run.previous.ROOT/'uniform_all_predictions.npz') as z:
        for k,p in predictions['frozen_uniform'].items():np.testing.assert_array_equal(p,z[k])
    run.write(run.ROOT/'integrity_receipt.json',dict(checkpoints_reloaded=models,heldout_prediction_arrays_recomputed=arrays,sampler_traces_replayed=traces,
        all_summaries_reproduced=True,baseline_arrays_equal_verified_parent=True,results_sha256=run.sha(run.ROOT/'results.json'),verifier_source_sha256=run.sha(__file__),
        raw_predictions_only=True,delivered_audio_qualification=False,production_eligible=False,research_validation_passed=False))
    print('TAIL_PRESERVATION_REPLAY_PASS',models,arrays,traces,flush=True)


if __name__=='__main__':main()
