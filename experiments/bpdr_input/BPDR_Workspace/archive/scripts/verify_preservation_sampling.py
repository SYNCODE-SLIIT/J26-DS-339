"""Reload fixed-budget models and replay training-only mining and held predictions."""
import hashlib
import numpy as np
import torch
import compare_preservation_sampling as run


def main():
    torch.set_num_threads(2);d,rows,originals=run.declare();root=run.ROOT
    result=run.read(root/'results.json');assert result['declaration_sha256']==run.sha(root/'declaration.json')
    for p,h in d['source_hashes'].items():assert run.sha(p)==h
    data=run.paired.records(rows,originals);predictions={}
    for arm in run.ARMS:
        with np.load(root/(arm+'_all_predictions.npz')) as z:predictions[arm]={k:z[k].copy() for k in z.files}
        assert set(predictions[arm])=={r['id'] for r in data}
    arrays=0;models=0;traces=0;teacher_arrays=0
    for fold,held in enumerate(d['folds'],1):
        train=[r for r in data if r['group'] not in held];test=[r for r in data if r['group'] in held]
        assert not {r['source_id'] for r in train}&{r['source_id'] for r in test}
        base=root/f'fold_{fold}';teacher_path=run.parent.ROOT/f'fold_{fold}/checkpoint.pt'
        assert d['parent_checkpoints'][str(teacher_path)]==run.sha(teacher_path)
        teacher,meta=run.paired.model_module.load(teacher_path);assert not set(meta['training_groups'])&set(held)
        p,protected=run.mining_evidence(train,teacher)
        with np.load(base/'mining_evidence.npz') as z:
            assert len(z.files)==2*len(train)
            for i in range(len(train)):
                np.testing.assert_array_equal(z['prediction_'+str(i)],p[i]);np.testing.assert_array_equal(z['protected_'+str(i)],protected[i]);teacher_arrays+=2
        trace,_=run.paired.previous.sample_trace(train,d)
        mined,stats=run.mining.mined_trace(train,trace,p,protected,d['seed']+fold)
        assert run.read(base/'mining_stats.json')==dict(stats,training_record_ids=[r['id'] for r in train])
        # Direct sampler invariants, independent of helper assertions.
        np.testing.assert_array_equal(mined[:,:,0],trace[:,:,0]);np.testing.assert_array_equal(mined[:,:12],trace[:,:12])
        for slot in list(range(12,18))+list(range(24,30)):
            for index,frame in mined[:,slot]:
                r=train[index];assert r['mask'][frame] and r['target'][frame]==0
        for arm,expected in [('uniform',trace),('preservation_mined',mined)]:
            folder=base/arm;np.testing.assert_array_equal(np.load(folder/'physical_trace.npy'),expected);traces+=1
            receipt=run.read(folder/'fit_receipt.json');assert receipt['physical_trace_sha256']==hashlib.sha256(expected.tobytes()).hexdigest()
            assert receipt['checkpoint_sha256']==run.sha(folder/'checkpoint.pt')
            assert receipt['mining_evidence_sha256']==run.sha(base/'mining_evidence.npz')
            model,meta=run.paired.model_module.load(folder/'checkpoint.pt')
            assert meta['training_groups']==sorted({r['group'] for r in train})
            assert meta['training_sources']==sorted({r['source_id'] for r in train})
            assert meta['declaration_sha256']==run.sha(root/'declaration.json') and meta['updates']==1200 and meta['seed']==339
            assert receipt['heldout_groups']==held and receipt['parameters']==21666
            with np.load(folder/'heldout_predictions.npz') as saved:
                assert set(saved.files)=={r['id'] for r in test}
                for r in test:
                    prediction=run.paired.previous.predict(model,r,'short_transformer')
                    np.testing.assert_array_equal(prediction,saved[r['id']]);np.testing.assert_array_equal(prediction,predictions[arm][r['id']]);arrays+=1
            models+=1
        print('PRESERVATION_SAMPLING_REPLAY_FOLD',fold,flush=True)
    for arm,pred in predictions.items():
        for name,subset in [('all',data),('humming',[r for r in data if r['group'].startswith('HumTrans:')]),('vocadito',[r for r in data if not r['group'].startswith('HumTrans:')])]:
            assert run.paired.previous.summarize(subset,pred)==result['arms'][arm][name]
    run.write(root/'integrity_receipt.json',dict(checkpoints_reloaded=models,heldout_prediction_arrays_recomputed=arrays,
        sampler_traces_replayed=traces,training_only_teacher_arrays_recomputed=teacher_arrays,all_summaries_reproduced=True,
        results_sha256=run.sha(root/'results.json'),verifier_source_sha256=run.sha(__file__),
        raw_predictions_only=True,delivered_audio_qualification=False,research_validation_passed=False,production_eligible=False))
    print('PRESERVATION_SAMPLING_REPLAY_PASS',models,arrays,traces,teacher_arrays,flush=True)


if __name__=='__main__':main()
