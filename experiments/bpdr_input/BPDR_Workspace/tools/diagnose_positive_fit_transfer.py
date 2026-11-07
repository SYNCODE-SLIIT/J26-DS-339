"""Sparse positive/paired-original head diagnosis on verified V68 checkpoints.

No new fit or counterfactual ungated-model claim. Head activity is not calibrated
musical-intention confidence. Query roles distinguish assignment from exposure.
"""
from collections import defaultdict
import numpy as np
import torch
import compare_tail_preservation as run


def metrics(p,logit,amplitude,ap,al,aa,target,mask):
    if not mask.any():return None
    p=p[mask];target=target[mask];ap=ap[mask]
    return dict(frames=int(mask.sum()),positive_mae_cents=float(np.mean(abs(p-target))),signed_response=float(np.mean(p*target)/np.mean(target**2)),
        correct_direction_fraction=float(np.mean(p*target>0)),activity_mean=float(np.mean(1/(1+np.exp(-logit[mask])))),
        amplitude_head_mae_cents=float(np.mean(abs(amplitude[mask]-target))),paired_original_mae_cents=float(np.mean(abs(ap))),
        paired_original_activity_mean=float(np.mean(1/(1+np.exp(-al[mask])))),paired_original_amplitude_abs_cents=float(np.mean(abs(aa[mask]))),
        paired_response=float(np.mean((p-ap)*target)/np.mean(target**2)))


def aggregate(rows,key,role):
    rows=[r for r in rows if r['role']==role and r[key] is not None];groups=sorted({r['group'] for r in rows})
    fields=['positive_mae_cents','signed_response','correct_direction_fraction','activity_mean','amplitude_head_mae_cents','paired_original_mae_cents','paired_original_activity_mean','paired_original_amplitude_abs_cents','paired_response']
    means={field:float(np.mean([np.mean([r[key][field] for r in rows if r['group']==group]) for group in groups])) if groups else None for field in fields}
    return dict(record_model_combinations=len(rows),positive_frames=sum(r[key]['frames'] for r in rows),groups=len(groups),**means)


def main():
    torch.set_num_threads(2);d,pairs,originals=run.declare();proof=run.read(run.ROOT/'integrity_receipt.json')
    assert proof['results_sha256']==run.sha(run.ROOT/'results.json') and d['config']['gated']
    sources={r['source_id']:r for r in originals if 'excluded' not in r};records=[];cache={};targets={};rows=[];checkpoints={};held_checks=0
    for source in sources.values():records.append(dict(id='A_'+source['source_id'],kind='original',source_id=source['source_id'],group=source['group']))
    for index,pair in enumerate(pairs):
        if 'excluded' not in pair:records.append(dict(id='C_'+str(index),kind='corrupted',source_id=pair['source_id'],group=pair['group'],pair_index=index))
    def features(path):
        if path not in cache:
            with np.load(path) as z:cache[path]=run.model_module.frame_features(z['raw_views'][0])
        return cache[path]
    def infer(model,path,frames):
        values=features(path);result=[np.empty(len(frames),np.float32) for _ in range(3)]
        # Reproduce the original128-row inference blocks exactly. Small selected
        # batches can select different float32 matrix kernels; do not relax checks.
        for start in np.unique((frames//128)*128):
            stop=min(int(start)+128,len(values));x,pad=run.model_module.tokens_from_features(values,np.arange(start,stop))
            with torch.inference_mode():outputs=model(torch.from_numpy(x),torch.from_numpy(pad),True)
            use=(frames>=start)&(frames<stop);index=frames[use]-start
            for destination,output in zip(result,outputs):destination[use]=output.numpy()[index]
        return tuple(result)
    with np.load(run.ROOT/'upper_tail_preservation_all_predictions.npz') as z:held={k:z[k].copy() for k in z.files}
    for fold,groups in enumerate(d['folds'],1):
        train=[r for r in records if r['group'] not in groups]
        assert [r['id'] for r in train]==run.read(run.previous.ROOT/f'fold_{fold}/mining_stats.json')['training_record_ids']
        trace=np.load(run.ROOT/f'fold_{fold}/upper_tail_preservation/physical_trace.npy');queried=defaultdict(set)
        for index,frame in trace[:,:12].reshape(-1,2):queried[train[index]['id']].add(int(frame))
        path=run.ROOT/f'fold_{fold}/upper_tail_preservation/checkpoint.pt';model,meta=run.model_module.load(path)
        checkpoints[str(path)]=run.sha(path);assert run.read(path.parent/'fit_receipt.json')['checkpoint_sha256']==checkpoints[str(path)]
        for record in records:
            if record['kind']!='corrupted':continue
            pair=pairs[record['pair_index']];source=sources[record['source_id']]
            if record['id'] not in targets:
                with np.load(pair['target_path']) as z:targets[record['id']]=(np.flatnonzero(z['plateau']),z['target'].copy())
            frames,target=targets[record['id']];target=target[frames]
            p,logit,amp=infer(model,pair['analysis_C'],frames);ap,al,aa=infer(model,source['analysis_A'],frames)
            fitted=record['group'] not in groups;assert (record['source_id'] in meta['training_sources'])==fitted
            seen=np.isin(frames,list(queried[record['id']])) if fitted else np.zeros(len(frames),bool)
            if not fitted:
                np.testing.assert_array_equal(p,held[record['id']][frames]);held_checks+=1
            row=dict(id=record['id'],fold=fold,source_id=record['source_id'],group=record['group'],role='training_assigned' if fitted else 'held_out',
                all=metrics(p,logit,amp,ap,al,aa,target,np.ones(len(frames),bool)),
                directly_queried=metrics(p,logit,amp,ap,al,aa,target,seen) if fitted else None,
                not_directly_queried=metrics(p,logit,amp,ap,al,aa,target,~seen) if fitted else None)
            rows.append(row)
        print('POSITIVE_FIT_DIAGNOSIS_FOLD',fold,flush=True)
    assert len(rows)==1780 and held_checks==445
    summary={name:aggregate(rows,key,role) for name,key,role in [('training_all','all','training_assigned'),('training_queried','directly_queried','training_assigned'),('training_not_queried','not_directly_queried','training_assigned'),('held_out','all','held_out')]}
    run.write(run.ROOT/'positive_fit_diagnosis.json',dict(summary=summary,rows=rows,checkpoints=checkpoints,source_sha256=run.sha(__file__),
        model_integrity_sha256=run.sha(run.ROOT/'integrity_receipt.json'),held_positive_predictions_exactly_replayed=held_checks,inference_batch_layout_matches_parent=True,
        scope='Inspected training-source sparse head diagnosis. Repeated fitted model/case/frame positions are not independent people. Activity is uncalibrated. Raw amplitude is a head output, not an evaluated ungated-model candidate; removing gating changes the model and requires its own training/evaluation.',
        new_model_fits=0,nontraining_audio_accessed=False,production_eligible=False,research_validation_passed=False))
    print('POSITIVE_FIT_DIAGNOSIS_COMPLETE',summary,flush=True)


if __name__=='__main__':main()
