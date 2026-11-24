"""Audit actual baseline training queries behind V67 fitted-original failures.

Counts direct original queries and byte-identical C-zero token queries only.
This does not measure all indirect gradients/generalization or prove causation.
No new fit, labels, waveform, operating threshold or held-data selection.
"""
from collections import defaultdict
import hashlib
import numpy as np
import compare_preservation_sampling as study

ROOT=study.WS/'results/transformer_usability/2026-11-04_v67_preservation_diagnosis'


def main():
    d,pairs,originals=study.declare();proof=study.read(ROOT/'integrity_receipt.json')
    assert proof['results_sha256']==study.sha(ROOT/'results.json')
    saved=study.read(ROOT/'results.json');by_key={(r['fold'],r['source_id']):r for r in saved['rows'] if r['arm']=='uniform'}
    sources={s['source_id']:s for s in originals if 'excluded' not in s};records=[];features={};masks={};times={};labels={};equivalence={}
    module=study.paired.model_module
    def load(path):
        if path not in features:
            with np.load(path) as z:
                raw=z['raw_views'][0].copy();masks[path]=z['reliable'].copy();times[path]=z['times_seconds'].copy()
            features[path]=module.frame_features(raw)
            np.testing.assert_array_equal(features[path][:,3]>.5,masks[path])
        return features[path],masks[path],times[path]
    for source in sources.values():records.append(dict(id='A_'+source['source_id'],kind='original',source_id=source['source_id'],group=source['group'],path=source['analysis_A']))
    for index,pair in enumerate(pairs):
        if 'excluded' not in pair:records.append(dict(id='C_'+str(index),kind='corrupted',source_id=pair['source_id'],group=pair['group'],path=pair['analysis_C'],target_path=pair['target_path']))
    def mask_target(record):
        path=record['target_path']
        if path not in labels:
            with np.load(path) as z:labels[path]=(z['mask'].copy(),z['target'].copy())
        return labels[path]
    rows=[];trace_hashes={};original_queries=0;zero_queries=0
    with np.load(ROOT/'original_predictions.npz') as z:predictions={k:z[k].copy() for k in z.files}
    for fold,held in enumerate(d['folds'],1):
        train=[r for r in records if r['group'] not in held]
        assert [r['id'] for r in train]==study.read(study.ROOT/f'fold_{fold}/mining_stats.json')['training_record_ids']
        path=study.ROOT/f'fold_{fold}/uniform/physical_trace.npy';trace=np.load(path)
        assert hashlib.sha256(trace.tobytes()).hexdigest()==study.read(path.parent/'fit_receipt.json')['physical_trace_sha256'];trace_hashes[str(path)]=study.sha(path)
        direct={r['source_id']:np.zeros(len(load(r['path'])[0]),bool) for r in train if r['kind']=='original'}
        equivalent={sid:np.zeros_like(mask) for sid,mask in direct.items()};zero_frames=defaultdict(list)
        for index,frame in trace[:,:12].reshape(-1,2):
            record=train[index];mask,target=mask_target(record);assert record['kind']=='corrupted' and mask[frame] and target[frame]!=0
            direct[record['source_id']][frame]=True;original_queries+=1
        for index,frame in trace[:,24:].reshape(-1,2):
            record=train[index];assert record['kind']=='original' and load(record['path'])[1][frame]
            direct[record['source_id']][frame]=True;original_queries+=1
        for index,frame in trace[:,12:24].reshape(-1,2):
            record=train[index];mask,target=mask_target(record);assert record['kind']=='corrupted' and mask[frame] and target[frame]==0
            zero_frames[int(index)].append(int(frame));zero_queries+=1
        for index,frames in zero_frames.items():
            record=train[index];path=record['path'];sid=record['source_id'];a=load(sources[sid]['analysis_A'])[0];c=load(path)[0]
            cache=equivalence.setdefault(record['id'],{})
            missing=np.array(sorted(set(frames)-set(cache)),dtype=int)
            if len(missing):
                ax,ap=module.tokens_from_features(a,missing);cx,cp=module.tokens_from_features(c,missing)
                same=np.all(ax==cx,axis=(1,2))&np.all(ap==cp,axis=1)
                cache.update({int(frame):bool(value) for frame,value in zip(missing,same)})
            for frame in set(frames):
                if cache[frame]:equivalent[sid][frame]=True
        for sid,direct_seen in direct.items():
            _,reliable,grid=load(sources[sid]['analysis_A']);combined=direct_seen|equivalent[sid]
            p=predictions[f'uniform_fold{fold}_'+sid];tail=reliable&(abs(p)>=12.)
            diagnostic=by_key[(fold,sid)];assert diagnostic['role']=='fitted_source'
            plan=np.zeros(len(grid),bool)
            for start,end in diagnostic['plan']['intervals_seconds']:plan|=(grid>=start)&(grid<end)
            plan&=reliable
            rows.append(dict(fold=fold,source_id=sid,group=sources[sid]['group'],reliable_frames=int(reliable.sum()),
                direct_original_frames_seen=int(np.sum(reliable&direct_seen)),direct_or_identical_zero_frames_seen=int(np.sum(reliable&combined)),
                activation_tail_frames=int(tail.sum()),activation_tail_frames_without_direct_query=int(np.sum(tail&~direct_seen)),
                activation_tail_frames_without_direct_or_identical_zero_query=int(np.sum(tail&~combined)),
                permitted_unwanted_proposal=diagnostic['permitted_nonidentity_proposal'],planned_reliable_frames=int(plan.sum()),
                planned_reliable_frames_without_direct_or_identical_query=int(np.sum(plan&~combined))))
        print('TRAINING_QUERY_COVERAGE_FOLD',fold,flush=True)
    assert len(rows)==159 and original_queries==115200 and zero_queries==57600
    fields=['reliable_frames','direct_original_frames_seen','direct_or_identical_zero_frames_seen','activation_tail_frames','activation_tail_frames_without_direct_query','activation_tail_frames_without_direct_or_identical_zero_query','planned_reliable_frames','planned_reliable_frames_without_direct_or_identical_query']
    summary={field:sum(r[field] for r in rows) for field in fields}
    summary.update(training_assigned_model_source_combinations=159,permitted_unwanted_proposals=sum(r['permitted_unwanted_proposal'] for r in rows),direct_original_query_draws=original_queries,zero_c_query_draws=zero_queries)
    study.write(ROOT/'training_query_coverage.json',dict(summary=summary,rows=rows,trace_file_hashes=trace_hashes,source_sha256=study.sha(__file__),
        diagnostic_integrity_sha256=study.sha(ROOT/'integrity_receipt.json'),feature_source_sha256=study.sha(module.__file__),
        scope='Baseline uniform training-assigned originals only; repeated frame/model combinations are not people. A seen query is not proof of optimization; indirect gradients from similar but nonidentical inputs are not counted. Coverage gaps alone are not a causal result.',
        new_model_fits=0,nontraining_audio_accessed=False,production_eligible=False,research_validation_passed=False))
    print('TRAINING_QUERY_COVERAGE_COMPLETE',summary,flush=True)


if __name__=='__main__':main()
