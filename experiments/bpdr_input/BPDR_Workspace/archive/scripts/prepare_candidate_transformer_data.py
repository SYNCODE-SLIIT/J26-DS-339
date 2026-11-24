"""Prospective candidate-level data preflight; no model fitting or evaluation access."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex,held_voice_training as held,model_recovery as recovery
from bpdr_input.experiments.learning.candidate_labels import label_candidate
from bpdr_input.modeling.step_candidates import measured_candidates,candidate_context

ROOT=WORKSPACE/'results/transformer_usability/2026-10-29_v21_candidate_data'
POOL=ROOT.parent/'2026-10-29_v17_audio_pool'
BASE=ROOT.parent/'2026-10-29_v18_audio_augmented'


def main():
    torch.set_num_threads(2);ROOT.mkdir(exist_ok=True)
    ready=ex.read(POOL/'acoustic_pool_receipt.json');assert ready['training_ready']
    assert ex.sha(ready['fit_manifest'])==ready['fit_manifest_sha256']
    files=[Path(__file__).resolve(),WORKSPACE/'source/bpdr_input/modeling/step_candidates.py',
           WORKSPACE/'source/bpdr_input/experiments/learning/candidate_labels.py',
           WORKSPACE/'source/bpdr_input/modeling/step_context.py',
           WORKSPACE/'source/bpdr_input/modeling/joint_transformer.py',
           WORKSPACE/'source/bpdr_input/modeling/held_voice_transformer.py']
    if not (ROOT/'declaration.json').exists():
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
                 role='Training-only candidate geometry/label preflight. Not model evaluation or completion.',
                 pool_receipt_sha256=ex.sha(POOL/'acoustic_pool_receipt.json'),
                 fit_manifest=str(ready['fit_manifest']),fit_manifest_sha256=ready['fit_manifest_sha256'],
                 anchors=str(BASE/'anchor_sources.json'),anchors_sha256=ex.sha(BASE/'anchor_sources.json'),
                 generation='Only contiguous nonzero input-only step evidence with>=43 frames. No expansion/trimming, reference interval or labels used to generate candidates. Context is51x8 broad+step tokens and an input-derived region pooling mask.',
                 labels='Training only.>=50% known frames; unchanged if all known labels0. Positive only with>=50% region positive,>=95% known-label purity and one constant bounded correction. Mixed/missing labels unassessed, never fabricated.',
                 original_control='Each original recording once from the50-source list, not repeated for every derivative. Original0 is engineering recovery control, not a human intended-pitch label.',
                 source_hashes={str(p):ex.sha(p) for p in files},
                 input_hashes=ex.read(BASE/'declaration.json')['input_hashes'],
                 numerical_guards_changed=False,new_evaluation_access=False,research_validation_passed=False))
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in files:(snapshot/p.name).write_bytes(p.read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for name in ('source_hashes','input_hashes'):
        for p,h in d[name].items():assert ex.sha(p)==h,p
    if (ROOT/'preflight.json').exists():raise ValueError('Existing candidate preflight retained')
    examples=[];cases=[];xx=[];pads=[];pools=[];labels=[];stats=Counter();groups=defaultdict(Counter)
    def add(raw,known,target,source,path,role):
        intervals=measured_candidates(raw);case=dict(source_sha256=source['sha256'],group=source['singer_id'],
                       path=path,role=role,measured_candidates=len(intervals))
        cases.append(case);stats['input_cases']+=1
        if not intervals:stats['cases_without_candidates']+=1
        for interval in intervals:
            label=label_candidate(interval,target,known)
            x,pad,pool=candidate_context(raw,interval)
            xx.append(x);pads.append(pad);pools.append(pool);labels.append(np.nan if label is None else label)
            kind='unassessed' if label is None else 'negative' if label==0 else 'positive'
            stats[kind]+=1;groups[source['singer_id']][kind]+=1
            examples.append(dict(index=len(examples),**case,start=interval[0],stop=interval[1],label_state=kind))
    cache={}
    for source in ex.read(BASE/'anchor_sources.json'):
        path=Path(source['working_audio'])
        _,analysis=recovery.from_cached(str(path),path.parent/'original_features.npz',cache)
        raw=analysis['views'][0]
        add(raw,analysis['reliable'],np.zeros(len(raw),np.float32),source['source'],str(path),'original_control')
    for pair in held.prepared(ex.read(ready['fit_manifest'])):
        if 'excluded' in pair:continue
        add(pair['C']['views'][0],pair['mask']&pair['C']['reliable'],pair['target'],pair['source'],pair['path_C'],'actual_corrupted_training_variant')
    if xx:
        np.savez_compressed(ROOT/'candidate_inputs.npz',features=np.stack(xx),padding=np.stack(pads),pool_mask=np.stack(pools))
        np.savez_compressed(ROOT/'candidate_labels_NOT_MODEL_INPUT.npz',target=np.asarray(labels,np.float32),known=np.isfinite(labels))
    ex.write(ROOT/'candidate_records.json',examples);ex.write(ROOT/'source_case_records.json',cases)
    result=dict(counts=dict(stats),by_singer={g:dict(v) for g,v in groups.items()},
                original_singers=10,original_sources=50,actual_training_variants=373,
                model_fitted=False,evaluation_denominator_unchanged='200 attempts/108 scoreable; not evaluated here',
                artifact_hashes={str(p):ex.sha(p) for p in ROOT.glob('candidate_*') if p.is_file()},
                declaration_sha256=ex.sha(ROOT/'declaration.json'),research_validation_passed=False)
    ex.write(ROOT/'preflight.json',result)
    for name in ('source_hashes','input_hashes'):
        for p,h in d[name].items():assert ex.sha(p)==h,p
    print('CANDIDATE_PREFLIGHT',dict(stats),result['by_singer'],flush=True)


if __name__=='__main__':main()
