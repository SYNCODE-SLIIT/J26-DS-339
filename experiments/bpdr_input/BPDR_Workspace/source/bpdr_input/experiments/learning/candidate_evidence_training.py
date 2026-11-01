"""Matched region head with direct input-cue summaries; same sampling and loss."""
from collections import defaultdict
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from torch.nn import functional as F
from bpdr_input.experiments.learning import context_improvement as ex,held_voice_training as held,model_recovery as recovery
from bpdr_input.modeling.joint_transformer import load as load_parent
from bpdr_input.modeling.candidate_evidence_transformer import EvidenceCandidateTransformer as CandidateTransformer,predict_raw,save,load
from bpdr_input.modeling.step_candidates import candidate_context
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.repair.held_voice_policy import HeldPolicy


def embeddings(model,contexts):
    xx,pad,pool=[np.stack([c[k] for c in contexts]) for k in range(3)]
    result=[]
    for start in range(0,len(xx),128):
        hidden=model.embed(torch.from_numpy(xx[start:start+128]),torch.from_numpy(pad[start:start+128]),torch.from_numpy(pool[start:start+128]))
        result.append(hidden.numpy().copy())
    return np.concatenate(result)


def fit(data_root,source_rows,excluded,seed,updates,folder,declaration_hash,parent_checkpoint):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    rng=np.random.default_rng(seed);parent,payload=load_parent(parent_checkpoint)
    model=CandidateTransformer(parent)
    original_state={k:v.detach().clone() for k,v in parent.state_dict().items()}
    with np.load(data_root/'candidate_inputs.npz') as z:features={k:z[k].copy() for k in z.files}
    with np.load(data_root/'candidate_labels_NOT_MODEL_INPUT.npz') as z:target=z['target'].copy();known=z['known'].copy()
    records=ex.read(data_root/'candidate_records.json')
    indices=[i for i,r in enumerate(records) if r['group'] not in excluded and known[i]]
    if not indices:raise ValueError('No supervised training candidates')
    contexts=[(features['features'][i],features['padding'][i],features['pool_mask'][i]) for i in indices]
    embedded=embeddings(model,contexts);mapping={i:j for j,i in enumerate(indices)}
    positive=defaultdict(list);hard=defaultdict(list)
    for i in indices:(positive if target[i]!=0 else hard)[records[i]['group']].append(i)
    if not positive:raise ValueError('No positive candidate supervision')
    base=ex.read(data_root/'declaration.json');pairs=ex.read(base['fit_manifest'])
    pair_map={p['path_C']:p for p in pairs if 'excluded' not in p}
    paired=[];positive_ids=[];cache={}
    for group in sorted(positive):
        for i in positive[group]:
            r=records[i];p=pair_map[r['path']]
            assert p['source']['sha256']==r['source_sha256'] and ex.group(p)==group
            path=Path(p['path_A']);_,analysis=recovery.from_cached(str(path),path.parent/'original_features.npz',cache)
            paired.append(candidate_context(analysis['views'][0],(r['start'],r['stop']),model.config));positive_ids.append(i)
    paired_embeddings=embeddings(model,paired);paired_map={i:j for j,i in enumerate(positive_ids)}
    anchors=held.anchor_pool(source_rows,excluded);anchor_contexts=[];anchor_records=[];by_anchor=defaultdict(list)
    for anchor in anchors:
        raw=anchor['A']['views'][0];valid=np.flatnonzero(anchor['A']['reliable'])
        for center in np.unique(valid[np.rint(np.linspace(0,len(valid)-1,4)).astype(int)]):
            start=int(np.clip(center-21,0,len(raw)-43));stop=start+43
            by_anchor[anchor['group']].append(len(anchor_contexts));anchor_records.append(anchor)
            anchor_contexts.append(candidate_context(raw,(start,stop),model.config))
    if not anchor_contexts:raise ValueError('No original control anchors')
    anchor_embeddings=embeddings(model,anchor_contexts)
    groups,agroups,hgroups=sorted(positive),sorted(by_anchor),sorted(hard)
    training_groups=set(payload['training_groups'])|{records[i]['group'] for i in indices}|set(agroups)
    training_sources=set(payload['training_sources'])|{records[i]['source_sha256'] for i in indices}|{a['source']['sha256'] for a in anchors}
    if training_groups&set(excluded):raise ValueError('Parent/head singer holdout overlap')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.001)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==2*(model.regression.in_features+1)
    folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter();history=[];sampler=hashlib.sha256()
    model.train()
    for update in range(updates):
        xx=[];yy=[];ww=[]
        for _ in range(12):
            g=groups[int(rng.integers(len(groups)))];i=int(rng.choice(positive[g]))
            xx.extend([paired_embeddings[paired_map[i]],embedded[mapping[i]]]);yy.extend([0.,float(target[i])/30.]);ww.extend([4.,1.])
            sampler.update(f'pair:{i}\n'.encode())
            if hgroups and rng.random()<.25:
                g=hgroups[int(rng.integers(len(hgroups)))];j=int(rng.choice(hard[g]));negative=embedded[mapping[j]];key=f'hard:{j}'
            else:
                g=agroups[int(rng.integers(len(agroups)))];j=int(rng.choice(by_anchor[g]));negative=anchor_embeddings[j];key=f'anchor:{j}'
            xx.append(negative);yy.append(0.);ww.append(4.);sampler.update((key+'\n').encode())
        y,w=torch.tensor(yy),torch.tensor(ww)
        out,logit,amp=model(torch.from_numpy(np.stack(xx)),True);pred=out/30.;positive_mask=y!=0
        anchor=((pred-y).square()*w).mean();triples=pred.reshape(12,3)
        response=(triples[:,1]-triples[:,0]-y.reshape(12,3)[:,1]).square().mean()
        activity=F.binary_cross_entropy_with_logits(logit,positive_mask.float())
        amplitude=(amp[positive_mask]/30.-y[positive_mask]).square().mean()
        loss=anchor+.25*response+.5*activity+.25*amplitude
        if not torch.isfinite(loss):raise ValueError('Nonfinite candidate head loss')
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step()
        if update%100==0 or update==updates-1:
            history.append(dict(update=update+1,loss=float(loss.detach())))
            ex.write(folder/'training_progress.json',dict(update=update+1,planned_updates=updates,history=history))
    assert all(torch.equal(v,model.parent.state_dict()[k]) for k,v in original_state.items())
    metadata=dict(arm='candidate_evidence',seed=seed,updates=updates,training_groups=sorted(training_groups),training_sources=sorted(training_sources),
                  declaration_sha256=declaration_hash,parent_checkpoint_sha256=ex.sha(parent_checkpoint),
                  frozen_parent_verified=True,parent_updates=payload['updates'],trainable_parameters=2*(model.regression.in_features+1),
                  positive_candidates=sum(len(v) for v in positive.values()),negative_candidates=sum(len(v) for v in hard.values()),
                  unchanged_original_queries=len(anchor_contexts),original_anchor_recordings=len(anchors),
                  sampling_sha256=sampler.hexdigest(),head_only_finetuning=True,eligible_for_demo=False,research_validation_passed=False)
    save(folder/'checkpoint.pt',model.eval(),metadata);restored,fitted=load(folder/'checkpoint.pt')
    probe=anchors[0]['A']['views'][0]
    np.testing.assert_allclose(predict_raw(model,probe)[0],predict_raw(restored,probe)[0],atol=1e-6)
    receipt=dict(**metadata,checkpoint_sha256=ex.sha(folder/'checkpoint.pt'),reload_verified=True,fit_seconds=time.perf_counter()-started,history=history)
    ex.write(folder/'fit_receipt.json',receipt);return restored,fitted,receipt


def predictions(model,pair,view):return np.stack([predict_raw(model,pair[view]['views'][0])[0]]*3)


def evaluate(model,payload,pairs,folder,policy):
    previous=held.predictions;held.predictions=predictions
    try:return held.evaluate(model,payload,pairs,folder,policy)
    finally:held.predictions=previous


def cv_worker(task):
    root,fold=task;root=Path(root);d=ex.read(root/'declaration.json');excluded=set(d['folds'][fold])
    rows=ex.read(root/'train_pairs.json');test=[p for p in rows if ex.group(p) in excluded]
    folder=root/f'cross_validation/fold_{fold+1}/candidate_evidence'
    if (folder/'validation/accuracy.json').exists():return ex.read(folder/'validation/accuracy.json')
    if (folder/'fit_receipt.json').exists():
        receipt=ex.read(folder/'fit_receipt.json');assert ex.sha(folder/'checkpoint.pt')==receipt['checkpoint_sha256']
        model,payload=load(folder/'checkpoint.pt')
    else:
        model,payload,_=fit(Path(d['candidate_data']),ex.read(root/'anchor_sources.json'),excluded,d['cv_seed'],d['cv_updates'],folder,
                            ex.sha(root/'declaration.json'),Path(d['cv_parents'][str(fold+1)]))
    assert_independent(payload['training_groups'],payload['training_sources'],test)
    return evaluate(model,payload,held.prepared(test),folder/'validation',HeldPolicy(**d['policy']))
