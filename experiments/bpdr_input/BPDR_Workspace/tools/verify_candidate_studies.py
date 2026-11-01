"""Verify frozen parents, candidate-only overrides, fitting counts and saved WAVs."""
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex,held_voice_training as held
from bpdr_input.modeling.correction_accuracy import assert_independent
from bpdr_input.modeling.step_candidates import measured_candidates
from bpdr_input.modeling.candidate_transformer import load as mean_load
from bpdr_input.modeling.candidate_evidence_transformer import load as evidence_load
from verify_frozen_audio_studies import outputs
import train_candidate_transformer as mean_study
import train_candidate_evidence_transformer as evidence_study
import evaluate_candidate_center_readout as center_study


def main():
    torch.set_num_threads(2)
    rows=ex.read(mean_study.BASE/'train_pairs.json');pairs=held.prepared(rows)
    pair_map={p['path_C']:p for p in pairs if 'excluded' not in p};reports={};anchor_cache={}
    for runner,loader,arm in ((mean_study,mean_load,'candidate_head'),(evidence_study,evidence_load,'candidate_evidence')):
        d=runner.declare();root=runner.ROOT;data=Path(d['candidate_data']);records=ex.read(data/'candidate_records.json')
        with np.load(data/'candidate_labels_NOT_MODEL_INPUT.npz') as z:known=z['known'];target=z['target']
        sources=ex.read(root/'anchor_sources.json')
        for r in records:assert d['approved_training_sources'][r['source_sha256']]==r['group']
        for fold,groups in enumerate(d['folds'],1):
            folder=root/f'cross_validation/fold_{fold}/{arm}';model,payload=loader(folder/'checkpoint.pt')
            receipt=ex.read(folder/'fit_receipt.json');parent=torch.load(d['cv_parents'][str(fold)],map_location='cpu',weights_only=True)
            assert receipt['checkpoint_sha256']==ex.sha(folder/'checkpoint.pt') and receipt['reload_verified']
            assert payload['declaration_sha256']==ex.sha(root/'declaration.json')
            assert payload['parent_checkpoint_sha256']==ex.sha(d['cv_parents'][str(fold)])
            assert all(torch.equal(v,model.parent.state_dict()[k]) for k,v in parent['state'].items())
            assert payload['frozen_parent_verified'] and payload['head_only_finetuning']
            assert sum(p.numel() for p in model.parameters() if p.requires_grad)==payload['trainable_parameters']
            indices=[i for i,r in enumerate(records) if known[i] and r['group'] not in groups]
            assert payload['positive_candidates']==sum(target[i]!=0 for i in indices)
            assert payload['negative_candidates']==sum(target[i]==0 for i in indices)
            key=tuple(groups)
            if key not in anchor_cache:anchor_cache[key]=held.anchor_pool(sources,groups)
            anchors=anchor_cache[key]
            count=sum(len(np.unique(np.flatnonzero(a['A']['reliable'])[np.rint(np.linspace(0,int(a['A']['reliable'].sum())-1,4)).astype(int)])) for a in anchors)
            assert payload['unchanged_original_queries']==count
            expected_groups=set(parent['training_groups'])|{records[i]['group'] for i in indices}|{a['group'] for a in anchors}
            expected_sources=set(parent['training_sources'])|{records[i]['source_sha256'] for i in indices}|{a['source']['sha256'] for a in anchors}
            assert set(payload['training_groups'])==expected_groups and set(payload['training_sources'])==expected_sources
            subset=[r for r in rows if ex.group(r) in groups]
            assert_independent(payload['training_groups'],payload['training_sources'],subset)
            with np.load(folder/'validation/predictions.npz') as student,np.load(mean_study.BASE/f'cross_validation/fold_{fold}/matched_baseline/validation/predictions.npz') as old:
                for i,row in enumerate(subset):
                    if 'excluded' in row:continue
                    pair=pair_map[row['path_C']]
                    for view in ('A','C'):
                        raw=pair[view]['views'][0];covered=np.zeros(len(raw),bool)
                        for a,b in measured_candidates(raw):
                            covered[a:b]=True;one=student[f'{i}_{view}'][0,a:b][raw[a:b,66]>.5]
                            if len(one):assert np.all(one==one[0])
                        np.testing.assert_array_equal(student[f'{i}_{view}'][:,~covered],old[f'{i}_{view}'][:,~covered])
        report=outputs(root/f'arms/{arm}/training/median11',root/'train_pairs.json',sources)
        reports[arm]=dict(**report,frozen_parent_folds=4,dense_fallback_exact_all_scored_cases=True)
        ex.write(root/'integrity_receipt.json',dict(**reports[arm],declaration_sha256=ex.sha(root/'declaration.json'),research_validation_passed=False))
    center_study.declare()
    if (center_study.ROOT/'arm/training/median11/evidence.json').exists():
        report=outputs(center_study.ROOT/'arm/training/median11',center_study.ROOT/'train_pairs.json',ex.read(center_study.ROOT/'anchor_sources.json'))
        reports['center_readout']=report
        ex.write(center_study.ROOT/'integrity_receipt.json',dict(**report,declaration_sha256=ex.sha(center_study.ROOT/'declaration.json'),research_validation_passed=False))
    protected=ex.read(mean_study.ROOT.parent/'2026-10-28_v7_guard/declaration.json')['protected_files']
    for path,digest in protected.items():assert ex.sha(path)==digest
    assert not (mean_study.ROOT.parent/'active_model.json').exists()
    print('CANDIDATE_STUDIES_VERIFIED',reports,'protected',len(protected),flush=True)


if __name__=='__main__':main()
