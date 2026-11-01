"""Declared unchanged-expression FEATURE controls; never additional real audio."""
from datetime import datetime,timezone
from pathlib import Path
import sys
import numpy as np
import torch

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex,held_voice_training as held
from bpdr_input.experiments.learning.joint_transformer_training import augmented_pairs
from bpdr_input.modeling.step_candidates import measured_candidates,candidate_context

ROOT=WORKSPACE/'results/transformer_usability/2026-10-29_v21_candidate_data/negative_controls'
BASE=ROOT.parent.parent/'2026-10-29_v18_audio_augmented'


def main():
    torch.set_num_threads(2);ROOT.mkdir(exist_ok=True)
    files=[Path(__file__).resolve(),WORKSPACE/'source/bpdr_input/experiments/learning/joint_transformer_training.py',
           WORKSPACE/'source/bpdr_input/modeling/step_candidates.py']
    if not (ROOT/'declaration.json').exists():
        ex.write(ROOT/'declaration.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
                 role='Synthetic unchanged-expression FEATURE controls; no WAVs or new people',seeds=[339,340,341],
                 recipe='Reuse existing V8 augmented_pairs CLEAN branch only: linear drift up to6c/s plus4-7Hz vibrato up to8c; delta recomputed, original voicing/confidence retained. Declared zero correction, no shifted branch.',
                 geometry='Unchanged input-only measured_candidates and pooling mask. No lower fitter threshold or candidate padding to obtain controls.',
                 training_rule='Inherit original singer/source folds. No candidate labels from held-out singers may enter a fit. These synthetic controls do not replace real original controls or listening.',
                 source_hashes={str(p):ex.sha(p) for p in files},input_hashes=ex.read(BASE/'declaration.json')['input_hashes'],
                 human_review=False,research_validation_passed=False))
        snapshot=ROOT/'executed_sources';snapshot.mkdir(exist_ok=True)
        for p in files:(snapshot/p.name).write_bytes(p.read_bytes())
    d=ex.read(ROOT/'declaration.json')
    for key in ('source_hashes','input_hashes'):
        for p,h in d[key].items():assert ex.sha(p)==h,p
    if (ROOT/'preflight.json').exists():raise ValueError('Existing negative-control preflight retained')
    anchors=held.anchor_pool(ex.read(BASE/'anchor_sources.json'));xx=[];pad=[];pool=[];rows=[];case_count=0
    for seed in d['seeds']:
        for case in augmented_pairs(anchors,seed):
            case_count+=1;raw=case['clean']
            for interval in measured_candidates(raw):
                x,p,m=candidate_context(raw,interval);xx.append(x);pad.append(p);pool.append(m)
                rows.append(dict(index=len(rows),group=case['group'],source_sha256=case['source']['sha256'],
                                 seed=seed,start=interval[0],stop=interval[1],role='synthetic_clean_feature_control'))
    if xx:
        np.savez_compressed(ROOT/'candidate_inputs.npz',features=np.stack(xx),padding=np.stack(pad),pool_mask=np.stack(pool))
        np.savez_compressed(ROOT/'candidate_labels_NOT_MODEL_INPUT.npz',target=np.zeros(len(rows),np.float32),known=np.ones(len(rows),bool))
    ex.write(ROOT/'candidate_records.json',rows)
    counts={group:sum(r['group']==group for r in rows) for group in sorted({r['group'] for r in rows})}
    result=dict(feature_control_cases=case_count,measured_negative_candidates=len(rows),by_singer=counts,
                additional_real_audio=0,additional_independent_singers=0,model_fitted=False,
                artifact_hashes={str(p):ex.sha(p) for p in ROOT.glob('candidate_*') if p.is_file()},
                declaration_sha256=ex.sha(ROOT/'declaration.json'),research_validation_passed=False)
    ex.write(ROOT/'preflight.json',result);print('CANDIDATE_NEGATIVES',result,flush=True)


if __name__=='__main__':main()
