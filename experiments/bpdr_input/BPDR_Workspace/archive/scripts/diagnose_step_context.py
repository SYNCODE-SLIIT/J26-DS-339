"""Training-only ceiling check for input-only local-step features."""
from pathlib import Path
import sys
import argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from bpdr_input.experiments.learning import held_voice_training as train
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.modeling.step_context import evidence
from bpdr_input.modeling.correction_accuracy import prediction_row,summarize

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pairs-manifest',type=Path);parser.add_argument('--output-root',type=Path);parser.add_argument('--anchor-sources',type=Path)
    args=parser.parse_args()
    root=args.output_root or ex.WORKSPACE/'results/transformer_usability/step_feature_diagnosis';root.mkdir(parents=True,exist_ok=True)
    manifest=args.pairs_manifest or ex.WORKSPACE/'results/transformer_usability/2026-10-28_v1/train_pairs.json'
    ex.write(root/'declaration.json',dict(training_manifest=str(manifest),training_manifest_sha256=ex.sha(manifest),source_sha256=ex.sha(Path(__file__)),feature_sha256=ex.sha(ex.WORKSPACE/'source/bpdr_input/modeling/step_context.py'),role='training-only feature ceiling; no model, no rendering, no research adoption'))
    pairs=train.prepared(ex.read(manifest));cache={};rows=[]
    for i,p in enumerate(pairs):
        if 'excluded' in p:
            rows.append(dict(group=p.get('group',p['source']['singer_id']),cents=p['cents'],source_sha256=p['source']['sha256'],excluded=p['excluded']));continue
        predictions={}
        for view in ('A','C'):
            path=p['path_'+view]
            if path not in cache:
                one=30*evidence(p[view]['views'][0])[:,0];cache[path]=np.stack([one]*3)
            predictions[view]=cache[path]
        rows.append(prediction_row(p,predictions['A'],predictions['C']))
        if i%20==0:print('STEP_CUE_CASE',i,'/',len(pairs),flush=True)
    result=summarize(rows);result['rows']=rows
    if args.anchor_sources:
        records=ex.read(args.anchor_sources)
        if isinstance(records,dict):records=records['source_records']
        originals=train.anchor_pool(records);controls=[]
        for p in originals:
            cue=30*evidence(p['A']['views'][0])[:,0];reliable=p['A']['reliable']
            controls.append(dict(group=p['group'],source_sha256=p['source']['sha256'],raw_original_abs_cents=float(abs(cue[reliable]).mean())))
        result['original_anchor_controls']=controls
    ex.write(root/'accuracy.json',result)
    print('STEP_CUE_TRAINING_DIAGNOSIS',result['summary'],flush=True)
