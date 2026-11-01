"""Training-side continuity metrics alongside, not instead of, WAV qualification."""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import median_filter

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.experiments.learning.support_supervision import block_starts, BLOCK_FRAMES


def report(root,arm):
    d=ex.read(root/'declaration.json');rows=ex.read(root/'train_pairs.json')
    pairs=held.prepared(rows);predictions={};hashes={}
    for fold,groups in enumerate(d['folds'],1):
        path=root/f'cross_validation/fold_{fold}/{arm}/validation/predictions.npz'
        hashes[str(path)]=ex.sha(path)
        indices=[i for i,p in enumerate(rows) if ex.group(p) in groups]
        with np.load(path) as z:
            for j,i in enumerate(indices):
                if 'excluded' not in pairs[i]:predictions[i]=z[f'{j}_C'][0].copy()
    cases=[];by_singer={}
    for i,pair in enumerate(pairs):
        group=ex.group(rows[i])
        summary=by_singer.setdefault(group,dict(attempted=0,scoreable=0,known_complete_block_cases=0,
                                              predicted_supported_block_cases=0))
        summary['attempted']+=1
        if 'excluded' in pair:continue
        summary['scoreable']+=1
        known=pair['mask'] & pair['A']['reliable'] & pair['C']['reliable']
        starts=block_starts(pair['target'],known)
        pred=median_filter(predictions[i],size=5,mode='nearest')
        good=0;minimum=[]
        for left in starts:
            sign=np.sign(pair['target'][left]);one=sign*pred[left:left+BLOCK_FRAMES]
            minimum.append(float(one.min()))
            good+=int(np.all(one>=4.) and one.max()>=8.)
        summary['known_complete_block_cases']+=int(len(starts)>0)
        summary['predicted_supported_block_cases']+=int(good>0)
        cases.append(dict(index=i,group=group,valid_block_count=len(starts),
                          supported_block_count=good,best_block_min_signed_cents=max(minimum) if minimum else None))
    result=dict(role='Training-side held-out continuity diagnostic; targets used only for scoring',
                arm=arm,attempted=len(rows),scoreable=len(cases),
                known_complete_block_cases=sum(v['known_complete_block_cases'] for v in by_singer.values()),
                predicted_supported_block_cases=sum(v['predicted_supported_block_cases'] for v in by_singer.values()),
                by_singer=by_singer,cases=cases,prediction_hashes=hashes,
                evaluator_sha256=ex.sha(Path(__file__)),
                definition='Exists a genuinely labelled43-frame constant-target block with all sign-correct filtered predictions>=4c and at least one>=8c. Before eligibility/protection, permission and rendering.',
                limitation='Model-only support is not usable repair coverage. All attempts and cases without complete labels retained. Not independent confirmation or general singing accuracy.',
                research_validation_passed=False)
    ex.write(root/f'{arm}_support_metrics.json',result)
    return {k:v for k,v in result.items() if k not in ('cases','prediction_hashes','by_singer')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('arm')
    args=parser.parse_args();print(report(args.root.resolve(),args.arm))
