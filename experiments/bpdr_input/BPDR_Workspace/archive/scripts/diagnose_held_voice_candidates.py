"""Explain identity/withheld cases using frozen predictions; never render or tune."""
import argparse
from collections import Counter
from pathlib import Path
import sys
import numpy as np

WORKSPACE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORKSPACE/'source'))
from bpdr_input.experiments.learning import context_improvement as ex
from bpdr_input.experiments.learning import held_voice_training as held
from bpdr_input.repair.held_voice_policy import HeldPolicy,proposal


def diagnose(root,arm):
    declaration=ex.read(root/'declaration.json')
    folder=root/f'arms/{arm}/training/median11'
    accuracy=ex.read(folder/'accuracy.json')
    pairs=held.prepared(ex.read(root/'train_pairs.json'))
    reasons=Counter();cases=[]
    with np.load(folder/'predictions.npz') as z:
        for i,(pair,row) in enumerate(zip(pairs,accuracy['rows'])):
            if 'excluded' in pair:continue
            analysis=pair['C']
            detail=proposal(pair['audio_C'],analysis['features']['times_seconds'],
                            analysis['views'],z[f'{i}_C'],HeldPolicy(**declaration['policy']))
            # These are existing input-only policy decisions. Targets/masks are
            # deliberately absent from this diagnostic's decision computation.
            rejected=sorted({c['reason'] for c in detail['candidates'] if c['reason']})
            if detail['plan']['identity']:
                labels=rejected or ['NO_PREDICTED_SUPPORTED_CANDIDATE']
                reasons.update(labels)
                cases.append(dict(index=i,group=pair['group'],reasons=labels,
                                  allowed_frames=int(detail['allowed'].sum()),
                                  candidates=detail['candidates']))
    report=dict(role='Training-side explanatory replay; no fitting or policy selection',
                declaration_sha256=ex.sha(root/'declaration.json'),
                prediction_file_sha256=ex.sha(folder/'predictions.npz'),
                identity_cases=len(cases),reason_counts=dict(reasons),reasons_can_overlap=True,
                cases=cases,thresholds_changed=False,rendering_performed=False,
                target_used_for_decisions=False,research_validation_passed=False)
    ex.write(root/f'{arm}_identity_diagnosis.json',report)
    return {k:v for k,v in report.items() if k!='cases'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('arm')
    args=parser.parse_args();print(diagnose(args.root.resolve(),args.arm))
