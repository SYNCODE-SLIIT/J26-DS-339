"""Read-only input diagnostics. Describes confounds; does not infer musical quality."""
from collections import defaultdict
from pathlib import Path
import uuid
import numpy as np
from bpdr_input.experiments.evaluation.transfer_study import COMPONENT, FINE, read, sha, write


def describe(a,times,mask,target):
    voiced=a[:,66]>.5
    adjacent=voiced[1:] & voiced[:-1]
    changes=np.abs(np.diff(a[:,64]))[adjacent]
    return dict(voiced_fraction=float(voiced.mean()),
        median_adjacent_pitch_change_cents=float(np.median(changes)) if len(changes) else None,
        p95_adjacent_pitch_change_cents=float(np.percentile(changes,95)) if len(changes) else None,
        labelled_correction_frames=int((mask & (target!=0)).sum()),
        labelled_preservation_frames=int((mask & (target==0)).sum()),
        span_seconds=float(times[-1]))


def main():
    out=COMPONENT/'runs/real_input_diagnosis'/uuid.uuid4().hex;out.mkdir(parents=True)
    groups=defaultdict(list);excluded=defaultdict(lambda:dict(frames=0,a4_frames_removed_by_legacy_mask=0,phrases_affected=0,phrases=0))
    hashes={}
    for split in ['training','development']:
        path=FINE/(split+'_catalog.json');hashes[str(path)]=sha(path)
        records=read(path)['records'];seen=set()
        for r in records:
            with np.load(Path(r['input_directory'])/'A_input.npz') as z:a=z['boundary_features_71'];times=z['times_seconds']
            with np.load(r['target_path']) as z: mask=z['common_reliable_mask'].astype(bool);target=z['correction_target_cents']
            key=f'{split}:{r["dataset"]}'
            groups[key].append(describe(a,times,mask,target))
            if r['phrase_id'] not in seen:
                seen.add(r['phrase_id'])
                with np.load(Path(r['quantized_input_directory'])/'A_input.npz') as z:raw=z['boundary_features_71']
                voiced=raw[:,66]>.5;lost=voiced & (raw[:,64]==0)
                e=excluded[key];e['frames']+=int(voiced.sum());e['a4_frames_removed_by_legacy_mask']+=int(lost.sum())
                e['phrases_affected']+=int(lost.any());e['phrases']+=1
    root=Path(read(COMPONENT/'runs/controlled_study/current.json')['path'])
    for r in read(root/'catalog.json')['records']:
        if r['split']!='train':continue
        with np.load(r['path']) as z:
            groups['training:synthetic'].append(describe(z['A'],z['times'],z['mask'].astype(bool),z['correction_target_cents']))
    summary={key:dict(variants=len(rr), medians={k:float(np.median([r[k] for r in rr if r[k] is not None])) for k in rr[0]})
             for key,rr in groups.items()}
    evidence=dict(summary=summary,legacy_adapter=excluded,input_catalog_sha256=hashes,final_test_access=False,
        interpretation='Descriptive confounds, not causal proof of model failure; pitch movement is not musical incorrectness. No dataset quality labels, targets or model weights changed.',
        limitations=['Variant medians; correlated variants are not independent people',
                     'Different support geometry and estimators across sources',
                     'Historical A4 masking defect quantified, not yet rebuilt/retrained; runtime v2 already preserves A4'])
    write(out/'evidence.json',evidence);write(out.parent/'current.json',dict(path=str(out)))
    print(__import__('json').dumps(evidence,indent=2));print('RUN',out)


if __name__=='__main__':main()
