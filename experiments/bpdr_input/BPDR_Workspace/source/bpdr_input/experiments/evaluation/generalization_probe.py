"""Challenge the frozen model when the detuned note is never the longest.

Twenty-four new constructed phrases, eight new synthetic voice draws. No model
training, threshold tuning or real-world accuracy claim. All cases are retained.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np

from bpdr_input.experiments.learning import controlled_study as cs
from bpdr_input.experiments.evaluation.transfer_study import ROOT, CONDITIONS, sha, read, write, predictions, score_predictions


def notes_for_voice(v, rng):
    durations = rng.permutation([1.10, 1.20, 1.35, 1.55])
    pitch = v['centre']+v['offset']; notes=[]; t=.25
    for duration in durations:
        notes.append((t,t+float(duration),pitch+float(rng.normal(0,v['intonation_sd']))))
        t += float(duration)+.05
        pitch += 100*int(rng.choice([-3,-2,-1,1,2,3]))
    eligible=[i for i in range(4) if i!=int(np.argmax(durations))]
    return notes, int(rng.choice(eligible))


def run():
    import torch
    import soundfile as sf
    from bpdr_input.repair.interventions import render_local, corrupt_boundaries
    from bpdr_input.preparation.pairs.shared_pair_audit import game_boundary_track
    from bpdr_input.experiments.learning.fine_pitch_relative_check import relative_pair
    from bpdr_input.reporting.bpdr_showcase import load_model
    from bpdr_input.experiments.learning.controlled_study import summary
    torch.set_num_threads(2)
    out=ROOT/uuid.uuid4().hex;out.mkdir(parents=True)
    (out/'executed_generalization_probe.py').write_bytes(Path(__file__).read_bytes())
    base=Path(read(cs.ROOT/'training_current.json')['path'])
    frozen={str(base/(c+'.pt')):sha(base/(c+'.pt')) for c in CONDITIONS}
    write(out/'declaration.json',dict(version='nonlongest-note-probe-1',declared_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),seed=20261017,voices=8,phrases_per_voice=3,records=24,
        design='Four held notes 1.10/1.20/1.35/1.55 seconds in random order; uniformly choose one of the three nonlongest notes',
        shifts=list(cs.SHIFTS),boundary_corruptions=list(cs.CORRUPTIONS),optimizer_updates=0,
        checkpoints=frozen,selection='All three existing seed-339 models, no retraining or selection',
        limitation='Synthetic challenge of location shortcut; not a final independent study or real singing'))
    rng=np.random.default_rng(20261017);catalog=[];pairs=[]
    for vi in range(8):
        v=cs.voice(rng)
        for pi in range(3):
            notes, target=notes_for_voice(v,rng); audio=cs.synthesize(v,notes,6.,rng)
            cents=cs.SHIFTS[(vi*3+pi)%6];corruption=cs.CORRUPTIONS[(vi+pi)%3]
            a,b,_=notes[target];support=(a+.15,b-.15)
            shifted,_=render_local(audio,cs.SR,cents,support)
            edges=np.asarray(sorted({x for n in notes for x in n[:2]}))
            times,A=cs.features(audio,edges);_,C=cs.features(shifted,edges)
            changed=corrupt_boundaries(edges,times,corruption,seed=20261017+vi*3+pi,
                split_time=sum(support)/2 if corruption=='split' else None)
            B,D=A.copy(),C.copy();B[:,70]=D[:,70]=game_boundary_track(times,changed)
            voiced=(A[:,66]>.5)&(C[:,66]>.5)
            ramp=((times>=support[0])&(times<support[0]+.1))|((times>support[1]-.1)&(times<=support[1]))
            mask=voiced&~ramp;plateau=(times>=support[0]+.1)&(times<=support[1]-.1)
            detuning=np.where(mask&plateau,cents,0.).astype(np.float32)
            if (mask&plateau).sum()<10:
                raise ValueError('Insufficient probe support; no silent exclusions')
            key=f'new_voice_{vi:02d}_phrase_{pi}'
            folder=out/'data'/key;folder.mkdir(parents=True)
            np.savez_compressed(folder/'views.npz',A=A,B=B,C=C,D=D,times=times,mask=mask,correction_target_cents=-detuning)
            sf.write(folder/'A.wav',audio,cs.SR,subtype='PCM_24');sf.write(folder/'C.wav',shifted,cs.SR,subtype='PCM_24')
            record=dict(path=str(folder/'views.npz'),voice_id=f'new_voice_{vi:02d}',phrase_id=key,
                        cents=cents,corruption=corruption,notes=notes,target_note=target,
                        target_is_longest=False,support_seconds=support)
            catalog.append(record)
            pairs.append(relative_pair(dict(inputs=dict(A=A,B=B,C=C,D=D),mask=mask,detuning=detuning)))
            print('PROBE BUILT',len(catalog),flush=True)
    write(out/'catalog.json',dict(records=catalog))
    results={}
    for condition in CONDITIONS:
        model,norm,width=load_model(base/(condition+'.pt'));rows=[]
        for record,pair in zip(catalog,pairs):
            r=score_predictions(pair,predictions(model,norm,width,pair,'cpu'))
            rows.append(dict(r,voice_id=record['voice_id'],cents=record['cents'],corruption=record['corruption']))
        results[condition]=dict(all=summary(rows),by_shift={str(c):summary([r for r in rows if r['cents']==c]) for c in cs.SHIFTS},rows=rows)
        print('PROBE RESULT',condition,json.dumps(results[condition]['all']),flush=True)
    assert all(sha(p)==h for p,h in frozen.items())
    write(out/'evidence.json',dict(declaration_sha256=sha(out/'declaration.json'),results=results,optimizer_updates=0))
    write(ROOT/'probe_current.json',dict(path=str(out)))
    return out


if __name__=='__main__':
    run()
