"""Paired group-bootstrap uncertainty from frozen model outputs; no fitting."""
from collections import defaultdict
from pathlib import Path
import uuid

import numpy as np

from bpdr_input.experiments.evaluation.transfer_study import COMPONENT, ROOT, CONDITIONS, read, write, sha, frozen_runs, predictions, score_predictions


def paired_interval(bpdr, control, unit='group'):
    if len(bpdr) != len(control):
        raise ValueError('Paired inventories differ')
    values=defaultdict(list)
    for b,c in zip(bpdr,control):
        if b.get('phrase_id') != c.get('phrase_id') or b['cents'] != c['cents'] or b['corruption'] != c['corruption']:
            raise ValueError('Pair order differs')
        values[b[unit]].append(b['correction_mae']-c['correction_mae'])
    x=np.array([np.mean(v) for _,v in sorted(values.items())]);rng=np.random.default_rng(20261017)
    samples=x[rng.integers(len(x),size=(5000,len(x)))].mean(axis=1)
    return dict(independent_groups=len(x),mean_bpdr_minus_control_cents=float(x.mean()),
                bootstrap_95_percentile_cents=np.quantile(samples,[.025,.975]).tolist(),
                interpretation='Negative favours BPDR; group means weighted equally; descriptive with few groups')


def run():
    import torch
    from bpdr_input.experiments.learning.controlled_study import load_pair
    from bpdr_input.experiments.learning.fine_pitch_relative_check import relative_pair
    from bpdr_input.reporting.bpdr_showcase import load_model
    torch.set_num_threads(2)
    out=ROOT/uuid.uuid4().hex;out.mkdir(parents=True)
    write(out/'declaration.json',dict(version='presentation-paired-statistics-1',source_sha256=sha(__file__),
        bootstrap_replicates=5000,bootstrap_seed=20261017,optimizer_updates=0,
        scope='Descriptive group bootstrap, seeds separately. Synthetic voice settings are not human participants.'))
    data=Path(read(COMPONENT/'runs/controlled_study/current.json')['path'])
    cat=[r for r in read(data/'catalog.json')['records'] if r['split']=='heldout' and r['plateau_frames']>=10]
    pairs=[relative_pair(load_pair(r)) for r in cat];synthetic={}
    for seed,base in frozen_runs():
        all_rows={}
        for c in CONDITIONS:
            model,norm,width=load_model(base/(c+'.pt'));rows=[]
            for record,pair in zip(cat,pairs):
                r=score_predictions(pair,predictions(model,norm,width,pair,'cpu'))
                rows.append(dict(r,group=record['voice_id'],phrase_id=record['phrase_id'],cents=record['cents'],corruption=record['corruption']))
            all_rows[c]=rows;print('STATS',seed,c,flush=True)
        synthetic[str(seed)]={c:paired_interval(all_rows['bpdr'],all_rows[c]) for c in CONDITIONS if c!='bpdr'}
        write(out/(str(seed)+'_synthetic_rows.json'),all_rows)
    transfer=Path(read(ROOT/'transfer_current.json')['path']);real={}
    for seed,_ in frozen_runs():
        rows={c:read(transfer/str(seed)/c/'rows.json') for c in CONDITIONS}
        real[str(seed)]={ds:{c:paired_interval([r for r in rows['bpdr'] if r['dataset']==ds],
                        [r for r in rows[c] if r['dataset']==ds]) for c in CONDITIONS if c!='bpdr'} for ds in ['vocadito','jamendo']}
    write(out/'evidence.json',dict(declaration_sha256=sha(out/'declaration.json'),synthetic=synthetic,real_transfer=real,
        source_transfer=str(transfer),optimizer_updates=0,final_test_accessed=False))
    write(ROOT/'statistics_current.json',dict(path=str(out)))
    return out


if __name__=='__main__':
    run()
