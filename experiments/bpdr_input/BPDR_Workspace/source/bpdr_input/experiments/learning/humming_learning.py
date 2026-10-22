"""One bounded real-humming engineering adaptation, retaining synthetic controls.

Two suitable training recordings are insufficient for a general musical model.
This diagnostic asks whether the current pipeline can learn known local errors
on real humming at all. It never adopts a model or claims BPDR superiority.
"""
from collections import defaultdict
from pathlib import Path
import json
import time
import uuid
import numpy as np
from bpdr_input.experiments.evaluation.transfer_study import COMPONENT, read, write, sha, score_predictions, summaries
from bpdr_input.experiments.learning.real_learning_v2 import relative, pair, initialise, balanced_objective, passes


def main():
    import torch
    from bpdr_input.modeling.examples import Normalizer, collate
    from bpdr_input.experiments.learning.controlled_study import load_pair
    from bpdr_input.experiments.learning.expanded_comparison import to_device
    if not torch.xpu.is_available():raise RuntimeError('Declared XPU unavailable')
    torch.set_num_threads(2)
    real=Path(read(COMPONENT/'runs/real_learning_v2/current.json')['path'])
    hum=Path(read(COMPONENT/'runs/humming_experiment/current.json')['path'])
    synthetic=Path(read(COMPONENT/'runs/controlled_study/current.json')['path'])
    hr=read(hum/'train_catalog.json')['records'];hd=read(hum/'development_catalog.json')['records']
    sr=[r for r in read(synthetic/'catalog.json')['records'] if r['split']=='train']
    sd=[r for r in read(synthetic/'catalog.json')['records'] if r['split']=='heldout']
    if len(hr)!=12 or len(hd)!=6:raise ValueError('Declared small humming pilot inventory changed')
    if {r['singer_id'] for r in hr}&{r['singer_id'] for r in hd}:raise ValueError('Singer overlap')
    with np.load(real/'normalizer.npz') as z:norm=Normalizer(z['mean'],z['scale'])
    out=COMPONENT/'runs/humming_learning'/uuid.uuid4().hex;out.mkdir(parents=True)
    groups=defaultdict(list)
    for i,r in enumerate(hr):groups[(r['phrase_id'],np.sign(r['cents']))].append(i)
    rng=np.random.default_rng(339);schedule=[]
    for _ in range(200):schedule.append([*[int(rng.choice(groups[g])) for g in sorted(groups)],*(len(hr)+rng.choice(len(sr),4,replace=False)).tolist()])
    declaration=dict(version='humming-learning-pilot-1',source_sha256=sha(__file__),updates=200,seed=339,condition='bpdr',
        training_records=len(hr),training_people=len({r['singer_id'] for r in hr}),development_records=len(hd),development_people=len({r['singer_id'] for r in hd}),
        initialization='Synthetic BPDR seed339, exact training-normalizer remap and output-gain30 parameterization',
        normalizer='Frozen new Vocadito/Jamendo training-only normalizer; no humming development fitting',
        sampling='One sign-balanced example per two humming training phrases (4) plus four synthetic train variants per update',
        optimizer='AdamW .001, weight decay .01, gradient clip1',objective='real_learning_v2 balanced normalized anchor, smoothness .1, BPDR response after100 updates',
        evaluation='Final endpoint only: all12 training hum variants, all6 development hum variants, all72 heldout synthetic variants; no early stopping',
        success='Report response>=.25, original alteration<=1c and positive correction gain separately for each source; no research or default model adoption',
        limitations=['Development humming is one prospectively score-selected recording/person','No independent musical suitability review','Known added-error repair is not arbitrary novice-intent recovery'],
        frozen_inputs={str(p):sha(p) for p in [hum/'train_catalog.json',hum/'development_catalog.json',real/'normalizer.npz',synthetic/'catalog.json']},
        final_test_audio_accessed=False,research_ready_examples=0)
    write(out/'declaration.json',declaration);np.save(out/'schedule.npy',np.asarray(schedule));(out/'executed_humming_learning.py').write_bytes(Path(__file__).read_bytes())
    model,width,saved=initialise('bpdr',norm);model.to('xpu').train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    cache={};history=[];started=time.monotonic();print('HUMMING_FIT',out,flush=True)
    for step,indices in enumerate(schedule,1):
        optimizer.zero_grad();total=0.
        for i in indices:
            if i not in cache:
                if i<len(hr):cache[i]=pair(hr[i])
                else:
                    p=load_pair(sr[i-len(hr)]);p['inputs']={v:relative(x) for v,x in p['inputs'].items()};cache[i]=p
            b=to_device(collate([cache[i]],norm,width),'xpu')
            pred={v:30.*model(x,b['padding']) for v,x in b['inputs'].items()}
            loss=balanced_objective(pred,b,'bpdr',step>100);(loss/8).backward();total+=float(loss.detach())/8
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();history.append(dict(update=step,loss=total))
        if step%25==0:print('humming',step,'loss',round(total,5),'seconds',round(time.monotonic()-started),flush=True)
    model.eval();results={};rows_by_split={}
    with torch.inference_mode():
        for split,records,loader,dataset in [('humming_training',hr,pair,'humtrans'),('humming_development',hd,pair,'humtrans'),('synthetic_heldout',sd,load_pair,'synthetic')]:
            rows=[]
            for r in records:
                p=loader(r)
                if dataset=='synthetic':p['inputs']={v:relative(x) for v,x in p['inputs'].items()}
                b=to_device(collate([p],norm,width),'xpu')
                pred={v:(30.*model(x,b['padding']))[0].cpu().numpy() for v,x in b['inputs'].items()}
                rows.append(dict(score_predictions(p,pred),dataset=dataset,group=p['singer_id'],phrase_id=p['phrase_id'],cents=p['cents'],corruption=p['corruption']))
            results[split]=summaries(rows);rows_by_split[split]=rows;write(out/(split+'_rows.json'),rows)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};state['head.weight']*=30.;state['head.bias']*=30.
    saved.update(state=state,normalizer=norm.state(),input_transform='explicit_voicing_phrase_relative_fine_pitch_v2',data_status='exploratory_humming_plus_synthetic',declaration_sha256=sha(out/'declaration.json'))
    torch.save(saved,out/'bpdr.pt');write(out/'history.json',history)
    for p,h in declaration['frozen_inputs'].items():
        if sha(p)!=h:raise ValueError('Frozen input changed')
    write(out/'evidence.json',dict(results=results,gates={k:passes(v) for k,v in results.items()},model_adopted=False,final_test_audio_accessed=False,checkpoint_sha256=sha(out/'bpdr.pt')))
    write(out.parent/'current.json',dict(path=str(out)))
    print('HUMMING_FIT_COMPLETE',json.dumps({k:r['all'] for k,r in results.items()}),out,flush=True)


if __name__=='__main__':main()
