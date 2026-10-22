"""Small local-evidence transformer, baselines, evaluation and inference."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import (FAMILIES, LABELS, FEATURE_DIM, MELODY_DIM, IGNORE,
                   melody_features, clean_notes, CleanConfig)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small sequences run much faster with bounded CPU thread overhead.
    torch.set_num_threads(min(4, torch.get_num_threads()))


def device_available():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class HarmonyDataset(Dataset):
    def __init__(self, arrays, indices):
        self.arrays = arrays
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return {k:torch.as_tensor(v[i]) for k,v in self.arrays.items()} | {'index':torch.tensor(i)}


def make_loader(arrays, examples, split, batch_size=64, seed=42, training=False):
    indices = np.flatnonzero(examples.split.to_numpy() == split)
    dataset = HarmonyDataset(arrays,indices)
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if training:
        # Equal expected song mass, prevents long/repetitive songs dominating.
        counts = examples.iloc[indices].track_id.value_counts()
        weights = [1./counts[tid] for tid in examples.iloc[indices].track_id]
        sampler = WeightedRandomSampler(weights,len(indices),replacement=True,generator=generator)
    return DataLoader(dataset,batch_size=batch_size,sampler=sampler,shuffle=False,
                      num_workers=0,generator=generator)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 96
    layers: int = 2
    heads: int = 4
    dropout: float = .15
    architecture: str = 'transformer'


class HarmonyModel(nn.Module):
    """Direct local beat evidence avoids the diffuse note cross-attention trap.

    Sub-beat slots preserve local order; a convolution shares short motifs.
    Bidirectional self-attention observes the supplied melody window, with
    explicit position and meter. Mode and multi-label genre condition every
    beat. Chords/tonic/absolute register are never fed into this network.
    Root and quality heads share evidence across rare joint chord classes.
    """
    def __init__(self, n_genres, config=ModelConfig()):
        super().__init__()
        self.config = config
        d = config.width
        self.input = nn.Sequential(nn.Linear(FEATURE_DIM,d),nn.GELU(),nn.LayerNorm(d))
        self.mode = nn.Embedding(2,d)
        self.genre = nn.Linear(n_genres,d,bias=False)
        self.local = nn.Conv1d(d,d,3,padding=1)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d,config.heads,4*d,config.dropout,
                                       activation='gelu',batch_first=True,norm_first=True)
            for _ in range(config.layers if config.architecture in {'transformer','harmonic_transformer'} else 0)])
        # Independent layers are initialized independently, rather than cloned.
        self.mlp = nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(2*d,d))
        self.norm = nn.LayerNorm(d)
        self.root = nn.Linear(d,12)
        self.quality = nn.Linear(d,len(FAMILIES))
        self.joint = nn.Linear(d,len(LABELS))
        self.change = nn.Linear(2*d,1)
        if config.architecture == 'harmonic_transformer':
            intervals = [(0,4,7),(0,3,7),(0,4,7,10),(0,3,6),(0,3,6,10),
                         (0,3,6,9),(0,4,8),(0,2,7),(0,5,7)]
            templates = torch.zeros(len(LABELS),12)
            for root in range(12):
                for quality,tones in enumerate(intervals):
                    for tone in tones:
                        templates[root*len(FAMILIES)+quality,(root+tone)%12]=1.
            self.register_buffer('chord_templates',templates)
            self.multiscale_input=nn.Linear(48,d)
            self.multiscale_logits=nn.Linear(48,len(LABELS))
            self.template_strength=nn.Parameter(torch.tensor([1.,.7,.4,.2]))

    def forward(self,batch):
        x = batch['x']
        pad = torch.arange(x.shape[1],device=x.device)[None,:] >= batch['lengths'][:,None]
        g = batch['genres'] / batch['genres'].sum(-1,keepdim=True).clamp_min(1.)
        h = self.input(x) + .5*(self.mode(batch['mode'])[:,None,:] + self.genre(g)[:,None,:])
        evidence = None
        if self.config.architecture == 'harmonic_transformer':
            chroma=sum(x[...,15*j:15*j+12] for j in range(4))/4
            chroma=chroma.masked_fill(pad[...,None],0)
            real=(~pad).float()[:,None,:]
            context=[]
            for kernel in [1,3,7,15]:
                pooled=F.avg_pool1d(chroma.transpose(1,2),kernel,stride=1,padding=kernel//2)
                mass=F.avg_pool1d(real,kernel,stride=1,padding=kernel//2).clamp_min(1e-6)
                context.append((pooled/mass).transpose(1,2))
            multi=torch.cat(context,-1)
            h=h+self.multiscale_input(multi)
            evidence=self.multiscale_logits(multi)
            for strength,features in zip(self.template_strength,context):
                # Chord tones are evidence, not a hard rule: non-chord tones stay.
                evidence=evidence+strength*(features@self.chord_templates.T)
        if self.config.architecture in {'transformer','harmonic_transformer'}:
            # Mask before/after convolution so padding cannot affect real beats.
            h = h.masked_fill(pad[...,None],0)
            h = h + .5*F.gelu(self.local(h.transpose(1,2)).transpose(1,2))
            position = torch.arange(x.shape[1],device=x.device,dtype=x.dtype)[:,None]
            freq = torch.exp(torch.arange(0,h.shape[-1],2,device=x.device,dtype=x.dtype)*(-np.log(10000.)/h.shape[-1]))
            pe = torch.zeros(x.shape[1],h.shape[-1],device=x.device,dtype=x.dtype)
            pe[:,0::2],pe[:,1::2] = torch.sin(position*freq),torch.cos(position*freq)
            h = self.dropout(h+.2*pe[None,:,:])
            for layer in self.layers:
                h = layer(h,src_key_padding_mask=pad)
        else:
            h = h + self.mlp(h)
        h = self.norm(h)
        root, quality = self.root(h), self.quality(h)
        logits = (root[..., :,None]+quality[...,None,:]).flatten(-2) + .5*self.joint(h)
        if evidence is not None:
            logits=logits+evidence
        boundary = self.change(torch.cat([h[:,:-1],h[:,1:]],-1)).squeeze(-1)
        return {'logits':logits,'root':root,'quality':quality,'change':boundary}


def loss_function(output, batch, weights=None, structured=False):
    y = batch['y']
    valid = y >= 0
    if not valid.any():
        return output['logits'].sum()*0
    ce = F.cross_entropy(output['logits'][valid],y[valid],weight=weights,reduction='none')
    certainty = batch['confidence'][valid]
    loss = (ce*certainty).sum()/certainty.sum().clamp_min(1.)
    if structured:
        loss = loss + .2*F.cross_entropy(output['root'][valid],y[valid]//len(FAMILIES))
        loss = loss + .2*F.cross_entropy(output['quality'][valid],y[valid]%len(FAMILIES))
        pair = valid[:,:-1]&valid[:,1:]
        if pair.any():
            target = (y[:,:-1] != y[:,1:]).float()
            # Capped imbalance compensation, computed on this TRAINING batch.
            pos_weight = ((1-target[pair]).sum()/target[pair].sum().clamp_min(1.)).clamp(1,5)
            loss = loss + .15*F.binary_cross_entropy_with_logits(output['change'][pair],target[pair],pos_weight=pos_weight)
    return loss


def class_weights(arrays, train_indices):
    y = arrays['y'][train_indices]
    counts = np.bincount(y[y>=0],minlength=len(LABELS))
    supported = counts>0
    weight = np.ones(len(LABELS),np.float32)
    weight[supported] = np.sqrt(np.median(counts[supported])/counts[supported])
    weight = np.clip(weight,.5,3.)
    weight /= np.average(weight[supported],weights=counts[supported])
    return torch.tensor(weight), counts


def move(batch,device):
    return {k:v.to(device) for k,v in batch.items()}


@torch.inference_mode()
def predict(model,loader,device='cpu',ablation=None):
    model.eval()
    predictions,logits,indices = [],[],[]
    for batch in loader:
        batch = move(batch,device)
        if ablation:
            batch = {k:v.clone() for k,v in batch.items()}
            if ablation == 'no_melody':
                batch['x'][...,:MELODY_DIM] = 0
            elif ablation == 'reverse_melody':
                for i,n in enumerate(batch['lengths'].tolist()):
                    batch['x'][i,:n,:MELODY_DIM] = batch['x'][i,:n,:MELODY_DIM].flip(0)
            elif ablation == 'wrong_pitch':
                for j in range(4):
                    a=15*j
                    batch['x'][...,a:a+12] = batch['x'][...,a:a+12].roll(5,-1)
                batch['x'][...,60:72] = batch['x'][...,60:72].roll(5,-1)
            elif ablation == 'no_genre':
                batch['genres'].zero_()
            elif ablation == 'wrong_mode':
                batch['mode'] = 1-batch['mode']
            else:
                raise ValueError(ablation)
        output = model(batch)['logits'].cpu().numpy()
        logits.append(output)
        indices.extend(batch['index'].cpu().tolist())
        predictions.append(output.argmax(-1))
    return np.concatenate(predictions),np.concatenate(logits),np.array(indices)


def collapse(sequence):
    result=[]
    for x in sequence:
        if not result or result[-1]!=x:
            result.append(int(x))
    return result


def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def metrics(reference,prediction,lengths,modes,track_ids=None,logits=None):
    valid=reference>=0
    yt,yp=reference[valid],prediction[valid]
    support=np.bincount(yt,minlength=len(LABELS))
    predicted=np.bincount(yp,minlength=len(LABELS))
    correct=np.bincount(yt[yt==yp],minlength=len(LABELS))
    f1=2*correct/np.maximum(support+predicted,1)
    pair=valid[:,:-1]&valid[:,1:]
    true_change=reference[:,:-1]!=reference[:,1:]
    pred_change=prediction[:,:-1]!=prediction[:,1:]
    tp=int((pair&true_change&pred_change).sum())
    fp=int((pair&~true_change&pred_change).sum())
    fn=int((pair&true_change&~pred_change).sum())
    tonic=np.where(np.asarray(modes)==0,0,1)[:,None]  # I:MAJ=0, i:MIN=1
    non_tonic=valid&(reference!=tonic)
    song_correct,song_count=Counter(),Counter()
    if track_ids is None:
        track_ids=np.arange(len(reference)).astype(str)
    segment_scores=[]
    for i,n in enumerate(lengths):
        song_correct[track_ids[i]]+=int(((reference[i]==prediction[i])&valid[i]).sum())
        song_count[track_ids[i]]+=int(valid[i].sum())
        # Never connect chord segments across a masked beat.
        positions=np.flatnonzero(valid[i,:n])
        for run in np.split(positions,np.where(np.diff(positions)>1)[0]+1):
            if not len(run):continue
            a,b=collapse(reference[i,run]),collapse(prediction[i,run])
            segment_scores.append(max(0.,1-edit_distance(a,b)/max(len(a),len(b),1)))
    result={'accuracy':float((yt==yp).mean()),'macro_f1':float(f1[support>0].mean()),
            'root_accuracy':float((yt//len(FAMILIES)==yp//len(FAMILIES)).mean()),
            'quality_accuracy':float((yt%len(FAMILIES)==yp%len(FAMILIES)).mean()),
            'non_tonic_accuracy':float((prediction[non_tonic]==reference[non_tonic]).mean()) if non_tonic.any() else 0.,
            'change_f1':2*tp/max(2*tp+fp+fn,1),
            'change_precision':tp/max(tp+fp,1),'change_recall':tp/max(tp+fn,1),
            'segment_edit_similarity':float(np.mean(segment_scores)),
            'song_macro_accuracy':float(np.mean([song_correct[k]/v for k,v in song_count.items() if v])),
            'predicted_tonic_fraction':float(((prediction==tonic)&valid).sum()/valid.sum()),
            'actual_tonic_fraction':float(((reference==tonic)&valid).sum()/valid.sum()),
            'valid_beats':int(valid.sum())}
    if logits is not None:
        z=logits[valid]
        result['top3_accuracy']=float(np.any(np.argsort(z,axis=-1)[:,-3:]==yt[:,None],axis=-1).mean())
        z=z-z.max(-1,keepdims=True)
        result['nll']=float(np.mean(np.log(np.exp(z).sum(-1))-z[np.arange(len(yt)),yt]))
    return result


def score(result):
    """Declared before training; favors minority chords and progression changes."""
    return .5*result['macro_f1']+.3*result['accuracy']+.2*result['change_f1']


def evaluate(model,loader,arrays,examples,device='cpu',ablation=None):
    p,z,idx=predict(model,loader,device,ablation)
    return metrics(arrays['y'][idx],p,arrays['lengths'][idx],arrays['mode'][idx],
                   examples.iloc[idx].track_id.to_numpy(),z)


def baseline_predictions(arrays,examples,split):
    train=np.flatnonzero(examples.split.to_numpy()=='train')
    selected=np.flatnonzero(examples.split.to_numpy()==split)
    y=arrays['y'][train]
    majority=Counter(y[y>=0]).most_common(1)[0][0]
    by_mode={}
    for mode in range(2):
        ym=y[arrays['mode'][train]==mode]
        by_mode[mode]=Counter(ym[ym>=0]).most_common(1)[0][0]
    return selected,{'majority':np.full_like(arrays['y'][selected],majority),
                     'mode_only':np.stack([np.full_like(arrays['y'][i],by_mode[arrays['mode'][i]]) for i in selected])}


def train_experiment(name,arrays,examples,n_genres,output_dir,epochs=20,balanced=False,
                     structured=False,config=ModelConfig(),seed=42,device='cpu',patience=6,
                     dataset_fingerprint=''):
    seed_everything(seed)
    output_dir=Path(output_dir)/name
    output_dir.mkdir(parents=True,exist_ok=True)
    params={'name':name,'epochs':epochs,'balanced':balanced,'structured':structured,
            'model':asdict(config),'seed':seed,'patience':patience,'lr':.001,
            'dataset_fingerprint':dataset_fingerprint}
    config_path=output_dir/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=params:
        raise ValueError(f'Experiment configuration changed: {name}; use a new run directory.')
    config_path.write_text(json.dumps(params,indent=2))
    model=HarmonyModel(n_genres,config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    train_loader=make_loader(arrays,examples,'train',training=True,seed=seed)
    val_loader=make_loader(arrays,examples,'validation')
    train_idx=np.flatnonzero(examples.split.to_numpy()=='train')
    weights,_=class_weights(arrays,train_idx)
    weights=weights.to(device) if balanced else None
    history,best,stale,start=[], -float('inf'),0,0
    last_path=output_dir/'last.pt'
    if last_path.exists():
        state=torch.load(last_path,map_location=device,weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history,best,stale,start=state['history'],state['best'],state['stale'],state['epoch']
        torch.set_rng_state(state['torch_rng'].cpu())
        train_loader.sampler.generator.set_state(state['sampler_rng'].cpu())
    for epoch in range(start,epochs):
        if stale>=patience:break
        begin=time.perf_counter()
        model.train()
        total_loss,steps=0.,0
        # Mild decay after warm-up; same schedule across comparable runs.
        lr=.001*(.15+.85*.5*(1+np.cos(np.pi*epoch/max(epochs-1,1))))
        for group in optimizer.param_groups:group['lr']=lr
        for batch in train_loader:
            batch=move(batch,device)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_function(model(batch),batch,weights,structured)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            total_loss+=loss.item();steps+=1
        result=evaluate(model,val_loader,arrays,examples,device)
        row={'experiment':name,'epoch':epoch+1,'train_loss':total_loss/max(steps,1),
             'seconds':time.perf_counter()-begin,**result}
        history.append(row)
        current=score(result)
        if current>best:
            best,stale=current,0
            torch.save({'model':{k:v.cpu().clone() for k,v in model.state_dict().items()},
                        'params':params,'epoch':epoch+1,'validation':result},output_dir/'best.pt')
        else:stale+=1
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,
                    'history':history,'best':best,'stale':stale,'torch_rng':torch.get_rng_state(),
                    'sampler_rng':train_loader.sampler.generator.get_state()},last_path)
        pd.DataFrame(history).to_csv(output_dir/'history.csv',index=False)
        print(f'{name} {epoch+1:02d}: acc={result["accuracy"]:.3f} macroF1={result["macro_f1"]:.3f} '
              f'changeF1={result["change_f1"]:.3f} ({row["seconds"]:.1f}s)',flush=True)
    state=torch.load(output_dir/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model'])
    return model,pd.DataFrame(history),state


def overfit_probe(arrays,examples,n_genres,steps=180,device='cpu'):
    """A plumbing diagnostic, not validation evidence: memorize 4 train windows."""
    seed_everything(7)
    idx=np.flatnonzero(examples.split.to_numpy()=='train')[:4]
    batch=move(next(iter(DataLoader(HarmonyDataset(arrays,idx),batch_size=4))),device)
    model=HarmonyModel(n_genres,ModelConfig(width=96,dropout=0.)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0.)
    for step in range(steps):
        model.train();optimizer.zero_grad()
        loss=loss_function(model(batch),batch)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
    model.eval()
    with torch.no_grad():
        valid=batch['y']>=0
        accuracy=float((model(batch)['logits'].argmax(-1)[valid]==batch['y'][valid]).float().mean())
    return {'windows':len(idx),'steps':steps,'training_accuracy':accuracy,'loss':float(loss.detach())}


def transition_log_probs(arrays,examples):
    """Train-only mode-conditioned transition matrix with additive smoothing."""
    counts=np.full((2,len(LABELS),len(LABELS)),.1,np.float64)
    for i in np.flatnonzero(examples.split.to_numpy()=='train'):
        y=arrays['y'][i,:arrays['lengths'][i]]
        for a,b in zip(y[:-1],y[1:]):
            if a>=0 and b>=0:counts[arrays['mode'][i],a,b]+=1
    return np.log(counts/counts.sum(-1,keepdims=True))


def viterbi(logits,lengths,modes,transition,alpha):
    """Decode the FULL melody beat grid, without reference-label masks."""
    result=np.zeros(logits.shape[:2],np.int64)
    for i,n in enumerate(lengths):
        score_t=logits[i,0].copy();backs=[]
        t=transition[modes[i]]*alpha
        for j in range(1,n):
            candidates=score_t[:,None]+t
            back=candidates.argmax(0)
            score_t=candidates[back,np.arange(len(LABELS))]+logits[i,j]
            backs.append(back)
        seq=[int(score_t.argmax())]
        for back in reversed(backs):seq.append(int(back[seq[-1]]))
        result[i,:n]=seq[::-1]
    return result


def progression(labels,boundaries):
    """Run-length encode a beat path; preserves repeated chords after changes."""
    if not len(labels):return []
    result=[];start=0
    for i in range(1,len(labels)+1):
        if i==len(labels) or labels[i]!=labels[start]:
            result.append({'start_beat':start,'end_beat':i,'start_sec':float(boundaries[start]),
                           'end_sec':float(boundaries[i]),'roman':LABELS[int(labels[start])]})
            start=i
    return result


def harmonize(model,notes,beat_boundaries,mode,genres,genre_vocab,tonic_pc=None,
              normalized=False,beats=None,transition=None,alpha=0.,device='cpu',hop=16):
    """Predict arbitrary-length songs using overlapping 48-beat melody windows.

    notes: onset_sec/offset_sec/midi_pitch_rounded dictionaries. If normalized,
    the reference tonic is C=0 for major and A=9 for minor. Otherwise tonic_pc
    is required. Supply beat boundaries and meter from user input/beat tracking;
    no hidden chord oracle or key estimator is used during inference.
    """
    if mode not in {'major','minor'}:raise ValueError('mode must be major or minor')
    if normalized:
        tonic_pc=0 if mode=='major' else 9
    if tonic_pc is None:raise ValueError('Provide tonic_pc, or mark a C-major/A-minor melody normalized=True.')
    bounds=np.asarray(beat_boundaries,float)
    if len(bounds)<2 or not np.isfinite(bounds).all() or np.any(np.diff(bounds)<=0):
        raise ValueError('Provide finite increasing beat boundaries.')
    if beats is None:
        beats=[{'beat_in_bar':i%4+1,'meter_numerator':4,'meter_denominator':4} for i in range(len(bounds)-1)]
    if len(beats)!=len(bounds)-1:raise ValueError('Meter record count differs from beat count.')
    # Clean BEFORE slicing to the requested interval, just as snapshot building
    # does. A valid long note may leave a <30ms tail at the final boundary;
    # rejecting that tail here would create training/inference skew.
    input_duration=float(bounds[-1])
    for note in notes:
        try:
            end=float(note['offset_sec'])
            if np.isfinite(end):input_duration=max(input_duration,end)
        except (KeyError,TypeError,ValueError):
            pass
    clean,_=clean_notes(notes,input_duration,CleanConfig())
    g=np.zeros(len(genre_vocab),np.float32)
    ids=[genre_vocab.index(v) for v in genres if v in genre_vocab]
    g[ids or [0]]=1
    n=len(bounds)-1
    logits=np.zeros((n,len(LABELS)),np.float64);mass=np.zeros(n)
    model.eval()
    starts=sorted(set(list(range(0,max(1,n-48+1),hop))+[max(0,n-48)]))
    with torch.inference_mode():
        for start in starts:
            end=min(start+48,n)
            x=melody_features(clean,bounds[start:end+1],tonic_pc,beats[start:end])
            batch={'x':torch.tensor(x)[None],'lengths':torch.tensor([len(x)]),
                   'mode':torch.tensor([int(mode=='minor')]),'genres':torch.tensor(g)[None]}
            z=model(move(batch,device))['logits'][0].cpu().numpy()
            weight=np.minimum(np.arange(len(x))+1,np.arange(len(x),0,-1)).clip(max=8)
            logits[start:end]+=z*weight[:,None];mass[start:end]+=weight
    logits/=mass[:,None]
    labels=(viterbi(logits[None],np.array([n]),np.array([mode=='minor'],int),transition,alpha)[0]
            if transition is not None and alpha else logits.argmax(-1))
    return {'beat_chords':[LABELS[int(v)] for v in labels],
            'progression':progression(labels,bounds),'logits':logits,
            'unknown_genres':[g for g in genres if g not in genre_vocab]}


def load_bundle(path,device='cpu'):
    """Load a locally authored portable bundle after checking its artifact hashes."""
    from .data import file_hash
    path=Path(path)
    meta=json.loads((path/'metadata.json').read_text())
    for filename,key in [('model.pt','model_sha256'),('transition.npy','transition_sha256')]:
        if file_hash(path/filename)!=meta[key]:raise ValueError(f'Bundle checksum failed: {filename}')
    for filename,expected in meta.get('implementation_sha256',{}).items():
        if file_hash(path/'phase2_harmony'/filename)!=expected:
            raise ValueError(f'Bundle implementation checksum failed: {filename}')
    if meta['labels']!=LABELS:raise ValueError('Incompatible chord vocabulary.')
    state=torch.load(path/'model.pt',map_location=device,weights_only=True)
    model=HarmonyModel(len(meta['genre_vocab']),ModelConfig(**state['model_config'])).to(device)
    model.load_state_dict(state['state_dict']);model.eval()
    transition=np.load(path/'transition.npy',allow_pickle=False)
    return model,meta,transition
