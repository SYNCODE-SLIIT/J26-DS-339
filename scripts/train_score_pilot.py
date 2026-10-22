"""Fixed-budget weak-score feasibility comparison; no final TEST evaluation."""
from pathlib import Path
import sys,json,random,time,hashlib
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
# Freeze the seed, CPU budget and final-epoch evaluation policy.
CONFIG={'seed':339,'epochs':5,'batch_size':8,'window_frames':256,'learning_rate':.001,'device':'cpu','threads':2,'selection':'fixed last epoch; no validation selection'}

# Defining the CNN and Optional Attention Classifier
class ScoreModel(nn.Module):
    """Matched convolutional front end, optionally followed by self-attention."""
    # Build a shared temporal encoder and a 13-class frame prediction head.
    def __init__(self,attention=False):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(80,32,5,padding=2),nn.ReLU(),nn.Conv1d(32,32,5,padding=2),nn.ReLU())
        self.attention=nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,128,.1,batch_first=True),1,enable_nested_tensor=False) if attention else None
        self.head=nn.Linear(32,13)
    # Encode Mel frames and optionally add sinusoidal positions before self-attention.
    def forward(self,x,padding):
        h=self.encoder(x).transpose(1,2)
        if self.attention is not None:
            position=torch.arange(h.shape[1],device=h.device)[:,None]
            rate=torch.exp(torch.arange(0,32,2,device=h.device)*(-np.log(10000.)/32))
            pe=torch.zeros(h.shape[1],32,device=h.device)
            pe[:,0::2]=torch.sin(position*rate);pe[:,1::2]=torch.cos(position*rate)
            h=self.attention(h+pe,src_key_padding_mask=padding)
        # Return per-frame class scores for rest and the twelve pitch classes.
        return self.head(h)

# Normalizing Features and Padding Fixed-Length Training Windows
def load_data(report):
    """Pad fixed windows; masked boundary labels never contribute to loss."""
    folder=ROOT/'datasets/humtrans/derived/research/weak_score_features'
    # Apply the saved TRAIN-only normalization to both splits.
    with np.load(folder/'train_normalization.npz') as z:mean,std=z['mean'],z['std']
    data={'TRAIN':[],'VALID':[]}
    for row in report['recordings']:
        with np.load(ROOT/row['path']) as z:
            x=(z['mel_db']-mean[:,None])/std[:,None];y=z['score_class'].copy();y[~z['supervision_mask']]=-100
        # Pad incomplete windows and use -100 to exclude padding and boundary labels from loss.
        for start in range(0,len(y),256):
            n=min(256,len(y)-start);xx=np.zeros((80,256),np.float32);yy=np.full(256,-100,np.int64);pad=np.ones(256,bool)
            xx[:,:n]=x[:,start:start+n];yy[:n]=y[start:start+n];pad[:n]=False
            data[row['split']].append((torch.from_numpy(xx),torch.from_numpy(yy),torch.from_numpy(pad),row['key']))
    return data

# Evaluating a Model or Constant Majority-Class Predictor
def evaluate(model,items,majority=None):
    """Compute pooled and recording-macro accuracy under the same label mask."""
    confusion=np.zeros((13,13),np.int64);counts={}
    # Accumulate confusion counts and accuracy using only supervised frames.
    with torch.no_grad():
        for x,y,pad,key in items:
            pred=model(x[None],pad[None]).argmax(-1)[0] if model is not None else torch.full_like(y,majority)
            keep=y!=-100;a=y[keep].numpy();b=pred[keep].numpy();np.add.at(confusion,(a,b),1)
            ok,n=counts.get(key,(0,0));counts[key]=(ok+int((a==b).sum()),n+len(a))
    # Compute class F1 and give each recording equal weight in macro accuracy.
    den=confusion.sum(0)+confusion.sum(1);f1=np.divide(2*np.diag(confusion),den,out=np.zeros(13),where=den>0)
    return {'frame_accuracy':float(np.trace(confusion)/confusion.sum()),'recording_macro_accuracy':float(np.mean([a/n for a,n in counts.values() if n])), 'macro_f1_reference_present_classes':float(f1[confusion.sum(1)>0].mean()),'confusion_matrix':confusion.tolist()}


# Comparing Cached pYIN on the Same Weak Score Task
def pitch_baseline(report):
    """Score cached pYIN on exactly the same frames and boundary mask."""
    output={}
    for split in ['TRAIN','VALID']:
        confusion=np.zeros((13,13),np.int64);accuracies=[]
        for row in report['recordings']:
            if row['split']!=split:continue
            with np.load(ROOT/row['path']) as z:labels=z['score_class'];keep=z['supervision_mask'];times=z['times_s']
            with np.load(ROOT/'datasets/humtrans/derived/research/pilot_features'/f"{row['key']}.npz") as z:
                n=len(times)
                # Require matching timestamps before converting pitch to octave-invariant classes.
                if not np.allclose(times,z['times_s'][:n],atol=1e-8):raise ValueError('Baseline timestamps differ')
                f0=z['f0_hz'][:n];voiced=z['voiced'][:n]&np.isfinite(f0)&(f0>0)
            # Encode unvoiced frames as rest and voiced MIDI pitch classes as 1 through 12.
            pred=np.zeros(n,np.int64)
            pred[voiced]=np.rint(69+12*np.log2(f0[voiced]/440)).astype(int)%12+1
            a,b=labels[keep],pred[keep];np.add.at(confusion,(a,b),1);accuracies.append(float((a==b).mean()))
        den=confusion.sum(0)+confusion.sum(1)
        f1=np.divide(2*np.diag(confusion),den,out=np.zeros(13),where=den>0)
        output[split]={'frame_accuracy':float(np.trace(confusion)/confusion.sum()),'recording_macro_accuracy':float(np.mean(accuracies)),'macro_f1_reference_present_classes':float(f1[confusion.sum(1)>0].mean()),'confusion_matrix':confusion.tolist()}
    return output

# Training Both Fixed Configurations and Saving Final Checkpoints
def main():
    """Train both frozen configurations and retain final checkpoints locally."""
    # Use deterministic CPU execution and load the preserved feature manifest.
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    path=ROOT/'experiments/vishwa/results/weak_score_dataset.json';report=json.loads(path.read_text());data=load_data(report)
    # Find the most frequent supervised TRAIN class for the constant baseline.
    counts=np.zeros(13,np.int64)
    for _,y,_,_ in data['TRAIN']:counts+=np.bincount(y[y>=0].numpy(),minlength=13)
    majority=int(counts.argmax())
    results={'config':CONFIG,'torch_version':torch.__version__,'numpy_version':np.__version__,'feature_manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'train_class_counts':counts.tolist(),'majority_class':majority,'models':{}}
    results['majority']={s:evaluate(None,v,majority) for s,v in data.items()}
    results['pyin']=pitch_baseline(report)
    # Reset all seeds so both architectures use reproducible initialization and ordering.
    for name,attention in [('cnn',False),('cnn_transformer',True)]:
        random.seed(339);np.random.seed(339);torch.manual_seed(339)
        model=ScoreModel(attention);opt=torch.optim.Adam(model.parameters(),lr=.001);loss_fn=nn.CrossEntropyLoss(ignore_index=-100)
        history=[];started=time.perf_counter();rng=np.random.default_rng(339)
        # Train for five epochs, skipping batches without supervised target frames.
        for epoch in range(5):
            model.train();loss_sum=0.;n_sum=0
            order=rng.permutation(len(data['TRAIN']))
            for start in range(0,len(order),8):
                rows=[data['TRAIN'][i] for i in order[start:start+8]]
                x,y,pad=[torch.stack([r[j] for r in rows]) for j in range(3)]
                n=int((y!=-100).sum())
                if not n:continue
                opt.zero_grad();loss=loss_fn(model(x,pad).transpose(1,2),y);loss.backward();opt.step();loss_sum+=loss.item()*n;n_sum+=n
            history.append({'epoch':epoch+1,'training_loss':loss_sum/n_sum})
        # Score the final epoch and save its checkpoint with the frozen configuration.
        model.eval();scores={s:evaluate(model,v) for s,v in data.items()}
        checkpoint=ROOT/'datasets/humtrans/derived/research/weak_score_features'/f'{name}.pt'
        torch.save({'state_dict':model.state_dict(),'config':CONFIG},checkpoint)
        results['models'][name]={'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-started,'history':history,'scores':scores,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    # Persist scores and limitations without selecting a model using VALID results.
    results['limitations']=['Single seed, 20 TRAIN / 10 reused development VALID recordings','Weak score-chroma task does not measure performed pitch or note transcription','Attention model has more parameters; comparison does not isolate architecture effect','Five epochs are a feasibility budget, not convergence evidence']
    (ROOT/'experiments/vishwa/results/score_model_pilot.json').write_text(json.dumps(results,indent=2)+'\n')
    for name,row in results['models'].items():print(name,row['parameters'],{s:v['recording_macro_accuracy'] for s,v in row['scores'].items()},flush=True)
    print('majority',{s:v['recording_macro_accuracy'] for s,v in results['majority'].items()})
if __name__=='__main__':main()
