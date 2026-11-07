"""Opt-in dense short-phrase context; no reference notes or intervention inputs."""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch import nn

VERSION='short-phrase-transformer-1'
OFFSETS=np.r_[np.arange(-100,-20,4),np.arange(-20,21),np.arange(24,101,4)]


@dataclass(frozen=True)
class ShortConfig:
    hidden:int=32
    layers:int=2
    heads:int=4
    feed_forward:int=96
    gated:bool=True


def frame_features(raw):
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim!=2 or raw.shape[1]!=71 or len(raw)==0 or not np.isfinite(raw).all():
        raise ValueError('Expected nonempty finite frames x71 acoustic input')
    voiced=raw[:,66]>.5;pitch=raw[:,64]
    contrasts=[]
    for width in (13,31,61):
        values=np.where(voiced,pitch,np.nan)
        windows=np.lib.stride_tricks.sliding_window_view(np.pad(values,width//2,constant_values=np.nan),width)
        count=np.isfinite(windows).sum(axis=1);ordered=np.sort(windows,axis=1)
        i=np.arange(len(raw));lo=np.maximum(count-1,0)//2;hi=count//2
        median=np.where(count>0,(ordered[i,lo]+ordered[i,hi])/2,0.)
        contrasts.append(np.where(voiced,np.clip((pitch-median)/30.,-4,4),0.))
    values=np.stack([pitch/100.,np.clip(raw[:,67]/30.,-4,4),raw[:,65],voiced.astype(float),
                     raw[:,68]/20.,*contrasts],axis=1).astype(np.float32)
    values[~voiced]=0.
    if not np.isfinite(values).all():raise ValueError('Nonfinite short-phrase context')
    return values


def tokens_from_features(features,centers):
    features=np.asarray(features,dtype=np.float32);centers=np.asarray(centers,dtype=int)
    if features.ndim!=2 or features.shape[1]!=8 or not len(features) or not np.isfinite(features).all():
        raise ValueError('Expected finite frames x8 short context')
    if centers.ndim!=1 or np.any(centers<0) or np.any(centers>=len(features)):
        raise ValueError('Invalid centers')
    positions=centers[:,None]+OFFSETS[None,:];present=(positions>=0)&(positions<len(features))
    x=features[np.clip(positions,0,len(features)-1)].copy()
    x[:,:,0]=np.clip(x[:,:,0]-features[centers,None,0],-12,12)
    x[:,:,4]=np.clip(x[:,:,4]-features[centers,None,4],-4,4)
    x[~present]=0.;x[x[:,:,3]<.5]=0.
    return x,~present


class ShortPhraseTransformer(nn.Module):
    def __init__(self,config=ShortConfig()):
        super().__init__();self.config=config
        self.project=nn.Linear(8,config.hidden)
        layer=nn.TransformerEncoderLayer(config.hidden,config.heads,config.feed_forward,
            dropout=0.,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,config.layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(config.hidden);self.regression=nn.Linear(config.hidden,1)
        self.activity=nn.Linear(config.hidden,1)
        nn.init.zeros_(self.regression.weight);nn.init.zeros_(self.regression.bias)
        nn.init.constant_(self.activity.bias,-2.)
        # Physical offsets in units of80ms, rather than pretending the mixed grid is uniform.
        pos=torch.tensor(OFFSETS,dtype=torch.float32)[:,None]/4
        rates=torch.exp(torch.arange(0,config.hidden,2)*(-np.log(10000.)/config.hidden))
        table=torch.zeros(len(OFFSETS),config.hidden)
        table[:,0::2]=torch.sin(pos*rates);table[:,1::2]=torch.cos(pos*rates)
        self.register_buffer('positions',table)
        self.center=int(np.flatnonzero(OFFSETS==0)[0])

    def forward(self,x,padding=None,details=False):
        if x.ndim!=3 or x.shape[1:]!=(len(OFFSETS),8):raise ValueError('Invalid token shape')
        h=self.norm(self.encoder(self.project(x)+self.positions,src_key_padding_mask=padding)[:,self.center])
        amplitude=30*torch.tanh(self.regression(h).squeeze(-1));logit=self.activity(h).squeeze(-1)
        prediction=amplitude*torch.sigmoid(logit) if self.config.gated else amplitude
        return (prediction,logit,amplitude) if details else prediction


def predict_raw(model,raw,batch_size=128):
    f=frame_features(raw);prediction=[];activity=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(f),batch_size):
            x,pad=tokens_from_features(f,np.arange(start,min(start+batch_size,len(f))))
            y,logit,_=model(torch.from_numpy(x),torch.from_numpy(pad),True)
            prediction.extend(y.numpy());activity.extend(torch.sigmoid(logit).numpy())
    p=np.asarray(prediction,np.float32);a=np.asarray(activity,np.float32)
    p[np.asarray(raw)[:,66]<=.5]=0.;a[np.asarray(raw)[:,66]<=.5]=0.
    if not np.isfinite(p).all():raise ValueError('Nonfinite short-phrase predictions')
    return p,a


def save(path,model,metadata):
    torch.save(dict(version=VERSION,config=asdict(model.config),state=model.state_dict(),
                    feature_version='short-relative-continuous-8ch-1',**metadata),path)


def load(path):
    p=torch.load(path,map_location='cpu',weights_only=True)
    if p['version']!=VERSION:raise ValueError('Unsupported short-phrase model')
    model=ShortPhraseTransformer(ShortConfig(**p['config']));model.load_state_dict(p['state'])
    return model.eval(),p
