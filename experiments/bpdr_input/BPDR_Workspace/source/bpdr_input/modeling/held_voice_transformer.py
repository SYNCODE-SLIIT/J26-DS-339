"""Learned activity and signed correction over four seconds of input-only pitch.

This is a new Transformer, not the historical context checkpoint or a rule
correction. Local centering uses this recording only; no chromatic grid, source
identity, clean reference or intervention interval enters prediction.
"""
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn

VERSION='held-voice-transformer-1'


@dataclass(frozen=True)
class HeldConfig:
    hidden:int=32
    layers:int=2
    heads:int=4
    feed_forward:int=96
    radius:int=100
    stride:int=4
    gated:bool=True
    centering:str='median'
    scope_same_run:bool=False
    flank_samples:int=0


def tokens(raw,centers,config=HeldConfig()):
    raw=np.asarray(raw,dtype=np.float32)
    centers=np.asarray(centers,dtype=int)
    if raw.ndim!=2 or raw.shape[1]!=71 or not np.isfinite(raw).all():
        raise ValueError('Expected finite input-only 71-column runtime features')
    if centers.ndim!=1 or np.any(centers<0) or np.any(centers>=len(raw)):
        raise ValueError('Invalid center indices')
    offsets=np.arange(-config.radius,config.radius+1,config.stride)
    if not np.any(offsets==0): raise ValueError('Window must contain its exact center')
    positions=centers[:,None]+offsets[None,:]
    present=(positions>=0)&(positions<len(raw))
    x=raw[np.clip(positions,0,len(raw)-1)].copy()
    voiced=present&(x[:,:,66]>.5)
    if config.scope_same_run:
        frame_voiced=raw[:,66]>.5
        cut=np.r_[True,(frame_voiced[1:]!=frame_voiced[:-1])|
            ((np.abs(np.diff(raw[:,64]))>=70.)&frame_voiced[1:]&frame_voiced[:-1])]
        regions=np.cumsum(cut)
        same=regions[np.clip(positions,0,len(raw)-1)]==regions[centers,None]
        voiced&=same&frame_voiced[centers,None]&(abs(x[:,:,64]-raw[centers,64,None])<=70.)
    pitch=np.where(voiced,x[:,:,64],np.nan)
    # All-unvoiced windows are valid identity inputs, never NaN outputs.
    med=np.array([np.median(p[v]) if v.any() else 0. for p,v in zip(pitch,voiced)])
    baseline=np.broadcast_to(med[:,None],pitch.shape).copy()
    if config.centering=='linear_flanks':
        for i,(p,v) in enumerate(zip(pitch,voiced)):
            usable=np.flatnonzero(v)
            if len(usable)<6:continue
            count=min(config.flank_samples,len(usable)//2) if config.flank_samples else max(3,len(usable)//4)
            left,right=usable[:count],usable[-count:]
            baseline[i]=np.interp(np.arange(len(p)),[np.median(left),np.median(right)],
                                  [np.median(p[left]),np.median(p[right])])
    elif config.centering!='median':raise ValueError('Unsupported pitch-centering contract')
    energy=np.array([np.median(p[v]) if v.any() else 0. for p,v in zip(x[:,:,68],voiced)])
    data=np.stack([np.where(voiced,np.clip((x[:,:,64]-baseline)/30.,-4,4),0.),
        np.clip(x[:,:,67]/30.,-4,4),x[:,:,65],voiced.astype(float),
        np.clip((x[:,:,68]-energy[:,None])/20.,-4,4),x[:,:,69],x[:,:,70]],axis=-1).astype(np.float32)
    data[~present]=0.
    padding=~present
    if config.scope_same_run:
        data[~voiced]=0.;padding=~voiced
        # Keep a finite dummy center for all-unvoiced identity inputs.
        padding[:,len(offsets)//2]=False
    return data,padding


class HeldVoiceTransformer(nn.Module):
    def __init__(self,config=HeldConfig()):
        super().__init__();self.config=config
        self.project=nn.Linear(7,config.hidden)
        layer=nn.TransformerEncoderLayer(config.hidden,config.heads,config.feed_forward,dropout=0.,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,config.layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(config.hidden)
        self.regression=nn.Linear(config.hidden,1)
        self.activity=nn.Linear(config.hidden,1)
        nn.init.zeros_(self.regression.weight);nn.init.zeros_(self.regression.bias)
        nn.init.constant_(self.activity.bias,-2.)

    def forward(self,x,padding=None,details=False):
        length=x.shape[1]
        pos=torch.arange(length,device=x.device,dtype=x.dtype)[:,None]
        rates=torch.exp(torch.arange(0,self.config.hidden,2,device=x.device,dtype=x.dtype)*(-np.log(10000.)/self.config.hidden))
        table=torch.zeros(length,self.config.hidden,device=x.device,dtype=x.dtype)
        table[:,0::2]=torch.sin(pos*rates);table[:,1::2]=torch.cos(pos*rates)
        hidden=self.norm(self.encoder(self.project(x)+table,src_key_padding_mask=padding)[:,length//2])
        amplitude=30.*torch.tanh(self.regression(hidden).squeeze(-1))
        logit=self.activity(hidden).squeeze(-1)
        output=amplitude*torch.sigmoid(logit) if self.config.gated else amplitude
        return (output,logit,amplitude) if details else output


def predict_raw(model,raw,batch_size=128):
    result=[];activity=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(raw),batch_size):
            data,padding=tokens(raw,np.arange(start,min(start+batch_size,len(raw))),model.config)
            output,logit,_=model(torch.from_numpy(data),torch.from_numpy(padding),True)
            result.extend(output.numpy());activity.extend(torch.sigmoid(logit).numpy())
    result,activity=np.array(result,dtype=np.float32),np.array(activity,dtype=np.float32)
    if model.config.scope_same_run:
        result[np.asarray(raw)[:,66]<=.5]=0.;activity[np.asarray(raw)[:,66]<=.5]=0.
    return result,activity


def save(path,model,metadata):
    torch.save(dict(version=VERSION,config=asdict(model.config),state=model.state_dict(),**metadata),path)


def load(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['version']!=VERSION: raise ValueError('Unsupported held-voice checkpoint')
    model=HeldVoiceTransformer(HeldConfig(**payload['config']));model.load_state_dict(payload['state'])
    return model.eval(),payload
