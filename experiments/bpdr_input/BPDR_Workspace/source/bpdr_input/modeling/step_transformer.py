"""A genuine Transformer calibrating input-only acoustic step evidence.

The three engineered input channels are a measured local-step level, fit gain
and residual scale. Invariance to provisional boundary metadata is by input
construction; this does NOT establish the proposed BPDR objective advantage.
"""
from dataclasses import asdict,dataclass
import numpy as np
import torch
from torch import nn
from bpdr_input.modeling.held_voice_transformer import HeldVoiceTransformer
from bpdr_input.modeling.step_context import evidence

VERSION='step-context-transformer-1'


@dataclass(frozen=True)
class StepConfig:
    hidden:int=32
    layers:int=2
    heads:int=4
    feed_forward:int=96
    radius:int=100
    stride:int=4
    gated:bool=True


class StepTransformer(HeldVoiceTransformer):
    def __init__(self,config=StepConfig()):
        super().__init__(config);self.project=nn.Linear(3,config.hidden)


def windows(features,centers,config=StepConfig()):
    centers=np.asarray(centers,dtype=int);offsets=np.arange(-config.radius,config.radius+1,config.stride)
    positions=centers[:,None]+offsets[None,:];valid=(positions>=0)&(positions<len(features))
    x=features[np.clip(positions,0,len(features)-1)].copy();x[~valid]=0.
    return x.astype(np.float32),~valid


def predict_raw(model,raw,batch_size=128):
    features=evidence(raw);result=[];activity=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(raw),batch_size):
            x,pad=windows(features,np.arange(start,min(start+batch_size,len(raw))),model.config)
            out,logit,_=model(torch.from_numpy(x),torch.from_numpy(pad),True)
            result.extend(out.numpy());activity.extend(torch.sigmoid(logit).numpy())
    result=np.array(result,np.float32);activity=np.array(activity,np.float32)
    result[np.asarray(raw)[:,66]<=.5]=0.;activity[np.asarray(raw)[:,66]<=.5]=0.
    if not np.isfinite(result).all():raise ValueError('Nonfinite Transformer output')
    return result,activity


def save(path,model,metadata):torch.save(dict(version=VERSION,config=asdict(model.config),state=model.state_dict(),feature_version='input-only-step-context-2',**metadata),path)


def load(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['version']!=VERSION:raise ValueError('Unsupported step-conditioned Transformer')
    model=StepTransformer(StepConfig(**payload['config']));model.load_state_dict(payload['state']);return model.eval(),payload
