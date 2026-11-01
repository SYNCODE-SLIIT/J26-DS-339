"""Learned region head over frozen Transformer features, with dense fallback."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from bpdr_input.modeling.joint_transformer import JointConfig,JointTransformer,predict_raw as dense_predict
from bpdr_input.modeling.step_candidates import measured_candidates,candidate_context

VERSION='candidate-transformer-head-1'


class CandidateTransformer(nn.Module):
    def __init__(self,parent):
        super().__init__()
        self.parent=parent.requires_grad_(False).eval()
        self.regression=deepcopy(parent.regression).requires_grad_(True)
        self.activity=deepcopy(parent.activity).requires_grad_(True)
        self.config=parent.config

    def train(self,mode=True):
        super().train(mode);self.parent.eval();return self

    def embed(self,x,padding,pool):
        if (x.ndim!=3 or x.shape[-1]!=8 or padding.shape!=x.shape[:2]
                or pool.shape!=padding.shape or pool.dtype!=torch.bool or padding.dtype!=torch.bool
                or torch.any((pool&~padding).sum(dim=1)==0)):
            raise ValueError('Invalid input-only candidate context')
        length=x.shape[1];pos=torch.arange(length,dtype=x.dtype,device=x.device)[:,None]
        rates=torch.exp(torch.arange(0,self.config.hidden,2,dtype=x.dtype,device=x.device)*(-np.log(10000.)/self.config.hidden))
        table=torch.zeros(length,self.config.hidden,dtype=x.dtype,device=x.device)
        table[:,0::2]=torch.sin(pos*rates);table[:,1::2]=torch.cos(pos*rates)
        with torch.no_grad():
            encoded=self.parent.encoder(self.parent.project(x)+table,src_key_padding_mask=padding)
            weights=(pool&~padding).to(x.dtype)[:,:,None]
            return self.parent.norm((encoded*weights).sum(dim=1)/weights.sum(dim=1))

    def forward(self,hidden,details=False):
        amplitude=30.*torch.tanh(self.regression(hidden).squeeze(-1))
        logit=self.activity(hidden).squeeze(-1);output=amplitude*torch.sigmoid(logit)
        return (output,logit,amplitude) if details else output


def predict_raw(model,raw,batch_size=128):
    output,activity=dense_predict(model.parent,raw,batch_size)
    intervals=measured_candidates(raw);model.eval()
    if intervals:
        contexts=[candidate_context(raw,interval,model.config) for interval in intervals]
        x,pad,pool=[torch.from_numpy(np.stack([c[k] for c in contexts])) for k in range(3)]
        with torch.inference_mode():
            values,logit,_=model(model.embed(x,pad,pool),True)
            for (start,stop),value,prob in zip(intervals,values.numpy(),torch.sigmoid(logit).numpy()):
                output[start:stop]=value;activity[start:stop]=prob
    unvoiced=np.asarray(raw)[:,66]<=.5
    output[unvoiced]=0.;activity[unvoiced]=0.
    return output,activity


def save(path,model,metadata):
    torch.save(dict(version=VERSION,config=asdict(model.config),state=model.state_dict(),
                    feature_version='input-step-region-pool-with-dense-fallback-1',**metadata),path)


def load(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['version']!=VERSION:raise ValueError('Unsupported candidate Transformer')
    if payload['feature_version']!='input-step-region-pool-with-dense-fallback-1':raise ValueError('Unknown candidate feature contract')
    model=CandidateTransformer(JointTransformer(JointConfig(**payload['config'])))
    model.load_state_dict(payload['state'],strict=True)
    return model.eval(),payload
