"""Frozen Transformer region readout augmented by three input-only cue summaries."""
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from bpdr_input.modeling.candidate_transformer import CandidateTransformer,predict_raw
from bpdr_input.modeling.joint_transformer import JointTransformer,JointConfig

VERSION='candidate-evidence-transformer-head-1'


class EvidenceCandidateTransformer(CandidateTransformer):
    def __init__(self,parent):
        super().__init__(parent)
        for name in ('regression','activity'):
            previous=getattr(self,name);expanded=nn.Linear(self.config.hidden+3,1)
            with torch.no_grad():
                expanded.weight.zero_();expanded.weight[:,:self.config.hidden]=previous.weight
                expanded.bias.copy_(previous.bias)
            setattr(self,name,expanded)

    def embed(self,x,padding,pool):
        hidden=super().embed(x,padding,pool)
        statistics=[]
        for values,mask in zip(x,pool&~padding):
            cue=values[mask,5:8]
            statistics.append(torch.stack([cue[:,0].median(),cue[:,1].max(),cue[:,2].max()]))
        return torch.cat([hidden,torch.stack(statistics)],dim=-1)


def save(path,model,metadata):
    torch.save(dict(version=VERSION,config=asdict(model.config),state=model.state_dict(),
                    feature_version='region-pool-plus-input-cue-statistics-1',**metadata),path)


def load(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['version']!=VERSION or payload['feature_version']!='region-pool-plus-input-cue-statistics-1':
        raise ValueError('Unsupported candidate-evidence Transformer')
    model=EvidenceCandidateTransformer(JointTransformer(JointConfig(**payload['config'])))
    model.load_state_dict(payload['state'],strict=True)
    return model.eval(),payload
