"""Aligned melody Transformer: parallel beat/bar functions and optional change loss."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    resolution: str = 'beat'
    genre: bool = False
    kind: str = 'transformer'
    d_model: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = .10


class FunctionTransformer(nn.Module):
    """Offline harmonization using the complete melody; no teacher-forced chords.

    Beat-aligned inputs remove the old decoder's alignment learning problem.
    Bar heads pool encoded beats; they never discard within-bar melody order.
    """
    def __init__(self, feature_dim, n_genres, bars, config=ModelConfig()):
        super().__init__()
        if config.resolution not in {'beat','bar'} or config.kind not in {'transformer','mlp'}:
            raise ValueError('Unsupported architecture')
        self.config,self.bars=config,bars
        d=config.d_model
        self.project=nn.Sequential(nn.Linear(feature_dim,d),nn.LayerNorm(d),nn.GELU())
        self.mode_embedding=nn.Embedding(2,d)
        self.genre_project=nn.Linear(n_genres,d,bias=False)
        layer=nn.TransformerEncoderLayer(d,config.heads,d*4,config.dropout,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,config.layers,norm=nn.LayerNorm(d),enable_nested_tensor=False)
        self.local=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(d*2,d))
        self.function_head=nn.Linear(d,3)
        self.change_head=nn.Linear(d*2,1)

    def forward(self,x,lengths,bar_index,mode,genres):
        # Inputs are explicit: labels and target-derived confidence cannot enter.
        h=self.project(x)+self.mode_embedding(mode)[:,None]
        if self.config.genre:
            genre=genres/genres.sum(-1,keepdim=True).clamp_min(1)
            if self.training:
                genre=genre*(torch.rand((len(x),1),device=x.device)>.1)
            h=h+self.genre_project(genre)[:,None]
        t=x.shape[1]
        pos=torch.arange(t,device=x.device,dtype=x.dtype)[:,None]
        freq=torch.exp(torch.arange(0,h.shape[-1],2,device=x.device,dtype=x.dtype)*(-math.log(10000.)/h.shape[-1]))
        pe=torch.zeros((t,h.shape[-1]),device=x.device,dtype=x.dtype)
        pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        pad=torch.arange(t,device=x.device)[None]>=lengths[:,None]
        h=self.encoder(h+pe[None],src_key_padding_mask=pad) if self.config.kind=='transformer' else self.local(h)
        if self.config.resolution=='bar':
            mask=(bar_index[:,:,None]==torch.arange(self.bars,device=x.device)[None,None,:])&(~pad[:,:,None])
            weights=mask.to(h.dtype)
            h=torch.einsum('btj,btd->bjd',weights,h)/weights.sum(1).clamp_min(1)[:,:,None]
        logits=self.function_head(h)
        change=self.change_head(torch.cat([h[:,:-1],h[:,1:]],dim=-1)).squeeze(-1)
        return logits,change
