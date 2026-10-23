"""Inference architecture copied exactly from the fixed-seed score pilot; no training side effects."""
import numpy as np
import torch
from torch import nn

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
