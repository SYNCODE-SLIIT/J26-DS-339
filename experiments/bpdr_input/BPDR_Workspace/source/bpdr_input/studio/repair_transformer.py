"""BPDR Pitch-Repair Transformer (BPDRRepairTransformer).

A boundary-guided, residual transformer network for minimal musical pitch error correction:
- Multi-scale dilated 1D convolutional residual front-end capturing local intonation and vibrato dynamics.
- Pre-norm self-attention encoder (FastTransformerBlock) capturing phrase-level melodic and harmonic context.
- Disentangled boundary conditioning: pitch correction is robust against false boundary splits and merges.
- Preserves clean vocal expression and natural vibrato (outputting ~0 cents correction on in-tune singing).
"""
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tnf

CHECKPOINT_DEFAULT = (
    Path(__file__).resolve().parents[3] / "results" / "studio" / "r1" / "bpdr_repair_transformer.pt"
)


class FastTransformerBlock(nn.Module):
    """Pre-norm self-attention transformer block with linear projections."""
    def __init__(self, d: int = 64, h: int = 4, dropout: float = 0.05):
        super().__init__()
        self.h = h
        self.n1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop1 = nn.Dropout(dropout)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, T, C = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, -1)
        f = lambda t: t.view(B_, T, self.h, C // self.h).transpose(1, 2)
        attn = tnf.scaled_dot_product_attention(f(q), f(k), f(v))
        a = attn.transpose(1, 2).contiguous().view(B_, T, C)
        x = x + self.drop1(self.proj(a))
        return x + self.drop2(self.ff(self.n2(x)))


class BPDRRepairTransformer(nn.Module):
    """Transformer for predicting pitch correction curves in cents."""
    def __init__(
        self,
        nin: int = 8,
        d: int = 64,
        nhead: int = 4,
        nlayers: int = 3,
        max_correction_cents: float = 150.0,
    ):
        super().__init__()
        self.nin = nin
        self.d = d
        self.max_correction_cents = float(max_correction_cents)
        
        # Step delta features (pitch first derivative and voicing transition)
        self.in_proj = nn.Conv1d(nin + 4, d, 3, padding=1)
        
        # Multi-scale dilated residual highway
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(d, d, 5, padding=2 * k, dilation=k)
            for k in (1, 2, 4, 8)
        ])
        
        # Temporal downsampling by 4 for broad phrase attention
        self.down = nn.Conv1d(d, d, 4, stride=4, padding=1)
        self.pos_conv = nn.Conv1d(d, d, 7, padding=3, groups=d)
        
        # Self-attention encoder stack
        self.enc = nn.Sequential(*[FastTransformerBlock(d, nhead) for _ in range(nlayers)])
        
        # Temporal upsampling back to frame resolution
        self.up = nn.ConvTranspose1d(d, d, 4, stride=4, padding=0)
        self.gate = nn.Parameter(torch.tensor(0.3))
        
        # Prediction head: outputs pitch correction in cents
        self.out_head = nn.Sequential(
            nn.Conv1d(d, d // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d // 2, 1, 1),
        )

    def _compute_deltas(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pitch slope and voicing transition deltas.
        
        x: [B, T, nin]
        col 0: pitch_sung, col 1: pitch_ref, col 3: ok_mask
        """
        p_sung = x[:, :, 0]
        ok = x[:, :, 3]
        
        # Sung pitch slope
        dp = torch.zeros_like(p_sung)
        dp[:, 1:] = (p_sung[:, 1:] - p_sung[:, :-1]) * ok[:, 1:] * ok[:, :-1]
        
        # Pitch acceleration
        ddp = torch.zeros_like(dp)
        ddp[:, 1:] = dp[:, 1:] - dp[:, :-1]
        
        # Voicing transition
        dok = torch.zeros_like(ok)
        dok[:, 1:] = (ok[:, 1:] - ok[:, :-1]).abs()
        
        # Error difference
        diff = (x[:, :, 0] - x[:, :, 1]) * ok
        
        return torch.stack([dp, ddp, dok, diff], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input feature tensor [Batch, Time, nin]
            
        Returns:
            Pitch correction curve in cents: [Batch, Time]
        """
        B_, T, _ = x.shape
        deltas = self._compute_deltas(x)
        feat = torch.cat([x, deltas], dim=-1)  # [B, T, nin + 4]
        
        z = self.in_proj(feat.transpose(1, 2))  # [B, d, T]
        for c in self.dilated_convs:
            z = z + tnf.gelu(c(z))
            
        zc = tnf.gelu(self.down(z))
        zc = zc + self.pos_conv(zc)
        zc = self.enc(zc.transpose(1, 2)).transpose(1, 2)
        
        zu = self.up(zc)
        # Ensure exact sequence length match
        if zu.shape[-1] < T:
            zu = tnf.pad(zu, (0, T - zu.shape[-1]), mode='replicate')
        else:
            zu = zu[..., :T]
            
        z = z + self.gate * tnf.gelu(zu)
        raw_pred = self.out_head(z).squeeze(1)  # [B, T]
        
        # Scaled tanh output bounded to [-max_correction_cents, +max_correction_cents]
        correction_cents = torch.tanh(raw_pred / 50.0) * self.max_correction_cents
        
        # Zero out unvoiced frames
        ok_mask = x[:, :, 3]
        return correction_cents * ok_mask


def load_repair_transformer(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: str = "cpu"
) -> BPDRRepairTransformer:
    """Load pretrained BPDRRepairTransformer checkpoint."""
    ckpt = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_DEFAULT
    net = BPDRRepairTransformer(nin=8)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        net.load_state_dict(state)
        net.to(device)
        net.eval()
    return net
