"""Explicit sinusoidal position formula, generated at the requested length."""
import math
import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def forward(self, x):
        length, dim = x.shape[1:]
        position = torch.arange(length, device=x.device, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) * (-math.log(10000.0)/dim))
        angles = position * frequency
        encoding = torch.zeros(length, dim, device=x.device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, :dim//2])
        return x + encoding.to(x.dtype).unsqueeze(0)
