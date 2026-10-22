"""Manual Q/K/V multi-head attention; True masks indicate real beats."""
import math
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model < 1 or num_heads < 1 or d_model % num_heads:
            raise ValueError('d_model must be positive and divisible by num_heads')
        self.num_heads, self.head_dim = num_heads, d_model//num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def project_qkv(self, x):
        batch, length, _ = x.shape
        return tuple(projection(x).reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
                     for projection in (self.query, self.key, self.value))

    def forward(self, x, beat_attention_mask, return_attention=False):
        x = x.masked_fill(~beat_attention_mask.unsqueeze(-1), 0)
        query, key, value = self.project_qkv(x)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        key_mask = beat_attention_mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, -torch.inf)  # Before softmax.
        # All-padding rows have no probability distribution: define it as zero.
        any_key = key_mask.any(-1, keepdim=True)
        safe_scores = torch.where(any_key, scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=-1).masked_fill(~key_mask, 0)
        weights = weights.masked_fill(~beat_attention_mask[:, None, :, None], 0)
        combined = (self.dropout(weights) @ value).transpose(1, 2).contiguous().reshape_as(x)
        result = self.output(combined).masked_fill(~beat_attention_mask.unsqueeze(-1), 0)
        return (result, weights) if return_attention else result
