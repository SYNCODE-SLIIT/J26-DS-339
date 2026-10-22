"""Pre-norm encoder: residual attention followed by residual GELU feed-forward."""
from torch import nn
from .attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(config.d_model, config.num_heads, config.dropout)
        self.feed_forward = nn.Sequential(nn.Linear(config.d_model, config.d_ff), nn.GELU(),
                                          nn.Dropout(config.dropout), nn.Linear(config.d_ff, config.d_model))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, beat_attention_mask, return_attention=False):
        attention, weights = self.attention(self.norm1(x), beat_attention_mask, True)
        x = x + self.dropout(attention)
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        x = x.masked_fill(~beat_attention_mask.unsqueeze(-1), 0)
        return (x, weights) if return_attention else x
