"""Phase 3 compact correction encoder: one relative correction in cents per frame."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    # Proposal starting settings; 71 inputs with the boundary score, 70 for the frame-only control.
    features: int = 71
    hidden: int = 128
    layers: int = 4
    heads: int = 4
    feed_forward: int = 512
    dropout: float = .1


class CorrectionEncoder(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.project = nn.Linear(config.features, config.hidden)
        layer = nn.TransformerEncoderLayer(config.hidden, config.heads, config.feed_forward, config.dropout, batch_first=True)
        # The nested-tensor fast path can shorten padded outputs; every frame must keep its time position.
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.head = nn.Linear(config.hidden, 1)

    def positions(self, length, device):
        position = torch.arange(length, device=device).unsqueeze(1)
        rate = torch.exp(torch.arange(0, self.config.hidden, 2, device=device)*(-math.log(10000.)/self.config.hidden))
        table = torch.zeros(length, self.config.hidden, device=device)
        table[:, 0::2], table[:, 1::2] = torch.sin(position*rate), torch.cos(position*rate)
        return table

    def forward(self, features, padding=None):
        """features: batch x frames x inputs; padding: True where a frame is padding. No causal mask: bidirectional."""
        if features.shape[-1] != self.config.features:
            raise ValueError(f'Expected {self.config.features} input features, got {features.shape[-1]}')
        hidden = self.project(features)+self.positions(features.shape[1], features.device)
        cents = self.head(self.encoder(hidden, src_key_padding_mask=padding)).squeeze(-1)
        return cents if padding is None else cents.masked_fill(padding, 0.)

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def describe(self):
        return dict(asdict(self.config), parameters=self.parameter_count())
