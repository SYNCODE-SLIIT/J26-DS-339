"""Basic chord representation; redundant and neighbor features are opt-in."""
import torch
from torch import nn


class ChordEncoder(nn.Module):
    def __init__(self, spec, config):
        super().__init__()
        keys = ['basic_chord_ids']
        if config.use_root_quality:
            keys += ['basic_root_ids', 'basic_quality_ids']
        if config.use_neighbor_chords:
            keys += ['previous_basic_chord_ids', 'next_basic_chord_ids']
        self.embeddings = nn.ModuleDict({k: nn.Embedding(spec.vocabulary_sizes[k], config.chord_dim, padding_idx=0) for k in keys})
        self.projection = nn.Identity() if len(keys) == 1 else nn.Linear(len(keys)*config.chord_dim, config.chord_dim)

    def forward(self, inputs):
        return self.projection(torch.cat([embedding(inputs[k]) for k, embedding in self.embeddings.items()], dim=-1))
