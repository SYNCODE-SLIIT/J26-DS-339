"""Input-side harmonic features and numerical timing, never reference labels."""
import torch
from torch import nn


class HarmonicContextEncoder(nn.Module):
    def __init__(self, spec, config):
        super().__init__()
        keys = (['key_ids', 'mode_ids'] if config.use_key_mode else [])
        keys += ['roman_ids', 'harmonic_function_ids'] if config.use_harmonic else []
        self.embeddings = nn.ModuleDict({k: nn.Embedding(spec.vocabulary_sizes[k], config.harmonic_dim, padding_idx=0) for k in keys})
        self.projection = nn.Linear(len(keys)*config.harmonic_dim, config.harmonic_dim)

    def forward(self, inputs):
        return self.projection(torch.cat([embedding(inputs[k]) for k, embedding in self.embeddings.items()], -1))


class GenreTemporalEncoder(nn.Module):
    def __init__(self, spec, config):
        super().__init__()
        self.genre = nn.Linear(spec.genre_dim, config.genre_temporal_dim) if config.use_genre else None
        self.use_absolute_timing = config.use_absolute_timing
        self.temporal = nn.Linear(7 if config.use_absolute_timing else 5, config.genre_temporal_dim) if config.use_temporal else None
        self.norm = nn.LayerNorm(config.genre_temporal_dim)

    @staticmethod
    def temporal_values(inputs, include_absolute_timing=False):
        # Saved float32 numerical values, not categorical embedding indices.
        # log1p gives a bounded-growth scale without fitting held-out statistics.
        known = inputs['beat_timing_known_mask']
        position = torch.where(known[..., 1], inputs['beat_position'], 0)
        bar = torch.where(known[..., 0], inputs['relative_bar_number'], 0)
        values = (torch.log1p(position.clamp_min(0)),
                            torch.log1p(bar.clamp_min(0)),
                            torch.log1p(inputs['event_duration'].clamp_min(0)),
                            known[..., 0].float(), known[..., 1].float())
        if include_absolute_timing:
            absolute_bar = torch.where(known[..., 0], inputs['bar_number'], 0)
            values += (torch.log1p(inputs['beat_index'].clamp_min(0)),
                       torch.log1p(absolute_bar.clamp_min(0)))
        return torch.stack(values, -1)

    def forward(self, inputs):
        length = inputs['beat_attention_mask'].shape[1]
        result = None
        if self.genre is not None:
            result = self.genre(inputs['genre_features']).unsqueeze(1).expand(-1, length, -1)
        if self.temporal is not None:
            timing = self.temporal(self.temporal_values(inputs, self.use_absolute_timing))
            result = timing if result is None else result + timing
        return self.norm(result)
