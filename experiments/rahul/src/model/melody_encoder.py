"""Pitch/timing note encoder with real-note masked mean pooling."""
import torch
from torch import nn


class MelodyEncoder(nn.Module):
    def __init__(self, spec, config):
        super().__init__()
        d = config.melody_dim
        self.pitch = nn.Embedding(spec.pitch_vocab_size, d, padding_idx=0)
        self.timing = nn.Linear(3, d)
        self.pitch_class = nn.Embedding(14, d, padding_idx=0) if config.use_pitch_class else None
        self.continuous_pitch = nn.Linear(1, d) if config.use_continuous_pitch else None
        self.note_norm = nn.LayerNorm(d)
        self.silence = nn.Embedding(2, d)

    def encode_notes(self, inputs):
        # Sanitize inactive slots BEFORE projections/lookup, so even arbitrary
        # inactive values cannot affect pooling or introduce non-finite values.
        mask = inputs['melody_note_mask']
        pitch = inputs['melody_pitch_ids'].masked_fill(~mask, 0)
        timing = inputs['melody_timing'].masked_fill(~mask.unsqueeze(-1), 0)
        notes = self.pitch(pitch) + self.timing(timing)
        if self.pitch_class is not None:
            notes = notes + self.pitch_class(inputs['melody_pitch_class_ids'].masked_fill(~mask, 0))
        if self.continuous_pitch is not None:
            notes = notes + self.continuous_pitch(inputs['melody_pitch_values'].masked_fill(~mask, 0).unsqueeze(-1))
        return self.note_norm(torch.nn.functional.gelu(notes)).masked_fill(~mask.unsqueeze(-1), 0)

    def forward(self, inputs):
        notes = self.encode_notes(inputs)
        count = inputs['melody_note_mask'].sum(-1, keepdim=True).clamp_min(1)
        pooled = notes.sum(-2) / count
        # The dummy melody_attention_mask is intentionally NOT a real-note mask.
        return pooled + self.silence(inputs['melody_silent'].long())
