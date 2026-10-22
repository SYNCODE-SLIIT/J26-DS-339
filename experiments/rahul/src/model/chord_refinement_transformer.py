"""Numerical musical features -> fused beats -> bidirectional encoder -> chords."""
import torch
from torch import nn
from .chord_encoder import ChordEncoder
from .melody_encoder import MelodyEncoder
from .context_encoder import HarmonicContextEncoder, GenreTemporalEncoder
from .positional_encoding import SinusoidalPositionalEncoding
from .transformer_block import TransformerBlock


class ChordRefinementTransformer(nn.Module):
    def __init__(self, spec, config):
        super().__init__()
        self.spec, self.config = spec, config
        encoders = {'chord': ChordEncoder(spec, config)}
        dimensions = [config.chord_dim]
        if config.use_melody:
            encoders['melody'] = MelodyEncoder(spec, config)
            dimensions.append(config.melody_dim)
        if config.use_key_mode or config.use_harmonic:
            encoders['harmonic'] = HarmonicContextEncoder(spec, config)
            dimensions.append(config.harmonic_dim)
        if config.use_genre or config.use_temporal:
            encoders['genre_temporal'] = GenreTemporalEncoder(spec, config)
            dimensions.append(config.genre_temporal_dim)
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = nn.Sequential(nn.Linear(sum(dimensions), config.d_model),
                                    nn.LayerNorm(config.d_model), nn.Dropout(config.dropout))
        self.position = SinusoidalPositionalEncoding() if config.positional_encoding else nn.Identity()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, spec.num_target_classes)

    def encode_features(self, inputs):
        mask = inputs['beat_attention_mask'].unsqueeze(-1)
        return {name: encoder(inputs).masked_fill(~mask, 0) for name, encoder in self.encoders.items()}

    def fuse_features(self, features, beat_attention_mask):
        fused = self.fusion(torch.cat(list(features.values()), dim=-1))
        return self.position(fused).masked_fill(~beat_attention_mask.unsqueeze(-1), 0)

    def forward(self, inputs, return_details=False):
        mask = inputs['beat_attention_mask']
        if mask.dtype != torch.bool or mask.ndim != 2:
            raise ValueError('beat_attention_mask must be bool [B,L]')
        features = self.encode_features(inputs)
        fused = self.fuse_features(features, mask)
        x, attention = fused, []
        for layer in self.layers:
            if return_details:
                x, weights = layer(x, mask, True)
                attention.append(weights)
            else:
                x = layer(x, mask)
        encoded = self.final_norm(x).masked_fill(~mask.unsqueeze(-1), 0)
        logits = self.classifier(encoded).masked_fill(~mask.unsqueeze(-1), 0)
        if return_details:
            return logits, {'features': features, 'fused': fused, 'encoded': encoded, 'attention': attention}
        return logits

    @torch.no_grad()
    def predict_topk(self, inputs, k=3):
        """Evaluation-mode predictions; PAD/UNK are not musical candidates.

        Confidence is softmax probability conditional on musical target classes.
        Padded beats have empty candidates. Original train/eval state is restored.
        """
        was_training = self.training
        self.eval()
        try:
            logits = self(inputs)
            musical = [i for i, label in enumerate(self.spec.target_labels) if label not in ('PAD', 'UNK')]
            if not 1 <= k <= len(musical):
                raise ValueError('k must fit the number of musical target classes')
            allowed = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.bool)
            allowed[musical] = True
            probabilities = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
            values, indices = probabilities.topk(k, dim=-1)
            return [[([{'id': int(i), 'label': self.spec.target_labels[int(i)], 'probability': float(p)}
                       for i, p in zip(indices[b, t], values[b, t])] if inputs['beat_attention_mask'][b, t] else [])
                     for t in range(logits.shape[1])] for b in range(logits.shape[0])]
        finally:
            self.train(was_training)
