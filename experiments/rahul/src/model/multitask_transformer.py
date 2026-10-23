"""Separate full-chord/quality experiment; original Transformer stays unchanged."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from experiments.rahul.src.normalize_chords import parse_chord_label
from experiments.rahul.src.weighted_training import supervised_mask
from .chord_refinement_transformer import ChordRefinementTransformer
from .config import DatasetSpec, ModelConfig
from .loss import masked_cross_entropy


@dataclass(frozen=True)
class QualityMapping:
    labels: tuple[str, ...]
    chord_to_quality: tuple[int, ...]

    @classmethod
    def from_chords(cls, chord_labels) -> 'QualityMapping':
        qualities = []
        for label in chord_labels:
            if label == 'N':
                qualities.append('N')
            else:
                parsed = parse_chord_label(label)
                qualities.append(parsed.quality if parsed.parsed and parsed.normalized else None)
        labels = tuple(sorted({quality for quality in qualities if quality is not None}))
        if not labels:
            raise ValueError('No usable quality labels in target vocabulary')
        indices = {label:index for index,label in enumerate(labels)}
        return cls(labels, tuple(indices[q] if q is not None else -100 for q in qualities))


class MultiTaskChordTransformer(ChordRefinementTransformer):
    """Reuse every existing shared layer and full-chord classifier; add one head."""
    def __init__(self, spec, config, quality_mapping=None):
        super().__init__(spec, config)
        self.quality_mapping = quality_mapping or QualityMapping.from_chords(spec.target_labels)
        if len(self.quality_mapping.chord_to_quality) != spec.num_target_classes:
            raise ValueError('Quality mapping must cover the full saved vocabulary')
        # Standard Linear initialization in a forked CPU RNG preserves the shared
        # baseline initialization and the RNG state available for training dropout.
        with torch.random.fork_rng(devices=[]):
            self.quality_classifier = nn.Linear(config.d_model, len(self.quality_mapping.labels))
        self.register_buffer('chord_to_quality', torch.tensor(self.quality_mapping.chord_to_quality, dtype=torch.int64))

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
        result = {
            'chord_logits':self.classifier(encoded).masked_fill(~mask.unsqueeze(-1), 0),
            'quality_logits':self.quality_classifier(encoded).masked_fill(~mask.unsqueeze(-1), 0),
        }
        if return_details:
            result['details'] = {'features':features, 'fused':fused, 'encoded':encoded, 'attention':attention}
        return result

    @torch.no_grad()
    def predict_topk(self, inputs, k=3):
        """Raw full-chord probabilities; quality re-scoring is deliberately absent."""
        if not 1 <= k <= self.spec.num_target_classes:
            raise ValueError('k is outside full target vocabulary')
        was_training = self.training
        self.eval()
        try:
            probabilities = self(inputs)['chord_logits'].softmax(-1)
            values, ids = probabilities.topk(k, -1)
            return [[([{'id':int(i), 'label':self.spec.target_labels[int(i)], 'probability':float(p)}
                       for i,p in zip(ids[b,t],values[b,t])] if inputs['beat_attention_mask'][b,t] else [])
                     for t in range(probabilities.shape[1])] for b in range(probabilities.shape[0])]
        finally:
            self.train(was_training)


def multitask_loss(output, targets, beat_mask, chord_to_quality, lambda_quality=0.5, reserved_ids=(0,1)):
    """Unweighted full-chord CE + lambda * unweighted quality CE; separate masks."""
    if not math.isfinite(lambda_quality) or lambda_quality < 0:
        raise ValueError('lambda_quality must be finite and nonnegative')
    chord_logits, quality_logits = output['chord_logits'], output['quality_logits']
    if chord_logits.shape[:2] != quality_logits.shape[:2]:
        raise ValueError('Prediction heads disagree on beat dimensions')
    mapping = torch.as_tensor(chord_to_quality, device=chord_logits.device, dtype=torch.int64)
    if mapping.shape != (chord_logits.shape[-1],):
        raise ValueError('Mapping must cover every full-chord coordinate')
    chord_mask = supervised_mask(targets, beat_mask, chord_logits.shape[-1], reserved_ids)
    chord_targets = {'target_chord_ids':targets['target_chord_ids'], 'target_loss_mask':chord_mask}
    chord_loss = masked_cross_entropy(chord_logits, chord_targets, beat_mask)
    ids = targets['target_chord_ids'].clamp(0, len(mapping)-1)
    quality_ids = mapping[ids]
    quality_mask = chord_mask & (quality_ids >= 0)
    if (quality_ids[quality_mask] >= quality_logits.shape[-1]).any():
        raise ValueError('Mapped quality ID outside auxiliary vocabulary')
    quality_loss = (F.cross_entropy(quality_logits[quality_mask], quality_ids[quality_mask])
                    if quality_mask.any() else quality_logits.sum()*0)
    quality_targets = quality_ids.masked_fill(~quality_mask, -100)
    return {'chord_loss':chord_loss, 'quality_loss':quality_loss,
        'total_loss':chord_loss+lambda_quality*quality_loss,
        'chord_mask':chord_mask, 'quality_mask':quality_mask, 'quality_targets':quality_targets}


def save_multitask_checkpoint(path, model, lambda_quality=0.5):
    torch.save({'format':'multitask_chord_quality_v1', 'config':asdict(model.config),
        'spec':asdict(model.spec), 'quality_mapping':asdict(model.quality_mapping),
        'lambda_quality':lambda_quality, 'state_dict':model.state_dict()}, Path(path))


def load_multitask_checkpoint(path, dataset_folder, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    spec = DatasetSpec.from_folder(dataset_folder)
    mapping = QualityMapping.from_chords(spec.target_labels)
    if saved.get('format') != 'multitask_chord_quality_v1' or saved['spec'] != asdict(spec):
        raise ValueError('Checkpoint format or saved dataset vocabulary differs')
    if saved['quality_mapping'] != asdict(mapping):
        raise ValueError('Quality mapping differs from saved target vocabulary')
    model = MultiTaskChordTransformer(spec, ModelConfig(**saved['config']), mapping).to(device)
    model.load_state_dict(saved['state_dict'])
    return model.eval()
