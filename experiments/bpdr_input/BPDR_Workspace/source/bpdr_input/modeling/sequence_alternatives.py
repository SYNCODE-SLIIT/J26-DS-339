"""Small sequence encoders with the joint model's exact input/output contract."""
from dataclasses import asdict
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from bpdr_input.modeling.joint_transformer import JointTransformer, JointConfig, predict_raw

VERSION = 'acoustic-sequence-alternatives-1'
FAMILIES = ('bigru', 'tcn')


class GRUEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gru = nn.GRU(width, width // 2, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, x, src_key_padding_mask=None):
        mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device) if src_key_padding_mask is None else src_key_padding_mask
        sequences = [one[~pad] for one, pad in zip(x, mask)]
        if any(len(one) == 0 for one in sequences):
            raise ValueError('Empty sequence context')
        packed, _ = self.gru(pack_sequence(sequences, enforce_sorted=False))
        encoded, lengths = pad_packed_sequence(packed, batch_first=True)
        result = torch.zeros_like(x)
        for i, length in enumerate(lengths):
            result[i, ~mask[i]] = encoded[i, :int(length)]
        return result


class TemporalEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(width, width, 3, padding=d, dilation=d), nn.GELU(),
            nn.Conv1d(width, width, 3, padding=d, dilation=d), nn.GELU()) for d in (1, 2, 4, 8)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in self.blocks])

    def forward(self, x, src_key_padding_mask=None):
        mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device) if src_key_padding_mask is None else src_key_padding_mask
        x = x.masked_fill(mask[..., None], 0.)
        for block, norm in zip(self.blocks, self.norms):
            x = norm(x + block(x.transpose(1, 2)).transpose(1, 2)).masked_fill(mask[..., None], 0.)
        return x


class SequenceAlternative(JointTransformer):
    def __init__(self, family, config=JointConfig()):
        if family not in FAMILIES or config.hidden % 2:
            raise ValueError('Unsupported alternative sequence encoder')
        super().__init__(config)
        self.family = family
        self.encoder = GRUEncoder(config.hidden) if family == 'bigru' else TemporalEncoder(config.hidden)


def save(path, model, metadata):
    torch.save(dict(version=VERSION, family=model.family, config=asdict(model.config),
                    feature_version='continuous-context-plus-step-1', state=model.state_dict(), **metadata), path)


def load(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['version'] != VERSION or payload['feature_version'] != 'continuous-context-plus-step-1':
        raise ValueError('Unsupported alternative model')
    model = SequenceAlternative(payload['family'], JointConfig(**payload['config']))
    model.load_state_dict(payload['state'], strict=True)
    return model.eval(), payload
