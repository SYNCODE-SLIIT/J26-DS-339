"""Continuous pitch context plus optional step evidence, with no target input."""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from bpdr_input.modeling.held_voice_transformer import HeldVoiceTransformer, tokens
from bpdr_input.modeling.step_context import evidence
from bpdr_input.modeling.step_transformer import windows

VERSION = 'joint-context-transformer-1'


@dataclass(frozen=True)
class JointConfig:
    hidden: int = 32
    layers: int = 2
    heads: int = 4
    feed_forward: int = 96
    radius: int = 100
    stride: int = 4
    gated: bool = True
    centering: str = 'linear_flanks'
    scope_same_run: bool = False
    flank_samples: int = 2


class JointTransformer(HeldVoiceTransformer):
    def __init__(self, config=JointConfig()):
        super().__init__(config)
        self.project = nn.Linear(8, config.hidden)


def joint_tokens(raw, centers, config=JointConfig()):
    # Only the first five continuous-context channels are used. Provisional
    # boundary channels 69/70 are deliberately absent from both inputs.
    context, pad = tokens(raw, centers, config)
    cues, _ = windows(evidence(raw), centers, config)
    return np.concatenate([context[:, :, :5], cues], axis=-1), pad


def predict_raw(model, raw, batch_size=128):
    result, activity = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(raw), batch_size):
            x, pad = joint_tokens(raw, np.arange(start, min(start+batch_size, len(raw))), model.config)
            out, logit, _ = model(torch.from_numpy(x), torch.from_numpy(pad), True)
            result.extend(out.numpy())
            activity.extend(torch.sigmoid(logit).numpy())
    result = np.asarray(result, np.float32)
    activity = np.asarray(activity, np.float32)
    unvoiced = np.asarray(raw)[:, 66] <= .5
    result[unvoiced] = 0.
    activity[unvoiced] = 0.
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite joint Transformer output')
    return result, activity


def save(path, model, metadata):
    torch.save(dict(version=VERSION, config=asdict(model.config), state=model.state_dict(),
                    feature_version='continuous-context-plus-step-1', **metadata), path)


def load(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['version'] != VERSION:
        raise ValueError('Unsupported joint-context Transformer')
    model = JointTransformer(JointConfig(**payload['config']))
    model.load_state_dict(payload['state'])
    return model.eval(), payload
