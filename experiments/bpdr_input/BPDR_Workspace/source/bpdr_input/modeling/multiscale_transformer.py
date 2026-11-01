"""Broad and current-region acoustic context plus step cues; input audio only."""
from dataclasses import asdict, replace

import numpy as np
import torch
from torch import nn

from bpdr_input.modeling.held_voice_transformer import HeldVoiceTransformer, tokens
from bpdr_input.modeling.joint_transformer import JointConfig
from bpdr_input.modeling.step_context import evidence
from bpdr_input.modeling.step_transformer import windows

VERSION = 'multiscale-context-transformer-1'
FEATURE_VERSION = 'broad-local-step-1'


class MultiscaleTransformer(HeldVoiceTransformer):
    def __init__(self, config=JointConfig()):
        if config.scope_same_run:
            raise ValueError('Multiscale representation manages its local channels internally')
        super().__init__(config)
        self.project = nn.Linear(13, config.hidden)


def multiscale_tokens(raw, centers, config=JointConfig()):
    broad, padding = tokens(raw, centers, replace(config, scope_same_run=False))
    local, _ = tokens(raw, centers, replace(config, scope_same_run=True))
    cues, _ = windows(evidence(raw), centers, config)
    # Preserve broad attention access. The local voiced channel explicitly
    # identifies which observations belong to the current acoustic region.
    return np.concatenate([broad[:, :, :5], local[:, :, :5], cues], axis=-1), padding


def predict_raw(model, raw, batch_size=128):
    result, activity = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(raw), batch_size):
            centers = np.arange(start, min(start+batch_size, len(raw)))
            x, pad = multiscale_tokens(raw, centers, model.config)
            out, logit, _ = model(torch.from_numpy(x), torch.from_numpy(pad), True)
            result.extend(out.numpy())
            activity.extend(torch.sigmoid(logit).numpy())
    result, activity = np.asarray(result, np.float32), np.asarray(activity, np.float32)
    unvoiced = np.asarray(raw)[:, 66] <= .5
    result[unvoiced] = 0.; activity[unvoiced] = 0.
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite multiscale Transformer output')
    return result, activity


def save(path, model, metadata):
    torch.save(dict(version=VERSION, config=asdict(model.config), state=model.state_dict(),
                    feature_version=FEATURE_VERSION, **metadata), path)


def load(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['version'] != VERSION or payload['feature_version'] != FEATURE_VERSION:
        raise ValueError('Unsupported multiscale model or feature contract')
    model = MultiscaleTransformer(JointConfig(**payload['config']))
    model.load_state_dict(payload['state'])
    return model.eval(), payload
