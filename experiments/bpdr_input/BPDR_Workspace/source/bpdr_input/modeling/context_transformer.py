"""Experimental windowed Transformer over single-recording pitch context.

Features reuse the tree model's 31 measured context columns; the optional 32nd
column is provisional boundary metadata. References and target masks belong
only to the trainer. This module has no demo activation or audio export path.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from bpdr_input.modeling.local_context_pilot import context_features
from bpdr_input.modeling.model import CorrectionEncoder, ModelConfig


@dataclass(frozen=True)
class ContextConfig:
    features: int = 32
    hidden: int = 48
    layers: int = 2
    heads: int = 4
    feed_forward: int = 128
    window: int = 161
    output_scale_cents: float = 30.


class ContextTransformer(nn.Module):
    def __init__(self, config=ContextConfig()):
        super().__init__()
        if config.window % 2 != 1 or config.features not in (31, 32):
            raise ValueError("Expected odd window and 31/32 context features")
        self.config = config
        self.encoder = CorrectionEncoder(ModelConfig(features=config.features,
            hidden=config.hidden, layers=config.layers, heads=config.heads,
            feed_forward=config.feed_forward, dropout=0.))
        nn.init.zeros_(self.encoder.head.weight)
        nn.init.zeros_(self.encoder.head.bias)

    def forward(self, features, padding=None):
        return self.config.output_scale_cents * self.encoder(features, padding)

    def describe(self):
        return dict(architecture="ContextTransformer", **asdict(self.config),
                    parameters=sum(p.numel() for p in self.parameters()))


def window_at(features, center, width):
    """Center a window, with explicit padding; keep the real frame time grid."""
    n = len(features)
    if not 0 <= center < n or width % 2 != 1:
        raise ValueError("Invalid center/window")
    offsets = np.arange(width) + center - width//2
    valid = (offsets >= 0) & (offsets < n)
    result = np.zeros((width, features.shape[1]), dtype=np.float32)
    result[valid] = features[offsets[valid]]
    return result, ~valid, offsets


def predict_features(model, features, batch_size=64):
    """One prediction per center frame: identical context at fit and inference."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != model.config.features or not np.isfinite(features).all():
        raise ValueError("Invalid context features")
    if not len(features):
        return np.empty(0, dtype=np.float32)
    width = model.config.window
    half = width//2
    padded = np.pad(features, ((half, half), (0, 0)))
    padded_valid = np.pad(np.ones(len(features), bool), (half, half))
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=0).transpose(0, 2, 1)
    valid = np.lib.stride_tricks.sliding_window_view(padded_valid, width)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(windows[start:start+batch_size].copy())
            padding = torch.from_numpy(~valid[start:start+batch_size].copy())
            outputs.append(model(x, padding)[:, half].cpu().numpy())
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite Transformer output")
    return result


def predict_raw(model, raw71):
    features = context_features(raw71, include_boundary=model.config.features == 32)
    return predict_features(model, features)


def save_checkpoint(path, model, metadata):
    torch.save(dict(config=asdict(model.config), state=model.state_dict(),
                    feature_version="single-input-local-context-1", **metadata), path)


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["feature_version"] != "single-input-local-context-1":
        raise ValueError("Unsupported context feature version")
    model = ContextTransformer(ContextConfig(**payload["config"]))
    model.load_state_dict(payload["state"])
    return model.eval(), payload
