"""Function-preserving dense adapters over frozen learned Transformer logits."""
from dataclasses import asdict
import numpy as np
import torch
from torch import nn
from bpdr_input.modeling.joint_transformer import JointTransformer, JointConfig, joint_tokens

VERSION = 'dense-residual-transformer-1'
FEATURE_VERSION = 'frozen-center-hidden-plus-local-cues-1'


class DenseResidualTransformer(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent.requires_grad_(False).eval()
        self.config = parent.config
        self.regression_delta = nn.Linear(self.config.hidden + 3, 1)
        self.activity_delta = nn.Linear(self.config.hidden + 3, 1)
        for layer in (self.regression_delta, self.activity_delta):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    @torch.no_grad()
    def embed(self, x, padding):
        length = x.shape[1]
        pos = torch.arange(length, device=x.device, dtype=x.dtype)[:, None]
        rates = torch.exp(torch.arange(0, self.config.hidden, 2, device=x.device, dtype=x.dtype)
                          * (-np.log(10000.) / self.config.hidden))
        table = torch.zeros(length, self.config.hidden, device=x.device, dtype=x.dtype)
        table[:, 0::2] = torch.sin(pos * rates)
        table[:, 1::2] = torch.cos(pos * rates)
        hidden = self.parent.norm(self.parent.encoder(self.parent.project(x) + table,
                                  src_key_padding_mask=padding)[:, length // 2])
        return torch.cat([hidden, x[:, length // 2, 5:8]], dim=-1)

    def forward_embeddings(self, features, details=False):
        hidden = features[:, :self.config.hidden].contiguous()
        amplitude = 30. * torch.tanh(self.parent.regression(hidden).squeeze(-1)
                                   + self.regression_delta(features).squeeze(-1))
        logit = self.parent.activity(hidden).squeeze(-1) + self.activity_delta(features).squeeze(-1)
        output = amplitude * torch.sigmoid(logit)
        return (output, logit, amplitude) if details else output

    def forward(self, x, padding=None, details=False):
        return self.forward_embeddings(self.embed(x, padding), details)


def embeddings_raw(model, raw, batch_size=128):
    result = []
    model.eval()
    for start in range(0, len(raw), batch_size):
        x, pad = joint_tokens(raw, np.arange(start, min(start + batch_size, len(raw))), model.config)
        result.append(model.embed(torch.from_numpy(x), torch.from_numpy(pad)).numpy().copy())
    return np.concatenate(result)


def predict_raw(model, raw, batch_size=128):
    features = embeddings_raw(model, raw, batch_size)
    outputs, activities = [], []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            output, logit, _ = model.forward_embeddings(torch.from_numpy(features[start:start + batch_size]), True)
            outputs.append(output.numpy().copy())
            activities.append(torch.sigmoid(logit).numpy().copy())
    output, activity = np.concatenate(outputs), np.concatenate(activities)
    unvoiced = np.asarray(raw)[:, 66] <= .5
    output[unvoiced] = 0.
    activity[unvoiced] = 0.
    if not np.isfinite(output).all() or np.any(np.abs(output) > 30.):
        raise ValueError('Invalid bounded dense-residual prediction')
    return output, activity


def save(path, model, metadata):
    torch.save(dict(version=VERSION, feature_version=FEATURE_VERSION, config=asdict(model.config),
                    state=model.state_dict(), **metadata), path)


def load(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['version'] != VERSION or payload['feature_version'] != FEATURE_VERSION:
        raise ValueError('Unsupported dense-residual Transformer')
    model = DenseResidualTransformer(JointTransformer(JointConfig(**payload['config'])))
    model.load_state_dict(payload['state'], strict=True)
    return model.eval(), payload
