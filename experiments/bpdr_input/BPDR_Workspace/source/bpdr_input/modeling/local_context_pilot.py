"""Exploratory local-pitch context MLP; inference sees one recording at a time.

This is a small learned alternative to the historical 71-feature Transformer,
not a renamed Transformer or an intention classifier. All context is derived
from this input's pitch, voicing and acoustic measurements. Targets, intervention
supports, clean references and source identities are absent from this interface.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
import torch
from torch import nn

FEATURE_VERSION = "single-input-local-context-1"
WINDOW_FRAMES = (11, 31, 61, 121)  # fixed 20ms grid: .22, .62, 1.22, 2.42 sec
FRAME_FEATURES = 31


def context_features(raw, include_boundary=True):
    """Return fixed-scale context features from one runtime-like 71-column view.

    The chromatic residual uses all 12 pitch classes and this performance's own
    circular tuning estimate; it is not a named key/scale or an intended melody.
    Gaps are interpolated solely for local statistics; explicit voiced/coverage
    features expose missing observations, and unvoiced outputs are masked later.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 71 or not np.isfinite(raw).all():
        raise ValueError("Expected finite frames x 71 runtime features")
    n = len(raw)
    if not n:
        raise ValueError("Cannot infer on an empty feature sequence")
    voiced = raw[:, 66] > .5  # A4=0 cents remains a valid voiced observation
    pitch = np.where(voiced, raw[:, 64], 0.).astype(np.float64)
    if voiced.any():
        indices = np.arange(n)
        interp = np.interp(indices, indices[voiced], pitch[voiced])
        offset = np.angle(np.mean(np.exp(2j*np.pi*pitch[voiced]/100.)))*100./(2*np.pi)
        residual = (interp-offset+50.) % 100.-50.
        relative = interp-np.median(pitch[voiced])
    else:
        interp = residual = relative = np.zeros(n)
        offset = 0.
    delta = np.r_[0., np.diff(interp)]
    energy = raw[:, 68].astype(np.float64)
    energy -= np.median(energy[voiced]) if voiced.any() else np.median(energy)
    features = [np.clip(relative/1200., -3, 3), residual/30.,
                np.sin(2*np.pi*residual/100.), np.cos(2*np.pi*residual/100.),
                raw[:, 65], voiced.astype(float), np.clip(delta/30., -4, 4),
                np.clip(energy/20., -4, 4), raw[:, 69]]
    for width in WINDOW_FRAMES:
        median_pitch = median_filter(interp, size=width, mode="nearest")
        median_residual = median_filter(residual, size=width, mode="nearest")
        mean_residual = uniform_filter1d(residual, size=width, mode="nearest")
        spread = np.sqrt(np.maximum(uniform_filter1d(residual**2, size=width, mode="nearest")-mean_residual**2, 0.))
        features.extend([np.clip((interp-median_pitch)/30., -4, 4),
                         ((residual-median_residual+50.) % 100.-50.)/30.,
                         mean_residual/30., spread/30.,
                         uniform_filter1d(voiced.astype(float), size=width, mode="nearest")])
    # A same-pitch-class segment is estimated from THIS audio only. The original
    # and detuned versions can produce different segments; no paired mask is used.
    note_bin = np.rint((interp-offset)/100.).astype(int)
    end_residual = np.zeros(n)
    groups = np.split(np.arange(n), np.flatnonzero((np.diff(note_bin) != 0) | (~voiced[1:]) | (~voiced[:-1]))+1)
    for group in groups:
        usable = group[voiced[group]]
        if len(usable) < 10:
            continue
        count = min(5, len(usable)//3)
        start_level, end_level = np.median(interp[usable[:count]]), np.median(interp[usable[-count:]])
        expected = np.interp(usable, [usable[0], usable[-1]], [start_level, end_level])
        end_residual[usable] = np.clip((interp[usable]-expected)/30., -4, 4)
    features.extend([end_residual, np.clip(np.abs(delta)/100., 0, 3)])
    result = np.column_stack(features).astype(np.float32)
    if result.shape != (n, FRAME_FEATURES):
        raise ValueError(f"Unexpected local feature shape {result.shape}")
    if include_boundary:
        result = np.column_stack([result, raw[:, 70]]).astype(np.float32)
    return result


@dataclass(frozen=True)
class LocalConfig:
    features: int = FRAME_FEATURES+1
    hidden: int = 32
    output_gain_cents: float = 30.


class LocalContextMLP(nn.Module):
    """Shared frame MLP over fixed multiscale context, with physical-cent output."""
    def __init__(self, config=LocalConfig()):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential(nn.Linear(config.features, config.hidden), nn.SiLU(),
                                    nn.Linear(config.hidden, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, voiced=None):
        if features.shape[-1] != self.config.features:
            raise ValueError("Local model feature width mismatch")
        result = self.config.output_gain_cents*self.layers(features).squeeze(-1)
        return result if voiced is None else result.masked_fill(~voiced, 0.)

    def describe(self):
        return dict(type="LocalContextMLP", **asdict(self.config),
                    feature_version=FEATURE_VERSION, parameters=sum(p.numel() for p in self.parameters()))


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["feature_version"] != FEATURE_VERSION:
        raise ValueError("Unknown runtime local-context feature version")
    model = LocalContextMLP(LocalConfig(**payload["config"]))
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload


def load_candidate(path):
    """Load only a checkpoint whose declared endpoint gates actually passed."""
    import json
    from pathlib import Path
    from bpdr_input.application.saved_results import checksum
    path = Path(path)
    evidence = json.loads((path.parent/"evidence.json").read_text(encoding="utf-8"))
    matching = [r for r in evidence["results"].values() if Path(r["checkpoint"]).resolve() == path.resolve()]
    if len(matching) != 1 or not matching[0]["eligible_for_demo"]:
        raise ValueError("This pilot did not pass the independent-group demonstration gates")
    record = matching[0]
    if checksum(path) != record["checkpoint_sha256"]:
        raise ValueError("Pilot checkpoint checksum mismatch")
    if not record["completed_budget"] or not all(r["passes"] for r in record["development"]["by_dataset"].values()):
        raise ValueError("Pilot eligibility is inconsistent with its endpoint evidence")
    model, payload = load_checkpoint(path)
    if not payload["eligible_for_demo"]:
        raise ValueError("Checkpoint itself is marked ineligible")
    if checksum(path.parent/"declaration.json") != payload["declaration_sha256"]:
        raise ValueError("Pilot declaration checksum mismatch")
    return model, payload


def predict_raw(model, raw):
    """Callable inference adapter: only one fine-pitch runtime feature array."""
    array = context_features(raw, model.config.features == FRAME_FEATURES+1)
    voiced = np.asarray(raw)[:, 66] > .5
    with torch.inference_mode():
        result = model(torch.from_numpy(array[None]), torch.from_numpy(voiced[None]))[0].numpy()
    if not np.isfinite(result).all():
        raise ValueError("Non-finite local model prediction")
    return result
