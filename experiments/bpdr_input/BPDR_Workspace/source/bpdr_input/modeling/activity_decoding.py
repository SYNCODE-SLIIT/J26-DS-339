"""Fixed decision ablation of separately trained amplitude/activity heads."""
import numpy as np
import torch
from bpdr_input.modeling.joint_transformer import joint_tokens

VERSION = 'joint-binary-activity-decoder-1'


def predict_raw(model, raw, batch_size=128, mode='binary'):
    if mode not in ('expected', 'binary'):
        raise ValueError('Unsupported activity decoder')
    model.eval(); outputs, activities = [], []
    with torch.inference_mode():
        for start in range(0, len(raw), batch_size):
            xx, pad = joint_tokens(raw, np.arange(start, min(start + batch_size, len(raw))), model.config)
            expected, logit, amplitude = model(torch.from_numpy(xx), torch.from_numpy(pad), True)
            output = expected if mode == 'expected' else torch.where(logit >= 0., amplitude, 0.)
            outputs.append(output.numpy().copy()); activities.append(torch.sigmoid(logit).numpy().copy())
    result, activity = np.concatenate(outputs), np.concatenate(activities)
    unvoiced = np.asarray(raw)[:, 66] <= .5
    result[unvoiced] = 0.; activity[unvoiced] = 0.
    if not np.isfinite(result).all() or np.any(abs(result) > 30.):
        raise ValueError('Invalid decoded correction')
    return result, activity
