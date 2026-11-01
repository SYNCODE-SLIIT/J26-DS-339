"""Classical context baseline; no IDs, boundaries or target-derived features."""
import joblib
import numpy as np
from bpdr_input.modeling.joint_transformer import JointConfig, joint_tokens

VERSION = 'context-extra-trees-1'


def features(raw, centers):
    x, pad = joint_tokens(raw, centers, JointConfig())
    return np.concatenate([x.reshape(len(x), -1), pad.astype(np.float32)], axis=1)


def predict_raw(model, raw, batch_size=128):
    output = np.concatenate([model.predict(features(raw, np.arange(i, min(i + batch_size, len(raw)))))
                             for i in range(0, len(raw), batch_size)]).astype(np.float32) * 30.
    output[np.asarray(raw)[:, 66] <= .5] = 0.
    if not np.isfinite(output).all() or np.any(abs(output) > 30.):
        raise ValueError('Invalid bounded tree prediction')
    # A regression-only comparator: no invented probability or confidence head.
    return output, None


def save(path, model, metadata):
    joblib.dump(dict(version=VERSION, feature_version='joint-context-flatten-plus-padding-1',
                     model=model, **metadata), path)


def load(path):
    payload = joblib.load(path)
    if payload['version'] != VERSION or payload['feature_version'] != 'joint-context-flatten-plus-padding-1':
        raise ValueError('Unsupported context tree')
    return payload['model'], payload
