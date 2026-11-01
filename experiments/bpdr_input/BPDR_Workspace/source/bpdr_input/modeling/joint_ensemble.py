"""Fixed three-Transformer median and member envelope; no calibrated confidence."""
import numpy as np
from bpdr_input.modeling.joint_transformer import predict_raw

VERSION='joint-median-envelope-1'


def combine(members):
    members=np.asarray(members,dtype=np.float32)
    if members.ndim!=2 or members.shape[0]!=3 or not np.isfinite(members).all() or np.any(abs(members)>30.):
        raise ValueError('Expected exactly three finite bounded member predictions')
    return np.stack([np.median(members,axis=0),members.min(axis=0),members.max(axis=0)])


def predict_views(models,raw):
    if len(models)!=3:raise ValueError('Exactly three fixed members required')
    return combine([predict_raw(model,raw)[0] for model in models])
