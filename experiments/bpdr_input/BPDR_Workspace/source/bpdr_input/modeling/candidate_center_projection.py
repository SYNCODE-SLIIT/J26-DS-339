"""Input-only region readout of the existing learned center prediction."""
import numpy as np
from bpdr_input.modeling.joint_transformer import predict_raw as dense_predict
from bpdr_input.modeling.step_candidates import measured_candidates

VERSION='input-candidate-center-readout-1'


def project(raw,values):
    values=np.asarray(values)
    if values.shape!=(len(raw),) or not np.isfinite(values).all() or np.any(abs(values)>30.):
        raise ValueError('Invalid bounded learned prediction')
    result=values.copy()
    # Bounds and centers come from input acoustic measurements, never labels.
    for start,stop in measured_candidates(raw):result[start:stop]=values[(start+stop-1)//2]
    result[np.asarray(raw)[:,66]<=.5]=0.
    return result


def predict_raw(model,raw,batch_size=128):
    output,activity=dense_predict(model,raw,batch_size)
    return project(raw,output),project(raw,activity)
