"""Global rank readout of learned predictions inside input-measured regions."""
import numpy as np
from bpdr_input.modeling.joint_transformer import predict_raw as dense_predict
from bpdr_input.modeling.step_candidates import measured_candidates

VERSION='input-candidate-rank-readout-1'
MODES={'median':.5,'upper_quartile':.75}


def representative_indices(raw,values,mode):
    values=np.asarray(values)
    if (mode not in MODES or values.shape!=(len(raw),) or not np.isfinite(values).all()
            or np.any(abs(values)>30.)):
        raise ValueError('Invalid readout mode or bounded learned prediction')
    indices=np.arange(len(raw))
    for start,stop in measured_candidates(raw):
        one=values[start:stop];direction=np.sign(np.median(one))
        if direction==0:indices[start:stop]=-1;continue
        order=np.argsort(direction*one,kind='stable')
        rank=int(np.rint(MODES[mode]*(len(one)-1)))
        indices[start:stop]=start+int(order[rank])
    indices[np.asarray(raw)[:,66]<=.5]=-1
    return indices


def apply_indices(values,indices):
    result=np.asarray(values)[np.maximum(indices,0)].copy()
    result[indices<0]=0.
    return result


def project(raw,values,mode):return apply_indices(values,representative_indices(raw,values,mode))


def predict_raw(model,raw,mode,batch_size=128):
    output,activity=dense_predict(model,raw,batch_size)
    indices=representative_indices(raw,output,mode)
    # Activity follows the same actual learned frame; not a confidence claim.
    return apply_indices(output,indices),apply_indices(activity,indices)
