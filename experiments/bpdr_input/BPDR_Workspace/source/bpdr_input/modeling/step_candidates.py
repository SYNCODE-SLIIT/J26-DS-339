"""Candidate geometry from input-only step evidence; never from training labels."""
import numpy as np
from bpdr_input.modeling.step_context import evidence
from bpdr_input.modeling.joint_transformer import joint_tokens,JointConfig
from bpdr_input.repair.held_voice_policy import spans


def measured_candidates(raw):
    # Keep only the measured nonzero support. No extension, rounding up or
    # repair around unreliable/protected frames; controller checks remain later.
    cues=evidence(raw)
    return [(int(a),int(b)) for a,b in spans(cues[:,0]!=0) if b-a>=43]


def candidate_context(raw,interval,config=JointConfig()):
    start,stop=interval
    if not 0<=start<stop<=len(raw) or stop-start<43:raise ValueError('Invalid measured candidate')
    center=(start+stop-1)//2
    x,pad=joint_tokens(raw,np.array([center]),config)
    positions=center+np.arange(-config.radius,config.radius+1,config.stride)
    pool=(positions>=start)&(positions<stop)&~pad[0]
    if not pool.any():raise ValueError('Candidate context has no measured tokens')
    return x[0],pad[0],pool
