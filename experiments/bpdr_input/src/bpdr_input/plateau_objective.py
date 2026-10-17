"""A versioned support-balanced objective; the historical objective is unchanged.

P is the frozen common reliable mask intersected with nonzero injected detuning;
O is its zero-detuning complement. These are training targets, never input
features or inference-time masks. No missing region is replaced with zero loss.
"""
from copy import deepcopy
import torch
from .losses import LossWeights, masked_mean_abs, smoothness, boundary_invariance

VERSION='plateau-balanced-objective-0.1'
SPEC=dict(version=VERSION,original_anchor=.5,plateau_repair=.25,outside_preservation=.25,
          plateau_response=.5,outside_response=.5,
          regions='P = common_reliable_mask & (injected_detuning != 0); O = common_reliable_mask & (injected_detuning == 0)',
          original_scope='A/B on the full unchanged common reliable mask',
          reduction='Per-example region means, then equal example mean; strict nonempty P and O for every example',
          unchanged='Boundary invariance on the full common mask; adjacent voiced smoothness; weights 1/1/0.1',
          missing_region='Refuse the batch; never silently skip, zero-fill or renormalize a missing protection region')


def regions(batch):
    mask,detuning=batch['mask'],batch['detuning']
    if mask.ndim!=2 or mask.dtype!=torch.bool or mask.shape!=detuning.shape or not torch.isfinite(detuning).all():
        raise ValueError('Invalid common mask or injected targets')
    padding=batch['padding']
    if padding.shape!=mask.shape or padding.dtype!=torch.bool or (mask&padding).any():
        raise ValueError('Padding may never enter a target region')
    plateau=mask&(detuning!=0);outside=mask&(detuning==0)
    if not plateau.any(dim=1).all() or not outside.any(dim=1).all():
        raise ValueError('Every example requires nonempty correction and outside preservation regions')
    return plateau,outside


def objective(predictions,batch,condition,weights=LossWeights(),response=True):
    if condition not in ('augmentation_only','frame_only','bpdr'):
        raise ValueError('Unknown condition')
    plateau,outside=regions(batch);mask=batch['mask'];target=-batch['detuning']
    if set(predictions)!=set('ABCD') or any(v.shape!=mask.shape or not torch.isfinite(v).all() for v in predictions.values()):
        raise ValueError('Invalid four-view predictions')
    r=predictions;zero=torch.zeros_like(target);parts={};skipped={}
    def mean(views,m,t):return sum(masked_mean_abs(r[v],t,m)[0] for v in views)/len(views)
    parts['original_anchor']=mean('AB',mask,zero)
    parts['plateau_repair']=mean('CD',plateau,target)
    parts['outside_preservation']=mean('CD',outside,zero)
    parts['anchor']=.5*parts['original_anchor']+.25*parts['plateau_repair']+.25*parts['outside_preservation']
    smooth=[smoothness(r[v],batch['voiced']['A' if v in 'AB' else 'C']) for v in 'ABCD']
    parts['smoothness']=sum(x[0] for x in smooth)/4
    skipped['smoothness']=sum(x[1] for x in smooth)
    total=parts['anchor']+weights.smoothness*parts['smoothness']
    if condition=='bpdr' and response:
        parts['boundary'],skipped['boundary']=boundary_invariance(r,mask)
        for name,m,t in [('plateau_response',plateau,target),('outside_response',outside,zero)]:
            parts[name]=sum(masked_mean_abs(r[v]-r[a],t,m)[0] for v,a in [('C','A'),('D','B')])/2
        parts['detuning']=.5*parts['plateau_response']+.5*parts['outside_response']
        total=total+weights.boundary*parts['boundary']+weights.detuning*parts['detuning']
    return total,parts,skipped


def locked_protocol(reference_protocol,reference_run):
    p=deepcopy(reference_protocol)
    p.update(version='plateau-objective-comparison-0.1',status='prepared_not_fitted',
        objective_changed=True,architecture_changed=False,objective=deepcopy(SPEC),reference_run=str(reference_run),
        loss_reduction=SPEC['reduction'],training_updates_executed=0,total_planned_optimizer_updates=900,
        comparison_design='Three new 300-update fits, paired by condition with the completed expanded comparison; reuse its 900-update control arm, do not refit it.',
        sole_declared_intervention='Support-balanced anchor and counter-detuning reductions; boundary/smoothness unchanged. This changes gradient magnitudes and is not proof that target dilution caused prior failure.',
        matched_artifacts='Exactly the same 995 training records, 300-by-8 schedule, normalization, 304 development records, 88 originals, partitions, initialization, optimizer, warm-up and checkpoint selection.',
        response_reporting='Report direct plateau error AND shifted-minus-original plateau response error/gain, both datasets, signed directions and corruptions; note whether selected update is within warm-up.',
        warmup_interpretation='New conditions share the revised anchor during updates 1-100. Their warm-up is not expected to equal the old-objective arm.',
        adoption='No model adoption. Numerical selection remains the unchanged engineering rule; useful response and musical benefit require separate evidence.',
        interpretation='One exploratory objective-family comparison. Existing development set is reused, so this is adaptive development, not independent confirmatory or held-out accuracy. One seed; no population or musical-quality claim.')
    return p
