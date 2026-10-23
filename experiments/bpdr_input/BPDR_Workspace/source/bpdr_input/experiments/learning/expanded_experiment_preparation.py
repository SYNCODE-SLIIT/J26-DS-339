"""Exact-contract quarantine and a locked exploratory fit protocol; never fits a model."""
from copy import deepcopy
import numpy as np
from bpdr_input.experiments.learning.comparison_preparation import checked_training_records, comparison_protocol

QUARANTINED_PHRASE='vocadito_37_p00'


def resolve_training_catalog(records,identity_audit):
    checked_training_records(records)
    if len(records)!=1013 or identity_audit['passed'] or all(identity_audit['acoustic_arrays_exact'].values()):
        raise ValueError('Expected the recorded exact-feature identity refusal and candidate catalog')
    if not all(identity_audit[k] for k in ['sample_passthrough_exact','float_wav_roundtrip_exact','source_yin_exact','checkpoint_exact']):
        raise ValueError('This decision is specific to an exact-audio/pitch feature-cache mismatch')
    selected=[];quarantined=[]
    for r in records:
        (quarantined if r['dataset']=='vocadito' and r['phrase_id']==QUARANTINED_PHRASE else selected).append(deepcopy(r))
    if len(quarantined)!=18 or len(selected)!=995 or {r['dataset'] for r in quarantined}!={'vocadito'}:
        raise ValueError('Only the eighteen named legacy variants may be quarantined')
    if {ds:sum(r['dataset']==ds for r in selected) for ds in ['vocadito','jamendo']}!={'vocadito':77,'jamendo':918}:
        raise ValueError('Expanded training dataset roles/counts changed')
    return selected,quarantined


def array_difference(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Cache shape or dtype differs')
    same=np.isclose(a,b,atol=0,rtol=0,equal_nan=True)
    result=dict(exact=bool(np.array_equal(a,b,equal_nan=True)),different_elements=int((~same).sum()),shape=list(a.shape),dtype=str(a.dtype))
    if result['exact']:return dict(result,maximum_absolute_difference=0.,maximum_ulp_distance=0)
    if a.dtype not in (np.dtype('float32'),np.dtype('float64')) or np.any(~np.isfinite(a[~same])) or np.any(~np.isfinite(b[~same])):
        raise ValueError('Different non-finite values or non-floating cache entries')
    width=a.dtype.itemsize*8;uint=np.uint32 if width==32 else np.uint64;sign=np.array(1<<(width-1),dtype=uint)
    def ordered(v):
        bits=np.ascontiguousarray(v).view(uint)
        return np.where(bits&sign,~bits,bits|sign)
    x=ordered(a[~same]);y=ordered(b[~same]);ulp=np.maximum(x,y)-np.minimum(x,y)
    return dict(result,maximum_absolute_difference=float(abs(a[~same].astype(np.float64)-b[~same].astype(np.float64)).max()),maximum_ulp_distance=int(ulp.max()))


def locked_protocol():
    p=deepcopy(comparison_protocol())
    p.update(version='expanded-training-experiment-0.1',status='prepared_not_fitted',training_examples=995,
        training_datasets=dict(vocadito=77,jamendo=918),quarantined_legacy_examples=18,development_examples=304,original_controls=88,
        objective_changed=False,architecture_changed=False,model_adopted=False,research_ready_examples=0,
        numerical_compatibility='Exact-feature rule retained. Named legacy eighteen excluded from this new experiment only; old artifacts and failed audit keys unchanged.',
        normalization='New candidate-only weighted real-frame moments fitted on exactly the 995 selected training records, four views per record, using hierarchical dataset/group/source/phrase/shift/corruption mass. Padding excluded; natural pauses retained. One shared 71-column state, first 70 for frame-only.',
        execution=dict(device='xpu',dtype='float32',cpu_threads=2,microbatch_size=1,effective_batch_size=8,gradient_accumulation=8,
            fallback='Stop on missing declared XPU runtime/device; no silent CPU fallback or budget change'),
        total_planned_optimizer_updates=900,training_updates_executed=0,
        interpretation='One fixed exploratory experiment. Data, normalization and checkpoint selection differ from historical run, so no causal attribution to source diversity alone. No production, musical or held-out accuracy claim.')
    p['checkpoint_selection']=dict(partition='development',candidates=list(range(25,301,25)),
        criterion='Minimize equal-dataset/group/recording/phrase/shift/corruption plateau-only repair_error_cents among preservation-qualified checkpoints; require strict improvement over zero correction; ties earliest update.',
        preservation_limit_cents=1.,preservation_scope='Separately in both datasets: original audited-plateau alteration, shifted outside alteration, and all-original voiced-frame alteration each <= 1 cent. Existing diagnostic one-cent engineering bound, not human calibration.',
        original_inventory='Retain all 88; all-voiced metrics available for 13/14 Vocadito and 74/74 Jamendo. vocadito_9_p01 remains unavailable, not zero.',
        whole_mask_mae='Report only; never use whole-common-mask error as repair-selection proxy.',
        no_qualified_improvement='Select zero correction; keep all model checkpoints and diagnostics, mark learned benefit unproven.',
        scope='Post-selection development engineering; no PP1/final-holdout access or model adoption.')
    p['development_preparation']=dict(status='reuse_verified_caches_only',paired_vocadito=19,paired_jamendo=285,original_controls=88,
        new_audio_or_acoustic_processing=False,partitions='Existing singer/artist partitions remain frozen')
    p['reporting']+='; include plateau paired response gain/counter-detuning and all-original voiced preservation, both dataset roles, directions/corruptions and every missing subset.'
    return p


def preservation_qualified(summary,limit=1.):
    if summary.get('pairs')!=304 or summary.get('missing_datasets') or set(summary.get('by_dataset',{}))!={'vocadito','jamendo'}:
        return False
    original=summary.get('original_preservation',{})
    if original.get('originals')!=88:return False
    for ds,expected in [('vocadito',(14,13)),('jamendo',(74,74))]:
        if ds not in original.get('by_dataset',{}):return False
        o=original['by_dataset'][ds]
        if (o.get('originals'),o.get('voiced_originals'))!=expected:return False
        m=summary['by_dataset'][ds]['metrics']
        values=[m.get('original_plateau_alteration_cents'),m.get('outside_alteration_cents'),o.get('mean_alteration_cents')]
        if any(v is None or not np.isfinite(v) or v<0 or v>limit for v in values):return False
    value=summary.get('dataset_balanced_macro',{}).get('repair_error_cents')
    return value is not None and np.isfinite(value) and value>=0


def select_checkpoint(candidates,zero_error):
    if not np.isfinite(zero_error) or zero_error<0:raise ValueError('Invalid zero-correction plateau error')
    updates=[c['update'] for c in candidates]
    if len(set(updates))!=len(updates) or any(u not in range(25,301,25) for u in updates):raise ValueError('Undeclared checkpoint or duplicate update')
    eligible=[c for c in candidates if preservation_qualified(c['summary']) and c['summary']['dataset_balanced_macro']['repair_error_cents']<zero_error]
    if not eligible:return None
    return min(eligible,key=lambda c:(c['summary']['dataset_balanced_macro']['repair_error_cents'],c['update']))['update']
