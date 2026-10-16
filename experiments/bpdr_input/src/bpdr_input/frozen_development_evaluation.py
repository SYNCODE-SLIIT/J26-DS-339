"""Post-selection development diagnosis; frozen checkpoints, no fitting or adoption."""
import numpy as np
import torch
from .matched_comparison import HIERARCHY,METRICS,macro,metric_row,check_catalog


SELECTED_UPDATES={'augmentation_only':25,'frame_only':50,'bpdr':300}
EXTRA_METRICS=('original_plateau_alteration_cents','plateau_counter_detuning_error_cents','plateau_response_gain')
ALL_METRICS=METRICS+EXTRA_METRICS


def frozen_model_inventory(manifest):
    if manifest['status']!='verified' or manifest['production_eligible'] is not False or set(manifest['models'])!=set(SELECTED_UPDATES):
        raise ValueError('Expected the three verified exploratory CPU exports')
    result=[]
    for name,update in SELECTED_UPDATES.items():
        r=manifest['models'][name]
        if r['selected_update']!=update or not r['existing_cpu_loader_passed'] or not r['all_state_tensors_cpu']:
            raise ValueError('Frozen historical checkpoint selection changed')
        result.append(dict(condition=name,**r))
    return result


def evaluation_inventory(records,originals):
    check_catalog(records,'development')
    if len(records)!=304 or {ds:sum(r['dataset']==ds for r in records) for ds in ['vocadito','jamendo']}!={'vocadito':19,'jamendo':285}:
        raise ValueError('Expected the explicit 304-example development inventory')
    if len(originals)!=88 or len({(r['dataset'],r['phrase_id']) for r in originals})!=88 or any(r['split']!='development' for r in originals):
        raise ValueError('Retain all 88 unique development original controls')
    for r in records+originals:
        expected=('human_annotation','annotated_singer_id') if r['dataset']=='vocadito' else ('model_estimate','artist_proxy_singer_unknown')
        if (r['reference_type'],r['group_kind'])!=expected:raise ValueError('Reference/singer evidence role changed')


def frozen_forward(model,normalizer,raw):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Requires eval mode and all parameters frozen')
    width=model.config.features
    if raw.ndim!=2 or raw.shape[1]!=width or width not in (70,71) or not np.isfinite(raw).all():
        raise ValueError('Invalid declared model input')
    with torch.inference_mode():
        value=model(torch.from_numpy(normalizer(raw)[None,:,:])).squeeze(0).cpu().numpy().copy()
    if value.shape!=(len(raw),) or not np.isfinite(value).all():raise ValueError('Invalid prediction')
    return value


def response_row(record,predictions,pair):
    row=metric_row(record,predictions,pair['inputs'],pair['mask'],pair['detuning'])
    plateau=pair['mask']&(pair['detuning']!=0);target=-pair['detuning'][plateau]
    row['original_plateau_alteration_cents']=float(np.mean([abs(predictions[v][plateau]).mean() for v in 'AB']))
    changes=[predictions[v][plateau]-predictions[a][plateau] for v,a in [('C','A'),('D','B')]]
    row['plateau_counter_detuning_error_cents']=float(np.mean([abs(delta-target).mean() for delta in changes]))
    row['plateau_response_gain']=float(np.mean([np.mean(delta/target) for delta in changes]))
    return row


def summarize(rows,originals):
    def block(group):return dict(pairs=len(group),metrics={m:macro(group,m) for m in ALL_METRICS})
    datasets=sorted({r['dataset'] for r in rows});missing=sorted({'vocadito','jamendo'}-set(datasets))
    return dict(pairs=len(rows),available_datasets=datasets,missing_datasets=missing,
        dataset_balanced_macro={m:macro(rows,m) for m in ALL_METRICS},
        full_two_dataset_macro=None if missing else {m:macro(rows,m) for m in ALL_METRICS},
        by_dataset={ds:dict(block([r for r in rows if r['dataset']==ds]),
            independent_groups=len({r['group_id'] for r in rows if r['dataset']==ds}),
            by_direction={str(c):block([r for r in rows if r['dataset']==ds and r['cents']==c]) for c in sorted({r['cents'] for r in rows if r['dataset']==ds})},
            by_corruption={k:block([r for r in rows if r['dataset']==ds and r['corruption']==k]) for k in sorted({r['corruption'] for r in rows if r['dataset']==ds})}) for ds in datasets},
        by_group={g:block([r for r in rows if r['group_id']==g]) for g in sorted({r['group_id'] for r in rows})},
        original_preservation=dict(originals=len(originals),
            mask='All original confidence >=0.8 voiced frames, independent of paired audit masks',
            by_dataset={ds:dict(originals=sum(r['dataset']==ds for r in originals),voiced_originals=sum(r['dataset']==ds and r['voiced_frames']>0 for r in originals),
                mean_alteration_cents=macro([r for r in originals if r['dataset']==ds],'original_all_voiced_alteration_cents'),
                edit_fraction=macro([r for r in originals if r['dataset']==ds],'original_all_voiced_edit_fraction')) for ds in sorted({r['dataset'] for r in originals})}),
        scope='Post-selection exploratory development diagnosis; no checkpoint selection or adoption')
