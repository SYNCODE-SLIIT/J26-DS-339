"""Independent geometry/unit audit of added pitch-only/reference diagnostics.

No model fit or reference-assisted feature implementation. Preserve prior tools
and results. Keep the user's original goal open; these probes are not substitutes.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
import compare_tail_preservation as run

ROOT=run.WS/'results/transformer_usability/2026-11-04_pitch_diagnostic_audit'
FL=7


def estimate(pitch,reliable,a,b):
    if a-FL<0 or b+FL>len(pitch) or b<=a or not reliable[a-FL:b+FL].all() or np.any(abs(np.diff(pitch[a-FL:b+FL]))>=70.):return None
    return float(np.clip(-(np.median(pitch[a:b])-np.median(np.r_[pitch[a-FL:a],pitch[b:b+FL]])),-30,30))


def macro(rows,key):
    groups=sorted({r['group'] for r in rows if r.get(key) is not None})
    return float(np.mean([np.mean([r[key] for r in rows if r['group']==group and r.get(key) is not None]) for group in groups])) if groups else None


def main():
    _,pairs,originals=run.declare();parent=run.read(run.DATA/'integrity_receipt.json');assert parent['staged_pairs_verified']==445
    sources={r['source_id']:r for r in originals if 'excluded' not in r};external=run.ROOT.parent/'2026-11-04_v71_oracle_level'
    paths=[run.WS/'tools/oracle_level_v71.py',run.WS/'tools/deviation_report_v74.py',external/'declaration.json',external/'results.json',run.DATA/'integrity_receipt.json']
    inputs={str(p):run.sha(p) for p in paths}
    declaration=dict(version='independent-pitch-diagnostic-audit-1',source_sha256=run.sha(__file__),input_hashes=inputs,
        protocol='Reproduce plateau-based oracle and compare actual support indices with identical seven-frame flanks/70c rule;report all445 attempts and intersection,not a new detector upper bound.',
        reference_scope='Compare centered raw64 and stored absolute YIN fine on identical registered frames. Known plateau flag rates are conditional measurements,not end-to-end localization.',
        toy='80-frame flat pitch,20c shift at[20,32),plateau[25,28),all reliable;old flanks overlap injected body;full support flanks do not.',
        no_model_fit=True,no_new_target_or_waveform=True,no_qualification_or_scope_change=True)
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'declaration.json'
    if path.exists():assert run.read(path)==declaration
    else:run.write(path,declaration)
    toy=np.zeros(80,np.float32);toy[20:32]=20.;valid=np.ones(80,bool)
    toy_result=dict(plateau_estimate=estimate(toy,valid,25,28),true_support_estimate=estimate(toy,valid,20,32),known_inverse_command=-20.)
    assert toy_result['plateau_estimate']==0 and toy_result['true_support_estimate']==-20
    cache={}
    def load(path):
        if path not in cache:
            with np.load(path) as z:cache[path]=(z['raw_views'][0].copy(),z['fine'].copy(),z['reliable'].copy(),z['times_seconds'].copy())
        return cache[path]
    rows=[];nulls=[];matched_nulls={}
    for source in sources.values():
        raw,fine,reliable,times=load(source['analysis_A'])
        for a in range(FL,len(raw)-12-FL,5):
            e=estimate(raw[:,64],raw[:,66]>.5,a,a+12)
            if e is not None:nulls.append(dict(group=source['group'],magnitude=abs(e),flag=float(abs(e)>=12.)))
    for index,pair in enumerate(pairs):
        if 'excluded' in pair:continue
        c,fc,rc,times=load(pair['analysis_C']);a,fa,ra,ta=load(sources[pair['source_id']]['analysis_A']);np.testing.assert_array_equal(times,ta)
        with np.load(pair['target_path']) as z:target=z['target'].copy();positive=z['plateau'].copy();zero=z['known_zero'].copy()
        frames=np.flatnonzero(positive);start,end=pair['support_seconds'];inside=(times>=start)&(times<end);true_frames=np.flatnonzero(inside)
        old=None
        if len(frames) and not np.any(np.diff(frames)>1):old=estimate(c[:,64],rc,int(frames[0]),int(frames[-1]+1))
        new=estimate(c[:,64],rc,int(true_frames[0]),int(true_frames[-1]+1))
        old_flanks=np.r_[np.arange(frames[0]-FL,frames[0]),np.arange(frames[-1]+1,frames[-1]+1+FL)]
        new_flanks=np.r_[np.arange(true_frames[0]-FL,true_frames[0]),np.arange(true_frames[-1]+1,true_frames[-1]+1+FL)]
        assert not inside[new_flanks].any()
        control=estimate(a[:,64],ra,int(true_frames[0]),int(true_frames[-1]+1))
        key=(pair['source_id'],start,end)
        if control is not None:matched_nulls[key]=dict(group=pair['group'],magnitude=abs(control),flag=float(abs(control)>=12.))
        dev_raw=c[:,64]-a[:,64];dev_absolute=fc-fa;truth=-target[positive]
        rows.append(dict(id='C_'+str(index),group=pair['group'],source_id=pair['source_id'],width_seconds=pair['width_seconds'],
            legacy_oracle_mae=float(np.mean(abs(old-target[positive]))) if old is not None else None,
            corrected_support_oracle_mae=float(np.mean(abs(new-target[positive]))) if new is not None else None,
            legacy_flank_frames_inside_injection=int(inside[old_flanks].sum()),true_support_flank_frames_inside_injection=0,
            centered_reference_mae=float(np.mean(abs(dev_raw[positive]-truth))),absolute_reference_mae=float(np.mean(abs(dev_absolute[positive]-truth))),
            centered_reference_oracle_flag=float(np.median(abs(dev_raw[positive]))>=12.),absolute_reference_oracle_flag=float(np.median(abs(dev_absolute[positive]))>=12.),
            centered_reference_false_frame_share=float(np.mean(abs(dev_raw[zero])>=12.)),absolute_reference_false_frame_share=float(np.mean(abs(dev_absolute[zero])>=12.)),
            positive_frames=int(positive.sum()),all_positive_frames_at_least_twelve=bool(positive.sum()>=12)))
    assert len(rows)==445
    common=[r for r in rows if r['legacy_oracle_mae'] is not None and r['corrected_support_oracle_mae'] is not None]
    reported=run.read(external/'results.json')['macro']
    summary=dict(pairs=445,legacy_scoreable=sum(r['legacy_oracle_mae'] is not None for r in rows),corrected_support_scoreable=sum(r['corrected_support_oracle_mae'] is not None for r in rows),intersection_pairs=len(common),
        legacy_oracle_macro_mae=macro(rows,'legacy_oracle_mae'),corrected_support_oracle_macro_mae=macro(rows,'corrected_support_oracle_mae'),
        intersection_legacy_macro_mae=macro(common,'legacy_oracle_mae'),intersection_corrected_macro_mae=macro(common,'corrected_support_oracle_mae'),
        legacy_null_macro_flag_share=macro(nulls,'flag'),matched_true_support_null_macro_flag_share=macro(list(matched_nulls.values()),'flag'),matched_null_spans=len(matched_nulls),
        pairs_with_contaminated_legacy_flanks=sum(r['legacy_flank_frames_inside_injection']>0 for r in rows),legacy_flank_contaminated_frame_fraction=sum(r['legacy_flank_frames_inside_injection'] for r in rows)/(445*14),
        centered_reference_macro_mae=macro(rows,'centered_reference_mae'),absolute_reference_macro_mae=macro(rows,'absolute_reference_mae'),
        centered_reference_oracle_flag_fraction=macro(rows,'centered_reference_oracle_flag'),absolute_reference_oracle_flag_fraction=macro(rows,'absolute_reference_oracle_flag'),
        centered_reference_false_frame_share=macro(rows,'centered_reference_false_frame_share'),absolute_reference_false_frame_share=macro(rows,'absolute_reference_false_frame_share'),
        pairs_with_fewer_than_twelve_registered_positive_frames=sum(r['positive_frames']<12 for r in rows))
    assert abs(summary['legacy_oracle_macro_mae']-reported['oracle_mae'])<1e-8
    assert abs(summary['legacy_null_macro_flag_share']-reported['null_share_ge12'])<1e-8
    for p,h in inputs.items():assert run.sha(p)==h
    run.write(ROOT/'results.json',dict(summary=summary,toy=toy_result,rows=rows,source_scope_findings=[
        'Original oracle-level code uses trimmed positive-mask bounds,not recorded support;its flanks can contain the intervention. The corrected fixed heuristic is still not a ceiling for every possible level estimator.',
        'Reference report flags use known positive masks and a median threshold,not a twelve-consecutive-frame end-to-end locator. Its flag rate cannot qualify deployment localization.',
        'Centered raw64 differences remove separate recording medians;absolute reference deviations require a consistent pitch origin. Exact aligned references remain an extra deployment requirement.',
        'Negative pitch-only probes do not establish failure of every richer representation. Reference-assisted design is separate from qualification of the original unreferenced Transformer goal.'
    ],declaration_sha256=run.sha(ROOT/'declaration.json'),new_model_fits=0,production_eligible=False,research_validation_passed=False))
    print('INDEPENDENT_PITCH_DIAGNOSTIC_AUDIT',summary,toy_result,flush=True)


if __name__=='__main__':main()
