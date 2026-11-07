"""Uniform source-region metadata after the independently replayed melodic pilot.

All input audio/context bytes stay fixed. Region constancy alone changes; no gaps
are bridged, original data roles remain, and no correction model runs here.
"""
from collections import Counter
from pathlib import Path
import numpy as np
from bpdr_input.repair.held_voice_policy import spans

import prepare_full_training_windows as parent

WS=parent.WS
ROOT=WS/'results/transformer_usability/2026-11-04_v63_reliable_region_inputs'
PARENTS=[ROOT.parent/'2026-11-04_v58_full_window_short_pairs']
read,write,norm,sha=parent.read,parent.write,parent.norm,parent.sha
check_lock=parent.common.check_lock


def main():
    check_lock();source=PARENTS[0];pilot=ROOT.parent/'2026-11-04_v60_melodic_construction'
    proof=read(source/'integrity_receipt.json');tested=read(pilot/'integrity_receipt.json')
    assert proof['staged_pairs_verified']==349 and tested['physical_audit_passes']==50
    assert tested['negative_controls_replayed']==224
    assert tested['results_sha256']==sha(pilot/'results.json')
    assert read(pilot/'results.json')['summary']['negative_false_accepts']==0
    original_rows=read(source/'train_originals.json');assert len(original_rows)==72
    assert proof['train_originals_sha256']==sha(source/'train_originals.json')
    rows=[];hashes={}
    for old in original_rows:
        row=dict(old,region_policy='longest_continuous_source_reliability',prior_region_frames=old.get('longest_region_frames'),
            source=dict(old['source'],prior_original=old),reuse_historical_case=False)
        if 'excluded' not in old:
            with np.load(old['analysis_A']) as z:mask=z['reliable'].copy()
            runs=spans(mask);region=list(map(int,max(runs,key=lambda p:p[1]-p[0]))) if runs else None
            widths=[w for w in (.24,.28,.36,.50) if region is not None and (region[1]-region[0])*.02>=w+.30-1e-12]
            row.update(longest_region_frames=region,widths_with_150ms_context=widths,
                identical_to_historical_working_pcm=True,
                reuse_historical_case=bool(region==old['longest_region_frames'] and widths==old['widths_with_150ms_context']))
            for p in [old['path_A'],old['analysis_A']]:hashes[p]=sha(p)
        rows.append(row)
    ROOT.mkdir(parents=True,exist_ok=True)
    declaration=norm(dict(version='uniform-reliable-region-inputs-1',source_sha256=sha(__file__),
        parent_originals_sha256=sha(source/'train_originals.json'),parent_integrity_sha256=sha(source/'integrity_receipt.json'),
        pilot_integrity_sha256=sha(pilot/'integrity_receipt.json'),source_windows=72,working_windows=53,
        source_recordings=33,planned_physical_attempts=1152,
        hypothesis='A constant-pitch source segment is unnecessary for controlled shifts of continuously reliable melody; V60 supports a uniform construction test',
        changed_variable='Source-region metadata only: longest continuous reliable span instead of70c pitch-constancy cuts',
        unchanged='All input audio/context, data roles,150ms context, center rule, renderer, audit errors/direction/coverage,3 common native observations, exact outside samples',
        budget='72 slots x4 widths x4 signed shifts;0 model fits. Reuse only unchanged input PCM plus identical region/width geometry; otherwise recreate/audit.',
        stopping='False acceptance of any new-case negative control invalidates the construction phase; no threshold sweep. Independent replay before fitting.',
        new_model_fits=0,nontraining_audio_accessed=False,research_training_eligible=False,production_eligible=False))
    path=ROOT/'declaration.json'
    if path.exists():assert read(path)==declaration
    else:write(path,declaration)
    write(ROOT/'train_originals.json',rows);write(ROOT/'input_hashes.json',hashes)
    valid=[r for r in rows if 'excluded' not in r]
    summary=dict(window_slots=72,valid_working_windows=53,unchanged_region_windows=sum(r['reuse_historical_case'] for r in valid),
        changed_region_windows=sum(not r['reuse_historical_case'] for r in valid),
        potential_window_width_combinations=sum(len(r['widths_with_150ms_context']) for r in valid),
        potential_construction_groups=sorted({r['group'] for r in valid if r['widths_with_150ms_context']}),
        excluded_windows=dict(Counter(r['excluded'] for r in rows if 'excluded' in r)),new_model_fits=0)
    write(ROOT/'summary.json',summary)
    # Independent metadata check against the immutable parent arrays, not a copy
    # of the cached proposed geometry; audio/context checksums are also rechecked.
    for old,new in zip(original_rows,read(ROOT/'train_originals.json')):
        assert (old['source_id'],old['group'],old['split'])==(new['source_id'],new['group'],new['split'])
        if 'excluded' in old:assert new['excluded']==old['excluded'];continue
        assert old['path_A']==new['path_A'] and old['analysis_A']==new['analysis_A']
        with np.load(old['analysis_A']) as z:mask=z['reliable'].copy()
        padded=np.r_[False,mask,False].astype(np.int8);changes=np.diff(padded)
        regions=list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))
        expected=list(map(int,max(regions,key=lambda p:p[1]-p[0]))) if regions else None
        assert new['longest_region_frames']==expected
    for p,h in hashes.items():assert sha(p)==h
    write(ROOT/'integrity_receipt.json',dict(window_slots_verified=72,valid_working_windows=53,
        source_and_partition_identity_checked=True,all_regions_rederived_from_parent_masks=True,no_audio_or_context_changed=True,
        declaration_sha256=sha(path),train_originals_sha256=sha(ROOT/'train_originals.json'),input_hashes_sha256=sha(ROOT/'input_hashes.json'),
        source_sha256=sha(__file__),new_model_fits=0,research_training_eligible=False,production_eligible=False))
    print('UNIFORM_RELIABLE_REGION_INPUTS_READY',summary,flush=True)


if __name__=='__main__':main()
