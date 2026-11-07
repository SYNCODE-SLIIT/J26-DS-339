"""Distinguish pitch-constancy eligibility from continuous estimator reliability.

Potential construction eligibility only; no waveform, label or model change.
"""
from pathlib import Path
import numpy as np
import prepare_full_training_windows as run
from bpdr_input.repair.held_voice_policy import spans

ROOT=run.WS/'results/transformer_usability/2026-11-04_v59_melodic_geometry'


def main():
    run.common.check_lock()
    receipt=run.read(run.ROOT/'integrity_receipt.json')
    assert receipt['window_slots_verified']==72
    assert receipt['train_originals_sha256']==run.sha(run.ROOT/'train_originals.json')
    originals=run.read(run.ROOT/'train_originals.json');rows=[]
    for r in originals:
        if 'excluded' in r:continue
        with np.load(r['analysis_A']) as z:mask=z['reliable'].copy()
        runs=spans(mask)
        longest=list(map(int,max(runs,key=lambda x:x[1]-x[0]))) if runs else None
        widths=[w for w in (.24,.28,.36,.50) if longest is not None and (longest[1]-longest[0])*.02>=w+.30-1e-12]
        rows.append(dict(source_id=r['source_id'],group=r['group'],note_region=r['longest_region_frames'],
            reliable_run=longest,current_widths=r['widths_with_150ms_context'],continuous_reliability_widths=widths))
    old=sorted({r['group'] for r in rows if r['current_widths']})
    new=sorted({r['group'] for r in rows if r['continuous_reliability_widths']})
    ROOT.mkdir(parents=True,exist_ok=True)
    run.write(ROOT/'results.json',dict(rows=rows,current_geometry_groups=old,continuous_reliability_groups=new,
        additional_potential_groups=sorted(set(new)-set(old)),
        current_window_width_combinations=sum(len(r['current_widths']) for r in rows),
        continuous_reliability_window_width_combinations=sum(len(r['continuous_reliability_widths']) for r in rows),
        same_150ms_context=True,existing_audit_thresholds_unchanged=True,
        source_sha256=run.sha(__file__),input_integrity_sha256=run.sha(run.ROOT/'integrity_receipt.json'),
        scope='Reliable melodic movement need not be a constant-pitch region; this is only a construction hypothesis requiring unchanged decoded-audio audits and controls.',
        new_waveforms=0,new_labels=0,new_model_fits=0,production_eligible=False,research_validation_passed=False))
    print('MELODIC_SOURCE_GEOMETRY_DIAGNOSIS',old,new,sorted(set(new)-set(old)),flush=True)


if __name__=='__main__':main()
