"""Prospective consecutive10s windows over all existing training sources.

Retain every final short slot as excluded; no model/outcome-based window selection.
The first-window canonical examples remain linked to their verified V55 parents.
"""
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

import prepare_canonical_training_inputs as common

WS=common.WS
ROOT=WS/'results/transformer_usability/2026-11-04_v57_full_training_windows'
PARENTS=[ROOT.parent/'2026-11-02_v55_canonical_short_pairs']
read,write,norm,sha=common.read,common.write,common.norm,common.sha
pipeline,recovery=common.pipeline,common.recovery


def declare():
    common.check_lock()
    parent=PARENTS[0]
    receipt=read(parent/'integrity_receipt.json')
    assert receipt['staged_pairs_verified']==188 and receipt['historical_valid_pairs_retained']==176
    originals=read(parent/'train_originals.json')
    assert receipt['train_originals_sha256']==sha(parent/'train_originals.json')
    slots=[]
    for original in originals:
        source=original['source']; path=source['native_path']; expected=source['native_sha256']
        assert sha(path)==expected and original['split']=='train'
        info=sf.info(path); step=10*info.samplerate
        for window,start in enumerate(range(0,info.frames,step)):
            count=min(step,info.frames-start)
            slots.append(dict(source_id=original['source_id'] if window==0 else original['source_id']+f'__window{window:02}',
                parent_source_id=original['source_id'],group=original['group'],domain=original['domain'],
                native_path=path,native_sha256=expected,source_channel=original['source_channel'],
                native_rate=info.samplerate,native_start=start,native_samples=count,window_index=window,
                duration_seconds=count/info.samplerate,split='train'))
    assert len(originals)==33 and len(slots)==72
    ROOT.mkdir(parents=True,exist_ok=True)
    d=norm(dict(version='full-source-consecutive-training-windows-1',source_sha256=sha(__file__),
        frontend_sha256=sha(pipeline.__file__),context_source_sha256=sha(recovery.__file__),
        parent_integrity_sha256=sha(parent/'integrity_receipt.json'),parent_originals_sha256=sha(parent/'train_originals.json'),
        slots=slots,original_source_count=33,window_slots=72,metadata_duration_eligible=53,
        plan='Every consecutive nonoverlapping10s window, including final shorter remainder; >=5s task minimum; no padding, extension, outcome search or favorable replacement',
        channels='Same predeclared full-source highest-energy native channel as V54; no window-specific channel search',
        first_windows='Reuse verified canonical first-window audio/context exactly; retain earlier artifacts and all528 first-window intervention attempts',
        later_windows='Preserve native slice as FLOAT; unmodified app process(); source-only context/70c regions; never load guide notes',
        prior_use='Historical training sources; extra windows are not extra people, independent source recordings or fresh confirmation',
        nontraining_audio_accessed=False,new_interventions=0,new_model_fits=0,
        research_training_eligible=False,production_eligible=False))
    p=ROOT/'declaration.json'
    if p.exists():assert read(p)==d
    else:write(p,d)
    return d,{r['source_id']:r for r in originals}


def main():
    torch.set_num_threads(2)
    d,parents=declare(); originals=[]; hashes={}
    for index,slot in enumerate(d['slots'],1):
        folder=ROOT/slot['source_id']; folder.mkdir(exist_ok=True)
        saved_path=folder/'result.json'
        if saved_path.exists():
            saved=read(saved_path); assert saved['declaration_sha256']==sha(ROOT/'declaration.json')
            for path,expected in saved['output_hashes'].items():assert sha(path)==expected
            originals.append(saved['original']);hashes.update(saved['output_hashes']);continue
        parent=parents[slot['parent_source_id']]
        paths=[]
        source=dict(native_path=slot['native_path'],native_sha256=slot['native_sha256'],split='train',
                    prior_original=parent if slot['window_index']==0 else None)
        if slot['window_index']==0:
            original=dict(parent,source=source,parent_source_id=slot['parent_source_id'],window_index=0,
                crop_offset_seconds=0.,identical_to_historical_working_pcm='excluded' not in parent)
            if 'path_A' in original:paths.append(Path(original['path_A']))
            if 'analysis_A' in original:paths.append(Path(original['analysis_A']))
            paths.append(Path(original['canonical_diagnostics']))
        else:
            original=dict(source_id=slot['source_id'],parent_source_id=slot['parent_source_id'],
                group=slot['group'],domain=slot['domain'],source=source,split='train',window_index=slot['window_index'],
                crop_offset_seconds=slot['native_start']/slot['native_rate'],crop_native_samples=slot['native_samples'],
                native_duration_seconds=slot['duration_seconds'],source_channel=slot['source_channel'],sample_rate_hz=22050,
                identical_to_historical_working_pcm=False,research_training_eligible=False)
            if slot['duration_seconds']<5.:
                original.update(excluded='SOURCE_SHORTER_THAN_FIVE_SECOND_TARGET',canonical_status='SHORT_SLOT_NOT_PROCESSED')
            else:
                native,sr=sf.read(slot['native_path'],start=slot['native_start'],frames=slot['native_samples'],dtype='float32',always_2d=True)
                assert sr==slot['native_rate'] and len(native)==slot['native_samples']
                crop=native[:,slot['source_channel']]
                crop_path=folder/'native_training_window.wav';sf.write(crop_path,crop,sr,subtype='FLOAT')
                decoded,rate=sf.read(crop_path,dtype='float32');assert rate==sr
                np.testing.assert_array_equal(crop,decoded)
                result,directory=pipeline.process(crop_path,folder/'canonical_process',WS/'checkpoints')
                diagnostics_path=directory/'diagnostics.json';diagnostics=read(diagnostics_path)
                original.update(canonical_status=result.status.value,canonical_diagnostics=str(diagnostics_path),canonical_operations=diagnostics['operations'])
                paths.extend([crop_path,diagnostics_path,Path(diagnostics['source']['preserved_path'])])
                if result.accepted_audio is None:
                    original.update(excluded='CANONICAL_FRONTEND_REJECTED',reason_codes=list(result.reason_codes))
                else:
                    accepted=result.accepted_audio
                    audio,rate=sf.read(accepted.path,dtype='float32');assert rate==22050
                    a=recovery.measured(audio)
                    context=folder/'canonical_repair_analysis.npz';common.save_analysis(context,a)
                    regions=common.segment_notes(a['views'][0][:,64],a['reliable'],common.RulePolicy(note_split_cents=70.))
                    region=max(regions,key=lambda r:r[1]-r[0]) if regions else None
                    original.update(path_A=accepted.path,analysis_A=str(context),duration_seconds=accepted.duration_seconds,
                        canonical_working_sha256=accepted.checksum,longest_region_frames=region,
                        widths_with_150ms_context=[w for w in (.24,.28,.36,.50) if region is not None and (region[1]-region[0])*.02>=w+.30-1e-12])
                    paths.extend([Path(accepted.path),context])
        assert sha(slot['native_path'])==slot['native_sha256']
        saved=dict(declaration_sha256=sha(ROOT/'declaration.json'),original=original,output_hashes={str(p):sha(p) for p in paths})
        write(saved_path,saved);originals.append(original);hashes.update(saved['output_hashes'])
        write(ROOT/'progress.json',dict(completed_slots=index,planned_slots=72))
        print('FULL_TRAIN_WINDOW',index,'/72',slot['source_id'],original.get('excluded',original.get('widths_with_150ms_context')),flush=True)
    write(ROOT/'train_originals.json',originals);write(ROOT/'input_hashes.json',hashes)
    valid=[r for r in originals if 'excluded' not in r]
    summary=dict(original_source_recordings=33,window_slots=72,metadata_duration_eligible=53,valid_working_windows=len(valid),
        exclusions=dict(Counter(r['excluded'] for r in originals if 'excluded' in r)),
        additional_valid_windows=sum(r['window_index']>0 for r in valid),
        groups_with_short_context=sorted({r['group'] for r in valid if r['widths_with_150ms_context']}),
        s25_windows=[dict(source_id=r['source_id'],offset=r['crop_offset_seconds'],excluded=r.get('excluded'),region=r.get('longest_region_frames'),widths=r.get('widths_with_150ms_context')) for r in originals if r['group']=='S25'],
        new_people=0,new_interventions=0,new_model_fits=0,nontraining_audio_accessed=False,
        research_training_eligible=False,production_eligible=False)
    write(ROOT/'summary.json',summary);print('FULL_TRAINING_WINDOWS_READY',summary,flush=True)


if __name__=='__main__':main()
