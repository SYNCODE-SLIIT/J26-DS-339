"""Replay first-window links, consecutive native offsets and all later contexts."""
from collections import defaultdict
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch

import prepare_full_training_windows as run
from verify_short_phrase_training import check_arrays


def main():
    torch.set_num_threads(2)
    d,parents=run.declare()
    assert d['source_sha256']==run.sha(run.__file__)
    assert d['frontend_sha256']==run.sha(run.pipeline.__file__)
    assert d['context_source_sha256']==run.sha(run.recovery.__file__)
    root=run.ROOT; originals=run.read(root/'train_originals.json')
    assert len(originals)==72
    for path,expected in run.read(root/'input_hashes.json').items():assert run.sha(path)==expected
    slots={s['source_id']:s for s in d['slots']}; offsets=defaultdict(list)
    counts=dict(first_windows=0,short_later_slots=0,later_working_copies=0,later_context_arrays=0,frontend_rejections=0)
    recreated=root/'verification_recreated';recreated.mkdir(exist_ok=True)
    for original in originals:
        slot=slots[original['source_id']];parent=parents[slot['parent_source_id']]
        assert original['split']=='train' and original['group']==parent['group']
        assert original['source_channel']==slot['source_channel']==parent['source_channel']
        info=sf.info(slot['native_path']);assert info.samplerate==slot['native_rate']
        assert run.sha(slot['native_path'])==slot['native_sha256']
        assert slot['native_start']==slot['window_index']*10*info.samplerate
        assert slot['native_samples']==min(10*info.samplerate,info.frames-slot['native_start'])
        offsets[slot['parent_source_id']].append((slot['native_start'],slot['native_samples']))
        assert original['crop_offset_seconds']==slot['native_start']/info.samplerate
        if slot['window_index']==0:
            source=dict(native_path=slot['native_path'],native_sha256=slot['native_sha256'],split='train',prior_original=parent)
            expected=dict(parent,source=source,parent_source_id=slot['parent_source_id'],window_index=0,
                crop_offset_seconds=0.,identical_to_historical_working_pcm='excluded' not in parent)
            assert original==expected;counts['first_windows']+=1;continue
        assert not original['identical_to_historical_working_pcm']
        if slot['duration_seconds']<5.:
            assert original['excluded']=='SOURCE_SHORTER_THAN_FIVE_SECOND_TARGET'
            assert 'path_A' not in original and 'analysis_A' not in original
            counts['short_later_slots']+=1;continue
        native,sr=sf.read(slot['native_path'],start=slot['native_start'],frames=slot['native_samples'],dtype='float32',always_2d=True)
        crop=native[:,slot['source_channel']]
        assert len(crop)==slot['native_samples']
        diagnostics=run.read(original['canonical_diagnostics'])
        assert diagnostics['operations']==original['canonical_operations']
        assert run.sha(diagnostics['source']['preserved_path'])==diagnostics['source']['sha256']
        preserved,rate=sf.read(diagnostics['source']['preserved_path'],dtype='float32');assert rate==sr
        np.testing.assert_array_equal(preserved,crop)
        if 'excluded' in original:
            assert original['excluded']=='CANONICAL_FRONTEND_REJECTED'
            counts['frontend_rejections']+=1;continue
        assert np.isfinite(crop).all() and abs(crop).max()<=1
        assert float(np.mean(abs(crop)>=.999))<=run.pipeline.Policy().max_clipped_fraction
        divisor=gcd(sr,22050)
        working=resample_poly(crop,22050//divisor,sr//divisor) if sr!=22050 else crop.copy()
        length=round(len(crop)*22050/sr);working=working[:length]
        if len(working)<length:working=np.pad(working,(0,length-len(working)))
        peak=float(abs(working).max());gain=.999/peak if peak>1 else 1.;working*=gain
        path=recreated/(original['source_id']+'.wav');sf.write(path,working,22050,subtype='PCM_24')
        assert run.sha(path)==run.sha(original['path_A'])==original['canonical_working_sha256']
        gains=[op['gain'] for op in diagnostics['operations'] if op['operation']=='peak_protection_gain']
        assert gains==([gain] if gain!=1. else [])
        audio,rate=sf.read(path,dtype='float32');assert rate==22050 and len(audio)==length and abs(audio).max()<1
        a=run.recovery.measured(audio)
        counts['later_context_arrays']+=check_arrays(original['analysis_A'],a)
        regions=run.common.segment_notes(a['views'][0][:,64],a['reliable'],run.common.RulePolicy(note_split_cents=70.))
        region=max(regions,key=lambda r:r[1]-r[0]) if regions else None
        assert run.norm(region)==original['longest_region_frames']
        widths=[w for w in (.24,.28,.36,.50) if region is not None and (region[1]-region[0])*.02>=w+.30-1e-12]
        assert original['widths_with_150ms_context']==widths
        counts['later_working_copies']+=1
        print('FULL_WINDOW_REPLAY',original['source_id'],widths,flush=True)
    for sid,intervals in offsets.items():
        intervals=sorted(intervals)
        assert intervals[0][0]==0
        assert all(a+n==b for (a,n),(b,_) in zip(intervals,intervals[1:]))
        path=parents[sid]['source']['native_path']
        assert sum(n for _,n in intervals)==sf.info(path).frames
    assert len(offsets)==33 and counts['first_windows']==33 and counts['short_later_slots']==18
    summary=run.read(root/'summary.json')
    assert summary['additional_valid_windows']==counts['later_working_copies']
    assert summary['valid_working_windows']==32+counts['later_working_copies']
    run.write(root/'integrity_receipt.json',dict(window_slots_verified=72,source_recordings=33,
        first_window_links_exactly_matched=33,all_native_intervals_contiguous_nonoverlapping_complete=True,**counts,
        declaration_sha256=run.sha(root/'declaration.json'),train_originals_sha256=run.sha(root/'train_originals.json'),
        input_hashes_sha256=run.sha(root/'input_hashes.json'),summary_sha256=run.sha(root/'summary.json'),
        verifier_source_sha256=run.sha(__file__),nontraining_audio_accessed=False,new_interventions=0,new_model_fits=0,
        research_training_eligible=False,production_eligible=False))
    print('FULL_TRAINING_WINDOW_INDEPENDENT_REPLAY_PASS',counts,flush=True)


if __name__=='__main__':main()
