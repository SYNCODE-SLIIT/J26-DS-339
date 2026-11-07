"""Stage all1152 consecutive-window attempts; preserve verified first-window pairs.

Reuse historical examples only with identical parent PCM and verified receipts.
This is training-data engineering, never research approval or model promotion.
"""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

import prepare_full_training_windows as windows
from types import SimpleNamespace
inputs=SimpleNamespace(WS=windows.WS,ROOT=windows.ROOT,PARENTS=windows.PARENTS,
    check_lock=windows.common.check_lock,read=windows.read,write=windows.write,norm=windows.norm,sha=windows.sha)
import calibrate_short_support_audit as calibration

WS = inputs.WS
ROOT = WS / 'results/transformer_usability/2026-11-04_v58_full_window_short_pairs'
read, write, norm, sha = inputs.read, inputs.write, inputs.norm, inputs.sha


def declare():
    inputs.check_lock()
    root = inputs.ROOT
    receipt = read(root / 'integrity_receipt.json')
    assert receipt['window_slots_verified'] == 72 and receipt['first_window_links_exactly_matched'] == 33
    originals = read(root / 'train_originals.json')
    assert receipt['train_originals_sha256'] == sha(root / 'train_originals.json')
    for path, expected in read(root / 'input_hashes.json').items():
        assert sha(path) == expected
    prior_rows = []
    for parent in inputs.PARENTS:
        prior_rows.extend(read(parent / 'train_pairs.json'))
    assert len(prior_rows) == 528
    prior = {(r['source_id'], r['width_seconds'], r['cents']): r for r in prior_rows}
    assert len(prior) == 528
    ROOT.mkdir(parents=True, exist_ok=True)
    d = norm(dict(version='full-window-short-physical-pairs-1', source_sha256=sha(__file__),
        canonical_input_receipt_sha256=sha(root / 'integrity_receipt.json'),
        canonical_originals_sha256=sha(root / 'train_originals.json'),
        parent_pair_manifests_sha256={str(p / 'train_pairs.json'):sha(p / 'train_pairs.json') for p in inputs.PARENTS},
        parent_integrities_sha256={str(p / 'integrity_receipt.json'):sha(p / 'integrity_receipt.json') for p in inputs.PARENTS},
        audit_source_sha256=sha(calibration.short.__file__), renderer=asdict(calibration.RENDER), audit=asdict(calibration.short.POLICY),
        planned_attempts=1152, planned_originals=72, original_source_recordings=33, widths_seconds=[.24,.28,.36,.50], cents=[-30.,-20.,20.,30.],
        support='Unchanged: longest source-only70c region, center fixed,150ms context each side',
        reuse='Only identical canonical and prior original PCM24 SHA256; verified old labels, candidate waveform and context remain unchanged; A path remapped to byte-identical canonical working copy',
        new_cases='Same decoded three-tracker audit and unchanged/opposite controls; >=3 common plateau frames and >=3 retained input-C anchor frames each side; no interpolation',
        sample_failures='Refuse nonfinite/overloaded rendered samples before PCM encoding; no candidate normalization',
        nontraining_audio_accessed=False, guide_notes_loaded=False, new_model_fits=0,
        research_training_eligible=False, production_eligible=False))
    path = ROOT / 'declaration.json'
    if path.exists():
        assert read(path) == d
    else:
        write(path, d)
    return d, originals, prior


def main():
    torch.set_num_threads(2)
    d, originals, prior = declare()
    rows, hashes = [], {}
    for original in originals:
        sid = original['source_id']; folder = ROOT / sid; folder.mkdir(exist_ok=True)
        saved_path = folder / 'source_result.json'
        if saved_path.exists():
            saved = read(saved_path)
            assert saved['declaration_sha256'] == sha(ROOT / 'declaration.json')
            for path, expected in saved['output_hashes'].items():
                assert sha(path) == expected
            rows.extend(saved['pairs']); hashes.update(saved['output_hashes'])
            print('FULL_WINDOW_PAIRS_SOURCE_REUSED',sid,flush=True)
            continue
        source_rows, paths = [], []
        if 'excluded' not in original:
            paths += [Path(original['path_A']), Path(original['analysis_A'])]
        reuse = original.get('identical_to_historical_working_pcm', False)
        if 'excluded' not in original and not reuse:
            audio, sr = sf.read(original['path_A'],dtype='float32'); assert sr == 22050
            with np.load(original['analysis_A']) as z:
                times=z['times_seconds'].copy(); reliable=z['reliable'].copy()
            region=original['longest_region_frames']
            if original['widths_with_150ms_context']:
                bf=calibration.coarse_features(audio,sr,WS/'checkpoints')
                yb=calibration.fine_hz(audio,sr,times); cb=calibration.short.cross_correlation_pitch(audio,sr,times)
                before=folder/'before_measurements.npz'; np.savez_compressed(before,**bf,yin=yb,cc=cb,reliable=reliable); paths.append(before)
        for width in d['widths_seconds']:
            for cents in d['cents']:
                row=dict(source_id=sid,group=original['group'],width_seconds=width,cents=cents,split='train',
                         research_training_eligible=False,reused_historical_case=False)
                if 'excluded' in original:
                    source_rows.append(dict(**row,excluded=original['excluded'])); continue
                if reuse:
                    previous=prior[(sid,width,cents)]
                    row=dict(previous,reused_historical_case=True)
                    if 'excluded' not in row:
                        row.update(path_A=original['path_A'],analysis_A=original['analysis_A'])
                        for key in ('path_C','analysis_C','target_path'):
                            paths.append(Path(row[key]))
                    source_rows.append(row); continue
                if width not in original['widths_with_150ms_context']:
                    source_rows.append(dict(**row,excluded='NO_SINGLE_RELIABLE_REGION_WITH_150MS_CONTEXT')); continue
                center=float(np.mean(region)*.02); support=(center-width/2,center+width/2)
                curve=np.where((times>=support[0])&(times<support[1]),cents,0.)
                case=folder/f'{width:.2f}_{cents:+g}'; case.mkdir(exist_ok=True)
                output,render_receipt=calibration.render_candidate(audio,22050,times,curve,[support],calibration.RENDER)
                row.update(path_A=original['path_A'],analysis_A=original['analysis_A'],support_seconds=support,render_receipt=render_receipt)
                if not np.isfinite(output).all() or abs(output).max()>=1:
                    source_rows.append(dict(**row,excluded='PHYSICAL_RENDER_PEAK_OR_SAMPLE_FAILURE')); continue
                c_path=case/'intervention.wav'; sf.write(c_path,output,22050,subtype='PCM_24')
                decoded,sr=sf.read(c_path,dtype='float32')
                assert sr==22050 and decoded.shape==audio.shape and np.isfinite(decoded).all() and abs(decoded).max()<1
                st=np.arange(len(audio))/sr; outside=(st<=support[0])|(st>=support[1])
                np.testing.assert_array_equal(decoded[outside],audio[outside])
                af=calibration.coarse_features(decoded,sr,WS/'checkpoints'); ya=calibration.fine_hz(decoded,sr,times)
                ca=calibration.short.cross_correlation_pitch(decoded,sr,times)
                audit=calibration.short.audit_short_curve(bf,af,yb,ya,cb,ca,reliable,[support],curve)
                controls={name:calibration.short.audit_short_curve(bf,af if name=='opposite' else bf,
                    yb,ya if name=='opposite' else yb,cb,ca if name=='opposite' else cb,reliable,[support],-curve if name=='opposite' else curve)
                    for name in ('unchanged','opposite')}
                assert not any(control['passed'] for control in controls.values())
                audit_path=case/'audit.json'; write(audit_path,dict(audit=audit,controls=controls))
                measured=case/'measurements.npz'; np.savez_compressed(measured,**af,yin=ya,cc=ca,requested=curve)
                paths += [c_path,audit_path,measured]
                row.update(path_C=str(c_path),audit_path=str(audit_path),audit=audit)
                if not audit['passed']:
                    source_rows.append(dict(**row,excluded='THREE_TRACKER_PHYSICAL_AUDIT_FAILED')); continue
                c=calibration.recovery.measured(decoded)
                common=reliable&c['reliable']&bf['voiced_mask']&af['voiced_mask']&np.isfinite(yb)&np.isfinite(ya)&(yb>0)&(ya>0)
                plateau=(times>=support[0]+.09)&(times<=support[1]-.09)&common&np.isfinite(cb)&np.isfinite(ca)&(cb>0)&(ca>0)
                zero=((times<support[0]-.07)|(times>support[1]+.07))&common
                left=(times>=region[0]*.02)&(times<support[0]-.07)&c['reliable']
                right=(times>support[1]+.07)&(times<region[1]*.02)&c['reliable']
                if plateau.sum()<3 or min(left.sum(),right.sum())<3:
                    source_rows.append(dict(**row,excluded='INSUFFICIENT_RETAINED_PLATEAU_OR_CONTEXT')); continue
                context=case/'input_context.npz'; np.savez_compressed(context,raw_views=np.stack(c['views']),reliable=c['reliable'],fine=c['fine'],times_seconds=times)
                target=case/'targets_NOT_MODEL_INPUT.npz'
                np.savez_compressed(target,target=np.where(plateau,-cents,0.).astype(np.float32),mask=plateau|zero,plateau=plateau,known_zero=zero,times=times)
                paths += [context,target]
                source_rows.append(dict(**row,analysis_C=str(context),target_path=str(target),positive_frames=int(plateau.sum()),zero_frames=int(zero.sum())))
        saved=dict(declaration_sha256=sha(ROOT/'declaration.json'),pairs=source_rows,output_hashes={str(p):sha(p) for p in paths})
        write(saved_path,saved); rows.extend(source_rows); hashes.update(saved['output_hashes'])
        print('FULL_WINDOW_PAIRS_SOURCE',sid,'valid',sum('excluded' not in r for r in rows),'/',len(rows),flush=True)
    assert len(rows)==1152
    write(ROOT/'train_pairs.json',rows); write(ROOT/'train_originals.json',originals); write(ROOT/'input_hashes.json',hashes)
    valid=[r for r in rows if 'excluded' not in r]
    summary=dict(attempted=1152,valid=len(valid),original_attempts=72,original_source_recordings=33,valid_originals=sum('excluded' not in s for s in originals),
        cases_reused=sum(r['reused_historical_case'] for r in rows),valid_pairs_reused=sum(r['reused_historical_case'] for r in valid),
        valid_pairs_new=sum(not r['reused_historical_case'] for r in valid),valid_by_group=dict(Counter(r['group'] for r in valid)),
        exclusions=dict(Counter(r['excluded'] for r in rows if 'excluded' in r)),new_model_fits=0,
        nontraining_audio_accessed=False,research_training_eligible=False,production_eligible=False)
    write(ROOT/'summary.json',summary)
    print('FULL_WINDOW_SHORT_PAIRS_READY',summary,flush=True)


if __name__=='__main__':
    main()
