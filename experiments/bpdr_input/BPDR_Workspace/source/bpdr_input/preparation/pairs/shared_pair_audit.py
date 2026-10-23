"""Declared Gate C: frozen training candidates -> audited exploratory four-view pairs."""
from bpdr_input import COMPONENT_ROOT, module_source
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

import numpy as np
import soundfile as sf

from bpdr_input.audio.analysis import extract
from bpdr_input.experiments.audio.audit_diagnostics import failure_reasons
from bpdr_input.preparation.sources.dataset import DataPolicy
from bpdr_input.repair.interventions import AuditPolicy, audit_shift, corrupt_boundaries, independent_pitch, render_local, select_support
from bpdr_input.preparation.pairs.laboratory import validate_pair_record
from bpdr_input.preparation.sources.shared_exports import digest, load_export


CORRUPTIONS = ('split','merge','jitter')


def training_candidates(payload):
    """Reject an unsafe selection before opening source WAVs or estimating pitch."""
    rows=payload['candidates']
    if not rows or any(r['split']!='train' or r['integrity']!='verified_export'
                       or not r['engineering_input_candidate'] or not r['candidate_for_pair_audit'] for r in rows):
        raise ValueError('Only declared, screened frozen-training candidates may be audited')
    if len({r['mtg_track_id'] for r in rows})!=len(rows) or len({r['audio_sha256'] for r in rows})!=len(rows):
        raise ValueError('Duplicate source crop in pair-audit selection')
    return rows


def game_boundary_track(times,endpoints):
    """Preserve the shared export's timing-only Gaussian boundary convention."""
    result=np.zeros(len(times),dtype=np.float32)
    for edge in endpoints:
        result=np.maximum(result,np.exp(-.5*((times-float(edge))/.04)**2).astype(np.float32))
    return result


def write_views(folder,times,before,after,nominal,corrupted):
    folder=Path(folder);folder.mkdir(parents=True)
    for view,frame,boundary in [('A',before,nominal),('B',before,corrupted),('C',after,nominal),('D',after,corrupted)]:
        np.savez_compressed(folder/(view+'_input.npz'),times_seconds=times,
            frame_features_70=frame,boundary_features_71=np.column_stack([frame,boundary]).astype(np.float32))


def validate_engineering_pair(record):
    """Stronger component adapter checks before the existing engineering loader."""
    if (record.get('split')!='train' or record.get('training_eligible') is not False
        or not record.get('exploratory') or record.get('review_status')!='pending'
        or record.get('reference_type')!='model_estimate' or record.get('data_status')!='exploratory_not_research'
        or not record.get('audit_passed')):
        raise ValueError('Expected audited training-only exploratory model-estimate pair')
    for path,key in [('manifest_path','manifest_sha256'),('target_path','targets_sha256'),
                     ('audit_path','audit_sha256'),('audit_policy_path','audit_policy_sha256')]:
        if digest(record[path])!=record[key]:raise ValueError('Pair provenance changed: '+path)
    audit=json.loads(Path(record['audit_path']).read_text())
    if not audit['passed']:raise ValueError('Saved audit did not pass')
    if audit['policy']!=json.loads(Path(record['audit_policy_path']).read_text()) or audit['requested_cents']!=record['cents']:
        raise ValueError('Audit policy or requested shift differs from pair')
    for view in 'ABCD':
        if digest(record['waveform_paths'][view])!=record['waveform_sha256'][view]:raise ValueError('Waveform changed')
        if digest(Path(record['input_directory'])/(view+'_input.npz'))!=record['input_sha256'][view]:raise ValueError('Input changed')
    validate_pair_record(record)
    with np.load(record['target_path'],allow_pickle=False) as z:
        if not z['detuned_plateau_mask'].any() or not z['zero_anchor_mask'].any():raise ValueError('Missing shift or preservation anchors')
        if np.any(z['detuned_plateau_mask']&z['zero_anchor_mask']):raise ValueError('Conflicting targets')
        if not np.array_equal(z['common_reliable_mask'],z['detuned_plateau_mask']|z['zero_anchor_mask']):
            raise ValueError('Target mask includes unaudited frames')
        if np.any(z['correction_target_cents'][~z['common_reliable_mask']]!=0):raise ValueError('Targets outside audit mask')
    return True


def run():
    root=COMPONENT_ROOT
    batch=Path(json.loads((root/'runs/shared_batch_intake_current.json').read_text())['path'])
    source=batch/'pair_audit_candidates.json'
    payload=json.loads(source.read_text());selected=training_candidates(payload)
    intake=json.loads((batch/'evidence.json').read_text())
    if intake['status']!='complete' or len(selected)!=intake['summary']['training_candidates_with_support']:
        raise ValueError('Incomplete or changed source intake')
    by_id={r['mtg_track_id']:r for r in intake['records']}
    if any(r!=by_id[r['mtg_track_id']] for r in selected):raise ValueError('Selection no longer matches intake')
    policy=AuditPolicy();seed=DataPolicy().split_seed
    destination=root/'runs/shared_pair_audit'/uuid.uuid4().hex;destination.mkdir(parents=True)
    def write(path,value):
        Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    code=['shared_pair_audit','analysis','interventions','laboratory','examples','shared_exports','audit_diagnostics']
    frozen={str(source):digest(source),str(batch/'evidence.json'):digest(batch/'evidence.json')}
    for name in code:
        p=module_source(name+'.py');frozen[str(p)]=digest(p)
        copied=destination/'source_code_at_run';copied.mkdir(exist_ok=True);(copied/p.name).write_bytes(p.read_bytes())
    pipeline_files=json.loads((batch/'declaration.json').read_text())['pipeline_files']
    for p,h in pipeline_files.items():
        if digest(p)!=h:raise ValueError('Shared pipeline changed since intake')
    for row in selected:
        frozen[row['record_path']]=row['record_sha256']
        folder=Path(row['record_path']).parent
        for name,sha in row['artifact_hashes'].items():frozen[str(folder/name)]=sha
        frozen[str(batch/row['screen_arrays'])]=intake['artifacts'][row['screen_arrays']]
        if 'reviews' in row:frozen[row['reviews']['path']]=row['reviews']['sha256']
    for p,h in frozen.items():
        if digest(p)!=h:raise ValueError('Source changed before pair audit: '+p)
    write(destination/'locked_audit_policy.json',asdict(policy))
    declaration=dict(source_selection=str(source),source_selection_sha256=digest(source),
        source_batch=str(batch),source_count=len(selected),artist_proxy_count=len({r['artist_id'] for r in selected}),
        ordered_tracks=[r['mtg_track_id'] for r in selected],shifts_cents=policy.shifts_cents,corruptions=CORRUPTIONS,
        seed=seed,policy=asdict(policy),frozen_inputs=frozen,pipeline_files=pipeline_files,
        feature_backend='Default frozen local PESTO mir-1k_g7 on CPU; no provider substitution or training',
        source_masks='Keep imported reliable mask and declared support; reuse saved source YIN after hash check',
        boundary_input='Crop GAME timing-only Gaussian sigma .04; pitch fields never become intended-note targets',
        targets='Negative injected cents on independently audited plateau; zero on audited outside; ramps excluded',
        pairs='All six shifts per source, all three corruptions for audit-passing shifts; failures retained, no outcome stopping',
        data_status='exploratory_not_research',no_heldout_compute=True,no_pipeline_execution=True,
        no_learner_training=True,no_review_approval=True,no_renderer_adoption=True)
    write(destination/'declaration.json',declaration)
    evidence=dict(status='running',declaration_sha256=digest(destination/'declaration.json'),sources=[],attempts=[],
        artifacts={},data_status='exploratory_not_research',research_ready_examples=0,training_updates=0,adopted=False)
    examples=[]
    def artifact(p):evidence['artifacts'][str(Path(p).relative_to(destination))]=digest(p)
    def save():
        write(destination/'evidence.json',evidence)
        write(destination/'summary.json',dict(mode='exploratory_not_research',examples=examples,
            training_eligible_pairs=0,research_ready_examples=0,run_status=evidence['status']))
    save();print('PAIR_AUDIT_RUN:',destination,flush=True)
    try:
        for index,row in enumerate(selected,1):
            started=time.monotonic();loaded=load_export(Path(row['record_path']).parent)
            audio=loaded['audio'];sr=loaded['sample_rate_hz'];times=loaded['features']['times_seconds']
            reliable=loaded['masks']['reliable_pitch_screen'];support=select_support(times,reliable,policy)
            if list(support)!=row['support_seconds']:raise ValueError('Declared support changed')
            with np.load(batch/row['screen_arrays'],allow_pickle=False) as z:yin_before=z['yin_hz']
            endpoints=np.asarray(sorted({float(n[k]) for n in loaded['boundaries']['notes'] for k in ('onset_sec','offset_sec')}))
            nominal=game_boundary_track(times,endpoints)
            if not np.allclose(nominal,loaded['model_inputs']['game_boundary_score'],rtol=0,atol=1e-6):
                raise ValueError('Original GAME boundary representation changed')
            folder=destination/row['mtg_track_id'];folder.mkdir()
            before,analyzer=extract(audio,sr,root/'checkpoints')
            if not np.array_equal(before['times_seconds'],times):raise ValueError('Local feature grid changed')
            original=Path(row['record_path']).parent/'vocal.wav'
            identity,_=render_local(audio,sr,0.,support,policy)
            sf.write(folder/'identity.wav',identity,sr,subtype='FLOAT')
            decoded_identity,identity_sr=sf.read(folder/'identity.wav',dtype='float32')
            repeated,_=extract(decoded_identity,identity_sr,root/'checkpoints')
            identity_ok=bool(np.array_equal(decoded_identity,audio) and np.array_equal(before['frame_features_70'],repeated['frame_features_70']))
            np.savez_compressed(folder/'baseline_acoustics.npz',**before)
            np.savez_compressed(folder/'reference_NOT_MODEL_INPUT.npz',times_seconds=times,reliable_mask=reliable,
                                source_yin_hz=yin_before,imported_f0_hz=loaded['features']['f0_hz'],nominal_boundary=nominal)
            write(folder/'baseline_analyzer.json',analyzer)
            source_row=dict(mtg_track_id=row['mtg_track_id'],artist_id=row['artist_id'],split='train',
                group_kind='artist_proxy_singer_unknown',reference_type='model_estimate',review_status='pending',
                original_audio_path=str(original),original_audio_sha256=row['audio_sha256'],record_path=row['record_path'],
                record_sha256=row['record_sha256'],support_seconds=list(support),identity_pass=identity_ok,
                nominal_boundary_seconds=endpoints.tolist(),nominal_boundary_kind='GAME Gaussian sigma .04',
                imported_local_voicing_agreement=float(np.mean(before['voiced_mask']==loaded['model_inputs']['voiced_mask'])),
                analyzer=analyzer,passed_shifts=0)
            evidence['sources'].append(source_row)
            for p in folder.iterdir():
                if p.is_file():artifact(p)
            for cents in policy.shifts_cents:
                print('AUDIT:',index,'/',len(selected),row['mtg_track_id'],cents,flush=True)
                attempt=dict(mtg_track_id=row['mtg_track_id'],artist_id=row['artist_id'],split='train',cents=cents,status='running',passed=False)
                evidence['attempts'].append(attempt)
                shifted_folder=folder/f'shift_{cents:+g}';shifted_folder.mkdir()
                try:
                    if not identity_ok:raise ValueError('Zero-shift identity failed')
                    shifted,blend=render_local(audio,sr,cents,support,policy)
                    wav=shifted_folder/'detuned.wav';sf.write(wav,shifted,sr,subtype='FLOAT')
                    decoded,rate=sf.read(wav,dtype='float32')
                    roundtrip=bool(rate==sr and np.array_equal(decoded,shifted))
                    after,after_analyzer=extract(decoded,rate,root/'checkpoints')
                    yin_after=independent_pitch(decoded,rate)
                    audit,anchors,target=audit_shift(before,after,yin_before,yin_after,reliable,support,cents,policy)
                    audit.update(requested_cents=cents,identity_pass=identity_ok,same_sample_count=len(decoded)==len(audio),
                        outside_support_exact=bool(np.array_equal(decoded[blend==0],audio[blend==0])),float_wav_roundtrip_exact=roundtrip)
                    audit['passed']=bool(audit['passed'] and identity_ok and audit['same_sample_count'] and audit['outside_support_exact'] and roundtrip)
                    audit['failure_reasons']=failure_reasons(audit,policy)
                    if not roundtrip:audit['failure_reasons'].append('FLOAT_WAV_ROUNDTRIP')
                    write(shifted_folder/'audit.json',audit);write(shifted_folder/'analyzer.json',after_analyzer)
                    np.savez_compressed(shifted_folder/'detuned_acoustics.npz',**after)
                    np.savez_compressed(shifted_folder/'audit_arrays_NOT_MODEL_INPUT.npz',yin_after=yin_after,blend=blend)
                    targets=shifted_folder/'targets_NOT_MODEL_INPUT.npz'
                    np.savez_compressed(targets,times_seconds=times,anchor_mask=anchors,common_reliable_mask=anchors,
                        detuned_plateau_mask=anchors&(target!=0),zero_anchor_mask=anchors&(target==0),correction_target_cents=target)
                    attempt.update(status='measured',passed=audit['passed'],audit_path=str(shifted_folder/'audit.json'),
                                   audit_sha256=digest(shifted_folder/'audit.json'),failure_reasons=audit['failure_reasons'])
                    if audit['passed']:
                        packed=[]
                        for kind in CORRUPTIONS:
                            altered=corrupt_boundaries(endpoints,times,kind,seed,split_time=sum(support)/2)
                            corrupted=game_boundary_track(times,altered)
                            views=shifted_folder/kind
                            write_views(views,times,before['frame_features_70'],after['frame_features_70'],nominal,corrupted)
                            pair=dict(phrase_id=row['mtg_track_id'],source_id=row['mtg_track_id'],singer_id=row['artist_id'],
                                artist_id=row['artist_id'],group_kind='artist_proxy_singer_unknown',split='train',cents=cents,corruption=kind,
                                reference_type='model_estimate',data_status='exploratory_not_research',review_status='pending',
                                exploratory=True,training_eligible=False,audit_passed=True,support_seconds=list(support),
                                waveform_paths={'A':str(original),'B':str(original),'C':str(wav),'D':str(wav)},
                                waveform_sha256={'A':digest(original),'B':digest(original),'C':digest(wav),'D':digest(wav)},
                                input_directory=str(views),input_sha256={v:digest(views/(v+'_input.npz')) for v in 'ABCD'},
                                target_path=str(targets),targets_sha256=digest(targets),common_mask_path=str(targets),
                                manifest_path=str(batch/'evidence.json'),manifest_sha256=digest(batch/'evidence.json'),crop_sha256=row['audio_sha256'],
                                audit_policy_path=str(destination/'locked_audit_policy.json'),audit_policy_sha256=digest(destination/'locked_audit_policy.json'),
                                audit_path=str(shifted_folder/'audit.json'),audit_sha256=digest(shifted_folder/'audit.json'),
                                reference_mask_path=str(folder/'reference_NOT_MODEL_INPUT.npz'),
                                nominal_boundary_seconds=endpoints.tolist(),corrupted_boundary_seconds=altered.tolist(),
                                boundary_provenance='Predicted crop GAME timing-only Gaussian sigma .04; no note pitches in boundary input')
                            validate_engineering_pair(pair);write(views/'pair_manifest.json',pair);packed.append(pair)
                        examples.extend(packed)
                    for p in shifted_folder.rglob('*'):
                        if p.is_file():artifact(p)
                except (ValueError,RuntimeError) as exc:
                    attempt.update(status='failed',passed=False,failure_reason=str(exc))
                    write(shifted_folder/'failure.json',attempt);artifact(shifted_folder/'failure.json')
                    # Preserve anything written before the failure, including measured audits.
                    for p in shifted_folder.rglob('*'):
                        if p.is_file():artifact(p)
                source_row['passed_shifts']=sum(a['passed'] for a in evidence['attempts'] if a['mtg_track_id']==row['mtg_track_id'])
                save()
            source_row['elapsed_seconds']=time.monotonic()-started;save()
            print('SOURCE_COMPLETE:',row['mtg_track_id'],'passed',source_row['passed_shifts'],'of 6',flush=True)
        for p,h in frozen.items():
            if digest(p)!=h:raise ValueError('Frozen source changed: '+p)
        for p,h in pipeline_files.items():
            if digest(p)!=h:raise ValueError('Common pipeline changed')
        passed=[a for a in evidence['attempts'] if a['passed']]
        # Pair count can be smaller only if an explicit corruption/packing failure was retained.
        evidence.update(status='complete',summary=dict(source_crops=len(selected),artist_proxies=len({r['artist_id'] for r in selected}),
            declared_shift_requests=len(selected)*len(policy.shifts_cents),recorded_shift_requests=len(evidence['attempts']),
            passed_shifts=len(passed),failed_or_withheld_shifts=len(evidence['attempts'])-len(passed),
            passed_crop_count=len({r['mtg_track_id'] for r in passed}),passed_artist_proxies=len({r['artist_id'] for r in passed}),
            all_six_pass_crop_count=sum(r['passed_shifts']==6 for r in evidence['sources']),
            engineering_pair_examples=len(examples),research_ready_examples=0,training_updates=0,
            failure_reasons=dict(Counter(reason for a in evidence['attempts'] for reason in a.get('failure_reasons',[])))))
        save()
        (root/'runs/shared_pair_audit_current.json').write_text(json.dumps(dict(path=str(destination)),indent=2),encoding='utf-8')
        print(json.dumps(evidence['summary'],indent=2),flush=True)
    except BaseException as exc:
        evidence.update(status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return destination


if __name__=='__main__':run()
