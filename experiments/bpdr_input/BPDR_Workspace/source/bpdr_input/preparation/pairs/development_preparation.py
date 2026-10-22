"""Declared development-only controlled examples; no fitting or shared pipeline calls."""
from bpdr_input import COMPONENT_ROOT, module_source
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import soundfile as sf

from bpdr_input.audio.analysis import extract
from bpdr_input.experiments.audio.audit_diagnostics import failure_reasons
from bpdr_input.preparation.sources.dataset import DataPolicy, reference_mask
from bpdr_input.repair.interventions import AuditPolicy, audit_shift, corrupt_boundaries, independent_pitch, nominal_boundaries, render_local, select_support
from bpdr_input.preparation.pairs.laboratory import validate_pair_record
from bpdr_input.audio.pipeline import process
from bpdr_input.preparation.sources.shared_exports import digest, load_export
from bpdr_input.preparation.pairs.shared_pair_audit import game_boundary_track, write_views
from bpdr_input.preparation.diagnostics.shared_screen_diagnostics import decompose


def development_selection(vocadito, intake, training):
    """Metadata-only partition guard before any acoustic access."""
    rows = []
    for p in vocadito['phrases']:
        if p['split'] == 'development':
            rows.append(dict(p, dataset='vocadito', group_id='vocadito:'+p['singer_id'],
                group_kind='annotated_singer_id', reference_type='human_annotation'))
    tracks = {r['mtg_track_id']:r for r in intake['tracks']}
    for p in intake['records']:
        if p['split'] == 'development' and p['integrity'] == 'verified_export' and p['screen_status'] == 'review_candidate':
            if not tracks[p['mtg_track_id']]['policy_export_eligible']:
                raise ValueError('Ineligible development export license/policy')
            rows.append(dict(p, dataset='jamendo', phrase_id=p['mtg_track_id'], source_id=p['mtg_track_id'],
                singer_id=p['artist_id'], group_id='jamendo:'+str(p['artist_id']),
                group_kind='artist_proxy_singer_unknown', reference_type='model_estimate'))
    train_groups={r['group_id'] for r in training}
    train_sources={(r['dataset'],r['source_id']) for r in training}
    for p in rows:
        if p['split'] != 'development' or p['group_id'] in train_groups or (p['dataset'],p['source_id']) in train_sources:
            raise ValueError('Development overlaps frozen training identifiers')
    if len({(r['dataset'],r['phrase_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate development source')
    return rows


def validate_development_record(r):
    if (r['split'] != 'development' or not r['audit_passed'] or not r['exploratory']
            or r['training_eligible'] is not False or r['data_status'] != 'exploratory_not_research'
            or r['review_status'] != 'pending'):
        raise ValueError('Expected development-only exploratory pair')
    for path,key in [('manifest_path','manifest_sha256'),('target_path','targets_sha256'),
                     ('audit_path','audit_sha256'),('audit_policy_path','audit_policy_sha256')]:
        if digest(r[path]) != r[key]:raise ValueError('Changed development provenance: '+path)
    audit=json.loads(Path(r['audit_path']).read_text())
    if not audit['passed'] or audit['requested_cents'] != r['cents']:
        raise ValueError('Development audit did not pass this requested shift')
    if audit['policy'] != json.loads(Path(r['audit_policy_path']).read_text()):
        raise ValueError('Development policy mismatch')
    for view in 'ABCD':
        if digest(r['waveform_paths'][view]) != r['waveform_sha256'][view]:raise ValueError('Changed development waveform')
        if digest(Path(r['input_directory'])/(view+'_input.npz')) != r['input_sha256'][view]:raise ValueError('Changed development inputs')
    validate_pair_record(r)
    with np.load(r['target_path']) as z:
        common,plateau,zero=(z[k] for k in ('common_reliable_mask','detuned_plateau_mask','zero_anchor_mask'))
        if not plateau.any() or not zero.any() or np.any(plateau & zero) or not np.array_equal(common,plateau|zero):
            raise ValueError('Invalid development target partition')
        if np.any(z['correction_target_cents'][~common] != 0):raise ValueError('Unmasked private development targets')
    return True


def run():
    root=COMPONENT_ROOT
    index=json.loads((root/'notebooks/evidence_index.json').read_text())['evidence']
    prep=Path(json.loads((root/'runs/comparison_preparation_current.json').read_text())['path'])
    train=json.loads((prep/'summary.json').read_text())['examples']
    protocol=json.loads((prep/'locked_protocol.json').read_text())
    if protocol['development_processing_executed'] or any(r['split']!='train' for r in train):
        raise ValueError('Unexpected preparation state')
    phase=root/index['phase2'];vpath=phase/'manifest.json'
    ipath=root/index['shared_batch_intake']
    vm=json.loads(vpath.read_text());intake=json.loads(ipath.read_text())
    selected=development_selection(vm,intake,train)
    policy=AuditPolicy();data_policy=DataPolicy(**vm['policy'])
    previous=json.loads((prep/'declaration.json').read_text())
    frozen=dict(previous['frozen_inputs'])
    for p in [vpath,ipath,phase/'reviews.csv',root/'runs/preparation/d274ed728d3d46f3b2661ba5291e61da/reviews.csv',
              prep/'summary.json',prep/'locked_protocol.json',prep/'normalizer.npz',prep/'sampling_schedule.npz']:
        frozen[str(p)]=digest(p)
    for r in selected:
        if r['dataset']=='vocadito':
            frozen[r['audio_path']]=r['sha256']
            frozen[r['f0_path']]=r['reference_sha256']['f0'];frozen[r['notes_path']]=r['reference_sha256']['notes']
        else:
            frozen[r['record_path']]=r['record_sha256']
            for name,h in r['artifact_hashes'].items():frozen[str(Path(r['record_path']).parent/name)]=h
            if 'reviews' in r:frozen[r['reviews']['path']]=r['reviews']['sha256']
    destination=root/'runs/development_preparation'/uuid.uuid4().hex;destination.mkdir(parents=True)
    copied=destination/'source_code_at_run';copied.mkdir()
    for name in ['development_preparation','analysis','dataset','pipeline','interventions','laboratory','shared_exports',
                 'shared_pair_audit','shared_screen_diagnostics','audit_diagnostics']:
        p=module_source(name+'.py');frozen[str(p)]=digest(p);shutil.copy2(p,copied/p.name)
    for p,h in {**frozen,**previous['pipeline_files']}.items():
        if digest(p)!=h:raise ValueError('Frozen input changed before development: '+p)
    def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write(destination/'locked_audit_policy.json',asdict(policy))
    write(destination/'declaration.json',dict(protocol_path=str(prep/'locked_protocol.json'),protocol_sha256=digest(prep/'locked_protocol.json'),
        frozen_inputs=frozen,pipeline_files=previous['pipeline_files'],selected_sources=selected,
        shifts_cents=policy.shifts_cents,corruptions=['split','merge','jitter'],seed=data_policy.split_seed,
        prospective_source_shift_keys=len(selected)*6,reference_roles='Primary human-reference Vocadito; supplementary predicted Jamendo',
        source_refusal='Record all six as not_attempted when source screen/support/technical check fails',
        no_training=True,no_train_normalizer_change=True,no_review_changes=True,no_threshold_changes=True,
        no_pp1_or_final_holdout_acoustic_access=True,no_shared_pipeline_execution=True))
    e=dict(status='running',sources=[],attempts=[],artifacts={},research_ready_examples=0,training_updates=0,
           data_status='exploratory_not_research',declaration_sha256=digest(destination/'declaration.json'))
    pairs=[]
    def save():
        write(destination/'evidence.json',e)
        write(destination/'summary.json',dict(mode='exploratory_not_research',split='development',examples=pairs,
            training_eligible_pairs=0,research_ready_examples=0,run_status=e['status']))
    save();print('DEVELOPMENT_RUN:',destination,flush=True)
    for index_number,r in enumerate(selected,1):
        folder=destination/r['dataset']/r['phrase_id'];folder.mkdir(parents=True)
        source=dict(dataset=r['dataset'],phrase_id=r['phrase_id'],source_id=r['source_id'],group_id=r['group_id'],
                    group_kind=r['group_kind'],reference_type=r['reference_type'],split='development',status='started')
        e['sources'].append(source)
        try:
            if r['dataset']=='jamendo':
                loaded=load_export(Path(r['record_path']).parent)
                audio=loaded['audio'];sr=loaded['sample_rate_hz'];times=loaded['model_inputs']['times_seconds']
                yin=independent_pitch(audio,sr)
                screen,arrays=decompose(loaded,yin)
                np.savez_compressed(folder/'screen_replay_NOT_MODEL_INPUT.npz',**arrays)
                reliable=loaded['masks']['reliable_pitch_screen']
                endpoints=np.array(sorted({float(n[k]) for n in loaded['boundaries']['notes'] for k in ('onset_sec','offset_sec')}))
                original=Path(r['record_path']).parent/'vocal.wav'
                source.update(screen_replay=screen,original_audio_path=str(original))
            else:
                result,phase1=process(r['audio_path'],folder/'phase1',root/'checkpoints')
                source['phase1_path']=str(phase1)
                if result.accepted_audio is None:raise ValueError('Phase 1 source refused: '+str(result.reason_codes))
                audio,sr=sf.read(result.accepted_audio.path,dtype='float32')
                times=np.arange(len(audio)//441+1)*441/sr
                f0=np.loadtxt(r['f0_path'],delimiter=',',ndmin=2)
                notes=json.loads(Path(r['notes_path']).read_text())
                reliable,reference=reference_mask(times,f0,notes,data_policy)
                endpoints=nominal_boundaries(times,notes)
                yin=independent_pitch(audio,sr)
                original=folder/'original.wav';sf.write(original,audio,sr,subtype='FLOAT')
                source.update(original_audio_path=str(original),source_crop_sha256=r['sha256'])
            before,analyzer=extract(audio,sr,root/'checkpoints')
            if not np.array_equal(times,before['times_seconds']):raise ValueError('Development feature grid differs')
            if r['dataset']=='vocadito':
                reliable &= before['voiced_mask'] & (reference>0) & (yin>0)
                mask=reliable.copy()
                reliable[mask] &= abs(1200*np.log2(yin[mask]/before['f0_hz'][mask]))<=policy.tracker_agreement_cents
                np.savez_compressed(folder/'reference_NOT_MODEL_INPUT.npz',times_seconds=times,reliable_mask=reliable,
                                    source_yin_hz=yin,reference_f0_hz=reference)
            else:
                np.savez_compressed(folder/'reference_NOT_MODEL_INPUT.npz',times_seconds=times,reliable_mask=reliable,source_yin_hz=yin)
            np.savez_compressed(folder/'baseline_acoustics.npz',**before);write(folder/'baseline_analyzer.json',analyzer)
            support=select_support(times,reliable,policy)
            nominal=game_boundary_track(times,endpoints)
            if r['dataset']=='jamendo' and not np.allclose(nominal,loaded['model_inputs']['game_boundary_score'],rtol=0,atol=1e-6):
                raise ValueError('Imported Gaussian boundary convention differs')
            identity,_=render_local(audio,sr,0.,support,policy)
            sf.write(folder/'identity.wav',identity,sr,subtype='FLOAT')
            decoded,rate=sf.read(folder/'identity.wav',dtype='float32')
            repeat,_=extract(decoded,rate,root/'checkpoints')
            identity_ok=bool(rate==sr and np.array_equal(decoded,audio) and np.array_equal(before['frame_features_70'],repeat['frame_features_70']))
            if not identity_ok:raise ValueError('Development zero-shift identity failed')
            source.update(status='eligible_for_audit',support_seconds=list(support),identity_pass=identity_ok,
                          original_audio_sha256=digest(original),nominal_boundary_seconds=endpoints.tolist())
        except (ValueError,KeyError,OSError,RuntimeError) as exc:
            source.update(status='withheld',reason=str(exc))
            for cents in policy.shifts_cents:
                e['attempts'].append(dict(dataset=r['dataset'],phrase_id=r['phrase_id'],cents=cents,
                    status='not_attempted_source_withheld',passed=False,reason=str(exc)))
            write(folder/'source_refusal.json',source);save()
            print('DEV_SOURCE_WITHHELD:',index_number,'/',len(selected),r['phrase_id'],str(exc),flush=True)
            continue
        for cents in policy.shifts_cents:
            shifted=folder/f'shift_{cents:+g}';shifted.mkdir()
            attempt=dict(dataset=r['dataset'],phrase_id=r['phrase_id'],cents=cents,status='started',passed=False)
            e['attempts'].append(attempt)
            try:
                candidate,blend=render_local(audio,sr,cents,support,policy)
                wav=shifted/'detuned.wav';sf.write(wav,candidate,sr,subtype='FLOAT')
                after_audio,after_rate=sf.read(wav,dtype='float32')
                after,metadata=extract(after_audio,after_rate,root/'checkpoints');yin_after=independent_pitch(after_audio,after_rate)
                audit,anchors,target=audit_shift(before,after,yin,yin_after,reliable,support,cents,policy)
                audit.update(requested_cents=cents,identity_pass=identity_ok,same_sample_count=len(after_audio)==len(audio),
                    outside_support_exact=bool(np.array_equal(after_audio[blend==0],audio[blend==0])),
                    float_wav_roundtrip_exact=bool(after_rate==sr and np.array_equal(after_audio,candidate)))
                audit['passed']=bool(audit['passed'] and all(audit[k] for k in ['identity_pass','same_sample_count','outside_support_exact','float_wav_roundtrip_exact']))
                audit['failure_reasons']=failure_reasons(audit,policy)
                write(shifted/'audit.json',audit);write(shifted/'analyzer.json',metadata)
                np.savez_compressed(shifted/'detuned_acoustics.npz',**after)
                np.savez_compressed(shifted/'audit_arrays_NOT_MODEL_INPUT.npz',yin_after=yin_after,blend=blend)
                targets=shifted/'targets_NOT_MODEL_INPUT.npz'
                np.savez_compressed(targets,times_seconds=times,anchor_mask=anchors,common_reliable_mask=anchors,
                    detuned_plateau_mask=anchors&(target!=0),zero_anchor_mask=anchors&(target==0),correction_target_cents=target)
                attempt.update(status='measured',passed=audit['passed'],audit_path=str(shifted/'audit.json'),
                    audit_sha256=digest(shifted/'audit.json'),failure_reasons=audit['failure_reasons'])
                if audit['passed']:
                    packed=[]
                    for kind in ['split','merge','jitter']:
                        altered=corrupt_boundaries(endpoints,times,kind,data_policy.split_seed,split_time=sum(support)/2)
                        views=shifted/kind;write_views(views,times,before['frame_features_70'],after['frame_features_70'],
                            nominal,game_boundary_track(times,altered))
                        pair=dict(dataset=r['dataset'],phrase_id=r['phrase_id'],source_id=r['source_id'],singer_id=r['singer_id'],
                            group_id=r['group_id'],group_kind=r['group_kind'],split='development',cents=cents,corruption=kind,
                            reference_type=r['reference_type'],data_status='exploratory_not_research',review_status='pending',
                            exploratory=True,training_eligible=False,audit_passed=True,support_seconds=list(support),
                            waveform_paths={v:str(original if v in 'AB' else wav) for v in 'ABCD'},
                            waveform_sha256={v:digest(original if v in 'AB' else wav) for v in 'ABCD'},
                            input_directory=str(views),input_sha256={v:digest(views/(v+'_input.npz')) for v in 'ABCD'},
                            target_path=str(targets),targets_sha256=digest(targets),common_mask_path=str(targets),
                            manifest_path=str(vpath if r['dataset']=='vocadito' else ipath),
                            manifest_sha256=digest(vpath if r['dataset']=='vocadito' else ipath),
                            crop_sha256=r['sha256'] if r['dataset']=='vocadito' else r['audio_sha256'],
                            audit_policy_path=str(destination/'locked_audit_policy.json'),audit_policy_sha256=digest(destination/'locked_audit_policy.json'),
                            audit_path=str(shifted/'audit.json'),audit_sha256=digest(shifted/'audit.json'),
                            reference_mask_path=str(folder/'reference_NOT_MODEL_INPUT.npz'),
                            nominal_boundary_seconds=endpoints.tolist(),corrupted_boundary_seconds=altered.tolist(),
                            boundary_encoding='gaussian_max_sigma_0.04_seconds',boundary_provenance='Saved timing-only edges; note pitches excluded')
                        validate_development_record(pair);write(views/'pair_manifest.json',pair);packed.append(pair)
                    pairs.extend(packed)
            except (ValueError,RuntimeError,OSError) as exc:
                attempt.update(status='failed',passed=False,reason=str(exc));write(shifted/'failure.json',attempt)
            save()
        source['passing_shifts']=sum(a['passed'] for a in e['attempts'] if a['dataset']==r['dataset'] and a['phrase_id']==r['phrase_id'])
        source['status']='audited';save()
        print('DEV_SOURCE_COMPLETE:',index_number,'/',len(selected),r['dataset'],r['phrase_id'],source['passing_shifts'],'of 6',flush=True)
    for p,h in {**frozen,**previous['pipeline_files']}.items():
        if digest(p)!=h:raise ValueError('Frozen input changed during development: '+p)
    e['artifacts']={str(p.relative_to(destination)):digest(p) for p in destination.rglob('*')
        if p.is_file() and p.name not in ['evidence.json','summary.json']}
    e.update(status='complete',summary=dict(selected_sources=len(selected),prospective_shift_keys=len(selected)*6,
        recorded_shift_keys=len(e['attempts']),attempt_status=dict(Counter(a['status'] for a in e['attempts'])),
        passing_shifts=sum(a['passed'] for a in e['attempts']),engineering_development_examples=len(pairs),
        by_dataset={ds:dict(selected_sources=sum(r['dataset']==ds for r in selected),
            sources_with_pairs=len({r['phrase_id'] for r in pairs if r['dataset']==ds}),
            group_count=len({r['group_id'] for r in pairs if r['dataset']==ds}),pairs=sum(r['dataset']==ds for r in pairs))
            for ds in ['vocadito','jamendo']},research_ready_examples=0,training_updates=0))
    save()
    (root/'runs/development_preparation_current.json').write_text(json.dumps(dict(path=str(destination)),indent=2)+'\n',encoding='utf-8')
    print('DEVELOPMENT_COMPLETE:',json.dumps(e['summary']),flush=True)
    return destination


if __name__=='__main__':run()
