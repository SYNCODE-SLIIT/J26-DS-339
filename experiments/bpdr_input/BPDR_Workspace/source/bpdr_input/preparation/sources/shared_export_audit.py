"""Import/audit shared outputs locally; no shared pipeline execution or training."""
from bpdr_input import COMPONENT_ROOT, module_source
import csv
from collections import Counter
import json
from pathlib import Path
import uuid
import numpy as np
from bpdr_input.preparation.sources.shared_exports import VERSION, import_archive, load_export, digest
from bpdr_input.preparation.sources.data_sources import audio_licenses
from bpdr_input.preparation.sources.data_builder import write_json


def pipeline_hashes(root):
    return {str(p):digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file()
            and not any(x.startswith('.venv') or x in ('__pycache__','.git','outputs','work') for x in p.relative_to(root).parts)}


def run(archive):
    component=COMPONENT_ROOT;dataset=component.parents[1]/'datasets/mtg-jamendo'
    pipeline=dataset/'melody-chord-extraction-pipeline';pipeline_before=pipeline_hashes(pipeline)
    prep_path=component/'runs/preparation/d274ed728d3d46f3b2661ba5291e61da/manifest.json'
    prep=json.loads(prep_path.read_text());selection=prep['source_selection']['jamendo'];splits=selection['artist_splits']
    old_inventory_path=Path(selection['metadata_path'])
    if digest(old_inventory_path)!=selection['metadata_sha256']:raise ValueError('Original frozen inventory changed')
    with old_inventory_path.open(encoding='utf-8',newline='') as f:old_inventory={r['mtg_track_id']:r for r in csv.DictReader(f)}
    annotations_path=dataset/'annotations/autotagging.tsv';annotation={}
    with annotations_path.open(encoding='utf-8',newline='') as f:
        reader=csv.reader(f,delimiter='\t');next(reader)
        for row in reader:annotation[row[0]]=dict(artist=row[1],album=row[2],path=row[3],duration=float(row[4]),tags=row[5:])
    license_path=dataset/'audio_licenses.txt';licenses=audio_licenses(license_path)
    run_root=component/'runs/shared_export_intake'/uuid.uuid4().hex;run_root.mkdir(parents=True)
    tracked={str(p):digest(p) for p in (prep_path,old_inventory_path,annotations_path,license_path,Path(__file__),module_source('shared_exports.py'))}
    declaration=dict(version=VERSION,archive=str(Path(archive).resolve()),archive_sha256=digest(archive),
        pipeline_read_only=True,pipeline_sha256=pipeline_before,original_artist_split_source=str(prep_path),
        tasks=['Safe metadata-clean import; full-track inventory and typed 16-crop adapter; local integrity checks.',
            'Keep frozen artist partitions. No held-out inference, tuning, pairs or training; withheld crops remain withheld.',
            'No shared pipeline edits/execution or new separation/GAME/PESTO extraction. No automatic musical approval.',
            'Single training-partition input may receive a saved engineering-checkpoint forward for I/O only, not accuracy or permission.'])
    write_json(run_root/'declaration.json',declaration)
    evidence=dict(run_status='running',declaration=declaration,declaration_sha256=digest(run_root/'declaration.json'),
        tracked_inputs=tracked,shared_pipeline_files=pipeline_before,tracks=[],crops=[],artifacts=[],
        training_updates=0,research_ready_examples=0,paired_training_examples=0,new_extraction_runs=0,
        shared_pipeline_changed=False,accepted_prepared_packages=0,adopted=False)
    def save():write_json(run_root/'evidence.json',evidence)
    def artifact(path):evidence['artifacts'].append(dict(path=str(path),sha256=digest(path)))
    save();print('SHARED_EXPORT_INTAKE_RUN:',run_root,flush=True)
    try:
        imported,receipt=import_archive(archive,component);evidence['import_root']=str(imported)
        evidence['import_receipt']=str(imported/'import_receipt.json');evidence['import_receipt_sha256']=digest(imported/'import_receipt.json')
        evidence['archive_summary']={k:v for k,v in receipt.items() if k!='files'}
        outputs=imported/'outputs';folders=sorted(outputs.glob('track_*'));save()
        for folder in folders:
            track_id=folder.name;meta=annotation.get(track_id);old=old_inventory.get(track_id)
            roles=[p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()]
            raw=json.loads((folder/'melody/game_vocal.json').read_text())
            issues=[]
            if raw.get('mtg_track_id')!=track_id or str(int(raw['track_id']))!=str(int(track_id[6:])):issues.append('TRACK_ID_MISMATCH')
            game_notes=raw['notes'];duration=float(raw['duration_sec'])
            for n in game_notes:
                if (not np.isfinite([n['onset_sec'],n['offset_sec'],n['midi_pitch'],n['model_midi_pitch']]).all()
                        or not 0<=n['onset_sec']<n['offset_sec']<=duration+.05
                        or abs(n['midi_pitch']-n['model_midi_pitch']-12)>1e-7):
                    issues.append('GAME_PITCH_OR_TIME_CONVENTION_MISMATCH');break
            with (folder/'melody/game_vocal.csv').open(encoding='utf-8',newline='') as f:game_rows=sum(1 for _ in csv.DictReader(f))
            if game_rows!=len(game_notes):issues.append('GAME_JSON_CSV_COUNT_MISMATCH')
            artist=meta['artist'] if meta else None;license_info=licenses.get(str(int(raw['track_id'])))
            eligible=bool(meta and 'instrument---voice' in meta['tags'] and license_info and license_info['kind'] in {'by','by-sa','by-nc','by-nc-sa'})
            row=dict(mtg_track_id=track_id,track_id=raw['track_id'],artist_id=artist,album_id=meta['album'] if meta else None,
                split=splits.get(artist,'unassigned_new_artist'),group_kind='artist_proxy_singer_unknown',duration_seconds=duration,
                annotation_present=bool(meta),tags=meta['tags'] if meta else [],license=license_info,policy_export_eligible=eligible,
                official_splits={k:old.get(k) for k in ('split_0','split_1','split_2','split_3','split_4')} if old else {},
                roles=roles,game_note_count=len(game_notes),bpdr_record_present=(folder/'bpdr/record.json').is_file(),
                bpdr_audio_present=(folder/'bpdr/vocal.wav').is_file(),issues=issues,all_outputs_are_model_estimates=True)
            for name in ('sheetsage2_instrumental','combined'):
                p=folder/'melody'/(name+'.json');row[name+'_note_count']=len(json.loads(p.read_text())['notes']) if p.is_file() else None
            evidence['tracks'].append(row)
        print('CATALOG_TRACKS:',len(evidence['tracks']),flush=True);save()
        for folder in sorted(outputs.glob('*/bpdr')):
            if not (folder/'record.json').is_file():continue
            original=json.loads((folder/'record.json').read_text());track=next(t for t in evidence['tracks'] if t['mtg_track_id']==folder.parent.name)
            row=dict(track_id=original['track_id'],mtg_track_id=folder.parent.name,crop_id=original['crop_id'],
                artist_id=track['artist_id'],split=track['split'],group_kind='artist_proxy_singer_unknown',singer_identity='unknown',
                record_path=str(folder/'record.json'),record_sha256=digest(folder/'record.json'),audio_path=str(folder/'vocal.wav'),
                exported_screen=original['screen'],integrity='pending',issues=[],paired_training_ready=False,research_eligible=False,
                saved_export_status=original['status'],saved_training_eligible=original['training_eligible'])
            try:
                loaded=load_export(folder);r=loaded['record'];x=loaded['model_inputs'];mask=loaded['masks']['reliable_pitch_screen']
                if r['artist_id_metadata_only']!=track['artist_id']:raise ValueError('Artist metadata mismatch')
                if not track['license'] or (track['license']['kind'],track['license']['version'])!=(r['license']['kind'],r['license']['version']):
                    raise ValueError('License provenance mismatch')
                provenance=r['feature_provenance']
                if (provenance['hop_samples']!=441 or provenance['sample_rate_hz']!=22050 or provenance['window_samples']!=1024
                        or provenance['voiced_confidence_threshold']!=.8 or not provenance['frozen']):raise ValueError('Feature contract mismatch')
                notes=loaded['boundaries']['notes'];f0=loaded['features']['f0_hz'];times=x['times_seconds']
                row.update(integrity='verified',sample_count=len(loaded['audio']),sample_rate_hz=loaded['sample_rate_hz'],
                    duration_seconds=len(loaded['audio'])/22050,frames=len(times),feature_width=71,
                    audio_sha256=r['artifacts']['vocal.wav'],source_mp3_sha256=r['source_mp3_sha256'],
                    crop_offset_seconds=r['crop']['offset_seconds'],voiced_frames=int(x['voiced_mask'].sum()),reliable_frames=int(mask.sum()),
                    boundary_kind=loaded['boundary_kind'],feature_provenance=provenance,
                    archive_source_mp3_present=False,archive_full_stem_present=False,
                    upstream_source_hash_status='recorded hash; full MP3/stem not supplied in this archive',
                    heuristic_equals_game_column=bool(np.array_equal(x['game_boundary_score'],x['heuristic_boundary_score'])),
                    valid_nan_f0_on_unvoiced=int(np.isnan(f0).sum()))
                review=folder/'reviews.csv'
                if review.exists():
                    with review.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
                    row['exported_reviews']=dict(path=str(review),sha256=digest(review),rows=len(rows),decisions=[v['decision'] for v in rows])
                # Holdout integrity is checked, but no new pitch audit or inference is run there.
                if row['split']=='train':
                    from bpdr_input.repair.interventions import independent_pitch
                    yin=independent_pitch(loaded['audio'],22050)
                    if yin.shape!=times.shape:raise ValueError('Independent tracker grid mismatch')
                    reliable=loaded['inside_notes']&(loaded['overlap']==1)&x['voiced_mask']&np.isfinite(yin)&(yin>0)&np.isfinite(f0)&(f0>0)&(loaded['note_hz']>0)
                    ids=np.flatnonzero(reliable);reliable[ids]&=abs(1200*np.log2(yin[ids]/f0[ids]))<=50
                    ids=np.flatnonzero(reliable);reliable[ids]&=abs(1200*np.log2(f0[ids]/loaded['note_hz'][ids]))<=100
                    row['screen_replay_mask_exact']=bool(np.array_equal(reliable,mask))
                    row['screen_replay_reliable_frames']=int(reliable.sum())
                    inside=loaded['inside_notes']&x['voiced_mask']&(loaded['overlap']==1)&(loaded['note_hz']>0)
                    delta=1200*np.log2(f0[inside]/loaded['note_hz'][inside])
                    row['training_only_pitch_agreement_diagnostic']=dict(frames=int(inside.sum()),
                        median_signed_cents=float(np.median(delta)) if len(delta) else None,
                        fraction_within_100_cents=float(np.mean(abs(delta)<=100)) if len(delta) else None,
                        fraction_near_octave_disagreement=float(np.mean(abs(abs(delta)-1200)<=100)) if len(delta) else None,
                        meaning='Model estimates versus model notes; not human truth or automatic octave correction')
                else:row['screen_replay_mask_exact']=None
                row['engineering_training_input_candidate']=bool(row['split']=='train' and r['screen']['status']=='review_candidate' and track['policy_export_eligible'])
                row['model_input_loadable']=True
            except (ValueError,KeyError,OSError) as exc:
                row.update(integrity='quarantined',issues=[str(exc)],model_input_loadable=False,engineering_training_input_candidate=False)
            evidence['crops'].append(row);save();print('CROP',row['track_id'],row['split'],row['integrity'],flush=True)
        candidates=[r for r in evidence['tracks'] if r['split']=='train' and r['policy_export_eligible'] and not r['bpdr_record_present'] and not r['issues']]
        priority=[];seen=set()
        for r in sorted(candidates,key=lambda r:int(r['track_id'])):
            if r['artist_id'] in seen:continue
            seen.add(r['artist_id']);priority.append(dict(track_id=r['track_id'],mtg_track_id=r['mtg_track_id'],artist_id=r['artist_id'],
                split='train',license=r['license'],reason='Existing frozen training artist; GAME output present but BPDR crop/features absent'))
        priority_path=run_root/'bpdr_export_priority.json';write_json(priority_path,dict(selection='One lowest numeric track per existing training artist; no outcome tuning',
            shared_pipeline_changes_required=False,tracks=priority,all_missing_training_tracks=len(candidates)));artifact(priority_path)
        sources_path=run_root/'source_inputs.json';write_json(sources_path,dict(version=VERSION,record_kind='unpaired_shared_source_inputs',
            records=evidence['crops'],paired_training_examples=0,research_ready_examples=0,
            targets='Not supplied: raw 71-feature crops are not A/B/C/D intervention pairs with known correction targets.'));artifact(sources_path)
        evidence['summary']=dict(tracks=len(evidence['tracks']),bpdr_records=len(evidence['crops']),
            tracks_without_bpdr=sum(not r['bpdr_record_present'] for r in evidence['tracks']),
            artifacts_verified=sum(r['integrity']=='verified' for r in evidence['crops']),
            quarantined=sum(r['integrity']!='verified' for r in evidence['crops']),
            bpdr_artist_groups=len({r['artist_id'] for r in evidence['crops']}),
            archive_artist_groups=len({r['artist_id'] for r in evidence['tracks'] if r['artist_id']}),
            screen_counts=dict(Counter(r['exported_screen']['status'] for r in evidence['crops'])),
            bpdr_split_counts=dict(Counter(r['split'] for r in evidence['crops'])),
            review_candidates_by_split=dict(Counter(r['split'] for r in evidence['crops'] if r['exported_screen']['status']=='review_candidate')),
            engineering_training_input_candidates=sum(r['engineering_training_input_candidate'] for r in evidence['crops']),
            unpaired_loadable_inputs=sum(r['model_input_loadable'] for r in evidence['crops']),
            missing_training_exports=len(candidates),priority_training_artist_groups=len(priority),
            tracks_with_issues=sum(bool(r['issues']) for r in evidence['tracks']),
            source_manifest_supplied=False,original_full_audio_supplied=False)
        if pipeline_hashes(pipeline)!=pipeline_before:raise ValueError('Shared pipeline changed during audit')
        if digest(archive)!=declaration['archive_sha256']:raise ValueError('Archive changed')
        for p,h in tracked.items():
            if digest(p)!=h:raise ValueError('Tracked input changed: '+p)
        evidence['shared_pipeline_unchanged_verified']=True;evidence['run_status']='complete';save()
        write_json(component/'runs/shared_export_intake_current.json',dict(path=str(run_root)))
        print(json.dumps(evidence['summary'],indent=2),flush=True)
    except BaseException as exc:
        evidence.update(run_status='failed',failure=dict(type=type(exc).__name__,reason=str(exc)));save();raise
    return run_root


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('archive');args=parser.parse_args();run(args.archive)
