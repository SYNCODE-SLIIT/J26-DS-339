"""Versioned intake of larger shared batches, including expected withheld records."""
from bpdr_input import COMPONENT_ROOT, module_source
from collections import Counter
import csv
import json
from pathlib import Path
import uuid
import numpy as np

from bpdr_input.preparation.sources.shared_exports import digest, import_archive, load_export
from bpdr_input.preparation.sources.shared_export_audit import pipeline_hashes
from bpdr_input.preparation.diagnostics.shared_screen_diagnostics import decompose
from bpdr_input.repair.interventions import independent_pitch, select_support


def validate_record_metadata(record, track, folder):
    """Withheld upstream records are evidence, not corrupt/missing crop packages."""
    if record.get('schema_version') != 'bpdr-track-v1':
        raise ValueError('Unknown BPDR record schema')
    if record.get('status') not in ('exported','withheld'):
        raise ValueError('Unknown export status')
    if record.get('reference_type') != 'model_estimate' or record.get('training_eligible') is not False:
        raise ValueError('Unexpected reference/training claim')
    if record['mtg_track_id'] != track['mtg_track_id'] or int(record['track_id']) != int(track['track_id']):
        raise ValueError('Record track identity mismatch')
    if record['artist_id_metadata_only'] != track['artist_id']:
        raise ValueError('Artist metadata mismatch')
    if not track['license'] or any(record['license'][k] != track['license'][k] for k in ('kind','version')):
        raise ValueError('License metadata mismatch')
    if digest(Path(folder).parent/'melody/game_vocal.json') != record['full_game_json_sha256']:
        raise ValueError('Full GAME provenance mismatch')
    if record['status'] == 'withheld':
        if record.get('artifacts') != {} or not record.get('reasons'):
            raise ValueError('Withheld record must give reasons without claimed artifacts')
        if any((Path(folder)/f).exists() for f in ('vocal.wav','features.npz','boundaries.json','masks.npz')):
            raise ValueError('Withheld record has contradictory crop artifacts')
    return record['status']


def run(archive):
    root = COMPONENT_ROOT
    previous_path = root/'runs/shared_export_intake/1998c76291664a6aa51253424e630a54/evidence.json'
    previous = json.loads(previous_path.read_text())
    if previous['run_status'] != 'complete':
        raise ValueError('Original intake is incomplete')
    previous_tracks = {r['mtg_track_id']: r for r in previous['tracks']}
    pipeline = root.parents[1]/'datasets/mtg-jamendo/melody-chord-extraction-pipeline'
    baseline = pipeline_hashes(pipeline)
    if baseline != previous['shared_pipeline_files']:
        raise ValueError('Shared pipeline differs from original snapshot')
    tracked = dict(previous['tracked_inputs'])
    tracked.update({str(p): digest(p) for p in (previous_path, Path(__file__),
                    module_source('shared_screen_diagnostics.py'),module_source('interventions.py'))})
    for path,sha in tracked.items():
        if digest(path) != sha:
            raise ValueError('Frozen metadata/source changed: '+path)
    out = root/'runs/shared_batch_intake'/uuid.uuid4().hex
    out.mkdir(parents=True)
    def save(name,obj):
        (out/name).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    declaration = dict(archive=str(Path(archive).resolve()),archive_sha256=digest(archive),
        original_intake=str(previous_path),tracked_inputs=tracked,pipeline_files=baseline,
        policy='Preserve separate batch. Frozen metadata/artist partitions. No targets from predicted notes.',
        compute='Integrity on supplied packages; YIN screen replay and support geometry on training crops only.',
        no_shared_pipeline_execution=True,no_model_training=True,no_audio_edits=True,no_review_changes=True,
        no_heldout_pitch_inference=True,no_new_pairs=True,no_threshold_changes=True)
    save('declaration.json',declaration)
    evidence=dict(status='running',declaration_sha256=digest(out/'declaration.json'),tracks=[],records=[],
        artifacts={},training_updates=0,new_pair_examples=0,research_ready_examples=0,adopted=False)
    save('evidence.json',evidence)
    print('BATCH_RUN:',out,flush=True)
    try:
        imported,receipt = import_archive(archive,root)
        evidence.update(import_root=str(imported),import_receipt_sha256=digest(imported/'import_receipt.json'),
                        archive_summary={k:v for k,v in receipt.items() if k!='files'})
        print('SAVED_FILES:',receipt['preserved_files'],flush=True)
        for folder in sorted((imported/'outputs').glob('track_*')):
            if folder.name not in previous_tracks:
                raise ValueError('New track needs explicit frozen metadata mapping: '+folder.name)
            original = previous_tracks[folder.name]
            row={k:v for k,v in original.items() if k not in ('roles','bpdr_record_present','bpdr_audio_present','issues')}
            row.update(roles=[p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()],issues=[],
                bpdr_record_present=(folder/'bpdr/record.json').is_file(),bpdr_audio_present=(folder/'bpdr/vocal.wav').is_file())
            raw=json.loads((folder/'melody/game_vocal.json').read_text())
            if raw['mtg_track_id'] != folder.name or int(raw['track_id']) != int(row['track_id']):
                row['issues'].append('GAME_TRACK_MISMATCH')
            duration=float(raw['duration_sec'])
            for n in raw['notes']:
                if (not np.isfinite([n['onset_sec'],n['offset_sec'],n['midi_pitch'],n['model_midi_pitch']]).all()
                    or not 0<=n['onset_sec']<n['offset_sec']<=duration+.05
                    or abs(n['midi_pitch']-n['model_midi_pitch']-12)>1e-7):
                    row['issues'].append('GAME_PITCH_OR_TIME_MISMATCH');break
            with (folder/'melody/game_vocal.csv').open(encoding='utf-8',newline='') as f:
                csv_count=sum(1 for _ in csv.DictReader(f))
            if csv_count != len(raw['notes']):row['issues'].append('GAME_ROW_COUNT_MISMATCH')
            row.update(game_note_count=len(raw['notes']),duration_seconds=duration)
            evidence['tracks'].append(row)
            p=folder/'bpdr/record.json'
            if not p.exists():continue
            record=json.loads(p.read_text())
            item=dict(mtg_track_id=folder.name,artist_id=row['artist_id'],split=row['split'],
                record_path=str(p),record_sha256=digest(p),export_status=record.get('status'),
                integrity='pending',screen_status=record.get('screen',{}).get('status'),
                upstream_reasons=record.get('reasons',[]),issues=[],engineering_input_candidate=False,
                paired_training_ready=False,research_eligible=False)
            try:
                status=validate_record_metadata(record,row,p.parent)
                if row['issues']:raise ValueError('Full-song GAME catalog has issues')
                if status=='withheld':
                    item['integrity']='verified_withheld_record'
                else:
                    loaded=load_export(p.parent)
                    prov=record['feature_provenance']
                    if (prov['hop_samples']!=441 or prov['sample_rate_hz']!=22050 or prov['window_samples']!=1024
                        or prov['voiced_confidence_threshold']!=.8 or not prov['frozen']):
                        raise ValueError('Feature provenance mismatch')
                    item.update(integrity='verified_export',audio_sha256=record['artifacts']['vocal.wav'],
                        artifact_hashes=record['artifacts'],screen=record['screen'],frames=len(loaded['features']['f0_hz']))
                    review=p.parent/'reviews.csv'
                    if review.exists():
                        with review.open(encoding='utf-8',newline='') as f:reviews=list(csv.DictReader(f))
                        item['reviews']=dict(path=str(review),sha256=digest(review),rows=len(reviews),
                                             decisions=dict(Counter(r['decision'] for r in reviews)))
                    if row['split']=='train':
                        screen,arrays=decompose(loaded,independent_pitch(loaded['audio'],22050))
                        arrays_path=out/(folder.name+'.npz');np.savez_compressed(arrays_path,**arrays)
                        evidence['artifacts'][arrays_path.name]=digest(arrays_path)
                        item.update(screen_replay=screen,screen_arrays=arrays_path.name,support_geometry_available=False)
                        try:
                            support=select_support(loaded['model_inputs']['times_seconds'],loaded['masks']['reliable_pitch_screen'])
                            item.update(support_geometry_available=True,support_seconds=list(support))
                        except ValueError as exc:item['support_geometry_reason']=str(exc)
                        item['engineering_input_candidate']=bool(row['policy_export_eligible'] and record['screen']['status']=='review_candidate')
                        item['candidate_for_pair_audit']=bool(item['engineering_input_candidate'] and item['support_geometry_available'])
                    else:
                        item.update(screen_replay=None,support_geometry_available=None,candidate_for_pair_audit=False)
            except (ValueError,KeyError,OSError) as exc:
                item.update(integrity='quarantined',issues=[str(exc)],engineering_input_candidate=False,candidate_for_pair_audit=False)
            evidence['records'].append(item)
            if len(evidence['records'])%25==0:
                save('evidence.json',evidence);print('CHECKED_RECORDS:',len(evidence['records']),flush=True)
        old_crops={r['mtg_track_id']:r for r in previous['crops']}
        old_hashes={r.get('audio_sha256') for r in previous['crops'] if r.get('audio_sha256')}
        exported=[r for r in evidence['records'] if r['integrity']=='verified_export']
        training=[r for r in exported if r['split']=='train']
        candidates=[r for r in training if r['candidate_for_pair_audit']]
        statuses={r['mtg_track_id']:r for r in evidence['records']}
        priority=json.loads((previous_path.parent/'bpdr_export_priority.json').read_text())['tracks']
        evidence['priority_receipt']=[dict(mtg_track_id=r['mtg_track_id'],artist_id=r['artist_id'],
            status=statuses.get(r['mtg_track_id'],{}).get('integrity','not_supplied'),
            screen_status=statuses.get(r['mtg_track_id'],{}).get('screen_status')) for r in priority]
        old_files={r['relative_path']:r['sha256'] for r in json.loads(Path(previous['import_receipt']).read_text())['files']}
        new_files={r['relative_path']:r['sha256'] for r in receipt['files']}
        common=set(old_files)&set(new_files)
        evidence['batch_comparison']=dict(previous_import_root=previous['import_root'],
            common_files=len(common),identical_common_files=sum(old_files[p]==new_files[p] for p in common),
            changed_common_files=sum(old_files[p]!=new_files[p] for p in common),
            new_files=len(set(new_files)-set(old_files)),previous_files_absent=len(set(old_files)-set(new_files)),
            previous_crop_records_seen=sum(r['mtg_track_id'] in old_crops for r in evidence['records']),
            previous_audio_hashes_repeated=sum(r['audio_sha256'] in old_hashes for r in exported),
            unique_exported_audio_hashes=len({r['audio_sha256'] for r in exported}))
        summary=dict(track_folders=len(evidence['tracks']),bpdr_records=len(evidence['records']),
            record_integrity=dict(Counter(r['integrity'] for r in evidence['records'])),
            record_artist_proxies=len({r['artist_id'] for r in evidence['records']}),
            exported_artist_proxies=len({r['artist_id'] for r in exported}),
            exported_by_split=dict(Counter(r['split'] for r in exported)),
            screen_candidates_by_split=dict(Counter(r['split'] for r in exported if r['screen_status']=='review_candidate')),
            exported_training_crops=len(training),training_artist_proxies=len({r['artist_id'] for r in training}),
            engineering_training_input_candidates=sum(r['engineering_input_candidate'] for r in training),
            training_candidates_with_support=len(candidates),pair_audit_artist_proxies=len({r['artist_id'] for r in candidates}),
            exact_training_screen_masks=sum(r.get('screen_replay',{}).get('mask_exact',False) for r in training),
            priority_receipt=dict(Counter(r['status'] for r in evidence['priority_receipt'])),
            withheld_reasons=dict(Counter(reason for r in evidence['records'] if r['integrity']=='verified_withheld_record' for reason in r['upstream_reasons'])),
            training_ready_pairs=0,research_ready_examples=0,root_manifest_supplied=False)
        save('source_inputs.json',dict(record_kind='unpaired_source_inputs',records=exported,paired_training_examples=0,research_ready_examples=0))
        save('pair_audit_candidates.json',dict(selection='All integrity-valid, policy-eligible frozen-training screen candidates with unchanged v0.3 support geometry',
            candidates=candidates,meaning='Prospective intervention audit only; no pair or repair approval',training_ready_pairs=0))
        for name in ('source_inputs.json','pair_audit_candidates.json'):
            evidence['artifacts'][name]=digest(out/name)
        if pipeline_hashes(pipeline)!=baseline:raise ValueError('Common pipeline changed')
        if digest(archive)!=declaration['archive_sha256']:raise ValueError('Source archive changed')
        for p,h in tracked.items():
            if digest(p)!=h:raise ValueError('Tracked input changed')
        evidence.update(summary=summary,status='complete',pipeline_unchanged=True)
        save('evidence.json',evidence)
        (root/'runs/shared_batch_intake_current.json').write_text(json.dumps(dict(path=str(out)),indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException as exc:
        evidence.update(status='failed',failure=dict(type=type(exc).__name__,message=str(exc)))
        save('evidence.json',evidence)
        raise
    return out


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive');args=parser.parse_args()
    run(args.archive)
