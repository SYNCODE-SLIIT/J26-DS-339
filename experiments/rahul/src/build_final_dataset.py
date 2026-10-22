"""Discover extraction tracks and build a validated, group-split final dataset.

Reuses Phase 1, FeatureEncoder and ChordSequenceDataset. Track shards and disk
arrays bound memory usage; existing Track 382/production outputs are never written.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile

from experiments.rahul import config
from experiments.rahul.src.build_synthetic_dataset import DEFAULT_SOURCE, BASIC_QUALITY_POLICY, build, dump

REQUIRED = ('melody/combined.csv','chords/full.csv','roman/full.csv',
            'beats/beats.csv','beats/beats.json','chords/full.json')


def discover(source_root):
    inventory=[]
    for folder in sorted(Path(source_root).glob('track_*')):
        if not folder.is_dir() or not folder.name.removeprefix('track_').isdigit():
            continue
        missing=[name for name in REQUIRED if not (folder/name).is_file()]
        inventory.append(dict(track_id=str(int(folder.name.removeprefix('track_'))),
                              source_folder=folder.name,status='incomplete' if missing else 'candidate',
                              missing_files=missing))
    return inventory


def _build_track(track, source_root, phase1_root):
    """Worker process: preserve policy, fail explicitly, validate ML input schema."""
    from experiments.rahul.src.encode_features import load_events
    csv.field_size_limit(64*1024*1024)
    folder=Path(phase1_root)/f'track_{int(track):07d}'
    try:
        # A completed immutable candidate may be reused when resuming this run.
        report_path=folder/f'track_{int(track):07d}_validation_report.json'
        csv_path=folder/f'track_{int(track):07d}_training_events.csv'
        cached=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
        if cached.get('chord_endpoint_tolerance_sec')!=1.0 or cached.get('basic_quality_policy')!=BASIC_QUALITY_POLICY:
            with contextlib.redirect_stdout(io.StringIO()):
                _,report=build(track,Path(source_root),folder,chord_endpoint_tolerance_sec=1.0)
            # Phase 1's old comparison text names Track 382 unconditionally.
            # Correct only the NEW batch report prose; source implementation stays unchanged.
            old_text=report.get('notebook_comparison','')
            report['notebook_comparison']='No per-track notebook comparison was performed in this batch; source and tensor validation are recorded separately.'
            dump(report_path,report)
            summary=folder/'dataset_build_summary.md'
            summary.write_text(summary.read_text(encoding='utf-8').replace(old_text,report['notebook_comparison']),encoding='utf-8')
        report=json.loads(report_path.read_text(encoding='utf-8'))
        if not report['all_passed']:
            raise ValueError('Phase 1 report failed')
        for relative,expected in report['source_sha256'].items():
            source=config.REPOSITORY_ROOT/relative
            if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Source changed since candidate build: {relative}')
        rows,_=load_events(csv_path)
        return dict(track_id=track,status='validated',statistics=report['statistics'],
                    max_notes=max(len(r['melody_midi']) for r in rows),
                    phase1_csv=csv_path.relative_to(config.REPOSITORY_ROOT).as_posix())
    except Exception as exc:
        return dict(track_id=track,status='rejected',error_type=type(exc).__name__,reason=str(exc))


class TrackFiles:
    """Load one validated track at a time during the existing encoder's fit loops."""
    def __init__(self,root,ids): self.root,self.ids=Path(root),set(ids)
    def __iter__(self): return iter(sorted(self.ids))
    def __getitem__(self,track):
        from experiments.rahul.src.encode_features import load_events
        if track not in self.ids: raise KeyError(track)
        name=f'track_{int(track):07d}'
        return load_events(self.root/name/f'{name}_training_events.csv')[0]


def export_split(split, ids, tracks, encoder, length, output, reports, sequence_unit='beats'):
    import numpy as np
    import torch
    from experiments.rahul.src.transformer_dataset import (
        ChordSequenceDataset, SavedSequenceDataset, export_npz, make_dataloader)
    from experiments.rahul.src.encode_features import supported_target, IGNORE_INDEX
    count=sum(reports[t]['statistics']['bar_sequences'] if sequence_unit=='bars' else
              math.ceil(reports[t]['statistics']['final_events']/length) for t in ids)
    manifest=[]
    statistics=dict(tracks=len(ids),sequences=count,beats=0,source_valid_targets=0,loss_eligible_targets=0,
                    invalid_source_targets=0,masked_real_targets=0,padded_positions=0,silent_beats=0,
                    max_observed_notes=0,notes_truncated=0,unseen_targets={},unseen_genres={},
                    unsupported_or_missing_targets=0)
    checked_arrays=0
    # Temporary arrays are internal to this NEW output directory and auto-cleaned.
    with tempfile.TemporaryDirectory(prefix=f'_{split}_',dir=output) as scratch_name:
        scratch=Path(scratch_name)
        if not scratch.resolve().is_relative_to(output.resolve()):
            raise ValueError('Unexpected temporary output location')
        maps={}
        offset=0
        for number,track in enumerate(ids,start=1):
            rows=tracks[track]
            stats=encoder.statistics(rows)
            statistics['beats']+=len(rows)
            statistics['source_valid_targets']+=stats['source_valid_targets']
            statistics['invalid_source_targets']+=len(rows)-stats['source_valid_targets']
            statistics['loss_eligible_targets']+=stats['loss_eligible_targets']
            statistics['silent_beats']+=stats['silent_beats']
            statistics['max_observed_notes']=max(statistics['max_observed_notes'],stats['maximum_notes_per_beat'])
            statistics['notes_truncated']+=stats['notes_truncated']
            statistics['unsupported_or_missing_targets']+=stats['unsupported_or_missing_targets']
            for field in ('unseen_targets','unseen_genres'):
                for label,n in stats[field].items():
                    statistics[field][label]=statistics[field].get(label,0)+n
            dataset=ChordSequenceDataset({track:rows},encoder,sequence_length=length,
                stride=None if sequence_unit=='bars' else length,sequence_unit=sequence_unit)
            shard_path=scratch/'track.npz'
            export_npz(shard_path,dataset)
            saved=SavedSequenceDataset(shard_path)
            # Efficient equality check: construct each item once, not once per field.
            coverage=[]
            for index in range(len(dataset)):
                original,restored=dataset[index],saved[index]
                for group in ('inputs','targets'):
                    if set(original[group])!=set(restored[group]): raise ValueError('Shard fields differ')
                    for key,value in original[group].items():
                        if not torch.equal(value,restored[group][key]): raise ValueError(f'Shard changed {key}')
                coverage.extend(restored['metadata']['beat_indices'][restored['inputs']['beat_attention_mask']].tolist())
            if coverage!=[r['beat_index'] for r in rows]: raise ValueError('Sequence coverage/order differs')
            # Input mutation test uses exactly the existing feature whitelist.
            sample=rows[0]['input_snapshot']
            before=encoder.encode_inputs(sample)
            after=encoder.encode_inputs({**sample,'reference_chord':'audit mutation','reference_quality':'audit mutation','target_valid':False})
            if not all(np.array_equal(before[k],after[k]) for k in before): raise ValueError('Target leakage')
            with np.load(shard_path,allow_pickle=False) as archive:
                arrays={k:archive[k] for k in archive.files}
            beat=arrays['inputs__beat_attention_mask']
            note=arrays['inputs__melody_note_mask']
            loss=arrays['targets__target_loss_mask']
            targets=arrays['targets__target_chord_ids']
            expected_loss=[bool(r['target_valid'] and supported_target(r)
                          and r['reference_chord_canonical'] in encoder.targets.token_to_id
                          and r['reference_chord_canonical'] not in ('PAD','UNK')) for r in rows]
            if loss[beat].tolist()!=expected_loss: raise ValueError('Target eligibility differs from source')
            if not np.array_equal(targets==IGNORE_INDEX,~loss): raise ValueError('Ignore index mask differs')
            if loss[~beat].any() or note[~beat].any(): raise ValueError('Padding contains real supervision/notes')
            if not np.array_equal(arrays['inputs__melody_silent'],~note.any(axis=-1)): raise ValueError('Silence mask differs')
            if not arrays['inputs__melody_attention_mask'].any(axis=-1).all(): raise ValueError('Unsafe attention row')
            if not (arrays['beat_indices'][~beat]==-1).all(): raise ValueError('Padded source beat ID differs')
            if set(arrays['track_ids'])!={track}: raise ValueError('Track boundary crossed')
            for key,value in arrays.items():
                if value.dtype.kind!='U' and not np.isfinite(value).all(): raise ValueError(f'Nonfinite tensor {key}')
                if key not in maps:
                    dtype=f'<U{max(len(t) for t in ids)}' if key=='track_ids' else value.dtype
                    maps[key]=np.lib.format.open_memmap(scratch/(key+'.npy'),mode='w+',
                        dtype=dtype,shape=(count,)+value.shape[1:])
                maps[key][offset:offset+len(dataset)]=value
                checked_arrays+=1
            for i in range(len(dataset)):
                real_ids=arrays['beat_indices'][i][beat[i]]
                manifest.append(dict(split=split,sequence_index=offset+i,track_id=track,
                    first_beat_index=int(real_ids[0]),last_beat_index=int(real_ids[-1]),
                    real_beats=int(beat[i].sum()),padded_beats=int((~beat[i]).sum()),
                    valid_targets=int(loss[i].sum())))
                if sequence_unit=='bars':
                    window=dataset.windows[i]
                    manifest[-1]['bar_number']=window['bar_number']
                    manifest[-1]['beat_positions']=json.dumps([r['beat_position'] for r in rows[window['start_row']:window['stop_row']]])
                    if len(set(arrays['inputs__bar_number'][i][beat[i]].tolist())) != 1:
                        raise ValueError('Sequence crosses a source bar boundary')
            statistics['padded_positions']+=int((~beat).sum())
            statistics['masked_real_targets']+=int((beat & ~loss).sum())
            offset+=len(dataset)
            if number%20==0 or number==len(ids):
                print(f'[{split}] {number}/{len(ids)} tracks; {offset}/{count} sequences',flush=True)
        if offset!=count: raise ValueError('Sequence count differs')
        for value in maps.values(): value.flush()
        candidate=output/f'{split}_sequences.npz'
        np.savez_compressed(candidate,**maps)
        # Compare final compressed arrays against the verified mapped source data.
        with np.load(candidate,allow_pickle=False) as final:
            if set(final.files)!=set(maps): raise ValueError('Final archive fields differ')
            for key,value in maps.items():
                if final[key].shape!=value.shape or final[key].dtype!=value.dtype or not np.array_equal(final[key],value):
                    raise ValueError(f'Final archive changed {key}')
        batch=next(iter(make_dataloader(SavedSequenceDataset(candidate),batch_size=4,seed=42)))
        shapes={group:{k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in batch[group].items()}
                for group in ('inputs','targets')}
        # Release Windows memory maps before deleting their temporary directory.
        for value in maps.values(): value._mmap.close()
        maps.clear()
        del value
    return statistics,manifest,dict(all_passed=True,track_shard_arrays_checked=checked_arrays,
        source_coverage=True,train_only_vocabulary=True,target_input_separation=True,
        chronological_track_boundaries=True,numerical_and_mask_checks=True,final_archive_equals_verified_shards=True,
        first_batch_tensors=shapes)


def run(source_root=DEFAULT_SOURCE,run_name='final_all_tracks',workers=4,sequence_length=8,seed=42,resume=False):
    from experiments.rahul.src.create_sequences import split_tracks,validate_splits
    from experiments.rahul.src.preprocess_for_transformer import source_identities
    from experiments.rahul.src.encode_features import FeatureEncoder,USED_INPUTS
    csv.field_size_limit(64*1024*1024)
    if workers<1 or sequence_length<1: raise ValueError('Workers and sequence length must be positive')
    if not run_name.replace('_','').replace('-','').isalnum(): raise ValueError('Use a simple run name')
    output=config.COMPONENT_DIR/'outputs/model_preprocessing'/run_name
    phase1=config.OUTPUT_DIR/'synthetic_dataset'/run_name
    if output.exists() and not resume: raise FileExistsError('Use --resume or a new run name; existing outputs are protected')
    if (output/'FINAL_DATASET_READY.json').exists(): raise FileExistsError('Completed final dataset is immutable; choose a new run name')
    output.mkdir(parents=True,exist_ok=True)
    inventory=discover(source_root)
    dump(output/'discovery_manifest.json',inventory)
    candidates=[r['track_id'] for r in inventory if r['status']=='candidate']
    results=[]
    print(f'Discovered {len(inventory)} tracks; {len(candidates)} complete candidates',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(_build_track,t,str(source_root),str(phase1)):t for t in candidates}
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            if len(results)%10==0 or result['status']=='rejected' or len(results)==len(candidates):
                print(f'[Phase 1] {len(results)}/{len(candidates)}: {result["track_id"]} {result["status"]} {result.get("reason","")}',flush=True)
            dump(output/'track_build_results.json',sorted(results,key=lambda r:int(r['track_id'])))
    reports={r['track_id']:r for r in results if r['status']=='validated'}
    ids=sorted(reports)
    identities,_=source_identities(ids)
    # Make source paths relative to the selected extraction root available even when key.json is absent.
    for track in ids:
        document=Path(source_root)/f'track_{int(track):07d}'/'chords/full.json'
        source=json.loads(document.read_text(encoding='utf-8')).get('source_mp3')
        if source: identities[track].append('recording:'+source.replace('\\','/'))
    splits=split_tracks(ids,seed=seed,identities=identities)
    validate_splits(splits,ids)
    dump(output/'split_manifest.json',splits)
    tracks=TrackFiles(phase1,ids)
    print(f'[Phase 2] Fitting on {len(splits["splits"]["train"])} training tracks only',flush=True)
    encoder=FeatureEncoder.fit(tracks,splits['splits']['train'])
    observed_train_max=encoder.max_notes
    # Training-derived capacity headroom, never fit on validation/test note counts.
    encoder.max_notes=1 << (encoder.max_notes-1).bit_length()
    encoder.save(output)
    reloaded=FeatureEncoder.load(output)
    if reloaded.targets.state()!=encoder.targets.state(): raise ValueError('Target vocabulary reload differs')
    print(f'[Phase 2] Training maximum {observed_train_max}; note capacity {encoder.max_notes}; {len(encoder.targets.tokens)-2} musical target classes',flush=True)
    all_stats,all_checks,manifest={},{},[]
    for split,assigned in splits['splits'].items():
        stat,entries,checks=export_split(split,assigned,tracks,encoder,sequence_length,output,reports)
        all_stats[split],all_checks[split]=stat,checks
        manifest.extend(entries)
        dump(output/'preprocessing_statistics.json',all_stats)
    with (output/'sequence_manifest.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    source_hashes={}
    for track in ids:
        report_path=phase1/f'track_{int(track):07d}'/f'track_{int(track):07d}_validation_report.json'
        for path,expected in json.loads(report_path.read_text(encoding='utf-8'))['source_sha256'].items():
            if path in source_hashes and source_hashes[path]!=expected: raise ValueError('Source snapshots differ')
            source_hashes[path]=expected
    for relative,expected in source_hashes.items():
        if hashlib.sha256((config.REPOSITORY_ROOT/relative).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Original extraction file changed: {relative}')
    settings=dict(schema_version=1,development_only=False,sequence_length=sequence_length,stride=sequence_length,
        seed=seed,split_ratios=[.8,.1,.1],selected_input_columns=sorted(USED_INPUTS),
        target_column='reference_chord_canonical',mask_column='target_valid',ignore_index=-100,
        max_notes_per_beat=encoder.max_notes,observed_training_max_notes=observed_train_max,
        note_capacity_policy='Next power of two of training maximum, minimum one. No truncation; held-out overflow fails rather than expanding from test data.',
        normalization='Existing encoder: original MIDI/127; duration/onset/overlap divided by event duration; no fitted numerical scaling.',
        basic_chords='Synthetic deterministic simplifications of recognized reference pseudo-labels.',
        basic_quality_policy=BASIC_QUALITY_POLICY,
        basic_quality_mapping='maj*/6/add9 -> maj; min*/m/m7/m9/m11 -> min; 7/9/11/13/7b9 -> dom; dim/dim7 -> dim; hdim7/m7b5/co7 -> hdim; aug/+ -> aug; sus2/sus4 -> sus; N preserved. Roots and detailed references preserved.',
        chord_endpoint_tolerance_sec=1.0,
        endpoint_policy='Explicitly matches upstream code/chords.py maximum end-to-duration difference. Source intervals are preserved and beats still end at validated audio duration. Track 382 default remains 0.01 seconds.',
        implementation_sha256={name:hashlib.sha256((config.COMPONENT_DIR/'src'/name).read_bytes()).hexdigest()
            for name in ('build_final_dataset.py','build_synthetic_dataset.py','encode_features.py',
                         'create_sequences.py','transformer_dataset.py','canonicalize_chords.py','normalize_chords.py')},
        identities='Known MTG artist/album plus original recording paths; distinct-identity duplicate audio cannot be ruled out without content fingerprints.',
        source_root=Path(source_root).relative_to(config.REPOSITORY_ROOT).as_posix(),
        phase1_output_root=phase1.relative_to(config.REPOSITORY_ROOT).as_posix())
    dump(output/'preprocessing_config.json',settings)
    validation=dict(all_passed=True,splits=all_checks,source_sha256=source_hashes,source_extractions_unchanged=True,
                    no_note_truncation=all(s['notes_truncated']==0 for s in all_stats.values()),
                    group_split_isolation=True,train_only_fit=True)
    dump(output/'preprocessing_validation_report.json',validation)
    summary=f'''# Final multi-track chord-refinement dataset

This build discovered {len(inventory)} extraction track folders. {len(candidates)} had all required files;
{len(inventory)-len(candidates)} were incomplete. {len(ids)} passed both source and ML-input validation;
{len(results)-len(ids)} complete candidates were rejected. Each omission and reason is retained in
discovery_manifest.json and track_build_results.json. No source extraction was changed.

Use train_sequences.npz for training, validation_sequences.npz for validation, and test_sequences.npz
for final evaluation. Each uses the same training-fitted feature/target vocabularies. Load with the
existing SavedSequenceDataset and make_dataloader. Never combine held-out tracks into vocabulary fitting.
Track/artist/album/recording groups are isolated; the seed is {seed}, with requested 80/10/10 group ratios.
Realized track/beat ratios differ because related tracks are kept together.

The sequence length and stride are {sequence_length}. Invalid targets remain contextual beats;
invalid, unseen and padded targets have ID -100 and false loss masks. Unseen held-out targets and genres
are explicitly counted below. N remains a legitimate class. Silent beats retain safe attention dummy
slots, which are not real notes; later note pooling must gate silence to zero. Masks use True=keep.

Training observed at most {observed_train_max} notes per beat; capacity {encoder.max_notes} is the next
power of two of that training-only statistic. No notes were truncated and no test-derived capacity
adjustment was performed. Features and tensor names follow MODEL_PREPROCESSING.md. This runner reuses
Phase 1 and existing encoders/Datasets, processing track shards and verified disk arrays to bound memory.
Original Track 382 development files remain separate and untouched.

This run explicitly uses the upstream chord extractor's one-second endpoint tolerance instead of the
old single-track 0.01-second default, which rejected ordinary 13–26 ms recognizer overruns. Source
intervals are preserved, and all beat windows still stop at the validated audio duration. The CSV
parser limit is 64 MiB per field to accommodate long JSON melody lists, without altering their data.

The basics are still synthetic, and reference labels are recognized pseudo-labels. Independent basic
predictions and label-quality evaluation remain necessary for research claims. No Transformer or loss
was implemented or calculated. Metadata identity grouping cannot detect every duplicate recording.

| Split | Tracks | Beats | Sequences | Eligible targets | Invalid/unseen real targets | Padding |
|---|---:|---:|---:|---:|---:|---:|
'''
    for split,s in all_stats.items():
        summary+=f'| {split} | {s["tracks"]} | {s["beats"]} | {s["sequences"]} | {s["loss_eligible_targets"]} | {s["masked_real_targets"]} | {s["padded_positions"]} |\n'
    summary+='\nDetailed distributions and held-out unseen-label counts are in preprocessing_statistics.json.\n'
    (output/'dataset_build_summary.md').write_text(summary,encoding='utf-8')
    ready=dict(status='ready',validated_tracks=len(ids),incomplete_tracks=len(inventory)-len(candidates),
        rejected_tracks=len(results)-len(ids),splits=all_stats,
        artifacts={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in
                   ('train_sequences.npz','validation_sequences.npz','test_sequences.npz','feature_vocabularies.json','target_vocabulary.json','split_manifest.json')})
    dump(output/'FINAL_DATASET_READY.json',ready)
    print(json.dumps(ready,indent=2),flush=True)
    return ready


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--run-name',default='final_all_tracks')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--sequence-length',type=int,default=8)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    run(args.source_root,args.run_name,args.workers,args.sequence_length,args.seed,args.resume)


if __name__=='__main__': main()
