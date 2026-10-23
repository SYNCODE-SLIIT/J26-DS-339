"""Extend the verified full-melody snapshot to variable-length multi-bar shards."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import numpy as np
import torch

from experiments.rahul import config
from experiments.rahul.src.create_sequences import validate_splits
from experiments.rahul.src.encode_features import FeatureEncoder
from experiments.rahul.src.rebuild_bar_dataset import BarTracks
from experiments.rahul.src.transformer_dataset import ChordSequenceDataset,SavedSequenceDataset,export_npz,make_dataloader,collate_sequences


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def run(previous_name='final_all_tracks_quality_v5_bars_full_melody',run_name='final_all_tracks_quality_v6_multibar',bars_per_sequence=4,batch_size=4,export_csv=True,compatibility_npz=False):
    if bars_per_sequence<1 or batch_size<1:raise ValueError('Positive bars_per_sequence and batch size required')
    if any(not n.replace('_','').replace('-','').isalnum() for n in (previous_name,run_name)):raise ValueError('Simple snapshot names required')
    csv.field_size_limit(64*1024*1024)
    base=config.COMPONENT_DIR/'outputs/model_preprocessing'
    previous,output=base/previous_name,base/run_name
    if output.exists():raise FileExistsError('Existing snapshots protected; choose another run name')
    read=lambda name:json.loads((previous/name).read_text(encoding='utf-8'))
    old_config,ready=read('preprocessing_config.json'),read('FINAL_DATASET_READY.json')
    for name,expected in ready['artifacts'].items():
        if sha(previous/name)!=expected:raise ValueError(f'Previous artifact changed: {name}')
    old_validation=read('preprocessing_validation_report.json')
    for name,expected in old_validation['source_sha256'].items():
        if sha(config.REPOSITORY_ROOT/name)!=expected:raise ValueError(f'Extraction changed: {name}')
    split_manifest=read('split_manifest.json')
    splits=split_manifest['splits']
    ids=[t for group in splits.values() for t in group]
    validate_splits(split_manifest,ids)
    encoder=FeatureEncoder.load(previous)
    if set(encoder.training_tracks)!=set(splits['train']) or encoder.truncate:raise ValueError('Expected full-note training-only mappings')
    grouping=read('bar_grouping_report.json')
    exclusion=read('outlier_exclusion.json')
    tracks=BarTracks(config.REPOSITORY_ROOT/old_config['phase1_output_root'],ids,
        exclusion['track_id'],exclusion['outlier_beat_index'],exclusion['sequence_length'])
    tracks.excluded_bars={}
    for bar in grouping['invalid_bars_excluded']:
        tracks.excluded_bars.setdefault(bar['track_id'],set()).add(bar['bar_number'])
    output.mkdir()
    for name in ('feature_vocabularies.json','target_vocabulary.json','split_manifest.json','bar_grouping_report.json','outlier_exclusion.json'):
        shutil.copyfile(previous/name,output/name)
    manifest=[]
    reports={}
    artifacts={}
    all_issues=[]
    actual_examples=[]
    raw_source_beats=0
    raw_source_bars=0
    meter_disagreements={(str(b['track_id']),int(b['bar_number'])) for b in grouping['partial_bars']}
    for split,assigned in splits.items():
        shard_dir=output/f'{split}_sequences'
        shard_dir.mkdir()
        shard_index=[]
        lengths=[]
        counts=Counter()
        per_track=[]
        offset=0
        for number,track in enumerate(assigned,1):
            rows=tracks[track]
            raw_source=config.REPOSITORY_ROOT/old_config['source_root']/f'track_{int(track):07d}'/'beats/beats.csv'
            with raw_source.open(encoding='utf-8',newline='') as stream:
                raw={int(r['beat_index']):r for r in csv.DictReader(stream)}
            raw_source_beats+=len(raw)
            raw_source_bars+=len({r['bar_index'] for r in raw.values()})
            for event in rows:event['source_meter_disagreement']=(track,event['bar_number']) in meter_disagreements
            # Confirm a complete ending bar only with both explicit source meter
            # coverage and ordinary beat timing. Otherwise retain and flag it.
            last_bar=[r for r in rows if r['bar_number']==rows[-1]['bar_number']]
            meter=raw[last_bar[-1]['beat_index']]
            durations=[r['event_duration'] for r in last_bar]
            ending_complete=(last_bar[0]['beat_position']==1 and
                [r['beat_position'] for r in last_bar]==list(range(1,int(meter['meter_numerator'])+1)) and
                durations[-1]<=1.75*statistics.median(durations[:-1] or durations))
            rows[-1]['bar_end_confirmed']=ending_complete
            dataset=ChordSequenceDataset({track:rows},encoder,sequence_unit='multi_bar',bars_per_sequence=bars_per_sequence)
            path=shard_dir/f'track_{int(track):07d}.npz'
            export_npz(path,dataset)
            saved=SavedSequenceDataset(path)
            assert len(saved)==len(dataset)
            covered=[]
            last_end=None
            for local in range(len(dataset)):
                item,restored=dataset[local],saved[local]
                window=dataset.windows[local]
                events=dataset.tracks[track][window['start_row']:window['stop_row']]
                real=len(events)
                for group in ('inputs','targets'):
                    for key,value in item[group].items():
                        assert torch.equal(value,restored[group][key]),(track,local,key)
                        assert value.dtype in (torch.int64,torch.float32,torch.bool)
                        assert torch.isfinite(value).all()
                assert item['metadata']['beat_indices'].tolist()==restored['metadata']['beat_indices'].tolist()
                assert all(item['metadata'][k]==restored['metadata'][k] for k in item['metadata'] if k!='beat_indices')
                assert item['inputs']['beat_attention_mask'].all()
                assert torch.equal(restored['targets']['target_chord_ids']==-100,~restored['targets']['target_loss_mask'])
                assert item['inputs']['melody_attention_mask'].any(dim=-1).all()
                assert int(item['inputs']['melody_note_mask'].sum())==sum(len(r['melody_midi']) for r in events)
                # Per-beat comparison independently verifies grouping did not
                # average chords, reset neighbors, or alter any existing feature.
                for p,event in enumerate(events):
                    for key,value in encoder.encode_inputs(event['input_snapshot']).items():
                        expected=torch.from_numpy(np.asarray(value).copy())
                        actual=item['inputs'][key] if key=='genre_features' else item['inputs'][key][p]
                        assert torch.equal(expected,actual),(track,event['beat_index'],key)
                    for key,value in encoder.encode_target(event).items():
                        assert item['targets'][key][p].item()==value.item()
                    assert item['inputs']['relative_bar_number'][p].item()==event['bar_number']-events[0]['bar_number']
                covered.extend(item['metadata']['beat_indices'].tolist())
                bar_numbers=list(dict.fromkeys(e['bar_number'] for e in events))
                assert bar_numbers==list(range(bar_numbers[0],bar_numbers[-1]+1))
                assert len(bar_numbers)<=bars_per_sequence
                meters=list(dict.fromkeys(f"{raw[e['beat_index']]['meter_numerator']}/{raw[e['beat_index']]['meter_denominator']}" for e in events))
                valid=int(item['targets']['target_loss_mask'].sum())
                entry=dict(split=split,track_id=track,sequence_id=offset+local,
                    sequence_start_bar=bar_numbers[0],sequence_end_bar=bar_numbers[-1],bars_in_sequence=len(bar_numbers),
                    real_beats_in_sequence=real,first_beat_index=events[0]['beat_index'],last_beat_index=events[-1]['beat_index'],
                    start_sec=events[0]['start_sec'],end_sec=events[-1]['end_sec'],source_meters=json.dumps(meters),
                    valid_targets=valid,invalid_targets=real-valid,final_partial_window=window['final_partial_window'],
                    complete_bar_count=window['complete_bar_count'],complete_bar_window=window['complete_four_bar_window'],
                    contains_partial_bar=window['contains_partial_bar'],contains_irregular_bar=window['contains_irregular_bar'],
                    partial_bars_count=window['partial_bars_count'],irregular_bars_count=window['irregular_bars_count'],
                    window_reason=window['window_reason'])
                manifest.append(entry)
                if len(actual_examples)<12 and (len(actual_examples)<3 or real not in {e['real_beats_in_sequence'] for e in actual_examples}):actual_examples.append(entry)
                if window['contains_partial_bar'] or window['contains_irregular_bar'] or window['window_reason'] in ('source_gap','before_internal_partial_bar'):
                    all_issues.append(entry)
                lengths.append(real)
                counts.update(real_beats=real,valid_targets=valid,invalid_targets=real-valid,
                    silent_beats=int(item['inputs']['melody_silent'].sum()),
                    complete_bar_windows=int(window['complete_four_bar_window']),final_partial_windows=int(window['final_partial_window']),
                    partial_bars=window['partial_bars_count'],irregular_bars=window['irregular_bars_count'],
                    bars=len(bar_numbers))
            assert covered==[r['beat_index'] for r in rows] and len(set(covered))==len(covered)
            if compatibility_npz:
                compat=output/'compatibility'/split
                compat.mkdir(parents=True,exist_ok=True)
                export_npz(compat/path.name,dataset,variable_length=False)
            relative=path.relative_to(output).as_posix()
            shard_index.append(dict(path=relative,sequences=len(dataset),real_beats=len(rows),sha256=sha(path)))
            artifacts[relative]=shard_index[-1]['sha256']
            per_track.append(dict(track_id=track,sequences=len(dataset),real_beats=len(rows)))
            offset+=len(dataset)
            if number%20==0 or number==len(assigned):print(f'[{split}] {number}/{len(assigned)} tracks: {offset} sequences',flush=True)
        assert counts['real_beats']==ready['splits'][split]['beats']
        index=dict(format='ragged_npz_shards_v1',sequences=offset,real_beats=counts['real_beats'],shards=shard_index)
        index_path=output/f'{split}_sequences.json'
        index_path.write_text(json.dumps(index,indent=2),encoding='utf-8')
        artifacts[index_path.name]=sha(index_path)
        loaded=SavedSequenceDataset(index_path)
        assert len(loaded)==offset
        batch=next(iter(make_dataloader(loaded,batch_size=batch_size)))
        dynamic_padding=sum(max(lengths[i:i+batch_size])*len(lengths[i:i+batch_size])-sum(lengths[i:i+batch_size]) for i in range(0,len(lengths),batch_size))
        reports[split]=dict(tracks=len(assigned),sequences=offset,**dict(counts),
            sequence_length_distribution=dict(Counter(lengths)),minimum_sequence_length=min(lengths),maximum_sequence_length=max(lengths),
            average_sequence_length=statistics.mean(lengths),per_track=per_track,
            source_beats_lost=0,source_beats_duplicated=0,stored_padding=0,
            previous_stored_padding=ready['splits'][split]['padded_positions'],
            dynamic_padding_positions_at_batch_size=dynamic_padding,batch_size=batch_size,
            sample_batch_shapes={g:{k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in batch[g].items()} for g in ('inputs','targets')})
    with (output/'sequence_manifest.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    # A mixed-length real batch illustrates actual, batch-local padding overhead.
    chosen=[]
    for desired in (12,16,20):
        found=next((e for e in manifest if e['real_beats_in_sequence']==desired),None)
        if found:chosen.append(found)
    datasets={s:SavedSequenceDataset(output/f'{s}_sequences.json') for s in splits}
    example_batch=collate_sequences([datasets[e['split']][e['sequence_id']] for e in chosen])
    settings={**old_config,'previous_snapshot':previous_name,'sequence_unit':'multi_bar','bars_per_sequence':bars_per_sequence,
        'bar_stride':bars_per_sequence,'sequence_length':None,'stride':None,'storage_format':'ragged_npz_shards_v1',
        'bar_padding_policy':'No stored beat padding; collate_sequences pads to batch maximum only.',
        'partial_bar_policy':'Retain partial opening/ending bars and final short windows. Isolate internal partial bars. Break windows at excluded-bar/beat/time gaps; never bridge them.',
        'completeness_policy':'Actual ordered beat events determine lengths. Next downbeat/bar transition confirms completion. Ending-bar completion needs source meter coverage plus ordinary ending duration; otherwise retained and flagged uncertain. Meter numerator never allocates beat tokens.',
        'context_policy':'Previous/next basics are genuine full-source track neighbors, including across bars/windows; START/END only at original track boundaries. Inference requires the same complete basic progression (offline context).',
        'vocabulary_policy':'Reuse v5 training-only vocabularies: exact same eligible beat population and feature semantics.',
        'implementation_sha256':{n:sha(config.COMPONENT_DIR/'src'/n) for n in ('create_sequences.py','transformer_dataset.py','rebuild_multi_bar_dataset.py','export_readable_sequences.py','encode_features.py')}}
    def dump(name,value):(output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    dump('preprocessing_config.json',settings)
    dump('preprocessing_statistics.json',reports)
    dump('sequence_issues.json',dict(policy=settings['partial_bar_policy'],windows=all_issues,
        inherited_exclusions=dict(outlier=exclusion,malformed_bars=grouping['invalid_bars_excluded']),
        legacy_partial_bar_catalog=grouping['partial_bars']))
    dump('actual_sequence_examples.json',dict(examples=actual_examples,mixed_batch_examples=chosen,
        batch_shapes={g:{k:list(v.shape) for k,v in example_batch[g].items()} for g in ('inputs','targets')},
        real_lengths=[e['real_beats_in_sequence'] for e in chosen]))
    dump('preprocessing_validation_report.json',dict(all_passed=True,split_isolation=True,train_only_vocabulary=True,
        all_shards_round_trip_verified=True,all_real_inputs_and_targets_equal_source_encoder=True,
        no_note_truncation=True,no_new_beat_exclusions=True,source_beats_lost=0,source_beats_duplicated=0,
        eligible_beats=sum(r['real_beats'] for r in reports.values()),
        inherited_excluded_source_beats=raw_source_beats-sum(r['real_beats'] for r in reports.values()),
        raw_source_beats=raw_source_beats,raw_source_bars=raw_source_bars,
        source_sha256=old_validation['source_sha256'],sample_mixed_batch_lengths=[e['real_beats_in_sequence'] for e in chosen]))
    if export_csv:
        from experiments.rahul.src.export_readable_sequences import export_folder
        export_folder(output)
    for name in ('feature_vocabularies.json','target_vocabulary.json','split_manifest.json','preprocessing_config.json','sequence_manifest.csv'):
        artifacts[name]=sha(output/name)
    dump('FINAL_DATASET_READY.json',dict(status='ready',validated_tracks=len(ids),bars_per_sequence=bars_per_sequence,
        sequences=sum(s['sequences'] for s in reports.values()),real_beats=sum(s['real_beats'] for s in reports.values()),
        artifacts=artifacts))
    print(json.dumps(dict(output=str(output),splits={s:{k:v for k,v in r.items() if k not in ('per_track','sample_batch_shapes')} for s,r in reports.items()}),indent=2),flush=True)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-name',default='final_all_tracks_quality_v5_bars_full_melody')
    parser.add_argument('--run-name',default='final_all_tracks_quality_v6_multibar')
    parser.add_argument('--bars-per-sequence',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--skip-readable-csv',action='store_true')
    parser.add_argument('--compatibility-npz',action='store_true')
    args=parser.parse_args()
    run(args.previous_name,args.run_name,args.bars_per_sequence,args.batch_size,not args.skip_readable_csv,args.compatibility_npz)
