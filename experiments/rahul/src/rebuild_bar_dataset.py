"""Build one source bar per sequence, retaining full source melody arrays."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil

from experiments.rahul import config
from experiments.rahul.src.create_sequences import bar_windows
from experiments.rahul.src.rebuild_four_note_dataset import RetainedTracks
from experiments.rahul.src.encode_features import FeatureEncoder
from experiments.rahul.src.build_final_dataset import export_split


class BarTracks(RetainedTracks):
    excluded_bars = None

    def __getitem__(self, track):
        rows=super().__getitem__(track)
        excluded=(self.excluded_bars or {}).get(track,set())
        return [r for r in rows if r['bar_number'] not in excluded]


def run(previous_name='final_all_tracks_quality_v4_m4', run_name='final_all_tracks_quality_v5_bars_full_melody'):
    csv.field_size_limit(64*1024*1024)
    if any(not n.replace('_','').replace('-','').isalnum() for n in (previous_name,run_name)):
        raise ValueError('Use simple snapshot names')
    base=config.COMPONENT_DIR/'outputs/model_preprocessing'
    previous,output=base/previous_name,base/run_name
    if output.exists(): raise FileExistsError('Existing snapshots protected; choose a new run name')
    read=lambda name: json.loads((previous/name).read_text(encoding='utf-8'))
    ready,settings,validation=read('FINAL_DATASET_READY.json'),read('preprocessing_config.json'),read('preprocessing_validation_report.json')
    for name,expected in ready['artifacts'].items():
        if hashlib.sha256((previous/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Previous artifact changed: {name}')
    for name,expected in validation['source_sha256'].items():
        if hashlib.sha256((config.REPOSITORY_ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Extraction changed: {name}')
    splits=read('split_manifest.json')['splits']
    ids=[t for group in splits.values() for t in group]
    exclusion=read('outlier_exclusion.json')
    tracks=BarTracks(config.REPOSITORY_ROOT/settings['phase1_output_root'],ids,
        exclusion['track_id'],exclusion['outlier_beat_index'],exclusion['sequence_length'])
    # Capacity is structural: support the source meters up to nine beat positions.
    # Real beats are never added, truncated or moved to another bar.
    length=9
    reports,bar_sizes,partial_bars,meters={},Counter(),[],Counter()
    invalid=[]
    excluded_bars={}
    for number,track in enumerate(ids,1):
        rows=tracks[track]
        source=config.REPOSITORY_ROOT/settings['source_root']/f'track_{int(track):07d}'/'beats/beats.csv'
        with source.open(encoding='utf-8',newline='') as f:
            raw={int(r['beat_index']):r for r in csv.DictReader(f)}
        groups={}
        for row in rows:
            groups.setdefault(row['bar_number'],[]).append(row)
        for bar_number,bar in groups.items():
            positions=[r['beat_position'] for r in bar]
            signatures={(raw[r['beat_index']]['meter_numerator'],raw[r['beat_index']]['meter_denominator']) for r in bar}
            reasons=[]
            if bar_number is None or any(p is None or p<1 for p in positions): reasons.append('missing bar/position metadata')
            if any(a>=b for a,b in zip(positions,positions[1:])): reasons.append('duplicate or decreasing beat positions')
            if len(signatures)!=1: reasons.append('meter changes inside source bar')
            if len(signatures)==1 and any(p>int(next(iter(signatures))[0]) for p in positions): reasons.append('position exceeds meter')
            if reasons:
                excluded_bars.setdefault(track,set()).add(bar_number)
                invalid.append(dict(track_id=track,bar_number=bar_number,beat_indices=[r['beat_index'] for r in bar],beat_positions=positions,reasons=reasons))
        rows=[r for r in rows if r['bar_number'] not in excluded_bars.get(track,set())]
        if not rows: raise ValueError(f'No valid bars remain for {track}')
        windows=bar_windows({track:rows},length)
        for w in windows:
            bar=rows[w['start_row']:w['stop_row']]
            signatures={(raw[r['beat_index']]['meter_numerator'],raw[r['beat_index']]['meter_denominator']) for r in bar}
            if len(signatures)!=1: raise ValueError('Time signature changes inside a source bar')
            numerator,denominator=next(iter(signatures))
            expected=int(numerator)
            if expected>length: raise ValueError('Meter exceeds configured bar padding capacity')
            positions=[r['beat_position'] for r in bar]
            if any(p>expected for p in positions): raise ValueError('Beat position exceeds source meter')
            meters[f'{numerator}/{denominator}']+=1
            bar_sizes[len(bar)]+=1
            if positions!=list(range(1,expected+1)):
                partial_bars.append(dict(track_id=track,bar_number=w['bar_number'],meter=f'{numerator}/{denominator}',beat_positions=positions))
        reports[track]=dict(statistics=dict(final_events=len(rows),bar_sequences=len(windows)))
        if number%100==0: print(f'Validated bar metadata: {number}/{len(ids)} tracks',flush=True)
    tracks.excluded_bars=excluded_bars
    print(f'Excluded {len(invalid)} malformed source bars; fitting full-note training encoder',flush=True)
    encoder=FeatureEncoder.fit(tracks,splits['train'])
    observed=encoder.max_notes
    encoder.max_notes=1 << (observed-1).bit_length()
    output.mkdir()
    encoder.save(output)
    stats,checks,manifest={},{},[]
    for split,assigned in splits.items():
        stat,entries,checked=export_split(split,assigned,tracks,encoder,length,output,reports,sequence_unit='bars')
        stats[split],checks[split]=stat,checked
        manifest.extend(entries)
    with (output/'sequence_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    for name in ('split_manifest.json','discovery_manifest.json','track_build_results.json','quality_rebuild_validation.json','outlier_exclusion.json'):
        shutil.copyfile(previous/name,output/name)
    def dump(name,value): (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    new_settings={**settings,'previous_snapshot':previous_name,'sequence_unit':'source_bar',
        'sequence_length':length,'stride':None,'bar_padding_policy':'One source bar per sequence; nine padded positions for meters up to nine. Short/partial bars retain their actual beats with false padding masks; no beats invented.',
        'partial_bar_policy':'Preserve source partial bars and flag in bar_grouping_report.json. Preserve previous outlier exclusion without restoring omitted beats.',
        'vocabulary_policy':'Fit exclusively on retained training bars after all exclusions.',
        'max_notes_per_beat':encoder.max_notes,'observed_training_max_notes':observed,'truncate_notes':False,
        'note_capacity_policy':'Training maximum rounded up to power of two; retain every source note. Fail on held-out overflow instead of truncating or fitting capacity on test data.',
        'invalid_bar_policy':'Exclude entire source bars with duplicate/decreasing positions, invalid meter or missing metadata; record exact beats/reasons. Never fabricate bar alignment.',
        'implementation_sha256':{name:hashlib.sha256((config.COMPONENT_DIR/'src'/name).read_bytes()).hexdigest()
            for name in ('rebuild_bar_dataset.py','create_sequences.py','transformer_dataset.py','build_final_dataset.py','encode_features.py')}}
    grouping=dict(all_passed=True,one_source_bar_per_sequence=True,bar_capacity=length,
        bar_size_distribution=dict(bar_sizes),meter_bar_counts=dict(meters),partial_bars=partial_bars,
        partial_bar_count=len(partial_bars),invalid_bars_excluded=invalid,no_beats_truncated=True,
        no_note_truncation=all(s['notes_truncated']==0 for s in stats.values()),
        preserved_excluded_beat_indices=exclusion['removed_beat_indices'])
    dump('bar_grouping_report.json',grouping)
    dump('preprocessing_config.json',new_settings)
    dump('preprocessing_statistics.json',stats)
    dump('preprocessing_validation_report.json',{**validation,'splits':checks,'bar_grouping':grouping,'all_passed':True,
        'note_truncation_explicit':False,'notes_truncated':0,'no_note_truncation':True})
    dump('FINAL_DATASET_READY.json',dict(status='ready',validated_tracks=len(ids),splits=stats,
        artifacts={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in
            ('train_sequences.npz','validation_sequences.npz','test_sequences.npz','feature_vocabularies.json','target_vocabulary.json','split_manifest.json')}))
    (output/'dataset_build_summary.md').write_text('# One bar per sequence\n\n'+new_settings['bar_padding_policy']+'\n\n'+new_settings['partial_bar_policy']+'\n\n'+json.dumps(dict(splits=stats,bar_sizes=dict(bar_sizes),meter_bars=dict(meters),partial_bars=len(partial_bars)),indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(output),splits=stats,bar_sizes=dict(bar_sizes),partial_bars=len(partial_bars)),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-name',default='final_all_tracks_quality_v4_m4')
    parser.add_argument('--run-name',default='final_all_tracks_quality_v5_bars_full_melody')
    args=parser.parse_args()
    run(args.previous_name,args.run_name)
