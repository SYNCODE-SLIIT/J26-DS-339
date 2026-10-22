"""Streaming readable export of fixed or ragged SavedSequenceDataset archives."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.rahul.src.encode_features import CATEGORICAL,FeatureEncoder,IGNORE_INDEX
from experiments.rahul.src.transformer_dataset import SavedSequenceDataset,collate_sequences


def readable_rows(dataset,encoder,split,padded_length=None):
    """All numerical input/target columns; list dimensions remain JSON lists."""
    for s in range(len(dataset)):
        item=dataset[s]
        if padded_length is not None:
            if padded_length<len(item['metadata']['beat_indices']):raise ValueError('Requested CSV padding too short')
            # Explicit compatibility CSV padding only; no dummy item or model change.
            from copy import deepcopy
            padded=deepcopy(item)
            extra=padded_length-len(item['metadata']['beat_indices'])
            if extra:
                import torch
                for group in ('inputs','targets'):
                    for key,value in padded[group].items():
                        if key=='genre_features':continue
                        fill=IGNORE_INDEX if key=='target_chord_ids' else (True if key=='melody_silent' else 0)
                        tail=torch.full((extra,)+tuple(value.shape[1:]),fill,dtype=value.dtype)
                        if key=='melody_attention_mask':tail[:,0]=True
                        padded[group][key]=torch.cat((value,tail))
                padded['metadata']['beat_indices']=torch.cat((item['metadata']['beat_indices'],torch.full((extra,),-1,dtype=torch.int64)))
            item=padded
        meta=item['metadata']
        real=int(item['inputs']['beat_attention_mask'].sum())
        for p in range(len(meta['beat_indices'])):
            if padded_length is None and not item['inputs']['beat_attention_mask'][p]:continue
            row=dict(split=split,track_id=meta['track_id'],sequence_id=s,position_in_sequence=p,
                sequence_start_bar=meta.get('sequence_start_bar',''),sequence_end_bar=meta.get('sequence_end_bar',''),
                bars_in_sequence=meta.get('bars_in_sequence',''),real_beats_in_sequence=real,
                beat_index=int(meta['beat_indices'][p]),bar_number=float(item['inputs']['bar_number'][p]),
                beat_position=float(item['inputs']['beat_position'][p]),is_padding=not bool(item['inputs']['beat_attention_mask'][p]))
            for group in ('inputs','targets'):
                for key,value in item[group].items():
                    row[group+'__'+key]=(value if key=='genre_features' else value[p]).tolist()
            for key,field in CATEGORICAL.items():
                row[key.removesuffix('_ids')+'_decoded']=encoder.vocabs[field].decode(int(row['inputs__'+key]))
            target=row['targets__target_chord_ids']
            row['reference_chord_decoded']='IGNORE' if target==IGNORE_INDEX else encoder.targets.decode(target)
            row['melody_sources_decoded']=[encoder.vocabs['melody_sources'].decode(v) for v in row['inputs__melody_source_ids']]
            row['melody_midi_rounded_decoded']=['PAD' if v==0 else 'UNK' if v==1 else v-2 for v in row['inputs__melody_pitch_ids']]
            row['melody_pitch_classes_decoded']=['PAD' if v==0 else 'UNK' if v==1 else v-2 for v in row['inputs__melody_pitch_class_ids']]
            row['genre_vocabulary']=encoder.genres
            row['genre_labels_decoded']=[g for g,v in zip(encoder.genres,row['inputs__genre_features']) if v!=0]
            yield row,item,p


def export_folder(folder,padded_length=None,verify=True):
    folder=Path(folder)
    destination=folder/'all_tracks_sequences_readable.csv'
    if padded_length is not None:destination=folder/'all_tracks_sequences_readable_padded.csv'
    if destination.exists():raise FileExistsError(destination)
    encoder=FeatureEncoder.load(folder)
    index_rows=[]
    total=0
    columns=None
    for_split={}
    with destination.open('x',encoding='utf-8',newline='') as stream:
        writer=None
        for split in ('train','validation','test'):
            source=folder/f'{split}_sequences.json'
            if not source.exists():source=folder/f'{split}_sequences.npz'
            dataset=SavedSequenceDataset(source)
            counts={}
            for row,item,p in readable_rows(dataset,encoder,split,padded_length):
                if writer is None:
                    columns=list(row)
                    writer=csv.DictWriter(stream,fieldnames=columns)
                    writer.writeheader()
                writer.writerow({k:json.dumps(v,separators=(',',':'),ensure_ascii=False,allow_nan=False) if isinstance(v,(list,bool)) else v for k,v in row.items()})
                count=counts.setdefault(row['track_id'],dict(sequences=0,real_positions=0,padded_positions=0,supervised_positions=0))
                count['sequences']+=int(p==0)
                count['real_positions']+=int(not row['is_padding'])
                count['padded_positions']+=int(row['is_padding'])
                count['supervised_positions']+=int(row['targets__target_loss_mask'])
                total+=1
            index_rows.extend(dict(split=split,track_id=track,**values) for track,values in counts.items())
            for_split[split]=sum(c['real_positions']+c['padded_positions'] for c in counts.values())
            print(f'CSV {split}: {for_split[split]} positions',flush=True)
    checked=0
    if verify:
        with destination.open(encoding='utf-8',newline='') as stream:
            reader=csv.DictReader(stream)
            for split in ('train','validation','test'):
                source=folder/f'{split}_sequences.json'
                if not source.exists():source=folder/f'{split}_sequences.npz'
                dataset=SavedSequenceDataset(source)
                for expected,item,p in readable_rows(dataset,encoder,split,padded_length):
                    actual=next(reader)
                    for key,value in expected.items():
                        if key.startswith(('inputs__','targets__')):
                            group,name=key.split('__',1)
                            original=item[group][name] if name=='genre_features' else item[group][name][p]
                            restored=np.asarray(json.loads(actual[key]),dtype=original.numpy().dtype)
                            assert restored.shape==original.numpy().shape and restored.tobytes()==original.numpy().tobytes(),(split,expected['sequence_id'],p,key)
                        else:
                            serialized=json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False) if isinstance(value,(list,bool)) else str(value)
                            assert actual[key]==serialized,(key,actual[key],serialized)
                    checked+=1
                print(f'CSV verified {split}',flush=True)
            assert next(reader,None) is None
        assert checked==total
    if padded_length is None:
        with (folder/'all_tracks_csv_index.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(index_rows[0]));writer.writeheader();writer.writerows(index_rows)
    suffix='_padded' if padded_length is not None else ''
    report=dict(status='passed' if verify else 'not_verified',rows=total,split_rows=for_split,
        numerical_comparison='Every exported input, target and mask restored to saved dtype and compared byte for byte.',
        padding='Explicit fixed-length inspection only' if padded_length is not None else 'No artificial beat padding rows.',columns=columns)
    (folder/f'readable_csv_validation{suffix}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--padded-length',type=int,default=None)
    args=parser.parse_args()
    export_folder(args.folder,args.padded_length)
