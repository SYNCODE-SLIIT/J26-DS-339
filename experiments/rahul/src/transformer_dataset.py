"""PyTorch Dataset and safe NPZ interchange; inputs/targets/metadata stay separate."""
from __future__ import annotations

from pathlib import Path
import bisect
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.rahul.src.create_sequences import sequence_windows, bar_windows, multi_bar_windows
from experiments.rahul.src.encode_features import IGNORE_INDEX


class ChordSequenceDataset(Dataset):
    def __init__(self, tracks, encoder, sequence_length=8, stride=None, debug=False, individual_beats=False, sequence_unit='beats', bars_per_sequence=4, bar_stride=None):
        self.variable_length=sequence_unit=='multi_bar'
        if self.variable_length:
            tracks={t:sorted(tracks[t],key=lambda r:(r.get('start_sec',r['beat_index']),r['beat_index'])) for t in tracks}
        self.tracks,self.encoder,self.debug = tracks,encoder,debug
        self.sequence_length = None if self.variable_length else (1 if individual_beats else sequence_length)
        if sequence_unit not in ('beats','bars','multi_bar'):
            raise ValueError('Sequence unit must be beats, bars or multi_bar')
        if sequence_unit == 'bars' and (individual_beats or stride is not None):
            raise ValueError('Bar sequences use source boundaries, not a beat stride')
        if self.variable_length:
            if stride is not None or individual_beats:raise ValueError('Use bar_stride for multi-bar windows')
            self.windows=multi_bar_windows(tracks,bars_per_sequence,bar_stride)
        else:
            self.windows = (bar_windows(tracks,self.sequence_length) if sequence_unit == 'bars' else
                            sequence_windows(tracks,self.sequence_length,1 if individual_beats else stride))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        rows = self.tracks[window['track_id']][window['start_row']:window['stop_row']]
        encoded = [self.encoder.encode_inputs(r['input_snapshot']) for r in rows]
        targets = [self.encoder.encode_target(r) for r in rows]
        length = len(rows) if self.variable_length else self.sequence_length
        inputs = {k:np.zeros((length,)+np.asarray(v).shape,dtype=np.asarray(v).dtype) for k,v in encoded[0].items()}
        # Genre is track-level [G]; require invariant tags within a sequence/track.
        genre = encoded[0]['genre_features']
        if any(not np.array_equal(e['genre_features'],genre) for e in encoded):
            raise ValueError('Genre tags vary within one track sequence')
        for i,e in enumerate(encoded):
            for k,v in e.items():
                inputs[k][i] = v
        inputs['genre_features'] = genre.copy()
        inputs['beat_attention_mask'] = np.arange(length) < len(rows)
        # Even padded beats have a dummy attention slot; real-note masks stay false.
        inputs['melody_attention_mask'][len(rows):,0] = True
        inputs['melody_silent'][len(rows):] = True
        # Bar indices remain numerical source values; add a separate relative view.
        inputs['relative_bar_number'] = np.zeros(length,dtype=np.float32)
        first_bar = next((r['bar_number'] for r in rows if r['bar_number'] is not None),0)
        for i,r in enumerate(rows):
            if r['bar_number'] is not None:
                inputs['relative_bar_number'][i] = r['bar_number']-first_bar
        target = dict(target_chord_ids=np.full(length,IGNORE_INDEX,dtype=np.int64),
                      target_loss_mask=np.zeros(length,dtype=np.bool_))
        for i,t in enumerate(targets):
            for k,v in t.items():
                target[k][i] = v
        metadata = dict(track_id=window['track_id'],sequence_index=index,
                        beat_indices=torch.tensor([r['beat_index'] for r in rows]+[-1]*(length-len(rows)),dtype=torch.int64))
        if self.variable_length:
            metadata.update({k:v for k,v in window.items() if k not in ('track_id','start_row','stop_row')})
        if self.debug:
            metadata['basic_labels'] = [r['basic_chord'] for r in rows]+['']*(length-len(rows))
            metadata['reference_labels'] = [r['reference_chord_canonical'] for r in rows]+['']*(length-len(rows))
        return dict(inputs={k:torch.from_numpy(np.asarray(v).copy()) for k,v in inputs.items()},
                    targets={k:torch.from_numpy(v) for k,v in target.items()},metadata=metadata)


def collate_sequences(examples):
    """Pad only to this batch's longest beat sequence. PAD=0 is the saved
    Vocabulary contract; N/UNK remain real categories and are never padding.
    """
    if not examples:raise ValueError('Cannot collate an empty batch')
    length=max(len(x['inputs']['beat_attention_mask']) for x in examples)
    result={'inputs':{},'targets':{},'metadata':[]}
    for group in ('inputs','targets'):
        keys=examples[0][group]
        if any(set(x[group])!=set(keys) for x in examples):raise ValueError('Batch feature schema mismatch')
        for key in keys:
            values=[]
            for example in examples:
                value=example[group][key]
                if key=='genre_features':
                    values.append(value)
                    continue
                fill=IGNORE_INDEX if group=='targets' and key=='target_chord_ids' else (True if key=='melody_silent' else 0)
                padded=torch.full((length,)+tuple(value.shape[1:]),fill,dtype=value.dtype,device=value.device)
                padded[:len(value)]=value
                if key=='melody_attention_mask':padded[len(value):,0]=True
                values.append(padded)
            result[group][key]=torch.stack(values)
    for example in examples:
        meta=dict(example['metadata'])
        beats=meta['beat_indices']
        padded=torch.full((length,),-1,dtype=beats.dtype)
        padded[:len(beats)]=beats
        meta['beat_indices']=padded
        meta['sequence_length']=len(beats)
        result['metadata'].append(meta)
    return result


def make_dataloader(dataset, batch_size=4, shuffle=False, seed=42):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=0,
                      generator=torch.Generator().manual_seed(seed),collate_fn=collate_sequences)


def export_npz(path, dataset, variable_length=None):
    """Only numerical arrays and Unicode track IDs; never pickle Python objects."""
    examples = [dataset[i] for i in range(len(dataset))]
    if not examples:
        raise ValueError('No sequences to export')
    variable_length=getattr(dataset,'variable_length',False) if variable_length is None else variable_length
    if variable_length:
        lengths=[len(x['inputs']['beat_attention_mask']) for x in examples]
        arrays={f'{group}__{k}':(np.stack if k=='genre_features' else np.concatenate)([x[group][k].numpy() for x in examples])
                for group in ('inputs','targets') for k in examples[0][group]}
        arrays['format_version']=np.asarray(2,dtype=np.int64)
        arrays['sequence_offsets']=np.asarray([0]+list(np.cumsum(lengths)),dtype=np.int64)
        arrays['track_ids']=np.asarray([x['metadata']['track_id'] for x in examples],dtype=np.str_)
        arrays['beat_indices']=np.concatenate([x['metadata']['beat_indices'].numpy() for x in examples])
        for key in examples[0]['metadata']:
            if key not in ('track_id','beat_indices','sequence_index','basic_labels','reference_labels'):
                arrays['metadata__'+key]=np.asarray([x['metadata'][key] for x in examples])
        np.savez_compressed(Path(path),**arrays)
        return
    if getattr(dataset,'variable_length',False):
        batch=collate_sequences(examples)
        examples=[dict(inputs={k:v[i] for k,v in batch['inputs'].items()},targets={k:v[i] for k,v in batch['targets'].items()},metadata=batch['metadata'][i]) for i in range(len(examples))]
    arrays = {f'{group}__{k}':np.stack([x[group][k].numpy() for x in examples])
              for group in ('inputs','targets') for k in examples[0][group]}
    arrays['track_ids'] = np.asarray([x['metadata']['track_id'] for x in examples],dtype=np.str_)
    arrays['beat_indices'] = np.stack([x['metadata']['beat_indices'].numpy() for x in examples])
    np.savez_compressed(Path(path),**arrays)


class SavedSequenceDataset(Dataset):
    """Load numerical exports with allow_pickle=False and the same Dataset interface."""
    def __init__(self, path):
        path=Path(path)
        self._shards=None
        if path.is_dir():path=path/'dataset_index.json'
        if path.suffix=='.json':
            index=json.loads(path.read_text(encoding='utf-8'))
            if index.get('format')!='ragged_npz_shards_v1':raise ValueError('Unknown dataset index format')
            self._shards=[]
            self._ends=[]
            count=0
            for shard in index['shards']:
                target=(path.parent/shard['path']).resolve()
                if not target.is_relative_to(path.parent.resolve()) or shard['sequences']<1:raise ValueError('Invalid shard path/count')
                if not target.is_file():raise FileNotFoundError(target)
                self._shards.append(target)
                count+=shard['sequences']
                self._ends.append(count)
            if count!=index['sequences']:raise ValueError('Index sequence count mismatch')
            self._cached_index,self._cached_dataset=None,None
            self.variable_length=True
            return
        with np.load(Path(path),allow_pickle=False) as archive:
            self.arrays = {k:archive[k] for k in archive.files}
        count = len(self.arrays['track_ids'])
        if any(v.dtype.hasobject for v in self.arrays.values()):raise ValueError('Object arrays forbidden')
        self.variable_length='sequence_offsets' in self.arrays
        if self.variable_length:
            offsets=self.arrays['sequence_offsets']
            if self.arrays.get('format_version')!=2 or offsets.dtype!=np.int64 or offsets.shape!=(count+1,) or offsets[0]!=0 or not (np.diff(offsets)>0).all():
                raise ValueError('Invalid ragged offsets/version')
            total=int(offsets[-1])
            for key,value in self.arrays.items():
                if key.startswith(('inputs__','targets__')) or key=='beat_indices':
                    expected=count if key=='inputs__genre_features' else total
                    if value.ndim<1 or len(value)!=expected:raise ValueError('Invalid ragged tensor lengths')
                elif key.startswith('metadata__') and len(value)!=count:raise ValueError('Invalid metadata length')
            if not self.arrays['inputs__beat_attention_mask'].all():raise ValueError('Ragged storage must contain only real beats')
        elif any(len(v)!=count for v in self.arrays.values()):
            raise ValueError('Invalid sequence archive')

    def __len__(self):
        if self._shards is not None:return self._ends[-1] if self._ends else 0
        return len(self.arrays['track_ids'])

    def __getitem__(self,index):
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        if self._shards is not None:
            shard=bisect.bisect_right(self._ends,index)
            if self._cached_index!=shard:
                self._cached_dataset=SavedSequenceDataset(self._shards[shard])
                expected=self._ends[shard]-(self._ends[shard-1] if shard else 0)
                if len(self._cached_dataset)!=expected:raise ValueError('Shard count differs from index')
                self._cached_index=shard
            item=self._cached_dataset[index-(self._ends[shard-1] if shard else 0)]
            item['metadata']['sequence_index']=index
            return item
        start,stop=(self.arrays['sequence_offsets'][index:index+2] if self.variable_length else (None,None))
        result={group:{k.split('__',1)[1]:torch.from_numpy((v[start:stop] if self.variable_length and k!='inputs__genre_features' else v[index]).copy())
                       for k,v in self.arrays.items() if k.startswith(group+'__')} for group in ('inputs','targets')}
        result['metadata'] = dict(track_id=str(self.arrays['track_ids'][index]),sequence_index=index,
                                  beat_indices=torch.from_numpy((self.arrays['beat_indices'][start:stop] if self.variable_length else self.arrays['beat_indices'][index]).copy()))
        if self.variable_length:
            result['metadata'].update({k.split('__',1)[1]:v[index].item() for k,v in self.arrays.items() if k.startswith('metadata__')})
        return result
