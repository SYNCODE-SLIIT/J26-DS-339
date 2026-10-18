"""PyTorch Dataset and safe NPZ interchange; inputs/targets/metadata stay separate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.rahul.src.create_sequences import sequence_windows
from experiments.rahul.src.encode_features import IGNORE_INDEX


class ChordSequenceDataset(Dataset):
    def __init__(self, tracks, encoder, sequence_length=8, stride=None, debug=False, individual_beats=False):
        self.tracks,self.encoder,self.debug = tracks,encoder,debug
        self.sequence_length = 1 if individual_beats else sequence_length
        self.windows = sequence_windows(tracks,self.sequence_length,1 if individual_beats else stride)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        rows = self.tracks[window['track_id']][window['start_row']:window['stop_row']]
        encoded = [self.encoder.encode_inputs(r['input_snapshot']) for r in rows]
        targets = [self.encoder.encode_target(r) for r in rows]
        length = self.sequence_length
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
        if self.debug:
            metadata['basic_labels'] = [r['basic_chord'] for r in rows]+['']*(length-len(rows))
            metadata['reference_labels'] = [r['reference_chord_canonical'] for r in rows]+['']*(length-len(rows))
        return dict(inputs={k:torch.from_numpy(np.asarray(v).copy()) for k,v in inputs.items()},
                    targets={k:torch.from_numpy(v) for k,v in target.items()},metadata=metadata)


def collate_sequences(examples):
    return {'inputs':{k:torch.stack([x['inputs'][k] for x in examples]) for k in examples[0]['inputs']},
            'targets':{k:torch.stack([x['targets'][k] for x in examples]) for k in examples[0]['targets']},
            'metadata':[x['metadata'] for x in examples]}


def make_dataloader(dataset, batch_size=4, shuffle=False, seed=42):
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=0,
                      generator=torch.Generator().manual_seed(seed),collate_fn=collate_sequences)


def export_npz(path, dataset):
    """Only numerical arrays and Unicode track IDs; never pickle Python objects."""
    examples = [dataset[i] for i in range(len(dataset))]
    if not examples:
        raise ValueError('No sequences to export')
    arrays = {f'{group}__{k}':np.stack([x[group][k].numpy() for x in examples])
              for group in ('inputs','targets') for k in examples[0][group]}
    arrays['track_ids'] = np.asarray([x['metadata']['track_id'] for x in examples],dtype=np.str_)
    arrays['beat_indices'] = np.stack([x['metadata']['beat_indices'].numpy() for x in examples])
    np.savez_compressed(Path(path),**arrays)


class SavedSequenceDataset(Dataset):
    """Load numerical exports with allow_pickle=False and the same Dataset interface."""
    def __init__(self, path):
        with np.load(Path(path),allow_pickle=False) as archive:
            self.arrays = {k:archive[k] for k in archive.files}
        count = len(self.arrays['track_ids'])
        if any(len(v)!=count or v.dtype.hasobject for v in self.arrays.values()):
            raise ValueError('Invalid sequence archive')

    def __len__(self):
        return len(self.arrays['track_ids'])

    def __getitem__(self,index):
        result = {group:{k.split('__',1)[1]:torch.from_numpy(v[index].copy())
                         for k,v in self.arrays.items() if k.startswith(group+'__')} for group in ('inputs','targets')}
        result['metadata'] = dict(track_id=str(self.arrays['track_ids'][index]),sequence_index=index,
                                  beat_indices=torch.from_numpy(self.arrays['beat_indices'][index].copy()))
        return result
