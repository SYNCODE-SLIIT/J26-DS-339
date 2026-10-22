import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.rahul.src.create_sequences import multi_bar_windows,validate_splits
from experiments.rahul.src.encode_features import FeatureEncoder
from experiments.rahul.src.transformer_dataset import ChordSequenceDataset,SavedSequenceDataset,export_npz,collate_sequences
from experiments.rahul.src.export_readable_sequences import readable_rows
from experiments.rahul.tests.test_model_preprocessing import fixture_rows,load_fixture


class MultiBarTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def data(self,sizes,track='101',pitches=(60.25,64),targets=None):
        raw=fixture_rows(track=track,count=sum(sizes),pitches=pitches)
        offset=0
        for bar,size in enumerate(sizes):
            for p in range(size):
                raw[offset+p]['bar_number']=bar
                raw[offset+p]['beat_position']=p+1
            offset+=size
        if targets:
            for row,target in zip(raw,targets):
                row['reference_chord']=target
                row['reference_chord_canonical']=target
        rows=load_fixture(raw,self.root)
        rows[-1]['bar_end_confirmed']=True
        return rows

    def dataset(self,rows,bars=4,encoder=None):
        tracks={rows[0]['track_id']:rows}
        encoder=encoder or FeatureEncoder.fit(tracks,list(tracks))
        return ChordSequenceDataset(tracks,encoder,sequence_unit='multi_bar',bars_per_sequence=bars)

    def test_four_four_four_bars(self):
        ds=self.dataset(self.data([4]*4))
        self.assertEqual(len(ds),1)
        self.assertEqual(ds[0]['inputs']['basic_chord_ids'].shape,(16,))
        self.assertTrue(ds[0]['metadata']['complete_four_bar_window'])

    def test_four_three_four_bars(self):
        self.assertEqual(len(self.dataset(self.data([3]*4))[0]['metadata']['beat_indices']),12)

    def test_four_five_four_bars(self):
        self.assertEqual(len(self.dataset(self.data([5]*4))[0]['metadata']['beat_indices']),20)

    def test_mixed_meters_use_actual_events(self):
        item=self.dataset(self.data([4,3,4,5]))[0]
        self.assertEqual(len(item['metadata']['beat_indices']),16)
        self.assertEqual(item['inputs']['bar_number'].tolist(),[0]*4+[1]*3+[2]*4+[3]*5)
        self.assertEqual(item['inputs']['relative_bar_number'].tolist(),[0]*4+[1]*3+[2]*4+[3]*5)

    def test_final_two_bars_preserved(self):
        ds=self.dataset(self.data([4]*6))
        self.assertEqual([len(ds[i]['metadata']['beat_indices']) for i in range(len(ds))],[16,8])
        self.assertTrue(ds[1]['metadata']['final_partial_window'])
        self.assertEqual(ds[1]['metadata']['bars_in_sequence'],2)

    def test_internal_chord_changes_preserved(self):
        rows=self.data([4],targets=['E:min','C:maj','C:maj','G:maj'])
        ds=self.dataset(rows)
        item=ds[0]
        self.assertEqual([ds.encoder.targets.decode(i) for i in item['targets']['target_chord_ids'].tolist()],['E:min','C:maj','C:maj','G:maj'])

    def test_ambiguous_real_beat_stays_in_context(self):
        rows=self.data([4])
        rows[1]['target_valid']=False
        item=self.dataset(rows)[0]
        self.assertTrue(item['inputs']['beat_attention_mask'][1])
        self.assertFalse(item['targets']['target_loss_mask'][1])
        self.assertEqual(item['targets']['target_chord_ids'][1],-100)

    def test_dynamic_batch_padding_12_16_20_and_all_features(self):
        items=[self.dataset(self.data([n]*4,track=str(n)))[0] for n in (3,4,5)]
        batch=collate_sequences(items)
        self.assertEqual(batch['inputs']['basic_chord_ids'].shape,(3,20))
        self.assertEqual(batch['inputs']['genre_features'].shape,(3,3))
        for i,length in enumerate((12,16,20)):
            for group in ('inputs','targets'):
                for key,value in items[i][group].items():
                    actual=batch[group][key][i] if key=='genre_features' else batch[group][key][i,:length]
                    self.assertTrue(torch.equal(actual,value),key)
            self.assertEqual(batch['inputs']['beat_attention_mask'][i].tolist(),[True]*length+[False]*(20-length))
            self.assertTrue((batch['targets']['target_chord_ids'][i,length:]==-100).all())
            self.assertFalse(batch['targets']['target_loss_mask'][i,length:].any())
            self.assertFalse(batch['inputs']['melody_note_mask'][i,length:].any())
            self.assertTrue(batch['inputs']['melody_silent'][i,length:].all())
            self.assertTrue(batch['inputs']['melody_attention_mask'][i,length:,0].all())
        short=collate_sequences(items[:1])
        self.assertEqual(short['inputs']['beat_attention_mask'].shape,(1,12))

    def test_track_isolation_and_no_loss_or_duplicate_events(self):
        a,b=self.data([4]*6,'101'),self.data([3]*5,'102')
        tracks={'101':a,'102':b}
        enc=FeatureEncoder.fit(tracks,['101'])
        ds=ChordSequenceDataset(tracks,enc,sequence_unit='multi_bar')
        coverage={t:[] for t in tracks}
        for i in range(len(ds)):
            item=ds[i]
            coverage[item['metadata']['track_id']]+=item['metadata']['beat_indices'].tolist()
        self.assertEqual(coverage,{'101':list(range(24)),'102':list(range(15))})

    def test_sorting_and_timestamp_validation(self):
        rows=self.data([4]*4)
        ds=self.dataset(rows[::-1])
        self.assertEqual(ds[0]['metadata']['beat_indices'].tolist(),list(range(16)))
        bad=copy.deepcopy(rows)
        bad[1]['start_sec']=0
        with self.assertRaisesRegex(ValueError,'timestamps'):
            self.dataset(bad)

    def test_silence_is_not_padding(self):
        ds=self.dataset(self.data([3],pitches=()))
        item=ds[0]
        self.assertTrue(item['inputs']['beat_attention_mask'].all())
        self.assertTrue(item['inputs']['melody_silent'].all())
        self.assertFalse(item['inputs']['melody_note_mask'].any())
        self.assertTrue(item['inputs']['melody_attention_mask'][:,0].all())

    def test_split_protection(self):
        with self.assertRaises(ValueError):
            validate_splits(dict(splits=dict(train=['101'],validation=['101'],test=['102'])),['101','102'])

    def test_vocab_stability_and_unknown_heldout_target(self):
        train=self.data([4]*4,'101')
        held=self.data([3]*4,'102',targets=['D:maj']*12)
        enc=FeatureEncoder.fit({'101':train},['101'])
        before=enc.targets.state()
        item=self.dataset(held,encoder=enc)[0]
        self.assertTrue((item['targets']['target_chord_ids']==-100).all())
        self.assertEqual(enc.targets.state(),before)
        self.assertTrue((item['inputs']['basic_chord_ids']==enc.vocabs['basic_chord_canonical'].encode('C')).all())

    def test_finite_correct_dtypes_and_no_four_note_cap(self):
        item=self.dataset(self.data([4]*4,pitches=tuple(range(60,67))))[0]
        self.assertEqual(item['inputs']['melody_note_mask'].sum().item(),16*7)
        self.assertEqual(item['inputs']['melody_timing'].dtype,torch.float32)
        self.assertEqual(item['targets']['target_chord_ids'].dtype,torch.int64)
        for group in ('inputs','targets'):
            self.assertTrue(all(torch.isfinite(v).all() for v in item[group].values()))

    def test_ragged_roundtrip_all_tensors_and_metadata(self):
        ds=self.dataset(self.data([4,3,4,5,3,3]))
        path=self.root/'ragged.npz'
        export_npz(path,ds)
        loaded=SavedSequenceDataset(path)
        with np.load(path,allow_pickle=False) as z:
            self.assertEqual(z['sequence_offsets'].tolist(),[0,16,22])
            self.assertEqual(z['inputs__basic_chord_ids'].shape,(22,))
            self.assertTrue(all(not z[k].dtype.hasobject for k in z.files))
        for i in range(len(ds)):
            a,b=ds[i],loaded[i]
            for group in ('inputs','targets'):
                for key in a[group]:self.assertTrue(torch.equal(a[group][key],b[group][key]),key)
            for key in a['metadata']:
                if key=='beat_indices':self.assertTrue(torch.equal(a['metadata'][key],b['metadata'][key]))
                else:self.assertEqual(a['metadata'][key],b['metadata'][key])

    def test_sharded_loader_and_fixed_compatibility(self):
        ds=self.dataset(self.data([4]*6))
        export_npz(self.root/'shard.npz',ds)
        index=dict(format='ragged_npz_shards_v1',sequences=2,shards=[dict(path='shard.npz',sequences=2)])
        (self.root/'index.json').write_text(json.dumps(index))
        loaded=SavedSequenceDataset(self.root/'index.json')
        self.assertEqual(len(loaded),2)
        self.assertEqual(len(loaded[-1]['metadata']['beat_indices']),8)
        export_npz(self.root/'compat.npz',ds,variable_length=False)
        legacy=SavedSequenceDataset(self.root/'compat.npz')
        self.assertEqual(legacy[1]['inputs']['beat_attention_mask'].tolist(),[True]*8+[False]*8)

    def test_source_neighbors_cross_sequence_boundaries(self):
        ds=self.dataset(self.data([4]*6))
        vocab=ds.encoder.vocabs['basic_chord_canonical']
        self.assertEqual(vocab.decode(int(ds[0]['inputs']['next_basic_chord_ids'][-1])),'C')
        self.assertEqual(vocab.decode(int(ds[1]['inputs']['previous_basic_chord_ids'][0])),'C')
        self.assertEqual(vocab.decode(int(ds[0]['inputs']['previous_basic_chord_ids'][0])),'START')
        self.assertEqual(vocab.decode(int(ds[1]['inputs']['next_basic_chord_ids'][-1])),'END')

    def test_source_gap_not_bridged(self):
        rows=self.data([4]*6)
        del rows[8:12]
        ds=self.dataset(rows)
        self.assertEqual([ds[i]['metadata']['bars_in_sequence'] for i in range(len(ds))],[2,3])
        self.assertEqual(sum(len(ds[i]['metadata']['beat_indices']) for i in range(len(ds))),20)

    def test_partial_and_unusual_compound_bar_not_dropped(self):
        rows=self.data([2,2,2,2])
        # Actual two-event bars must not be expanded to a numerator of six.
        for row in rows:row['meter_numerator']=6
        rows[0]['beat_position']=2
        rows[1]['beat_position']=3
        ds=self.dataset(rows)
        self.assertEqual(len(ds[0]['metadata']['beat_indices']),8)
        self.assertTrue(ds[0]['metadata']['contains_partial_bar'])

    def test_bar_metadata_validation_and_stride(self):
        rows=self.data([4]*4)
        rows[4]['bar_number']=0
        with self.assertRaises(ValueError):self.dataset(rows)
        with self.assertRaises(ValueError):multi_bar_windows({'101':self.data([4]*4)},4,2)

    def test_readable_export_contains_real_positions_and_decoded_targets(self):
        ds=self.dataset(self.data([4]*6))
        rows=list(readable_rows(ds,ds.encoder,'train'))
        self.assertEqual(len(rows),24)
        self.assertTrue(all(not r['is_padding'] for r,_,_ in rows))
        self.assertEqual(rows[16][0]['bars_in_sequence'],2)
        self.assertEqual(rows[0][0]['reference_chord_decoded'],'C:maj')
        padded=list(readable_rows(ds,ds.encoder,'train',padded_length=16))
        self.assertEqual(len(padded),32)
        self.assertEqual(sum(r['is_padding'] for r,_,_ in padded),8)

    def test_corrupt_offsets_are_rejected_without_pickle(self):
        ds=self.dataset(self.data([4]*6))
        export_npz(self.root/'good.npz',ds)
        with np.load(self.root/'good.npz',allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        arrays['sequence_offsets']=np.array([0,16,16],dtype=np.int64)
        np.savez_compressed(self.root/'bad.npz',**arrays)
        with self.assertRaisesRegex(ValueError,'offsets'):
            SavedSequenceDataset(self.root/'bad.npz')

    def test_shard_index_cannot_escape_dataset_directory(self):
        path=self.root/'index.json'
        path.write_text(json.dumps(dict(format='ragged_npz_shards_v1',sequences=1,
            shards=[dict(path='../outside.npz',sequences=1)])))
        with self.assertRaisesRegex(ValueError,'path'):
            SavedSequenceDataset(path)
