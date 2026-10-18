"""Small synthetic fixtures plus read-only Track 382 integration checks."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.rahul import config
from experiments.rahul.src import build_synthetic_dataset as phase1
from experiments.rahul.src.create_sequences import sequence_windows, split_tracks, validate_splits
from experiments.rahul.src.encode_features import (
    CATEGORICAL, IGNORE_INDEX, FeatureEncoder, Vocabulary, load_events, parse_list, verify_schema,
)
from experiments.rahul.src.transformer_dataset import (
    ChordSequenceDataset, SavedSequenceDataset, export_npz, make_dataloader,
)
from experiments.rahul.src import preprocess_for_transformer as pipeline


def fixture_rows(track='101', count=3, pitches=(60.25,64), basic='C', reference='C:maj'):
    rows=[]
    for i in range(count):
        quality = 'N' if basic=='N' else 'maj'
        basic_fields=dict(basic_chord=basic,basic_root='' if basic=='N' else basic,basic_quality=quality)
        roman,function=phase1.context(basic_fields,'C','major')
        n=len(pitches)
        rows.append(dict(track_id=track,beat_index=i,start_sec=i*.5,end_sec=(i+1)*.5,event_duration=.5,
            bar_number=i//4,beat_position=i%4+1,**basic_fields,simplification_status='supported',
            basic_chord_canonical=basic,basic_root_pitch_class=None if basic=='N' else 0,melody=[],
            melody_midi=list(pitches),melody_pitch_classes=[round(p)%12 for p in pitches],
            melody_durations=[1.0]*n,melody_relative_start=[-.25]*n,melody_overlap_durations=[.5]*n,
            melody_sources=['fixture']*n,genre_tags=['jazz','classical'],key='C',mode='major',key_confidence='high',
            roman_numeral=roman,harmonic_function=function,reference_chord=reference,reference_chord_canonical=reference,
            reference_root_pitch_class=0,reference_quality='maj',reference_bass_degree=None,
            reference_overlap_ratio=1.,reference_candidates=[],reference_ambiguous=False,target_valid=True,
            source_chord_event_id=i,basic_source='fixture',previous_basic_chord=basic if i else 'START',
            next_basic_chord=basic if i+1<count else 'END'))
    return rows


def fixture_schema(rows):
    return dict(input_columns=phase1.INPUTS,target_columns=phase1.TARGETS,
                columns={k:dict(role='input' if k in phase1.INPUTS else 'target' if k in phase1.TARGETS else 'audit') for k in rows[0]})


def load_fixture(rows, parent):
    folder=Path(parent)/f"track_{int(rows[0]['track_id']):07d}"
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"track_{int(rows[0]['track_id']):07d}_training_events.csv"
    phase1.write_csv(path,rows)
    phase1.dump(folder/'dataset_schema.json',fixture_schema(rows))
    return load_events(path)[0]


class ModelPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.rows=load_fixture(fixture_rows(),self.root)
        self.tracks={'101':self.rows}
        self.encoder=FeatureEncoder.fit(self.tracks,['101'])

    def test_json_parsing(self):
        self.assertEqual(parse_list('[1,2]','notes'),[1,2])
        self.assertEqual(parse_list('[]','notes'),[])
        for value in ('', '{}', 'null', '1', '__import__("os")', '[NaN,'):
            with self.subTest(value=value),self.assertRaises(ValueError): parse_list(value,'notes')

    def test_vocabulary_roundtrip_and_order(self):
        a=Vocabulary.fit(['G','C','G'])
        b=Vocabulary.fit(['C','G'])
        self.assertEqual(a.state(),b.state())
        for value in ('G','C','PAD','UNK','N','START','END','unknown'):
            self.assertEqual(a.decode(a.encode(value)),value)
        self.assertEqual(a.encode('unseen'),1)
        self.assertNotEqual(a.encode('N'),a.encode('PAD'))

    def test_shared_chord_vocab_and_track_context(self):
        mappings=[self.encoder.vocabs[f].state() for f in ('basic_chord_canonical','previous_basic_chord','next_basic_chord')]
        self.assertEqual(mappings[0],mappings[1]); self.assertEqual(mappings[0],mappings[2])
        first=self.encoder.encode_inputs(self.rows[0])
        last=self.encoder.encode_inputs(self.rows[-1])
        self.assertEqual(first['previous_basic_chord_ids'],3)
        self.assertEqual(last['next_basic_chord_ids'],4)

    def test_fractional_absolute_pitch_and_pc(self):
        result=self.encoder.encode_inputs(self.rows[0])
        np.testing.assert_array_equal(result['melody_pitch_ids'],[62,66])
        np.testing.assert_array_equal(result['melody_pitch_class_ids'],[2,6])
        self.assertAlmostEqual(float(result['melody_pitch_values'][0]),60.25/127,places=6)

    def test_midi_zero_is_real_note(self):
        rows=load_fixture(fixture_rows(track='102',pitches=(0,)),self.root)
        encoded=self.encoder.encode_inputs(rows[0])
        self.assertEqual(encoded['melody_pitch_ids'][0],2)
        self.assertTrue(encoded['melody_note_mask'][0])
        self.assertEqual(encoded['melody_pitch_ids'][1],0)

    def test_negative_relative_timing_and_conversion(self):
        timing=self.encoder.encode_inputs(self.rows[0])['melody_timing']
        np.testing.assert_allclose(timing,[[2,-.5,1],[2,-.5,1]])

    def test_empty_melody_safe_attention(self):
        rows=load_fixture(fixture_rows(track='102',pitches=()),self.root)
        encoder=FeatureEncoder.fit({'102':rows},['102'])
        self.assertEqual(encoder.max_notes,1)
        encoded=encoder.encode_inputs(rows[0])
        self.assertFalse(encoded['melody_note_mask'].any())
        self.assertTrue(encoded['melody_attention_mask'][0])
        self.assertTrue(encoded['melody_silent'])
        self.assertTrue(np.isfinite(encoded['melody_timing']).all())
        logits=torch.zeros(encoder.max_notes)
        weights=torch.softmax(logits.masked_fill(~torch.from_numpy(encoded['melody_attention_mask']),float('-inf')),dim=-1)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(encoder.statistics(rows)['silent_beats'],3)

    def test_variable_note_padding(self):
        rows=load_fixture(fixture_rows(track='102',pitches=(60,)),self.root)
        encoded=self.encoder.encode_inputs(rows[0])
        np.testing.assert_array_equal(encoded['melody_note_mask'],[True,False])
        self.assertEqual(encoded['melody_pitch_ids'][1],0)

    def test_explicit_overflow_and_truncation(self):
        with self.assertRaises(ValueError): FeatureEncoder.fit(self.tracks,['101'],max_notes=1)
        small=FeatureEncoder.fit(self.tracks,['101'],max_notes=1,truncate=True)
        np.testing.assert_array_equal(small.encode_inputs(self.rows[0])['melody_pitch_ids'],[62])
        self.assertEqual(small.statistics(self.rows)['notes_truncated'],3)
        overflow=load_fixture(fixture_rows(track='102',pitches=(60,62,64)),self.root)
        with self.assertRaises(ValueError): self.encoder.encode_inputs(overflow[0])

    def test_multigenre_and_unseen_genre(self):
        encoded=self.encoder.encode_inputs(self.rows[0])
        np.testing.assert_array_equal(encoded['genre_features'],[0,1,1])
        row=copy.deepcopy(self.rows[0]); row['genre_tags']=['jazz','metal']
        np.testing.assert_array_equal(self.encoder.encode_inputs(row)['genre_features'],[1,0,1])
        row['genre_tags']=[]
        self.assertFalse(self.encoder.encode_inputs(row)['genre_features'].any())

    def test_no_chord_target_and_missing_root(self):
        rows=load_fixture(fixture_rows(track='102',basic='N',reference='N'),self.root)
        encoder=FeatureEncoder.fit({'102':rows},['102'])
        encoded=encoder.encode_inputs(rows[0])
        self.assertEqual(encoded['basic_chord_ids'],2)
        self.assertEqual(encoded['basic_root_ids'],2)
        self.assertEqual(encoded['roman_ids'],2)
        self.assertEqual(encoded['harmonic_function_ids'],5)
        self.assertEqual(encoder.encode_target(rows[0])['target_chord_ids'],2)
        self.assertTrue(encoder.encode_target(rows[0])['target_loss_mask'])

    def test_missing_basic_and_unknown_categories(self):
        row=copy.deepcopy(self.rows[0]); row.update(basic_chord='',basic_chord_canonical='',basic_root='',basic_quality='',key='Unseen')
        result=self.encoder.encode_inputs(row)
        self.assertEqual(result['basic_chord_ids'],1)
        self.assertEqual(result['basic_root_ids'],1)
        self.assertEqual(result['key_ids'],1)

    def test_unknown_target_never_becomes_trainable(self):
        row=copy.deepcopy(self.rows[0]); row.update(reference_chord='G:7',reference_chord_canonical='G:7')
        result=self.encoder.encode_target(row)
        self.assertEqual(result['target_chord_ids'],IGNORE_INDEX)
        self.assertFalse(result['target_loss_mask'])
        self.assertEqual(self.encoder.statistics([row])['unseen_targets'],{'G:7':1})

    def test_invalid_and_unsupported_targets(self):
        for changes in (dict(target_valid=False),dict(reference_chord='C:mystery',reference_chord_canonical='C:mystery'),
                        dict(reference_chord='',reference_chord_canonical=''),dict(reference_chord_canonical='D:maj')):
            row={**self.rows[0],**changes}
            self.assertEqual(self.encoder.encode_target(row)['target_chord_ids'],IGNORE_INDEX)
            self.assertFalse(self.encoder.encode_target(row)['target_loss_mask'])

    def test_inputs_invariant_to_target_audit_mutation(self):
        original=self.encoder.encode_inputs(self.rows[0])
        changed=copy.deepcopy(self.rows[0])
        for field in ('reference_chord','reference_chord_canonical','reference_quality','reference_bass_degree',
                      'reference_candidates','reference_overlap_ratio','target_valid','source_chord_event_id','melody'):
            changed[field]='arbitrary target/audit payload'
        result=self.encoder.encode_inputs(changed)
        for key in original:
            np.testing.assert_array_equal(original[key],result[key])
        schema=fixture_schema(fixture_rows()); schema['input_columns']=schema['input_columns']+['reference_quality']
        with self.assertRaises(ValueError): verify_schema(schema)

    def test_train_only_fit_does_not_see_heldout(self):
        heldout=load_fixture(fixture_rows(track='102',basic='D',reference='D:maj'),self.root)
        for row in heldout:
            row['genre_tags']=['unseen']; row['input_snapshot']['genre_tags']=['unseen']
        tracks={**self.tracks,'102':heldout}
        encoder=FeatureEncoder.fit(tracks,['101'])
        self.assertNotIn('D',encoder.vocabs['basic_chord_canonical'].tokens)
        self.assertNotIn('D:maj',encoder.targets.tokens)
        self.assertNotIn('unseen',encoder.genres)
        self.assertFalse(encoder.encode_target(heldout[0])['target_loss_mask'])

    def test_context_validation_rejects_target_derived_roman(self):
        rows=fixture_rows(); rows[0]['roman_numeral']='V'
        with self.assertRaisesRegex(ValueError,'context'): load_fixture(rows,self.root)

    def test_malformed_numerical_values_fail_explicitly(self):
        for field,value in (('melody_midi',[float('nan'),64]),('melody_midi',[-1,64]),
                            ('melody_durations',[None,1]),('melody_overlap_durations',[2,2]),
                            ('event_duration',0),('melody_pitch_classes',[12,4]),
                            ('melody_relative_start',[float('inf'),0]),('melody_durations',[1])):
            rows=fixture_rows(); rows[0][field]=value
            # JSON encoder itself rejects nonfinite data as well as the loader.
            with self.subTest(field=field,value=value),self.assertRaises(ValueError): load_fixture(rows,self.root)

    def test_track_order_and_padding(self):
        ds=ChordSequenceDataset(self.tracks,self.encoder,sequence_length=8,debug=True)
        self.assertEqual(len(ds),1)
        sample=ds[0]
        self.assertEqual(sample['metadata']['beat_indices'].tolist(),[0,1,2,-1,-1,-1,-1,-1])
        self.assertEqual(sample['inputs']['beat_attention_mask'].sum(),3)
        self.assertFalse(sample['targets']['target_loss_mask'][3:].any())
        self.assertTrue((sample['targets']['target_chord_ids'][3:]==IGNORE_INDEX).all())
        self.assertFalse(sample['inputs']['melody_note_mask'][3:].any())
        self.assertTrue(sample['inputs']['melody_attention_mask'].any(dim=-1).all())
        self.assertIn('reference_labels',sample['metadata'])
        self.assertNotIn('reference_labels',sample['inputs'])

    def test_invalid_beats_remain_context(self):
        rows=copy.deepcopy(self.rows); rows[1]['target_valid']=False
        sample=ChordSequenceDataset({'101':rows},self.encoder,sequence_length=3)[0]
        self.assertTrue(sample['inputs']['beat_attention_mask'][1])
        self.assertFalse(sample['targets']['target_loss_mask'][1])
        self.assertNotEqual(sample['inputs']['basic_chord_ids'][1],0)

    def test_track_boundaries_and_window_lengths(self):
        second=load_fixture(fixture_rows(track='102',count=10),self.root)
        tracks={**self.tracks,'102':second}
        for length in (8,16,32):
            ds=ChordSequenceDataset(tracks,self.encoder,length)
            for i in range(len(ds)):
                item=ds[i]
                self.assertEqual(len(item['metadata']['beat_indices']),length)
                self.assertIn(item['metadata']['track_id'],tracks)
                source={r['beat_index'] for r in tracks[item['metadata']['track_id']]}
                self.assertTrue(set(item['metadata']['beat_indices'][item['inputs']['beat_attention_mask']].tolist())<=source)
        debug=ChordSequenceDataset(tracks,self.encoder,individual_beats=True)
        self.assertEqual(len(debug),13)

    def test_overlap_preserves_all_beats(self):
        ds=ChordSequenceDataset(self.tracks,self.encoder,sequence_length=2,stride=1)
        self.assertEqual(len(ds),3)
        with self.assertRaises(ValueError): sequence_windows(self.tracks,2,3)
        backwards=copy.deepcopy(self.tracks); backwards['101'].reverse()
        with self.assertRaises(ValueError): sequence_windows(backwards)

    def test_batch_shapes_finite_and_dtypes(self):
        ds=ChordSequenceDataset(self.tracks,self.encoder,sequence_length=2,stride=2)
        batch=next(iter(make_dataloader(ds,batch_size=2)))
        self.assertEqual(tuple(batch['inputs']['melody_timing'].shape),(2,2,2,3))
        self.assertEqual(tuple(batch['inputs']['genre_features'].shape),(2,3))
        self.assertEqual(batch['targets']['target_chord_ids'].dtype,torch.int64)
        self.assertEqual(batch['inputs']['melody_note_mask'].dtype,torch.bool)
        for group in ('inputs','targets'):
            self.assertTrue(all(torch.isfinite(v).all() for v in batch[group].values()))

    def test_track_splits_isolated_deterministic(self):
        ids=[str(i) for i in range(10)]
        identities={'0':['artist:a'],'1':['artist:a'],'2':['recording:b'],'3':['recording:b']}
        a=split_tracks(ids,seed=19,identities=identities)
        self.assertEqual(a,split_tracks(ids[::-1],seed=19,identities=identities))
        owners={t:s for s,ts in a['splits'].items() for t in ts}
        self.assertEqual(owners['0'],owners['1']); self.assertEqual(owners['2'],owners['3'])
        self.assertTrue(all(a['splits'].values()))
        bad=copy.deepcopy(a); bad['splits']['test'].append(bad['splits']['train'][0])
        with self.assertRaises(ValueError): validate_splits(bad,ids)

    def test_single_track_requires_development_mode(self):
        with self.assertRaises(ValueError): split_tracks(['101'])
        split=split_tracks(['101'],development_only=True)
        self.assertEqual(split['splits'],{'train':['101'],'validation':[],'test':[]})
        with self.assertRaises(ValueError): split_tracks(['1','2','3'],identities={t:['same'] for t in ('1','2','3')})

    def test_saved_encoder_reload_and_fixed_seed_batches(self):
        self.encoder.save(self.root)
        restored=FeatureEncoder.load(self.root)
        for row in self.rows:
            for k,v in self.encoder.encode_inputs(row).items():
                np.testing.assert_array_equal(v,restored.encode_inputs(row)[k])
        ds=ChordSequenceDataset(self.tracks,self.encoder,individual_beats=True)
        a=list(make_dataloader(ds,1,shuffle=True,seed=7))
        b=list(make_dataloader(ds,1,shuffle=True,seed=7))
        self.assertEqual([x['metadata'][0]['beat_indices'].item() for x in a],
                         [x['metadata'][0]['beat_indices'].item() for x in b])

    def test_npz_roundtrip_without_pickle(self):
        ds=ChordSequenceDataset(self.tracks,self.encoder,2)
        path=self.root/'sequences.npz'
        export_npz(path,ds)
        restored=SavedSequenceDataset(path)
        for i in range(len(ds)):
            self.assertEqual(restored[i]['metadata']['track_id'],ds[i]['metadata']['track_id'])
            for group in ('inputs','targets'):
                for key in ds[i][group]:
                    self.assertTrue(torch.equal(ds[i][group][key],restored[i][group][key]))
        with np.load(path,allow_pickle=False) as arrays:
            self.assertTrue(all(not arrays[k].dtype.hasobject for k in arrays.files))

    def test_multitrack_end_to_end(self):
        for track in ('102','103','104'):
            load_fixture(fixture_rows(track=track,count=5),self.root)
        output=self.root/'phase2'
        with patch.object(pipeline,'DEFAULT_OUTPUT',output),contextlib.redirect_stdout(io.StringIO()):
            result=pipeline.run(['101','102','103','104'],input_root=self.root,output_dir=output,sequence_length=2)
        self.assertTrue(result['report']['all_passed'])
        manifest=json.loads((output/'split_manifest.json').read_text())
        self.assertEqual(set(result['encoder'].training_tracks),set(manifest['splits']['train']))
        self.assertTrue(all((output/f'{s}_sequences.npz').exists() for s in ('train','validation','test')))

    def test_track382_sources_unchanged_and_actual_statistics(self):
        folder=config.OUTPUT_DIR/'synthetic_dataset/track_0000382'
        paths=list(folder.glob('*'))
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
        rows,_=load_events(folder/'track_0000382_training_events.csv')
        encoder=FeatureEncoder.fit({'382':rows},['382'])
        stats=encoder.statistics(rows)
        self.assertEqual(stats['beats'],428)
        self.assertEqual(stats['source_valid_targets'],425)
        self.assertEqual(stats['silent_beats'],57)
        self.assertEqual(stats['notes_truncated'],0)
        ds=ChordSequenceDataset({'382':rows},encoder,8)
        self.assertEqual(len(ds),54)
        self.assertEqual(sum(int(ds[i]['targets']['target_loss_mask'].sum()) for i in range(len(ds))),425)
        self.assertEqual(int(ds[-1]['inputs']['beat_attention_mask'].sum()),4)
        for length,expected in ((16,27),(32,14)):
            self.assertEqual(len(ChordSequenceDataset({'382':rows},encoder,length)),expected)
        after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
        self.assertEqual(hashes,after)


if __name__=='__main__':
    unittest.main()
