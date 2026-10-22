import tempfile
import unittest
from pathlib import Path
import numpy as np

from experiments.rahul.src.create_sequences import bar_windows
from experiments.rahul.src.encode_features import FeatureEncoder
from experiments.rahul.src.transformer_dataset import ChordSequenceDataset
from experiments.rahul.tests.test_model_preprocessing import fixture_rows,load_fixture


class BarSequenceTests(unittest.TestCase):
    def test_mixed_meters_stay_in_separate_bars_and_padding_is_masked(self):
        raw=fixture_rows(count=9)
        for i,row in enumerate(raw):
            row['bar_number']=0 if i<3 else 1
            row['beat_position']=i+1 if i<3 else i-2
        with tempfile.TemporaryDirectory() as folder:
            rows=load_fixture(raw,Path(folder))
            tracks={'101':rows}
            encoder=FeatureEncoder.fit(tracks,['101'],max_notes=4,truncate=True)
            dataset=ChordSequenceDataset(tracks,encoder,sequence_length=9,sequence_unit='bars')
            self.assertEqual(len(dataset),2)
            for index,count in enumerate((3,6)):
                item=dataset[index]
                mask=item['inputs']['beat_attention_mask'].numpy()
                np.testing.assert_array_equal(mask,np.arange(9)<count)
                self.assertEqual(len(set(item['inputs']['bar_number'][mask].tolist())),1)
                self.assertFalse(item['targets']['target_loss_mask'][~mask].any())
                self.assertEqual(item['inputs']['melody_pitch_ids'].shape,(9,4))
            self.assertEqual(dataset[1]['metadata']['beat_indices'][:6].tolist(),list(range(3,9)))

    def test_partial_bar_preserves_original_positions(self):
        rows=[dict(track_id='1',beat_index=i,bar_number=4,beat_position=i+2) for i in range(2)]
        self.assertEqual(bar_windows({'1':rows},9)[0]['real_beats'],2)

    def test_missing_metadata_fails_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError,'Missing'):
            bar_windows({'1':[dict(track_id='1',beat_index=0,bar_number=None,beat_position=1)]},9)

    def test_oversized_bar_is_not_truncated(self):
        rows=[dict(track_id='1',beat_index=i,bar_number=0,beat_position=i+1) for i in range(5)]
        with self.assertRaisesRegex(ValueError,'exceeds'):
            bar_windows({'1':rows},4)

    def test_repeated_position_rejected(self):
        rows=[dict(track_id='1',beat_index=i,bar_number=0,beat_position=1) for i in range(2)]
        with self.assertRaisesRegex(ValueError,'duplicate'):
            bar_windows({'1':rows},9)

    def test_full_melody_retains_more_than_four_notes(self):
        raw=fixture_rows(count=4,pitches=(60,61,62,63,64,65,66))
        with tempfile.TemporaryDirectory() as folder:
            rows=load_fixture(raw,Path(folder))
            tracks={'101':rows}
            encoder=FeatureEncoder.fit(tracks,['101'])
            self.assertFalse(encoder.truncate)
            item=ChordSequenceDataset(tracks,encoder,sequence_length=9,sequence_unit='bars')[0]
            self.assertEqual(item['inputs']['melody_note_mask'][0].sum().item(),7)
            self.assertEqual(item['inputs']['melody_pitch_ids'][0].tolist(),[62,63,64,65,66,67,68])
