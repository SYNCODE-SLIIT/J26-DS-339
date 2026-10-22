"""Batch discovery, endpoint compatibility and disk-assembled NPZ checks."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments.rahul import config
from experiments.rahul.src import build_final_dataset as batch
from experiments.rahul.src.build_synthetic_dataset import DEFAULT_SOURCE,build
from experiments.rahul.src.encode_features import FeatureEncoder
from experiments.rahul.src.transformer_dataset import SavedSequenceDataset
from experiments.rahul.tests.test_model_preprocessing import fixture_rows,load_fixture


class FinalDatasetTests(unittest.TestCase):
    def test_discovery_records_all_missing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            complete=root/'track_0000001'
            for relative in batch.REQUIRED:
                path=complete/relative;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
            (root/'track_0000002').mkdir()
            (root/'other_folder').mkdir()
            records=batch.discover(root)
            self.assertEqual([r['track_id'] for r in records],['1','2'])
            self.assertEqual(records[0]['status'],'candidate')
            self.assertEqual(set(records[1]['missing_files']),set(batch.REQUIRED))

    def test_explicit_endpoint_policy_preserves_raw_source(self):
        # Keep fixture paths within the repo so per-track source provenance remains portable.
        with tempfile.TemporaryDirectory(dir=config.COMPONENT_DIR/'outputs') as temporary:
            root=Path(temporary)
            source=root/'sources/track_0000382'
            for relative in (*batch.REQUIRED,'key.json'):
                target=source/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(DEFAULT_SOURCE/'track_0000382'/relative,target)
            for relative in ('chords/full.csv','roman/full.csv'):
                path=source/relative
                with path.open(encoding='utf-8',newline='') as f:
                    reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
                rows[-1]['end_sec']='211.09'
                with path.open('w',encoding='utf-8',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            with patch.object(config,'OUTPUT_DIR',root):
                with self.assertRaisesRegex(ValueError,'source_intervals'),contextlib.redirect_stdout(io.StringIO()):
                    build('382',source.parent,root/'synthetic_dataset/strict/track_0000382')
                result=batch._build_track('382',source.parent,root/'synthetic_dataset/compatible')
                self.assertEqual(result['status'],'validated',result)
                report_path=root/'synthetic_dataset/compatible/track_0000382/track_0000382_validation_report.json'
                report=json.loads(report_path.read_text(encoding='utf-8'))
                self.assertEqual(report['chord_endpoint_tolerance_sec'],1.0)
                self.assertAlmostEqual(report['chord_endpoint_overrun_sec'],.05)
                self.assertEqual(report['statistics']['final_events'],428)
                # Values are accepted explicitly, never clipped or rewritten in the source.
                self.assertIn('211.09',(source/'chords/full.csv').read_text(encoding='utf-8'))

    def test_invalid_tolerance_rejected(self):
        for value in (-1,2,float('nan')):
            with self.assertRaises(ValueError): build(chord_endpoint_tolerance_sec=value)

    def test_null_selected_key_is_preserved_as_unknown(self):
        with tempfile.TemporaryDirectory(dir=config.COMPONENT_DIR/'outputs') as temporary:
            root=Path(temporary)
            with patch.object(config,'OUTPUT_DIR',root),contextlib.redirect_stdout(io.StringIO()):
                rows,report=build('205076',output=root/'synthetic_dataset/track_0205076',chord_endpoint_tolerance_sec=1.0)
            self.assertTrue(rows)
            self.assertTrue(all(r['key'] is None and r['mode'] is None for r in rows))
            self.assertTrue(all(r['key_confidence']=='insufficient_evidence' for r in rows))
            self.assertIn('key',report['missing_metadata'])

    def test_streamed_archives_preserve_masks_and_unseen_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            train=load_fixture(fixture_rows(track='101',count=3),root)
            heldout=load_fixture(fixture_rows(track='102',count=5,reference='D:maj'),root)
            tracks=batch.TrackFiles(root,['101','102'])
            encoder=FeatureEncoder.fit(tracks,['101'])
            reports={t:dict(statistics=dict(final_events=len(rows))) for t,rows in [('101',train),('102',heldout)]}
            with contextlib.redirect_stdout(io.StringIO()):
                a,_,checks=batch.export_split('train',['101'],tracks,encoder,8,root,reports)
                b,entries,_=batch.export_split('validation',['102'],tracks,encoder,8,root,reports)
            self.assertTrue(checks['all_passed'])
            self.assertEqual(a['loss_eligible_targets'],3)
            self.assertEqual(b['loss_eligible_targets'],0)
            self.assertEqual(b['unseen_targets'],{'D:maj':5})
            self.assertEqual(b['padded_positions'],3)
            dataset=SavedSequenceDataset(root/'validation_sequences.npz')
            self.assertEqual(len(dataset),1)
            self.assertEqual(dataset[0]['metadata']['track_id'],'102')
            self.assertEqual(dataset[0]['metadata']['beat_indices'].tolist(),[0,1,2,3,4,-1,-1,-1])
            self.assertFalse(dataset[0]['targets']['target_loss_mask'].any())
            self.assertEqual(entries[0]['real_beats'],5)
            self.assertFalse(list(root.glob('_train_*')))
            self.assertFalse(list(root.glob('_validation_*')))


if __name__=='__main__': unittest.main()
