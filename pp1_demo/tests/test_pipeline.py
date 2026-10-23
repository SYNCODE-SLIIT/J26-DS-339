"""Checks for time alignment, checkpoint selection and the actual model handoff."""
import unittest
from pathlib import Path
import json
import numpy as np
from pp1_demo.config import configure_runtime
configure_runtime()
from pp1_demo.config import PASAN, APP, RUNTIME
from pp1_demo.contracts import Settings, sha
from pp1_demo.dataset import catalogue, context, crop_rows, verify_audio, track_preview
from pp1_demo.adapters import pasan, rahul


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalogue=catalogue()
        cls.c=context(cls.catalogue[0]['id'])

    def test_library_exposes_available_aligned_vocals(self):
        self.assertGreater(len(self.catalogue),0)
        self.assertNotIn('waveform',self.catalogue[0])
        self.assertTrue(any(t['key']['mode']=='minor' for t in self.catalogue))
        for t in self.catalogue[:6]+self.catalogue[-1:]:
            c=context(t['id']);verify_audio(c)
            self.assertEqual(sorted({b['bar_number'] for b in c['beats']}),list(range(16)))
            self.assertGreaterEqual(c['crop_offset_s'],c['context_start_s'])
            self.assertGreaterEqual(c['context_end_s']+1e-6,c['crop_offset_s']+c['duration_s'])
            self.assertTrue(c['notes'])

    def test_visible_timestamps_preserve_crop_offset(self):
        for t in self.catalogue[:6]+self.catalogue[-1:]:
            c=context(t['id'])
            for n in c['display_notes']:
                self.assertGreaterEqual(n['onset_s'],0)
                self.assertGreater(n['duration_s'],0)
                self.assertLessEqual(n['onset_s']+n['duration_s'],c['duration_s']+1e-7)
            for b in c['display_beats']:
                self.assertGreaterEqual(b['start_s'],0)
                self.assertLessEqual(b['end_s'],c['duration_s'])
                self.assertGreater(b['end_s'],b['start_s'])

    def test_crossing_intervals_are_clipped_and_no_outside_predictions_leak(self):
        rows=[{'start_s':2,'end_s':4,'label':'C:maj'}, {'start_s':4,'end_s':6,'label':'F:maj'}, {'start_s':9,'end_s':10,'label':'G:maj'}]
        self.assertEqual(crop_rows(rows,3,2),[{'start_s':0.,'end_s':1,'label':'C:maj'},{'start_s':1,'end_s':2,'label':'F:maj'}])

    def test_latest_checkpoint_and_real_prediction_contract(self):
        self.assertTrue(PASAN.name.endswith('_latest.pt'))
        c=self.c;settings=Settings(tonic_pc=c['reference_pc'],genre=c['genre'])
        predicted=pasan.predict(c['notes'],c['beats'],settings)
        self.assertEqual(predicted['metadata']['epoch'],9)
        self.assertEqual(predicted['window_bars'],16)
        self.assertEqual(len(predicted['chords']),len(c['beats']))
        refined=rahul.predict(c['notes'],predicted['chords'],settings)
        self.assertEqual(refined['mode'],'major')
        self.assertEqual(len(refined['chords']),len(predicted['chords']))
        for basic,detail in zip(predicted['chords'],refined['chords']):
            self.assertEqual((basic['start_s'],basic['end_s']),(detail['start_s'],detail['end_s']))
            self.assertEqual(len(detail['top3']),3)
            self.assertTrue(np.isfinite([x['probability'] for x in detail['top3']]).all())
        self.assertEqual(refined['input_summary']['full_chord_classes'],402)

    def test_short_context_is_not_silently_used_as_sixteen_bars(self):
        c=self.c
        with self.assertRaisesRegex(ValueError,'sixteen-bar'):
            pasan.predict(c['notes'],c['beats'][:8],Settings(tonic_pc=c['reference_pc']))

    def test_recording_id_cannot_escape_dataset(self):
        for invalid in ['../config.py','track_../x','track_0000382/../secret']:
            with self.assertRaises(ValueError):context(invalid)

    def test_runtime_is_inside_demo(self):
        self.assertTrue(RUNTIME.resolve().is_relative_to(APP.resolve()))
        for name in ['MPLCONFIGDIR','NUMBA_CACHE_DIR','TORCH_HOME']:
            import os
            self.assertTrue(Path(os.environ[name]).resolve().is_relative_to(RUNTIME.resolve()))


if __name__=='__main__':unittest.main()
