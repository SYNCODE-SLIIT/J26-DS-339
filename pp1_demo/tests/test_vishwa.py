"""Portable checkpoint and crop-handoff checks; no HumTrans dataset required."""
import unittest
from unittest.mock import patch
import numpy as np
from pp1_demo.config import configure_runtime
configure_runtime()
from pp1_demo.adapters import vishwa

class VishwaTests(unittest.TestCase):
    def test_export_integrity_and_original_training_identity(self):
        m=vishwa.verify_bundle()
        self.assertEqual(m['models']['cnn']['parameters'],18413)
        self.assertEqual(m['models']['cnn_transformer']['parameters'],31117)
        self.assertEqual(m['training_config']['epochs'],5)

    def test_both_checkpoints_predict_finite_13_class_probabilities(self):
        features={'mel_db':np.full((80,257),-40,dtype=np.float32)}
        predictions=vishwa.pilot_predictions(features)
        for model in predictions.values():
            p=np.asarray(model['probabilities'])
            self.assertEqual(p.shape,(257,13))
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
        repeated=vishwa.pilot_predictions(features)
        np.testing.assert_array_equal(predictions['cnn_transformer']['probabilities'],repeated['cnn_transformer']['probabilities'])

    def test_crop_replaces_saved_notes_including_silence_and_crossing_notes(self):
        saved=[{'onset_s':0.,'duration_s':12.,'pitch_midi':60,'source':'game_vocal'}]
        live=[{'onset_s':1.,'duration_s':1.,'pitch_midi':67}]
        notes=vishwa.merge_context(saved,live,4.,4.)
        self.assertEqual([(n['onset_s'],n['duration_s'],n['pitch_midi']) for n in notes],[(0.,4.,60),(5.,1.,67),(8.,4.,60)])
        self.assertEqual(notes[1]['source'],'pyin_vocal')
        self.assertEqual(saved[0]['duration_s'],12.)

    def test_out_of_crop_note_rejected(self):
        with self.assertRaises(ValueError):
            vishwa.merge_context([], [{'onset_s':3.,'duration_s':2.,'pitch_midi':60}],0.,4.)

    def test_bad_checkpoint_digest_fails_closed(self):
        with patch.object(vishwa,'sha',return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'integrity'):
                vishwa.verify_bundle()

if __name__=='__main__':unittest.main()
