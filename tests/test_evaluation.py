"""Check metric identity, octave errors and smoothing across rests."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from melody_understanding.evaluation import score_notes, score_frames, notes_with_smoothing

class EvaluationTests(unittest.TestCase):
    def test_perfect_notes_and_octave_error(self):
        ref=[(0., .5, 69)]
        pred=[{'onset_s':0.,'duration_s':.5,'pitch_midi':69}]
        self.assertAlmostEqual(score_notes(ref,pred)['note_f1'],1.)
        self.assertAlmostEqual(score_notes(ref,pred)['note_offset_f1'],1.)
        pred[0]['pitch_midi']=81
        self.assertEqual(score_notes(ref,pred)['note_f1'],0.)
        self.assertEqual(score_notes(ref,[])['note_f1'],0.)

    def test_rest_masking_and_frame_agreement(self):
        f={'times_s':np.array([0.,.1,.2,.3,.4]),
           'f0_hz':np.array([440.,440.,np.nan,440.,440.]),
           'voiced':np.array([True,True,False,True,True]),
           'voiced_probability':np.ones(5)}
        ref=[(0.,.2,69),(.3,.5,69)]
        self.assertEqual(score_frames(ref,f)['midi_frame_pitch_agreement_50c'],1.)
        self.assertEqual(score_frames(ref,f)['midi_frame_overall_agreement_50c'],1.)
        self.assertEqual(len(notes_with_smoothing(f,.5,5)),2)
        with self.assertRaises(ValueError):notes_with_smoothing(f,.5,2)
