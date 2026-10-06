"""Real frozen-model smoke check, independent of injected unit-test pitch."""
import unittest
from pathlib import Path
import numpy as np
from bpdr_input.analysis import pesto_pitch, _pitch_model


class FrozenPitchSmoke(unittest.TestCase):
    def test_frozen_model_hz_and_timestamp_units(self):
        sr = 22050
        audio = (.2*np.sin(2*np.pi*220*np.arange(sr)/sr)).astype('float32')
        cache = str(Path(__file__).resolve().parents[1]/'checkpoints')
        times, f0, confidence = pesto_pitch(audio, sr, cache)
        reliable = (times > .2) & (times < .8) & (confidence >= .8)
        self.assertTrue(reliable.any())
        self.assertGreater(np.median(f0[reliable]), 210)
        self.assertLess(np.median(f0[reliable]), 230)
        self.assertAlmostEqual(times[1]-times[0], .02)
        model = _pitch_model(str(Path(cache).resolve()))
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertIs(model, _pitch_model(str(Path(cache).resolve())))
