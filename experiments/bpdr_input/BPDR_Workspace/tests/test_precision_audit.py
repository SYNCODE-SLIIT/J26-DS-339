import unittest
from unittest.mock import patch
import numpy as np
import torch
from bpdr_input.repair.precision_audit import coarse_features

class FakePitch:
    reduction='alwa';bins_per_semitone=3;hop_size=20.
    def __call__(self,audio,sr,**kwargs):
        n=len(audio)//441+1;a=torch.zeros((n,384));a[:,207]=1
        c=torch.ones(n);c[0]=.79;c[1]=.8
        return torch.full((n,),440.),c,None,a

class PrecisionAuditTests(unittest.TestCase):
    def test_existing_grid_voicedness_and_double_measurements(self):
        with patch('bpdr_input.repair.precision_audit._pitch_model',return_value=FakePitch()):
            f=coarse_features(np.zeros(22050,np.float32),22050,'.')
        np.testing.assert_array_equal(f['times_seconds'],np.arange(51)*441/22050)
        self.assertFalse(f['voiced_mask'][0]);self.assertTrue(f['voiced_mask'][1])
        self.assertTrue(np.isnan(f['f0_hz'][0]));self.assertEqual(f['f0_hz'][1],440.)
        self.assertEqual(f['f0_hz'].dtype,np.float64)
        self.assertEqual(float(f['confidence'][0]),float(np.float32(.79)))

    def test_unplanned_sample_rate_rejected(self):
        with self.assertRaises(ValueError):coarse_features(np.zeros(16000,np.float32),16000,'.')

if __name__=='__main__':unittest.main()
