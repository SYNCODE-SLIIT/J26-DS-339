"""Ensure energy gating cannot invent voiced frames or alter pitch values."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from melody_understanding.audio import PreparedAudio
from melody_understanding.energy import gate_voicing

class EnergyGateTests(unittest.TestCase):
    def test_mask_only_and_silence(self):
        sr=22050
        samples=np.r_[np.zeros(sr),.2*np.sin(2*np.pi*220*np.arange(sr)/sr)]
        audio=PreparedAudio(samples,sr,2.,0.,{})
        times=np.arange(0,len(samples),256)/sr
        voiced=np.ones(len(times),dtype=bool);voiced[-1]=False
        features={'times_s':times,'f0_hz':np.full(len(times),220.),'voiced':voiced.copy()}
        gated,metadata=gate_voicing(features,audio,'relative_20db')
        self.assertFalse(gated['voiced'][times<.9].any())
        self.assertTrue(gated['voiced'][(times>1.1)&(times<1.8)].all())
        self.assertFalse(gated['voiced'][-1])
        np.testing.assert_array_equal(features['voiced'],voiced)
        np.testing.assert_array_equal(gated['f0_hz'],features['f0_hz'])
        self.assertGreater(metadata['removed_voiced_frames'],0)
        with self.assertRaises(ValueError):gate_voicing(features,audio,'unknown')
