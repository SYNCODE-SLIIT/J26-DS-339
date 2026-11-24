import unittest
from unittest.mock import patch
import numpy as np

from bpdr_input.noise_preview import attenuate, screen_reference


class NoiseReferenceTests(unittest.TestCase):
    def setUp(self):
        self.sr=22050
        self.noise=np.random.default_rng(17).normal(0,.01,self.sr*3)

    def test_no_declaration_is_bitwise_identity(self):
        output,r=attenuate(self.noise,self.sr,None)
        np.testing.assert_array_equal(output,self.noise)
        self.assertEqual(r['outcome'],'WITHHELD_UNCHANGED')

    def test_hum_misdeclared_as_noise_is_withheld(self):
        t=np.arange(len(self.noise))/self.sr
        hum=.2*np.sin(2*np.pi*220*t)+self.noise*.01
        output,r=attenuate(hum,self.sr,(0,.8))
        np.testing.assert_array_equal(output,hum)
        self.assertFalse(r['reference']['eligible'])

    def test_silence_short_or_invalid_reference_is_withheld(self):
        for audio,interval in [(self.noise,(0,.1)),(self.noise,(-1,.8)),
                               (self.noise,(0,float('nan'))),(np.zeros(self.sr), (0,.8))]:
            self.assertFalse(screen_reference(audio,self.sr,interval)['eligible'])

    def test_known_noise_reduced_without_time_or_pitch_warp(self):
        t=np.arange(len(self.noise))/self.sr
        clean=np.where(t>.9,.15*np.sin(2*np.pi*220*t),0.)
        mixed=clean+self.noise
        out,r=attenuate(mixed,self.sr,(0,.8))
        self.assertEqual(r['outcome'],'UNVERIFIED_NOISE_CANDIDATE')
        self.assertEqual(len(out),len(mixed))
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(np.mean((out-clean)**2),np.mean(self.noise**2))
        # Frequency peak unchanged; the operation does not resample or transpose.
        a=np.fft.rfft(mixed[self.sr:]);b=np.fft.rfft(out[self.sr:])
        self.assertEqual(np.argmax(np.abs(a)),np.argmax(np.abs(b)))

    def test_audit_blocks_voice_pitch_damage_even_when_noise_drops(self):
        from bpdr_input.noise_preview import audit
        n=self.sr*2;audio=self.noise[:n];candidate=audio*.8
        times=np.arange(101)*.02
        f=dict(times_seconds=times)
        with patch('bpdr_input.rule_correction.fine_pitch',side_effect=[np.zeros(101),np.full(101,20.)]), \
             patch('bpdr_input.rule_correction.reliable_frames',return_value=np.ones(101,bool)):
            result=audit(audio,candidate,self.sr,f,f,(0,.7))
        self.assertFalse(result['passes'])
        self.assertIn('PITCH_TAIL_CHANGED',result['reasons'])

    def test_colored_reference_policy_still_withholds_humming(self):
        from bpdr_input.noise_policy_v2 import ColoredNoisePolicy
        t=np.arange(len(self.noise))/self.sr
        hum=.1*np.sin(2*np.pi*220*t)+self.noise*.1
        out,r=attenuate(hum,self.sr,(0.,.7),ColoredNoisePolicy())
        np.testing.assert_array_equal(out,hum)
        self.assertFalse(r['reference']['eligible'])



if __name__=='__main__': unittest.main()
