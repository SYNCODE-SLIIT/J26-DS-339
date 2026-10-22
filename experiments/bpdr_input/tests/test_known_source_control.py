import unittest
import numpy as np
from bpdr_input.anchor_feasibility import solve_support
from bpdr_input.framewise_fixtures import constructed,signal
from bpdr_input.known_source_control import make_known_control,strict_eligible,amplitude_metrics


class KnownSourceControlTests(unittest.TestCase):
    def setUp(self):
        self.profile=dict(hz=220,kind='rest',seconds=8)
        self.st,self.f0,amp=constructed(self.profile);self.audio=signal(self.f0,amp);self.supports=[]
        for start,end in ((1.,2.),(5.,6.)):
            mask=(self.st>=start)&(self.st<end);curve=np.linspace(-30,30,int(mask.sum()))
            result,a=solve_support(self.f0[mask],curve,22050,1764)
            self.assertTrue(result['sign_preserving']['phase_closed_solution'])
            self.supports.append((start,end,dict(times_seconds=self.st[mask],**a)))

    def test_whole_case_withheld_if_one_support_lacks_strict_solution(self):
        good=dict(sign_preserving=dict(phase_closed_solution=True));bad=dict(sign_preserving=dict(phase_closed_solution=False))
        self.assertFalse(strict_eligible(dict(supports=[])))
        self.assertFalse(strict_eligible(dict(supports=[good,bad])))
        self.assertTrue(strict_eligible(dict(supports=[good,good])))

    def test_constructed_control_preserves_source_rest_endpoints_and_sample_count(self):
        original=self.audio.copy();output,diag=make_known_control(self.profile,self.audio,self.supports)
        inside=np.zeros(len(output),dtype=bool)
        for start,end,_ in self.supports:inside|=(self.st>=start)&(self.st<end)
        np.testing.assert_array_equal(output[~inside],self.audio[~inside])
        np.testing.assert_array_equal(output[(self.st>=2.5)&(self.st<4.5)],0.)
        np.testing.assert_array_equal(self.audio,original);self.assertEqual(output.shape,self.audio.shape)
        self.assertTrue(all(s['endpoints_exact'] for s in diag['supports']))
        self.assertFalse(diag['ordinary_vocal_renderer']);self.assertFalse(diag['production_eligible']);self.assertIsNone(diag['accepted_audio'])

    def test_empty_correction_is_exact_identity(self):
        result,diag=make_known_control(self.profile,self.audio,[])
        np.testing.assert_array_equal(result,self.audio);self.assertTrue(diag['identity'])

    def test_different_waveform_cannot_claim_known_source_oracle(self):
        changed=self.audio.copy();changed[0]+=.001
        with self.assertRaises(ValueError):make_known_control(self.profile,changed,self.supports)

    def test_relaxed_only_solution_cannot_be_used_as_fallback(self):
        start,end,a=self.supports[0];a=dict(a);a.pop('sign_preserving_cents')
        with self.assertRaises(ValueError):make_known_control(self.profile,self.audio,[(start,end,a)])

    def test_phase_mismatch_and_wrong_direction_are_refused(self):
        for key in ('sign_preserving_added_cycles','sign_preserving_cents'):
            start,end,a=self.supports[0];a={k:v.copy() for k,v in a.items()}
            if key.endswith('cycles'):a[key][-1]=.1
            else:a[key][100]=20.
            with self.assertRaises(ValueError):make_known_control(self.profile,self.audio,[(start,end,a)])

    def test_output_amplitude_dip_is_not_excluded_from_diagnostic_support(self):
        source=np.ones(len(self.st),dtype=complex);low=np.zeros(len(self.st),dtype=complex)
        rows,_=amplitude_metrics(source,source,low,self.st,[(1.,2.)],np.ones(len(self.st),dtype=bool))
        self.assertTrue(all(r['control']['minimum_ratio']==0. and r['control']['below_half'] for r in rows))
        self.assertTrue(all(r['baseline']['minimum_ratio']==1. and not r['baseline']['below_half'] for r in rows))
