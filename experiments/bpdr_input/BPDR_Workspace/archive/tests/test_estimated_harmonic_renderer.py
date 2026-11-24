import unittest
from unittest.mock import patch
import numpy as np
from bpdr_input.estimated_harmonic_renderer import render_estimated_candidate
from bpdr_input.harmonic_separation import separate_source


class EstimatedHarmonicRendererTests(unittest.TestCase):
    def example(self):
        sr=22050;t=np.arange(4*sr)/sr;phase=2*np.pi*239*t+.37;f=np.full(len(t),239.)
        audio=(.09*np.sin(phase)+.023*np.cos(2*phase)+.011*np.sin(3*phase)).astype(np.float32)
        prediction=dict(status='estimated',arrays=dict(sample_times=t,estimated_sine_phase=phase,
            estimated_frequency_hz=f,retained_support=(t>=.1)&(t<3.9)))
        separated=separate_source(audio,sr,prediction)
        times=np.arange(len(audio)//441+1)*441/sr;request=np.where((times>=1)&(times<2),-30+60*(times-1),0.)
        return audio,sr,prediction,separated,times,request

    def test_audio_only_closed_edit_preserves_outside_endpoints_and_residual(self):
        audio,sr,p,d,times,request=self.example();source=audio.copy();residual=d['arrays']['residual'].copy()
        candidate,diag,a=render_estimated_candidate(audio,sr,p,d,times,request,[(1.,2.)])
        self.assertIsNotNone(candidate);self.assertEqual(diag['outcome'],'EXPERIMENTAL_CANDIDATE')
        mask=a['inside_support'];np.testing.assert_array_equal(candidate[~mask],audio[~mask])
        self.assertEqual(candidate[mask][0],audio[mask][0]);self.assertEqual(candidate[mask][-1],audio[mask][-1])
        np.testing.assert_array_equal(d['arrays']['residual'],residual);np.testing.assert_array_equal(audio,source)
        np.testing.assert_array_equal(candidate[mask],(a['edited_harmonic_model'][mask]+residual[mask]).astype(np.float32))
        self.assertFalse(diag['oracle_inputs_used']);self.assertFalse(diag['production_eligible']);self.assertIsNone(diag['accepted_audio'])

    def test_zero_request_is_exact_bypass(self):
        audio,sr,p,d,times,_=self.example()
        candidate,diag,_=render_estimated_candidate(audio,sr,p,d,times,np.zeros(len(times)),[])
        np.testing.assert_array_equal(candidate,audio);self.assertTrue(diag['identity'])

    def test_prior_source_refusal_cannot_render(self):
        audio,sr,p,d,times,request=self.example();p['status']='refused'
        candidate,diag,_=render_estimated_candidate(audio,sr,p,d,times,request,[(1.,2.)])
        self.assertIsNone(candidate);self.assertEqual(diag['reasons'],['PRIOR_SOURCE_REFUSAL'])

    def test_missing_strict_solution_withholds_without_relaxed_fallback(self):
        audio,sr,p,d,times,request=self.example()
        with patch('bpdr_input.estimated_harmonic_renderer.solve_support',return_value=(dict(sign_preserving=dict(phase_closed_solution=False)),dict(relaxed_cents=np.ones(22050)))):
            candidate,diag,a=render_estimated_candidate(audio,sr,p,d,times,request,[(1.,2.)])
        self.assertIsNone(candidate);self.assertEqual(diag['reasons'],['NO_STRICT_PHASE_SOLUTION'])
        self.assertFalse(any('relaxed' in k for k in a))

    def test_unreliable_samples_withhold_entire_request(self):
        audio,sr,p,d,times,request=self.example();p['arrays']['retained_support'][30000]=False
        d['arrays']['retained_support']=p['arrays']['retained_support'].copy()
        candidate,diag,_=render_estimated_candidate(audio,sr,p,d,times,request,[(1.,2.)])
        self.assertIsNone(candidate);self.assertEqual(diag['reasons'],['UNSUPPORTED_SOURCE_SAMPLES'])

    def test_changed_source_or_residual_refuses(self):
        for key in ('source','residual'):
            audio,sr,p,d,times,request=self.example()
            if key=='source':audio[30]+=.01
            else:d['arrays']['residual'][30]+=.01
            with self.assertRaises(ValueError):render_estimated_candidate(audio,sr,p,d,times,request,[(1.,2.)])

    def test_over_limit_or_outside_request_is_not_clipped(self):
        for index,value in [(60,31.),(5,1.)]:
            audio,sr,p,d,times,request=self.example();request[index]=value
            with self.assertRaises(ValueError):render_estimated_candidate(audio,sr,p,d,times,request,[(1.,2.)])

    def test_any_missing_support_solution_withholds_all_supports(self):
        audio,sr,p,d,times,_=self.example();request=np.zeros(len(times))
        for start in (1.,2.5):
            m=(times>=start)&(times<start+1);request[m]=-30+60*(times[m]-start)
        from bpdr_input.anchor_feasibility import solve_support
        with patch('bpdr_input.estimated_harmonic_renderer.solve_support',side_effect=[
                solve_support(np.full(22050,239.),np.linspace(-30,30,22050),sr,1764),
                (dict(sign_preserving=dict(phase_closed_solution=False)),{})]):
            candidate,diag,_=render_estimated_candidate(audio,sr,p,d,times,request,[(1.,2.),(2.5,3.5)])
        self.assertIsNone(candidate);self.assertEqual(len(diag['supports']),2)
