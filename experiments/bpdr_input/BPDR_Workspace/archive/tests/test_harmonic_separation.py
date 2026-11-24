import unittest
import numpy as np
from bpdr_input.harmonic_separation import separate_source,zero_reconstruction,relative_rmse


class HarmonicSeparationTests(unittest.TestCase):
    def example(self):
        sr=22050;t=np.arange(sr)/sr;phase=2*np.pi*197*t+.37
        coefficients=np.array([.11,.013,.037,-.009,.019,.004])
        basis=np.column_stack([fn(k*phase) for k in (1,2,3) for fn in (np.sin,np.cos)])
        audio=(basis@coefficients).astype(np.float32);support=(t>=.1)&(t<.9)
        prediction=dict(status='estimated',arrays=dict(sample_times=t,estimated_sine_phase=phase,
            retained_support=support,estimated_frequency_hz=np.full(len(t),197.)))
        return audio,sr,prediction,coefficients

    def test_source_only_coefficients_recover_unlabelled_partial_amplitudes_and_phases(self):
        audio,sr,prediction,coefficients=self.example();r=separate_source(audio,sr,prediction)
        self.assertEqual(r['status'],'decomposed');np.testing.assert_allclose(r['coefficients'],coefficients,atol=1e-8)
        self.assertLess(r['harmonic_source_evaluation']['relative_rmse'],1e-6)
        self.assertFalse(r['production_eligible']);self.assertIsNone(r['accepted_audio'])

    def test_prior_refusal_withholds_without_loading_or_fitting_arrays(self):
        r=separate_source(None,None,dict(status='refused'))
        self.assertEqual(r['status'],'withheld');self.assertEqual(r['arrays'],{})

    def test_outside_impulse_stays_in_residual_and_input_is_unchanged(self):
        audio,sr,prediction,_=self.example();audio[20]=.75;original=audio.copy()
        r=separate_source(audio,sr,prediction);a=r['arrays'];outside=~prediction['arrays']['retained_support']
        np.testing.assert_array_equal(a['supported_harmonic_model'][outside],0.)
        np.testing.assert_array_equal(a['residual'][outside],audio[outside])
        np.testing.assert_array_equal(a['zero_edit_bypass'],audio);np.testing.assert_array_equal(audio,original)
        self.assertEqual(a['raw_recombination'].shape,audio.shape)

    def test_evaluation_amplitude_change_does_not_refit_coefficients(self):
        audio,sr,prediction,coefficients=self.example();audio[sr//2:]*=2
        r=separate_source(audio,sr,prediction)
        np.testing.assert_allclose(r['coefficients'],coefficients,atol=1e-8)
        self.assertGreater(r['harmonic_source_evaluation']['relative_rmse'],.49)

    def test_rank_deficiency_refuses_without_regularization(self):
        audio,sr,prediction,_=self.example();prediction['arrays']['estimated_sine_phase'][:]=0.
        r=separate_source(audio,sr,prediction)
        self.assertEqual(r['status'],'refused');self.assertIn('RANK_DEFICIENT_OR_NONFINITE_FIT',r['reasons'])

    def test_changed_grid_or_nonpositive_frozen_frequency_refuses(self):
        for key in ('sample_times','estimated_frequency_hz'):
            audio,sr,prediction,_=self.example()
            prediction['arrays'][key][100]=0. if key.endswith('hz') else 123.
            with self.assertRaises(ValueError):separate_source(audio,sr,prediction)

    def test_arithmetic_recombination_differences_are_not_forced_to_identity(self):
        audio=np.array([1e-30,.1],dtype=np.float32);model=np.array([.1,0.],dtype=float)
        residual,raw,cast,identity=zero_reconstruction(audio,model)
        self.assertNotEqual(cast[0],audio[0]);np.testing.assert_array_equal(identity,audio)
        self.assertEqual(raw[1],float(audio[1]));self.assertEqual(residual[1],float(audio[1]))

    def test_relative_error_uses_reference_energy_without_an_epsilon(self):
        x=np.array([1.,-1.]);mask=np.ones(2,dtype=bool)
        self.assertEqual(relative_rmse(2*x,x,mask)['relative_rmse'],1.)
        with self.assertRaises(ValueError):relative_rmse(x,np.zeros(2),mask)
