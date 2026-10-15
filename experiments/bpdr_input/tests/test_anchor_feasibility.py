import unittest
import numpy as np
from bpdr_input.anchor_feasibility import nearest_integer_in_interval,solve_support


class AnchorFeasibilityTests(unittest.TestCase):
    def test_integer_intervals_and_deterministic_ties(self):
        self.assertIsNone(nearest_integer_in_interval(.2,.8,.5))
        self.assertEqual(nearest_integer_in_interval(1.9,2.3,2.1),2)
        self.assertEqual(nearest_integer_in_interval(-1.,1.,.5),0)
        self.assertEqual(nearest_integer_in_interval(2.,2.,2.),2)
        with self.assertRaises(ValueError):nearest_integer_in_interval(1.,0.,.5)

    def test_constant220_plus20_cannot_close_under_sign_preserving_ramp_budget(self):
        sr=22050;n=sr
        record,_=solve_support(np.full(n,220.),np.full(n,20.),sr,1764)
        self.assertFalse(record['sign_preserving']['necessary_contains_integer'])
        self.assertTrue(record['relaxed']['necessary_contains_integer'])
        self.assertFalse(record['sign_preserving']['phase_closed_solution'])

    def test_closed_solution_is_bounded_and_endpoints_and_interior_are_preserved(self):
        sr=22050;n=sr
        record,arrays=solve_support(np.full(n,220.),np.linspace(-30.,30.,n),sr,1764)
        self.assertTrue(record['sign_preserving']['phase_closed_solution'])
        solution=arrays['sign_preserving_cents'];r=record['sign_preserving']
        self.assertLess(abs(r['phase_residual_cycles']),1e-10)
        np.testing.assert_allclose(solution[1764:-1764],arrays['requested'][1764:-1764],rtol=0,atol=1e-8)
        self.assertEqual(solution[0],0);self.assertEqual(solution[-1],0)
        self.assertTrue(np.all(solution>=np.minimum(0,arrays['requested'])-1e-8))
        self.assertTrue(np.all(solution<=np.maximum(0,arrays['requested'])+1e-8))

    def test_zero_request_has_exact_zero_phase_solution(self):
        record,arrays=solve_support(np.full(22050,220.),np.zeros(22050),22050,1764)
        for name in ('sign_preserving','relaxed'):
            self.assertTrue(record[name]['phase_closed_solution'])
            self.assertEqual(record[name]['selected_integer'],0)
        np.testing.assert_array_equal(arrays['sign_preserving_cents'],0.)

    def test_bad_inputs_refuse_without_clipping(self):
        with self.assertRaises(ValueError):solve_support(np.zeros(22050),np.zeros(22050),22050,1764)
        with self.assertRaises(ValueError):solve_support(np.full(22050,220.),np.full(22050,31.),22050,1764)
        with self.assertRaises(ValueError):solve_support(np.ones(3000),np.zeros(3000),22050,1764)
