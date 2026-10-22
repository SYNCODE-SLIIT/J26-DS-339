import unittest
import torch
from bpdr_input.learning_control import balanced_objective, make_control, regions, zero_gradient_check


def batch(length=12, plateau=2):
    e=torch.zeros(1,length);e[:,2:2+plateau]=-30
    return dict(inputs={v:torch.randn(1,length,71) for v in 'ABCD'},detuning=e,
                mask=torch.ones(1,length,dtype=torch.bool),padding=torch.zeros(1,length,dtype=torch.bool),
                voiced={v:torch.ones(1,length,dtype=torch.bool) for v in 'AC'})


class LearningControlTests(unittest.TestCase):
    def test_oracle_cannot_mutate_runtime_inputs(self):
        b=batch();before={v:x.clone() for v,x in b['inputs'].items()};c=make_control(b,True)
        for v in 'ABCD':
            self.assertTrue(torch.equal(before[v],b['inputs'][v]))
            self.assertTrue(torch.equal(c['inputs'][v][:,:,:70],b['inputs'][v][:,:,:70]))
            self.assertTrue(torch.equal(c['inputs'][v][:,:,70],torch.zeros_like(b['detuning']) if v in 'AB' else -b['detuning']/30))

    def test_real_control_is_exact_copy(self):
        b=batch();c=make_control(b,False)
        for v in 'ABCD':self.assertTrue(torch.equal(b['inputs'][v],c['inputs'][v]))

    def test_balancing_is_invariant_to_outside_duration_at_zero(self):
        values=[]
        for n in [12,40]:
            b=batch(n);r={v:torch.zeros_like(b['detuning']) for v in 'ABCD'}
            values.append(float(balanced_objective(r,b)[0]))
        self.assertEqual(values,[22.5,22.5])

    def test_zero_plateau_gradient_has_correct_correction_sign_for_both_injections(self):
        for sign in [-1,1]:
            b=batch();b['detuning']*=sign
            for balanced in [False,True]:
                c=zero_gradient_check(b,balanced)
                self.assertTrue(c['finite_gradients']);self.assertTrue(c['detuned_plateau_descent_sign_correct'])

    def test_invalid_regions_are_rejected(self):
        b=batch();b['detuning'].zero_()
        with self.assertRaisesRegex(ValueError,'plateau'):regions(b)

    def test_perfect_compensation_zeroes_anchor_and_response(self):
        b=batch();r={v:torch.zeros_like(b['detuning']) if v in 'AB' else -b['detuning'] for v in 'ABCD'}
        _,parts=balanced_objective(r,b)
        for key in ['anchor','boundary','detuning']:self.assertEqual(float(parts[key]),0.)
        self.assertGreater(float(parts['smoothness']),0.)
