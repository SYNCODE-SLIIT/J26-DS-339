import unittest
import numpy as np
import torch
from bpdr_input.input_response_diagnosis import feature_stats,group_replacement,blend,response_row,aligned_arrays


def pair():
    a=np.zeros((10,71),dtype=np.float32);a[:,66]=1;c=a.copy();c[2:5,64]=30
    e=np.zeros(10,dtype=np.float32);e[2:5]=30
    return dict(inputs=dict(A=a,C=c),mask=np.ones(10,dtype=bool),detuning=e)


class InputResponseTests(unittest.TestCase):
    def test_invalid_normalization_is_refused(self):
        for scale in [np.zeros(71),np.ones(70),np.full(71,np.nan)]:
            with self.assertRaisesRegex(ValueError,'scales'):feature_stats(dict(source_id='x',dataset='jamendo',cents=30),pair(),scale)

    def test_signed_feature_shift_is_measured_independently_of_prediction(self):
        p=pair();r=feature_stats(dict(source_id='x',dataset='jamendo',cents=30),p,np.ones(71))
        self.assertEqual(r['pitch_delta_mae_cents'],0);self.assertEqual(r['pitch_delta_correct_direction_fraction'],1)
        self.assertEqual(r['pitch_delta_numerical_zero_fraction'],0)
        p['inputs']['C'][:,64]=0;r=feature_stats(dict(source_id='x',dataset='jamendo',cents=30),p,np.ones(71))
        self.assertEqual(r['pitch_delta_mae_cents'],30);self.assertEqual(r['pitch_delta_numerical_zero_fraction'],1)

    def test_group_change_never_mutates_inputs_or_other_columns(self):
        a=torch.zeros(1,10,71);c=torch.ones_like(a);r=group_replacement(a,c,'pitch')
        self.assertTrue(torch.equal(c,torch.ones_like(c)));self.assertEqual(float(r[:,:,64].sum()),0)
        self.assertTrue(torch.equal(r[:,:,:64],c[:,:,:64]));self.assertTrue(torch.equal(r[:,:,65:],c[:,:,65:]))

    def test_interpolation_endpoints_are_bitwise_exact_copies(self):
        a=torch.randn(1,10,71);c=torch.randn_like(a)
        self.assertTrue(torch.equal(blend(a,c,0),a));self.assertTrue(torch.equal(blend(a,c,1),c))
        self.assertTrue(torch.allclose(blend(a,c,.5),(a+c)/2))
        with self.assertRaises(ValueError):blend(a,c,2)

    def test_boundary_negative_control_is_identity_when_metadata_match(self):
        a=torch.randn(1,10,71);c=torch.randn_like(a);c[:,:,70]=a[:,:,70]
        self.assertTrue(torch.equal(group_replacement(a,c,'boundary_negative_control'),c))

    def test_nonfinite_or_misaligned_inputs_are_refused(self):
        p=pair();p['inputs']['C'][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):aligned_arrays(p)
        p=pair();p['mask']=p['mask'][:-1]
        with self.assertRaisesRegex(ValueError,'Misaligned'):aligned_arrays(p)

    def test_response_measurement_does_not_confuse_shift_with_repair(self):
        p=pair();prediction=np.zeros(10);prediction[2:5]=-10
        row=response_row(dict(source_id='x',dataset='jamendo',cents=30),p,prediction,np.zeros(10),np.zeros(10))
        self.assertEqual(row['response_from_A_cents'],-10)
        self.assertEqual(row['plateau_prediction_change_from_C_cents'],10)
        self.assertEqual(row['outside_prediction_change_from_C_cents'],0)
