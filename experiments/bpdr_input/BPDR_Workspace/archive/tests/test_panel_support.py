"""Numerical guards for the panel's metrics and immutable scoring denominator."""
import unittest
import numpy as np
from bpdr_input.reporting.panel_support import fixed_metrics, paired_reduction
from bpdr_input.modeling.losses import boundary_invariance, counter_detuning, masked_mean_abs, objective
import torch


class PanelMetricTests(unittest.TestCase):
    def test_tolerance_is_inclusive_and_sign_matters(self):
        m=np.array([True,True,True])
        result=fixed_metrics(np.array([-20.,-21.,30.]),np.array([-30.,-30.,-30.]),m)
        self.assertEqual(result['mae_cents'],79/3)
        self.assertAlmostEqual(result['within_tolerance_percent'],200/3)

    def test_missing_scored_predictions_cannot_shrink_denominator(self):
        with self.assertRaises(ValueError):fixed_metrics([np.nan,0.],[0.,0.],np.array([True,True]))
        self.assertEqual(fixed_metrics([np.nan,0.],[0.,0.],np.array([False,True]))['frames'],1)

    def test_empty_or_misaligned_masks_do_not_create_perfect_scores(self):
        for prediction,target,mask in [([0.],[0.],np.array([False])),([0.,0.],[0.],np.array([True])),([0.],[0.],np.array([1]))]:
            with self.assertRaises(ValueError):fixed_metrics(prediction,target,mask)

    def test_relative_reduction_uses_paired_mean_errors(self):
        r=paired_reduction([30.,20.],[10.,20.])
        self.assertEqual(r['cases'],2);self.assertEqual(r['mean_paired_improvement_cents'],10.)
        self.assertAlmostEqual(r['relative_reduction_percent'],40.)

    def test_invalid_baseline_or_missing_pairs_is_explicit(self):
        for a,b in [([],[]),([0.],[0.]),([30.],[1.,2.]),([30.],[np.nan])]:
            with self.assertRaises(ValueError):paired_reduction(a,b)

    def test_zero_model_passes_boundary_but_fails_detuning_response(self):
        zero=torch.zeros((1,4));r={v:zero.clone() for v in 'ABCD'}
        m=torch.ones((1,4),dtype=torch.bool);e=torch.tensor([[0.,0.,20.,0.]])
        self.assertEqual(float(boundary_invariance(r,m)[0]),0.)
        self.assertEqual(float(counter_detuning(r,e,m)[0]),5.)
        r['C']=-e;r['D']=-e
        self.assertEqual(float(counter_detuning(r,e,m)[0]),0.)

    def test_empty_mask_loss_is_zero_and_reports_exclusions(self):
        a=torch.ones((2,3),requires_grad=True);mask=torch.zeros((2,3),dtype=torch.bool)
        loss,skipped=masked_mean_abs(a,a*0,mask)
        self.assertEqual(float(loss.detach()),0.);self.assertEqual(skipped,2)
        loss.backward();self.assertTrue(torch.equal(a.grad,torch.zeros_like(a)))


if __name__=='__main__':unittest.main()
