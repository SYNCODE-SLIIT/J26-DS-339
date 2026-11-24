"""Scientific accounting, valid-gradient masks and split isolation."""
import unittest

import numpy as np
import torch

from bpdr_input.modeling.correction_accuracy import (assert_independent, improvement_loss,
    prediction_row, summarize)


class CorrectionAccuracyTests(unittest.TestCase):
    def test_identity_and_repair_targets_have_zero_loss(self):
        target=torch.tensor([1.,-2/3,0.])
        predictions=torch.stack([target*0,target*0,target,target],-1)
        loss,_=improvement_loss(predictions,target,arm='baseline',response_active=True)
        self.assertAlmostEqual(loss.item(),0.,places=7)

    def test_zero_output_cannot_pass_detuning_response(self):
        target=torch.ones(3)
        loss,parts=improvement_loss(torch.zeros(3,4),target,arm='baseline',response_active=True)
        self.assertGreater(loss.item(),0.)
        self.assertEqual(parts['response'].item(),1.)

    def test_unlabelled_gap_is_excluded_from_derivative(self):
        target=torch.ones(1)
        pred=torch.zeros(1,4,requires_grad=True)
        neighbours=torch.zeros(1,3,4,requires_grad=True)
        targets=torch.tensor([[1000.,1.,1000.]])
        loss,parts=improvement_loss(pred,target,arm='continuity',response_active=False,
            neighbour_predictions=neighbours,neighbour_targets=targets,
            neighbour_valid=torch.tensor([[False,True,False]]))
        self.assertEqual(parts['gradient'].item(),0.)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_preservation_and_limit_penalties_are_active(self):
        target=torch.zeros(2)
        pred=torch.full((2,4),1.5)
        base,_=improvement_loss(pred,target,arm='baseline',response_active=False)
        preserve,_=improvement_loss(pred,target,arm='preservation',response_active=False,
            original_predictions=torch.ones(4))
        limited,parts=improvement_loss(pred,target,arm='correction_limit',response_active=False)
        self.assertGreater(preserve.item(),base.item())
        self.assertGreater(limited.item(),base.item())
        self.assertAlmostEqual(parts['over_limit'].item(),.25)

    def test_same_singer_or_source_is_rejected(self):
        rows=[dict(group='s2',source=dict(singer_id='s2',sha256='b'))]
        assert_independent(['s1'],['a'],rows)
        with self.assertRaises(ValueError):
            assert_independent(['s2'],['a'],rows)
        with self.assertRaises(ValueError):
            assert_independent(['s1'],['b'],rows)

    def test_singer_and_seed_macros_do_not_count_frames_as_people(self):
        def row(g,seed,error):
            return dict(group=g,seed=seed,source_sha256=g,source_id=g,cents=20.,
                baseline_mae_cents=20.,raw_mae_cents=error,original_abs_cents=1.,
                outside_abs_cents=1.,response_gain=.5,within_10c_percent=50.)
        rows=[row('s1',339,0.),row('s1',340,10.)]+[row('s2',339,20.)]*8
        rows.append(dict(group='s3',seed=339,source_sha256='c',cents=20.,excluded='AUDIT_FAILED'))
        result=summarize(rows)
        self.assertEqual(result['summary']['singers'],2)
        self.assertEqual(result['summary']['excluded_case_evaluations'],1)
        self.assertEqual(result['summary']['raw_mae_cents'],12.5)
        self.assertFalse(result['prediction_gate_passed'])

    def test_prediction_shapes_and_reference_masks_are_checked(self):
        pair=dict(target=np.array([0.,-20.,-20.]),mask=np.ones(3,bool),
            A=dict(reliable=np.ones(3,bool)),group='s',source_id='x',source=dict(sha256='abc'),cents=20.)
        a=np.zeros((3,3)); c=np.tile([0.,-20.,-20.],(3,1))
        result=prediction_row(pair,a,c)
        self.assertEqual(result['raw_mae_cents'],0.)
        self.assertEqual(result['within_10c_percent'],100.)
        with self.assertRaises(ValueError):
            prediction_row(pair,a,c[:1])


if __name__=='__main__':
    unittest.main()
