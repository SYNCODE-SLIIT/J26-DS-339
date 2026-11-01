import unittest
import numpy as np
import torch

from bpdr_input.experiments.learning.support_supervision import block_starts, block_objective


class SupportSupervisionTests(unittest.TestCase):
    def test_labels_must_be_contiguous_known_and_constant(self):
        y=np.r_[np.zeros(5),np.full(50,20.),np.zeros(5)]
        known=np.ones(60,bool)
        np.testing.assert_array_equal(block_starts(y,known),np.arange(5,13))
        known[30]=False
        self.assertEqual(len(block_starts(y,known)),0)
        known[:]=True;y[30]=-20.
        self.assertEqual(len(block_starts(y,known)),0)
        self.assertEqual(len(block_starts(np.full(42,20.),np.ones(42,bool))),0)

    def test_one_gap_cannot_hide_in_same_mean_and_both_signs_work(self):
        target=torch.stack([torch.zeros(43),torch.full((43,),20.)])
        logits=torch.stack([torch.full((43,),-4.),torch.full((43,),4.)])
        good=target.clone()
        bad=target.clone();bad[1,0]=0.;bad[1,1:]+=20./42
        self.assertAlmostEqual(float(good[1].mean()),float(bad[1].mean()),places=4)
        _, m_good=block_objective(good,logits,target)
        _, m_bad=block_objective(bad,logits,target)
        self.assertEqual(float(m_good),0.)
        self.assertGreater(float(m_bad),float(m_good))
        loss,_=block_objective(-good,logits,-target)
        self.assertAlmostEqual(float(loss),float(block_objective(good,logits,target)[0]),places=6)

    def test_false_clean_correction_penalized_and_invalid_block_rejected(self):
        target=torch.stack([torch.zeros(43),torch.full((43,),20.)])
        output=target.clone().requires_grad_();logits=torch.zeros_like(output)
        loss,_=block_objective(output,logits,target)
        wrong=target.clone();wrong[0]=10.
        self.assertGreater(float(block_objective(wrong,logits,target)[0]),float(loss.detach()))
        loss.backward();self.assertTrue(torch.isfinite(output.grad).all())
        target[1,20]=0.
        with self.assertRaises(ValueError):block_objective(output,logits,target)


if __name__=='__main__':unittest.main()
