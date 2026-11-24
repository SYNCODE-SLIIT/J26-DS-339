import unittest
import torch
from bpdr_input.experiments.learning.tail_preservation import upper_tail_preservation


class TailPreservationTests(unittest.TestCase):
    def test_rare_large_error_is_not_hidden_by_small_negative_rows(self):
        c=torch.tensor([.9]+[.1]*11);a=torch.zeros(12)
        actual=upper_tail_preservation(c,a)
        self.assertAlmostEqual(float(actual),.5*(.9**2+3*.1**2)/4,places=6)
        self.assertGreater(float(actual),float(.5*c.square().mean()))

    def test_sign_permutation_and_constant_class_weight_are_preserved(self):
        c=torch.arange(12,dtype=torch.float32)/12;a=torch.arange(12,dtype=torch.float32)/30
        expected=upper_tail_preservation(c,a)
        self.assertTrue(torch.equal(expected,upper_tail_preservation(-c.flip(0),a.flip(0))))
        self.assertAlmostEqual(float(upper_tail_preservation(torch.full((12,),.2),torch.full((12,),.4))),.1,places=6)

    def test_gradients_focus_on_worst_rows_in_both_classes(self):
        c=(torch.arange(12,dtype=torch.float32)/12).requires_grad_();a=(torch.arange(12,dtype=torch.float32)/120).requires_grad_()
        upper_tail_preservation(c,a).backward()
        for value in [c,a]:
            self.assertTrue(torch.equal(value.grad[:8],torch.zeros(8)))
            self.assertTrue((value.grad[8:]>0).all())
        zeros=torch.zeros(12,requires_grad=True);loss=upper_tail_preservation(zeros,zeros);loss.backward()
        self.assertEqual(float(loss.detach()),0.);self.assertTrue(torch.isfinite(zeros.grad).all())

    def test_invalid_class_size_type_or_nonfinite_values_refused(self):
        good=torch.zeros(12)
        for bad in [torch.zeros(11),torch.zeros(12,dtype=torch.int64),torch.full((12,),float('nan'))]:
            with self.assertRaisesRegex(ValueError,'twelve finite'):upper_tail_preservation(bad,good)


if __name__=='__main__':unittest.main()
