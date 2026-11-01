import unittest
import torch

from bpdr_input.experiments.learning.response_transformer_training import weighted_anchor


class ResponseWeightingTests(unittest.TestCase):
    def test_unit_weight_exactly_preserves_existing_anchor(self):
        pred = torch.tensor([.2, -.3, .4])
        target = torch.tensor([0., -1., 1.])
        weight = torch.tensor([4., 1., 1.])
        self.assertTrue(torch.equal(weighted_anchor(pred, target, weight, 1.),
                                    ((pred-target).square()*weight).mean()))

    def test_original_gradient_unchanged_and_both_signs_get_correct_response(self):
        target = torch.tensor([0., -1., 1.])
        weight = torch.tensor([4., 1., 1.])
        gradients = []
        for factor in [1., 2.]:
            pred = torch.tensor([.2, -.3, .4], requires_grad=True)
            weighted_anchor(pred, target, weight, factor).backward()
            gradients.append(pred.grad)
        self.assertEqual(float(gradients[0][0]), float(gradients[1][0]))
        torch.testing.assert_close(gradients[1][1:], 2*gradients[0][1:])
        self.assertGreater(float(gradients[1][1]), 0.)
        self.assertLess(float(gradients[1][2]), 0.)

    def test_rejects_negative_reweighting_or_nonfinite_factor(self):
        one = torch.ones(3)
        for factor in [0., .5, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                weighted_anchor(one, one, one, factor)


if __name__ == '__main__':
    unittest.main()
