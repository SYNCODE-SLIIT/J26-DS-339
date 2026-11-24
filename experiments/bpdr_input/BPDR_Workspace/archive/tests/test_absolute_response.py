import unittest
import torch
from bpdr_input.experiments.learning.absolute_response_training import paired_response, weighted_anchor
from bpdr_input.experiments.learning.response_transformer_training import weighted_anchor as reference_anchor


class AbsoluteResponseTests(unittest.TestCase):
    def test_mse_control_and_preservation_are_unchanged(self):
        changed = torch.tensor([.4, -.6, 0.]); original = torch.tensor([.1, -.1, 0.])
        wanted = torch.tensor([.7, -.9, 0.])
        self.assertTrue(torch.equal(paired_response(changed, original, wanted, 'mse'),
                                    (changed - original - wanted).square().mean()))
        weights = torch.tensor([1., 1., 4.])
        self.assertTrue(torch.equal(weighted_anchor(changed, wanted, weights, 1.),
                                    reference_anchor(changed, wanted, weights, 1.)))

    def test_absolute_response_pushes_correct_relative_direction(self):
        for wanted, predicted in ((.8, .5), (-.8, -.5)):
            original = torch.tensor([0.], requires_grad=True)
            changed = torch.tensor([predicted], requires_grad=True)
            loss = paired_response(changed, original, torch.tensor([wanted]), 'mae')
            loss.backward()
            self.assertEqual(float(changed.grad), -1. if wanted > 0 else 1.)
            self.assertEqual(float(original.grad), 1. if wanted > 0 else -1.)
        x = torch.tensor([.3], requires_grad=True)
        paired_response(x, torch.tensor([0.]), torch.tensor([.3]), 'mae').backward()
        self.assertEqual(float(x.grad), 0.)
        with self.assertRaises(ValueError):
            paired_response(x, x, x, 'tuned')


if __name__ == '__main__':
    unittest.main()
